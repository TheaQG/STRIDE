"""Single-process, incremental NetCDF output for one physical variable/grid.

The caller supplies physical members [member, y, x], mean [y, x], and PMM
[y, x]. This module neither computes products nor transforms/regrids values.
Float32 is the default output precision; choose dtype="f8" for float64 inputs.
NaNs/masked values are stored as missing data; infinities are rejected.

Use a NEW, dedicated output directory (e.g. generation/netcdf). Files are
written inside its sibling '<name>.incomplete' directory. A successful context
exit or finalize() closes all files and renames the directory, publishing the
three products and a completion manifest together. close() WITHOUT finalize()
leaves incomplete output. Existing destinations/staging directories are never
reused. There is no resume, concurrent writing, or NPZ deletion in this version.

Grid coordinates, units, calendar and scientific attributes come from the
caller/source dataset; this writer does not certify their scientific accuracy
or infer an accumulation period. with_time_bounds=True requires interval bounds
on every append. Timestamps must be increasing datetime or cftime objects.

Example (grid and arrays supplied by the caller)::

    with NetCDFWriter(output_dir, variable_name="prcp", units="mm",
                      grid=grid, member_count=16,
                      calendar="proleptic_gregorian") as writer:
        writer.append(timestamp, members=members, ensemble_mean=mean, pmm=pmm)

NetCDF API reference: https://unidata.github.io/netcdf4-python/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import os
import re

import cftime
import netCDF4 as nc
import numpy as np


PRODUCTS = ("ensemble_members", "ensemble_mean", "pmm")
_ENCODING_ATTRIBUTES = {
    "_FillValue", "missing_value", "scale_factor", "add_offset", "_Unsigned",
    "valid_min", "valid_max", "valid_range", "least_significant_digit",
}


@dataclass(frozen=True)
class NetCDFGrid:
    """Actual output-grid coordinates, already cropped/oriented like the fields.

    x/y must be monotonic 1-D coordinates. Latitude/longitude may be either
    lat(y), lon(x), or 2-D arrays (y, x). x_attributes/y_attributes must include
    units; optional standard names and projection parameters are caller-owned.
    """

    x: np.ndarray
    y: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    x_attributes: Mapping[str, Any]
    y_attributes: Mapping[str, Any]
    grid_mapping: Mapping[str, Any] | None = None


def _attributes(values: Mapping[str, Any], reserved: set[str]) -> dict[str, Any]:
    """Copy simple NetCDF attributes; prohibit metadata that packs/masks data."""
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or key in reserved:
            raise ValueError(f"Invalid or writer-owned attribute: {key!r}")
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        array = np.asarray(value)
        if array.ndim > 1 or not array.size or array.dtype.kind not in "iufSU":
            raise ValueError(f"Attribute {key!r} must be text or numeric; serialize nested metadata as JSON")
        result[key] = array.item() if array.ndim == 0 else array.copy()
    return result


def _require_text(value: Any, name: str) -> str:
    """Ensure the value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


class NetCDFWriter:
    """Writer for NetCDF output files. Handles creation and management of NetCDF datasets with specified grids, variables, and attributes."""
    def __init__(
        self, output_dir: str | Path, *, variable_name: str, units: str,
        grid: NetCDFGrid, member_count: int, calendar: str,
        time_units: str = "seconds since 1970-01-01 00:00:00",
        with_time_bounds: bool = False, expected_timesteps: int | None = None,
        dtype: str = "f4", compression_level: int = 1,
        variable_attributes: Mapping[str, Any] | None = None,
        global_attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize the NetCDFWriter with the specified configuration.

        Args:
            output_dir (str | Path): Directory to store the output NetCDF files.
            variable_name (str): Name of the primary variable in the NetCDF files.
            units (str): Units of the primary variable.
            grid (NetCDFGrid): Grid definition for the NetCDF files.
            member_count (int): Number of ensemble members.
            calendar (str): Calendar type for time coordinates.
            time_units (str, optional): Units for the time coordinate. Defaults to "seconds since 1970-01-01 00:00:00".
            with_time_bounds (bool, optional): Whether to include time bounds. Defaults to False.
            expected_timesteps (int | None, optional): Expected number of time steps. Defaults to None.
            dtype (str, optional): Data type for the primary variable. Defaults to "f4".
            compression_level (int, optional): Compression level for NetCDF files. Defaults to 1.
            variable_attributes (Mapping[str, Any] | None, optional): Additional attributes for the primary variable. Defaults to None.
            global_attributes (Mapping[str, Any] | None, optional): Global attributes for the NetCDF files. Defaults to None.
        """
        self.output_dir = Path(output_dir).absolute()
        if not self.output_dir.name:
            raise ValueError("output_dir must name a new, dedicated output directory")
        self.incomplete_dir = self.output_dir.with_name(self.output_dir.name + ".incomplete")
        self.paths = {product: self.output_dir / f"{product}.nc" for product in PRODUCTS}
        reserved_names = {"time", "member", "y", "x", "lat", "lon", "crs", "time_bounds", "nv"}
        _require_text(variable_name, "variable_name")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable_name) or variable_name in reserved_names:
            raise ValueError("variable_name must be a non-reserved NetCDF identifier")
        self.variable_name = variable_name
        self.units = _require_text(units, "units")
        for name, value in (("member_count", member_count), ("expected_timesteps", expected_timesteps)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if member_count is None:
            raise ValueError("member_count is required")
        self.member_count = int(member_count)
        self.expected_timesteps = None if expected_timesteps is None else int(expected_timesteps)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("f4"), np.dtype("f8")):
            raise ValueError("dtype must be f4 (float32) or f8 (float64)")
        if isinstance(compression_level, bool) or not isinstance(compression_level, int) or not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be an integer from 0 to 9")
        self.compression_level = compression_level
        if not isinstance(with_time_bounds, bool):
            raise ValueError("with_time_bounds must be a boolean")
        self.with_time_bounds = with_time_bounds
        reference_time = cftime.datetime(2000, 1, 1, calendar=_require_text(calendar, "calendar"))
        self.calendar = reference_time.calendar  # Normalize aliases such as 365_day.
        self.time_units = _require_text(time_units, "time_units")
        nc.date2num(reference_time, units=self.time_units, calendar=self.calendar)
        self._variable_attributes = _attributes(
            variable_attributes or {}, _ENCODING_ATTRIBUTES | {"units", "coordinates", "grid_mapping"},
        )
        self._global_attributes = _attributes(
            global_attributes or {}, {"stride_product", "stride_output_space", "written_timesteps"},
        )
        self._coordinates, self._grid_mapping = self._prepare_grid(grid)
        self._shape = (len(self._coordinates["y"][1]), len(self._coordinates["x"][1]))
        self._files: dict[str, nc.Dataset] = {}
        self._closed = False
        self.completed = False
        self.timesteps_written = 0
        self._last_time: float | None = None

        if os.path.lexists(self.output_dir) or os.path.lexists(self.incomplete_dir):
            raise FileExistsError(f"Output or incomplete output already exists for {self.output_dir}")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.incomplete_dir.mkdir()  # Exclusive ownership; also blocks a second writer.
        try:
            for product in PRODUCTS:
                dataset = nc.Dataset(self.incomplete_dir / f"{product}.nc", "w", clobber=False, format="NETCDF4")
                self._files[product] = dataset
                self._initialize_file(dataset, product)
        except BaseException:
            self._abort()
            raise

    @staticmethod
    def _prepare_grid(grid: NetCDFGrid) -> tuple[dict, dict]:
        """
        Prepare the grid coordinates and mapping for the NetCDF files.

        Args:
            grid (NetCDFGrid): Grid definition containing x, y, latitude, longitude, and grid mapping information.
        Returns:
            tuple[dict, dict]: A tuple containing the coordinates dictionary and the grid mapping dictionary.
        Raises:
            ValueError: If the grid coordinates are not valid (e.g., non-monotonic, non-finite, or out of bounds).
        """
        coordinates = {}
        for name, values, attributes in (("x", grid.x, grid.x_attributes), ("y", grid.y, grid.y_attributes)):
            array = np.ma.asarray(values, dtype="f8").filled(np.nan).copy()
            if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                raise ValueError(f"{name} coordinates must be finite, non-empty and 1-D")
            delta = np.diff(array)
            if array.size > 1 and not (np.all(delta > 0) or np.all(delta < 0)):
                raise ValueError(f"{name} coordinates must be strictly monotonic")
            attrs = _attributes(attributes, _ENCODING_ATTRIBUTES)
            _require_text(attrs.get("units"), f"{name} coordinate units")
            if attrs.get("axis", name.upper()) != name.upper():
                raise ValueError(f"Incorrect axis attribute for {name}")
            coordinates[name] = ((name,), array, {**attrs, "axis": name.upper()})
        shape = (len(coordinates["y"][1]), len(coordinates["x"][1]))
        for name, values, axis, units, standard_name in (
            ("lat", grid.latitude, "y", "degrees_north", "latitude"),
            ("lon", grid.longitude, "x", "degrees_east", "longitude"),
        ):
            array = np.ma.asarray(values, dtype="f8").filled(np.nan).copy()
            dimensions = (axis,) if array.ndim == 1 else ("y", "x")
            expected = (len(coordinates[axis][1]),) if array.ndim == 1 else shape
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"{name} must match the output grid and contain finite coordinates")
            if name == "lat" and np.any(np.abs(array) > 90):
                raise ValueError("Latitude must be between -90 and 90 degrees")
            coordinates[name] = (dimensions, array, {"units": units, "standard_name": standard_name})
        mapping = _attributes(grid.grid_mapping or {}, _ENCODING_ATTRIBUTES)
        if grid.grid_mapping is not None:
            _require_text(mapping.get("grid_mapping_name"), "grid_mapping_name")
        return coordinates, mapping

    def _initialize_file(self, dataset: nc.Dataset, product: str) -> None:
        """
        Initialize the NetCDF file with the necessary dimensions, variables, and attributes.

        Args:
            dataset (nc.Dataset): The NetCDF dataset to initialize.
            product (str): The type of product (e.g., "ensemble_members") to configure the dataset for.
        """
        dataset.setncatts({
            **self._global_attributes, "stride_product": product,
            "stride_output_space": "physical", "written_timesteps": 0,
        })
        dataset.createDimension("time", None)
        dataset.createDimension("y", self._shape[0])
        dataset.createDimension("x", self._shape[1])
        time = dataset.createVariable("time", "f8", ("time",))
        time.setncatts({"standard_name": "time", "axis": "T", "units": self.time_units, "calendar": self.calendar})
        if self.with_time_bounds:
            dataset.createDimension("nv", 2)
            bounds = dataset.createVariable("time_bounds", "f8", ("time", "nv"))
            bounds.setncatts({"units": self.time_units, "calendar": self.calendar})
            time.bounds = "time_bounds"
        for name, (dimensions, values, attrs) in self._coordinates.items():
            coordinate = dataset.createVariable(name, "f8", dimensions, fill_value=False)
            coordinate.setncatts(attrs)
            coordinate[:] = values
        dimensions = ("time", "y", "x")
        chunks = (1, min(256, self._shape[0]), min(256, self._shape[1]))
        if product == "ensemble_members":
            dataset.createDimension("member", self.member_count)
            member = dataset.createVariable("member", "i4", ("member",))
            member.setncatts({"standard_name": "realization", "long_name": "ensemble member index"})
            member[:] = np.arange(self.member_count, dtype="i4")
            dimensions = ("time", "member", "y", "x")
            chunks = (1, 1, *chunks[1:])
        variable = dataset.createVariable(
            self.variable_name, self.dtype, dimensions, fill_value=np.nan,
            zlib=self.compression_level > 0, complevel=self.compression_level,
            shuffle=True, chunksizes=chunks,
        )
        variable.setncatts({**self._variable_attributes, "units": self.units, "coordinates": "lat lon"})
        if self._grid_mapping:
            crs = dataset.createVariable("crs", "i4")
            crs.setncatts(self._grid_mapping)
            crs.assignValue(0)
            variable.grid_mapping = "crs"
        dataset.sync()

    def _encode_time(self, timestamp: datetime | cftime.datetime) -> float:
        """
        Encode a timestamp into a numeric value suitable for the NetCDF time coordinate.

        Args:
            timestamp (datetime | cftime.datetime): The timestamp to encode.
        Returns:
            float: The encoded time value.
        Raises:
            ValueError: If the timestamp's calendar does not match the writer's calendar or if the encoded value is not finite.
            TypeError: If the timestamp is not a datetime or cftime.datetime instance.
        """
        if isinstance(timestamp, cftime.datetime):
            if timestamp.calendar != self.calendar:
                raise ValueError("Timestamp calendar does not match the writer's calendar")
        elif isinstance(timestamp, datetime):
            if self.calendar not in {"standard", "proleptic_gregorian"}:
                raise ValueError("Use a matching cftime datetime for this calendar")
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            raise TypeError("timestamp must be a datetime or cftime.datetime, not a date label/string")
        value = float(nc.date2num(timestamp, units=self.time_units, calendar=self.calendar))
        if not np.isfinite(value):
            raise ValueError("Timestamp must encode to a finite time coordinate")
        return value

    def _data_array(self, values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        """
        Validate and convert an array to the appropriate data type for NetCDF output.

        Args:
            values (np.ndarray): The array of values to validate and convert.
            shape (tuple[int, ...]): The expected shape of the array.
            name (str): The name of the array (used for error messages).
        Returns:
            np.ndarray: The validated and converted array.
        Raises:
            ValueError: If the array does not match the expected shape or contains invalid values.
        """
        array = np.asanyarray(values)
        if array.shape != shape or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real numeric array with shape {shape}, got {array.shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.ma.asarray(array, dtype=self.dtype).filled(np.nan)
        if np.isinf(array).any():
            raise ValueError(f"{name} contains infinity or overflows {self.dtype}")
        return array

    def append(
        self, timestamp: datetime | cftime.datetime, *, members: np.ndarray,
        ensemble_mean: np.ndarray, pmm: np.ndarray,
        time_bounds: tuple[datetime | cftime.datetime, datetime | cftime.datetime] | None = None,
    ) -> None:
        """Validate all three fields before writing one aligned timestep.

        Validation errors leave the writer usable. An I/O error closes it and
        leaves staging output incomplete; partial records cannot be resumed.
        """
        self._require_open()
        if self.expected_timesteps is not None and self.timesteps_written >= self.expected_timesteps:
            raise ValueError("Cannot exceed expected_timesteps")
        time_value = self._encode_time(timestamp)
        if self._last_time is not None and time_value <= self._last_time:
            raise ValueError("Timestamps must be strictly increasing; duplicates are not allowed")
        bounds = None
        if self.with_time_bounds:
            if time_bounds is None or len(time_bounds) != 2:
                raise ValueError("Exactly two time_bounds are required for every timestep")
            bounds = tuple(self._encode_time(value) for value in time_bounds)
            if not bounds[0] <= time_value <= bounds[1] or bounds[0] >= bounds[1]:
                raise ValueError("time_bounds must be ordered and contain the timestamp")
        elif time_bounds is not None:
            raise ValueError("Set with_time_bounds=True to write time_bounds")
        arrays = {
            "ensemble_members": self._data_array(members, (self.member_count, *self._shape), "members"),
            "ensemble_mean": self._data_array(ensemble_mean, self._shape, "ensemble_mean"),
            "pmm": self._data_array(pmm, self._shape, "pmm"),
        }
        try:
            for product, array in arrays.items():
                self._write_record(self._files[product], array, time_value, bounds)
        except BaseException:
            self._abort()
            raise
        self.timesteps_written += 1
        self._last_time = time_value

    def _write_record(self, dataset: nc.Dataset, array: np.ndarray, time_value: float, bounds: tuple | None) -> None:
        """
        Write a single record to the NetCDF dataset.

        Args:
            dataset (nc.Dataset): The NetCDF dataset to write to.
            array (np.ndarray): The data array to write.
            time_value (float): The encoded time value for the record.
            bounds (tuple | None): The time bounds for the record, if applicable.
        """
        index = self.timesteps_written
        dataset.variables[self.variable_name][index] = array
        dataset.variables["time"][index] = time_value
        if bounds is not None:
            dataset.variables["time_bounds"][index] = bounds
        dataset.sync()

    def _require_open(self) -> None:
        """Ensure that the NetCDF writer is open."""
        if self._closed:
            raise RuntimeError("NetCDF writer is closed")

    def close(self) -> None:
        """Close all handles without publishing output. Safe to call repeatedly."""
        files, self._files = self._files, {}
        self._closed = True
        error = None
        for dataset in files.values():
            try:
                dataset.close()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def _abort(self) -> None:
        """Abort the NetCDF writing process, attempting to close all open datasets."""
        try:
            self.close()
        except Exception:
            pass  # Preserve the original failure after attempting every close.

    def finalize(self) -> dict[str, Path]:
        """Publish the complete directory; reject empty or short expected runs."""
        if self.completed:
            return dict(self.paths)
        self._require_open()
        try:
            if not self.timesteps_written:
                raise ValueError("Cannot finalize an empty NetCDF output")
            if self.expected_timesteps is not None and self.timesteps_written != self.expected_timesteps:
                raise ValueError("Written timestep count does not match expected_timesteps")
            for dataset in self._files.values():
                dataset.setncattr("written_timesteps", self.timesteps_written)
            self.close()
            manifest = {
                "schema_version": 1, "status": "complete", "timesteps": self.timesteps_written,
                "variable": self.variable_name, "units": self.units, "calendar": self.calendar,
                "time_units": self.time_units, "member_count": self.member_count,
                "files": {product: path.name for product, path in self.paths.items()},
            }
            with (self.incomplete_dir / "netcdf_manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
            if os.path.lexists(self.output_dir):
                raise FileExistsError(f"Refusing to replace {self.output_dir}")
            self.incomplete_dir.rename(self.output_dir)
            self.completed = True
        except BaseException:
            self._abort()
            raise
        return dict(self.paths)

    def __enter__(self) -> NetCDFWriter:
        """Enter the context manager, ensuring the NetCDF writer is open."""
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager, finalizing or aborting the NetCDF writer as appropriate."""
        if exc_type is None and not self._closed:
            self.finalize()
        else:
            self._abort()
