"""Read export metadata from the same NorCP target file used by the adapter.

This helper supports the inspected NorCP layout: time/y/x fields, x/y in
projection coordinates, and 2-D lat/lon. It reads coordinates and headers, not
the target field values. Orientation is applied before cropping, just as in
the field loader. Missing projection information or time bounds stay None.

Physical units come from the variable registry AFTER the adapter's unit
conversion: precipitation is a rate in mm day-1, not a six-hour accumulation.
The source standard_name is retained only as provenance because conversion
can change the physical representation (e.g. mass flux to water-depth rate).

NorCP indexing currently uses Python datetimes; this helper keeps that calendar
limitation explicit. It does not add non-Gregorian support to the adapter.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import xarray as xr

from data_adapters.norcp.features import _resolve_file_timestamp, _to_python_datetime
from data_adapters.norcp.regions import CropSpec, validate_crop_within_shape
from data_adapters.norcp.variable_registry import get_source_spec, get_variable_spec
from stride_core.generation.netcdf_output import NetCDFGrid


@dataclass(frozen=True)
class NorCPOutputMetadata:
    """Metadata for one target record; the grid is already oriented and cropped.

    timestamp is the source's representative time (e.g. 03:00 for a 00–06 mean).
    aligned_timestamp is STRIDE's matching key (00:00 with a -3-hour offset).
    time_bounds retain the source interval; neither bound is shifted or inferred.
    """

    grid: NetCDFGrid
    variable_name: str
    units: str
    variable_attributes: dict[str, str]
    timestamp: datetime
    aligned_timestamp: datetime
    calendar: str
    time_bounds: tuple[datetime, datetime] | None


def load_output_metadata(
    file_path: str | Path,
    variable: str,
    source: str,
    aligned_timestamp: datetime,
    *,
    file_time_offset_hours: float = 0.0,
    crop: CropSpec | None = None,
) -> NorCPOutputMetadata:
    """Select the target record and return metadata ready for the output writer."""
    variable_spec = get_variable_spec(variable)
    source_spec = get_source_spec(variable, source)
    file_timestamp = _resolve_file_timestamp(
        aligned_timestamp, offset_hours=file_time_offset_hours,
    )

    with xr.open_dataset(file_path, decode_times=True) as dataset:
        # CF's default is standard when a file does not specify a calendar.
        calendar = dataset.time.encoding.get("calendar", "standard")
        if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError(f"NorCP timestamp indexing does not yet support calendar={calendar!r}")
        record = dataset.sel(time=file_timestamp)
        field = record[source_spec.raw_field_name]
        if field.dims != ("y", "x"):
            raise ValueError(f"Expected NorCP target dimensions (y, x), got {field.dims}")
        for name in ("x", "y", "lat", "lon"):
            if name not in record.coords:
                raise ValueError(f"NorCP output requires source coordinate {name!r}; none will be invented")

        # Named-dimension slicing moves fields and coordinates together.
        record = record.isel(
            y=slice(None, None, -1 if source_spec.flip_ud else 1),
            x=slice(None, None, -1 if source_spec.flip_lr else 1),
        )
        if crop is not None:
            validate_crop_within_shape(crop, full_height=record.sizes["y"], full_width=record.sizes["x"])
            record = record.isel(y=slice(crop.top, crop.bottom), x=slice(crop.left, crop.right))

        # Copy descriptive attributes only: packing and links to source-only
        # bounds variables must not be carried into new coordinate variables.
        coordinate_attributes = {}
        for name in ("x", "y"):
            attrs = record[name].attrs
            if not attrs.get("units"):
                raise ValueError(f"NorCP source coordinate {name!r} has no units")
            coordinate_attributes[name] = {
                key: attrs[key] for key in ("units", "standard_name", "long_name") if key in attrs
            }
        mapping_name = field.attrs.get("grid_mapping")
        grid = NetCDFGrid(
            x=record.x.values.copy(), y=record.y.values.copy(),
            latitude=record.lat.transpose("y", "x").values.copy(),
            longitude=record.lon.transpose("y", "x").values.copy(),
            x_attributes=coordinate_attributes["x"],
            y_attributes=coordinate_attributes["y"],
            grid_mapping=dict(dataset[mapping_name].attrs) if mapping_name else None,
        )

        time_bounds = None
        bounds_name = dataset.time.attrs.get("bounds")
        if bounds_name:
            bounds = record[bounds_name].values
            if bounds.shape != (2,):
                raise ValueError(f"Expected two time bounds, got shape {bounds.shape}")
            time_bounds = (_to_python_datetime(bounds[0]), _to_python_datetime(bounds[1]))

        attributes = {"long_name": variable_spec.long_name, "source_variable": source_spec.raw_field_name}
        for key in ("units", "standard_name"):
            if key in field.attrs:
                attributes[f"source_{key}"] = field.attrs[key]
        if "cell_methods" in field.attrs:
            attributes["cell_methods"] = field.attrs["cell_methods"]

        return NorCPOutputMetadata(
            grid=grid, variable_name=variable_spec.key, units=variable_spec.units,
            variable_attributes=attributes,
            timestamp=_to_python_datetime(record.time.values[()]),
            aligned_timestamp=aligned_timestamp, calendar=calendar, time_bounds=time_bounds,
        )
