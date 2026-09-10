"""Small synthetic NorCP files: check metadata against the real field loader.

Run: python -m unittest test_scripts.test_norcp_output_metadata -v
No source data, model or GPU is required.
"""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from data_adapters.norcp.adapter import NorCPDataset
from data_adapters.norcp.features import load_field_at_timestamp
from data_adapters.norcp.output_metadata import load_output_metadata
from data_adapters.norcp.regions import CropSpec, maybe_crop_2d
from data_adapters.norcp.variable_registry import VARIABLE_REGISTRY
from stride_core.generation.netcdf_output import NetCDFWriter


class NorCPOutputMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "target.nc"
        self.aligned_time = datetime(2024, 2, 29)
        self.pixels = np.arange(20, dtype="f4").reshape(4, 5)
        self.x = np.arange(5) * 1000.
        self.y = np.array([4000., 3000., 2000., 1000.])
        self.latitude = 50. + self.pixels / 100.
        self.longitude = 10. + self.pixels / 100.
        self.source = xr.Dataset(
            data_vars={
                "pr": (("time", "y", "x"), np.stack([self.pixels, self.pixels + 20]) / 86400., {
                    "units": "kg m-2 s-1", "standard_name": "precipitation_flux",
                    "cell_methods": "time: mean", "grid_mapping": "Lambert_Conformal",
                }),
                "tas": (("time", "y", "x"), np.full((2, 4, 5), 280., dtype="f4"), {
                    "units": "K", "standard_name": "air_temperature", "cell_methods": "time: point",
                }),
                "Lambert_Conformal": ((), 0, {
                    "grid_mapping_name": "lambert_conformal_conic",
                    "standard_parallel": 62.2, "longitude_of_central_meridian": 10.,
                    "latitude_of_projection_origin": 62.2, "earth_radius": 6371220.,
                }),
                "time_bnds": (("time", "bnds"), [[0., 6.], [6., 12.]], {
                    "units": "hours since 2024-02-29 00:00:00", "calendar": "standard",
                }),
            },
            coords={
                "time": ("time", [3., 9.], {
                    "units": "hours since 2024-02-29 00:00:00", "calendar": "standard",
                    "bounds": "time_bnds",
                }),
                "x": ("x", self.x, {"units": "m", "standard_name": "projection_x_coordinate"}),
                "y": ("y", self.y, {"units": "m", "standard_name": "projection_y_coordinate"}),
                "lat": (("y", "x"), self.latitude, {"units": "degrees_north"}),
                "lon": (("y", "x"), self.longitude, {"units": "degrees_east"}),
            },
        )
        self.source.to_netcdf(self.path)

    def metadata(self, **overrides):
        options = dict(variable="prcp", source="NORCP_HR", aligned_timestamp=self.aligned_time,
                       file_time_offset_hours=-3.)
        options.update(overrides)
        return load_output_metadata(self.path, **options)

    def test_source_time_bounds_and_canonical_rate_units(self):
        for offset in (0, 6):
            aligned = self.aligned_time + timedelta(hours=offset)
            metadata = self.metadata(aligned_timestamp=aligned)
            self.assertEqual(metadata.aligned_timestamp, aligned)
            self.assertEqual(metadata.timestamp, aligned + timedelta(hours=3))
            self.assertEqual(metadata.time_bounds, (aligned, aligned + timedelta(hours=6)))
            self.assertEqual(metadata.calendar, "standard")
            self.assertEqual(metadata.variable_name, "prcp")
            self.assertEqual(metadata.units, "mm day-1")
            self.assertEqual(metadata.variable_attributes["source_units"], "kg m-2 s-1")
            self.assertEqual(metadata.variable_attributes["source_standard_name"], "precipitation_flux")
            self.assertEqual(metadata.variable_attributes["cell_methods"], "time: mean")
            self.assertNotIn("standard_name", metadata.variable_attributes)
            self.assertNotIn("units", metadata.variable_attributes)  # Writer owns converted units.
            np.testing.assert_array_equal(metadata.grid.x, self.x)
            np.testing.assert_array_equal(metadata.grid.y, self.y)
            np.testing.assert_array_equal(metadata.grid.latitude, self.latitude)
            self.assertEqual(metadata.grid.grid_mapping, self.source.Lambert_Conformal.attrs)

    def test_coordinates_follow_field_orientation_then_crop(self):
        crop = CropSpec(top=1, left=1, height=2, width=3)
        sources = VARIABLE_REGISTRY["prcp"].sources
        original_spec = sources["NORCP_HR"]
        for flip_ud, flip_lr in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(flip_ud=flip_ud, flip_lr=flip_lr), patch.dict(sources, {
                "NORCP_HR": replace(original_spec, flip_ud=flip_ud, flip_lr=flip_lr),
            }):
                metadata = self.metadata(crop=crop)
                field = load_field_at_timestamp(
                    self.path, "prcp", "NORCP_HR", self.aligned_time, file_time_offset_hours=-3.,
                )
                physical = NorCPDataset._maybe_crop_2d_field(field.array, crop)
                # Each pixel's latitude encodes its original target value, so
                # this catches flips/crops that preserve shape but move locations.
                np.testing.assert_allclose(metadata.grid.latitude, 50. + physical / 100.)
                np.testing.assert_allclose(metadata.grid.longitude, 10. + physical / 100.)
                np.testing.assert_array_equal(metadata.grid.x, (self.x[::-1] if flip_lr else self.x)[1:4])
                np.testing.assert_array_equal(metadata.grid.y, (self.y[::-1] if flip_ud else self.y)[1:3])

    def test_temperature_units_and_absent_optional_metadata(self):
        self.source.time.attrs.pop("bounds")
        self.source.time.attrs.pop("calendar")  # CF default; no custom guess.
        self.source = self.source.drop_vars("time_bnds")
        self.source.to_netcdf(self.path)
        metadata = self.metadata(variable="temp", aligned_timestamp=self.aligned_time + timedelta(hours=3),
                                 file_time_offset_hours=0.)
        self.assertEqual(metadata.timestamp, metadata.aligned_timestamp)
        self.assertEqual(metadata.units, "degC")
        self.assertEqual(metadata.variable_attributes["source_units"], "K")
        self.assertEqual(metadata.variable_attributes["cell_methods"], "time: point")
        self.assertEqual(metadata.calendar, "standard")
        self.assertIsNone(metadata.time_bounds)
        self.assertIsNone(metadata.grid.grid_mapping)

    def test_missing_coordinates_or_units_are_not_invented(self):
        missing_lat = self.source.drop_vars("lat")
        missing_units = self.source.copy(deep=True)
        missing_units.x.attrs.pop("units")
        for source in (missing_lat, missing_units):
            with self.subTest(coords=list(source.coords)):
                source.to_netcdf(self.path)
                with self.assertRaisesRegex(ValueError, "coordinate"):
                    self.metadata()

    def test_non_gregorian_calendar_is_explicitly_outside_adapter_scope(self):
        self.source.time.attrs["calendar"] = "360_day"
        self.source.time_bnds.attrs["calendar"] = "360_day"
        self.source.to_netcdf(self.path)
        with self.assertRaisesRegex(ValueError, "timestamp indexing does not yet support"):
            self.metadata()

    def test_dataset_method_uses_target_file_offset_and_fixed_crop(self):
        # Isolate the new method from index/statistics setup, using its real
        # file selection and metadata loading code.
        dataset = NorCPDataset.__new__(NorCPDataset)
        timestamp = self.aligned_time.isoformat()
        dataset.timestamps = [timestamp]
        dataset._sample_lookup = {timestamp: {"target_files": {"prcp": self.path}}}
        dataset.target_variable = "prcp"
        dataset.target_source = "NORCP_HR"
        dataset.target_time_offsets = {"prcp": -3.}
        dataset.hr_crop_spec = CropSpec(1, 1, 2, 3)
        dataset.spatial_shuffle = False
        metadata = dataset.get_output_metadata(0)
        np.testing.assert_array_equal(metadata.grid.latitude, self.latitude[1:3, 1:4])
        self.assertEqual(metadata.timestamp, self.aligned_time + timedelta(hours=3))
        dataset.spatial_shuffle = True
        with self.assertRaisesRegex(ValueError, "fixed grid"):
            dataset.get_output_metadata(0)

    def test_metadata_and_loaded_physical_field_round_trip_through_writer(self):
        crop = CropSpec(1, 1, 2, 3)
        metadata = self.metadata(crop=crop)
        field = load_field_at_timestamp(
            self.path, "prcp", "NORCP_HR", self.aligned_time, file_time_offset_hours=-3.,
        )
        physical = maybe_crop_2d(field.array, crop)
        with NetCDFWriter(
            self.root / "output", grid=metadata.grid, variable_name=metadata.variable_name,
            units=metadata.units, variable_attributes=metadata.variable_attributes,
            member_count=1, calendar=metadata.calendar, with_time_bounds=True,
        ) as writer:
            writer.append(metadata.timestamp, time_bounds=metadata.time_bounds,
                          members=physical[None], ensemble_mean=physical, pmm=physical)
        with xr.open_dataset(writer.paths["ensemble_members"]) as output:
            np.testing.assert_array_equal(output.prcp[0, 0], physical)
            np.testing.assert_array_equal(output.lat, self.latitude[1:3, 1:4])
            self.assertEqual(output.prcp.units, "mm day-1")
            np.testing.assert_array_equal(output.time, [np.datetime64("2024-02-29T03:00:00", "ns")])
            np.testing.assert_array_equal(output.time_bounds,
                np.array([["2024-02-29T00:00:00", "2024-02-29T06:00:00"]], dtype="datetime64[ns]"))


if __name__ == "__main__":
    unittest.main()
