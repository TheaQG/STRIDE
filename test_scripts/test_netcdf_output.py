"""CPU-only NetCDF writer checks; no model, checkpoint or source data needed.

Run from the repository root:
    python -m unittest test_scripts.test_netcdf_output -v
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import cftime
import netCDF4 as nc
import numpy as np
import xarray as xr

from stride_core.generation.netcdf_output import NetCDFGrid, NetCDFWriter, PRODUCTS


class NetCDFOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name) / "netcdf"
        self.grid = NetCDFGrid(
            x=np.array([1000., 2000., 3000.]),
            y=np.array([2000., 1000.]),  # Preserve descending spatial orientation.
            latitude=np.array([[56.1, 56.1, 56.1], [56., 56., 56.]]),
            longitude=np.array([[14., 14.1, 14.2], [14., 14.1, 14.2]]),
            x_attributes={"units": "m", "standard_name": "projection_x_coordinate"},
            y_attributes={"units": "m", "standard_name": "projection_y_coordinate"},
            grid_mapping={
                "grid_mapping_name": "lambert_conformal_conic",
                "standard_parallel": np.array([55., 65.]),
                "longitude_of_central_meridian": 15.,
                "latitude_of_projection_origin": 56.,
            },
        )
        self.timestamp = datetime(2024, 2, 29)
        self.members = np.arange(12, dtype="f4").reshape(2, 2, 3) / 10
        # Deliberately unlike aggregates of members: the writer must not
        # recompute products, invert transforms, clip, flip or regrid them.
        self.mean = np.full((2, 3), 42., dtype="f4")
        self.pmm = np.full((2, 3), 7.5, dtype="f4")

    def writer(self, **overrides):
        options = dict(
            variable_name="prcp", units="mm", grid=self.grid,
            member_count=2, calendar="proleptic_gregorian",
        )
        options.update(overrides)
        writer = NetCDFWriter(self.output_dir, **options)
        self.addCleanup(writer.close)
        return writer

    def append(self, writer, timestamp=None, **overrides):
        fields = dict(members=self.members, ensemble_mean=self.mean, pmm=self.pmm)
        fields.update(overrides)
        writer.append(self.timestamp if timestamp is None else timestamp, **fields)

    def assert_staged_count(self, writer, count):
        for product in PRODUCTS:
            with nc.Dataset(writer.incomplete_dir / f"{product}.nc") as dataset:
                self.assertEqual(len(dataset.dimensions["time"]), count)

    def assert_incomplete(self, writer):
        self.assertFalse(writer.completed)
        self.assertFalse(self.output_dir.exists())
        self.assertTrue(writer.incomplete_dir.is_dir())

    def test_three_product_round_trip_preserves_values_coordinates_and_metadata(self):
        timestamps = [self.timestamp, self.timestamp + timedelta(hours=6), datetime(2024, 3, 2)]
        with self.writer(
            expected_timesteps=3,
            variable_attributes={"long_name": "Precipitation amount"},
            global_attributes={"source": "synthetic test", "provenance": json.dumps({"seed": 12})},
        ) as writer:
            for index, timestamp in enumerate(timestamps):
                self.append(writer, timestamp, members=self.members + index,
                            ensemble_mean=self.mean + index, pmm=self.pmm + index)
        self.assertTrue(writer.completed)
        self.assertFalse(writer.incomplete_dir.exists())
        self.assertEqual(writer.finalize(), writer.paths)  # Idempotent publication.
        manifest = json.loads((self.output_dir / "netcdf_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["timesteps"], 3)
        self.assertEqual(manifest["files"], {key: f"{key}.nc" for key in PRODUCTS})
        expected_fields = dict(ensemble_members=self.members, ensemble_mean=self.mean, pmm=self.pmm)
        for product, field in expected_fields.items():
            with self.subTest(product=product), xr.open_dataset(writer.paths[product]) as dataset:
                dimensions = ("time", "member", "y", "x") if product == "ensemble_members" else ("time", "y", "x")
                self.assertEqual(dataset.prcp.dims, dimensions)
                self.assertEqual(dataset.prcp.dtype, np.dtype("f4"))
                np.testing.assert_array_equal(dataset.prcp, np.stack([field + i for i in range(3)]))
                np.testing.assert_array_equal(dataset.time, np.array(timestamps, dtype="datetime64[ns]"))
                for name, values in (("x", self.grid.x), ("y", self.grid.y),
                                     ("lat", self.grid.latitude), ("lon", self.grid.longitude)):
                    np.testing.assert_array_equal(dataset[name], values)
                self.assertEqual(dataset.lat.dims, ("y", "x"))
                self.assertEqual(dataset.prcp.units, "mm")
                self.assertEqual(dataset.prcp.grid_mapping, "crs")
                self.assertEqual(dataset.crs.grid_mapping_name, "lambert_conformal_conic")
                np.testing.assert_array_equal(dataset.crs.standard_parallel, [55., 65.])
                self.assertEqual(dataset.attrs["stride_product"], product)
                self.assertEqual(dataset.attrs["stride_output_space"], "physical")
                self.assertEqual(dataset.attrs["written_timesteps"], 3)
                self.assertEqual(json.loads(dataset.provenance), {"seed": 12})
                if product == "ensemble_members":
                    np.testing.assert_array_equal(dataset.member, [0, 1])

    def test_records_are_flushed_incrementally_with_time_chunks(self):
        writer = self.writer()
        self.grid.x[:] = -999  # Writer owns a snapshot of the supplied coordinates.
        for count in (1, 2):
            self.append(writer, self.timestamp + timedelta(hours=6 * (count - 1)))
            self.assert_incomplete(writer)
            self.assert_staged_count(writer, count)
            for product in PRODUCTS:
                with nc.Dataset(writer.incomplete_dir / f"{product}.nc") as dataset:
                    self.assertTrue(dataset.dimensions["time"].isunlimited())
                    field = dataset.variables["prcp"]
                    chunks = [1, 1, 2, 3] if product == "ensemble_members" else [1, 2, 3]
                    self.assertEqual(field.chunking(), chunks)
                    self.assertTrue(field.filters()["zlib"])
                    self.assertTrue(field.filters()["shuffle"])
                    np.testing.assert_array_equal(dataset.variables["x"][:], [1000., 2000., 3000.])
        writer.finalize()

    def test_bad_record_is_rejected_before_any_product_is_written(self):
        writer = self.writer()
        self.append(writer)
        next_time = self.timestamp + timedelta(hours=6)
        invalid_fields = [
            {"members": self.members[:1]},
            {"members": self.members[:, None]},
            {"ensemble_mean": self.mean[:1]},
            {"pmm": self.pmm.astype(complex)},
            {"pmm": self.pmm.astype(str)},
            {"pmm": np.full((2, 3), np.inf)},
            {"pmm": np.full((2, 3), 1e100)},
            {"time_bounds": (self.timestamp, next_time)},
        ]
        for fields in invalid_fields:
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                self.append(writer, next_time, **fields)
            self.assertEqual(writer.timesteps_written, 1)
            self.assert_staged_count(writer, 1)
        for timestamp in (self.timestamp, self.timestamp - timedelta(hours=6), "2024-02-29"):
            with self.subTest(timestamp=timestamp), self.assertRaises((ValueError, TypeError)):
                self.append(writer, timestamp)
            self.assert_staged_count(writer, 1)
        self.append(writer, next_time)  # Validation errors are recoverable.
        writer.finalize()

    def test_invalid_configuration_fails_without_creating_output(self):
        invalid_options = [
            {"variable_name": None}, {"variable_name": "time"}, {"units": ""},
            {"member_count": 0}, {"member_count": True}, {"expected_timesteps": -1},
            {"dtype": "i2"}, {"compression_level": 10}, {"with_time_bounds": "false"},
            {"grid": replace(self.grid, x=np.array([1., 1., 2.]))},
            {"grid": replace(self.grid, y_attributes={})},
            {"grid": replace(self.grid, latitude=np.ones((3, 2)))},
            {"grid": replace(self.grid, latitude=np.full((2, 3), np.nan))},
            {"grid": replace(self.grid, latitude=np.full((2, 3), 100.))},
            {"grid": replace(self.grid, grid_mapping={})},
            {"variable_attributes": {"scale_factor": 2.}},
            {"variable_attributes": {"units": "unknown"}},
            {"global_attributes": {"stride_output_space": "model"}},
            {"global_attributes": {"nested": {"seed": 12}}},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.writer(**options)
            self.assertFalse(self.output_dir.exists())
            self.assertFalse(self.output_dir.with_name("netcdf.incomplete").exists())

    def test_non_gregorian_calendar_and_rectilinear_coordinates(self):
        grid = replace(self.grid, latitude=np.array([56.1, 56.]),
                       longitude=np.array([14., 14.1, 14.2]), grid_mapping=None)
        timestamps = [cftime.Datetime360Day(2001, 2, 30, 6), cftime.Datetime360Day(2001, 3, 1, 6)]
        with self.writer(calendar="360_day", grid=grid) as writer:
            for invalid in (self.timestamp, cftime.DatetimeNoLeap(2001, 2, 28)):
                with self.assertRaises(ValueError):
                    self.append(writer, invalid)
            for timestamp in timestamps:
                self.append(writer, timestamp)
        for path in writer.paths.values():
            with xr.open_dataset(path, use_cftime=True) as dataset:
                self.assertEqual(list(dataset.time.values), timestamps)
                self.assertEqual(dataset.lat.dims, ("y",))
                self.assertEqual(dataset.lon.dims, ("x",))
                self.assertNotIn("crs", dataset)

    def test_timezone_and_explicit_interval_bounds(self):
        timestamp = datetime(2024, 2, 29, 8, tzinfo=timezone(timedelta(hours=2)))
        bounds = (timestamp - timedelta(hours=6), timestamp)
        with self.writer(with_time_bounds=True) as writer:
            for invalid in (None, (timestamp,), bounds[::-1], (timestamp, timestamp),
                            (timestamp + timedelta(hours=1), timestamp + timedelta(hours=2))):
                with self.subTest(bounds=invalid), self.assertRaises(ValueError):
                    self.append(writer, timestamp, time_bounds=invalid)
                self.assert_staged_count(writer, 0)
            self.append(writer, timestamp, time_bounds=bounds)
        for path in writer.paths.values():
            with xr.open_dataset(path) as dataset:
                np.testing.assert_array_equal(dataset.time, [np.datetime64("2024-02-29T06:00:00", "ns")])
                np.testing.assert_array_equal(dataset.time_bounds,
                    np.array([["2024-02-29T00:00:00", "2024-02-29T06:00:00"]], dtype="datetime64[ns]"))
                self.assertEqual(dataset.time.bounds, "time_bounds")

    def test_float64_and_missing_values_round_trip(self):
        members = self.members.astype("f8") + 1e-10
        members[0, 0, 0] = np.nan
        pmm = np.ma.array(self.pmm, mask=[[True, False, False], [False, False, False]])
        with self.writer(dtype="f8", compression_level=0) as writer:
            self.append(writer, members=members, pmm=pmm)
        with xr.open_dataset(writer.paths["ensemble_members"]) as dataset:
            self.assertEqual(dataset.prcp.dtype, np.dtype("f8"))
            np.testing.assert_array_equal(dataset.prcp[0], members)
        with xr.open_dataset(writer.paths["pmm"]) as dataset:
            np.testing.assert_array_equal(dataset.prcp[0], pmm.filled(np.nan))

    def test_exception_closes_files_and_leaves_only_incomplete_output(self):
        writer = self.writer()
        handles = list(writer._files.values())
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with writer:
                self.append(writer)
                raise RuntimeError("generation failed")
        self.assert_incomplete(writer)
        self.assertTrue(all(not handle.isopen() for handle in handles))
        self.assertFalse((writer.incomplete_dir / "netcdf_manifest.json").exists())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.append(writer)
        with self.assertRaises(FileExistsError):
            self.writer()  # Incomplete output is not silently reused or removed.

    def test_partial_write_failure_cannot_publish(self):
        writer = self.writer()
        self.append(writer)
        handles = list(writer._files.values())
        original = writer._write_record
        calls = 0

        def fail_on_second_file(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated disk failure")
            original(*args)

        with patch.object(writer, "_write_record", side_effect=fail_on_second_file):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.append(writer, self.timestamp + timedelta(hours=6))
        self.assert_incomplete(writer)
        self.assertEqual(writer.timesteps_written, 1)
        self.assertTrue(all(not handle.isopen() for handle in handles))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            writer.finalize()

    def test_initialization_failure_closes_opened_files(self):
        handles = []
        original = NetCDFWriter._initialize_file

        def fail_on_second_file(writer, dataset, product):
            handles.append(dataset)
            if len(handles) == 2:
                raise OSError("simulated initialization failure")
            original(writer, dataset, product)

        with patch.object(NetCDFWriter, "_initialize_file", new=fail_on_second_file):
            with self.assertRaisesRegex(OSError, "initialization failure"):
                self.writer()
        self.assertEqual(len(handles), 2)
        self.assertTrue(all(not handle.isopen() for handle in handles))
        self.assertFalse(self.output_dir.exists())
        self.assertTrue(self.output_dir.with_name("netcdf.incomplete").exists())

    def test_existing_output_is_never_reused(self):
        self.output_dir.mkdir()
        sentinel = self.output_dir / "existing.npz"
        np.savez(sentinel, data=self.members)
        before = sentinel.read_bytes()
        with self.assertRaises(FileExistsError):
            self.writer()
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertFalse(self.output_dir.with_name("netcdf.incomplete").exists())

    def test_exclusive_staging_and_late_destination_conflict(self):
        writer = self.writer()
        with self.assertRaises(FileExistsError):
            self.writer()
        self.append(writer)
        self.output_dir.mkdir()  # Another actor creates the destination meanwhile.
        sentinel = self.output_dir / "existing.npz"
        np.savez(sentinel, data=self.members)
        before = sentinel.read_bytes()
        with self.assertRaises(FileExistsError):
            writer.finalize()
        self.assertFalse(writer.completed)
        self.assertTrue(writer.incomplete_dir.exists())
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertFalse((self.output_dir / "ensemble_members.nc").exists())

    def test_manual_close_does_not_publish_or_delete_npz(self):
        sentinel = self.output_dir.parent / "existing.npz"
        np.savez(sentinel, data=self.members)
        before = sentinel.read_bytes()
        writer = self.writer()
        self.append(writer)
        writer.close()
        writer.close()
        self.assert_incomplete(writer)
        self.assertEqual(sentinel.read_bytes(), before)

    def test_empty_or_short_output_is_not_published(self):
        for records in (0, 1):
            with self.subTest(records=records):
                self.output_dir = self.output_dir.parent / f"short_{records}"
                writer = self.writer(expected_timesteps=2)
                if records:
                    self.append(writer)
                with self.assertRaises(ValueError):
                    writer.finalize()
                self.assert_incomplete(writer)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    self.append(writer)

    def test_cannot_exceed_expected_length(self):
        with self.writer(expected_timesteps=1) as writer:
            self.append(writer)
            with self.assertRaisesRegex(ValueError, "exceed"):
                self.append(writer, self.timestamp + timedelta(hours=6))
            self.assert_staged_count(writer, 1)
        self.assertTrue(writer.completed)


if __name__ == "__main__":
    unittest.main()
