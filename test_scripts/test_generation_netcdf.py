"""NetCDF-only generation integration, with CPU tensors and synthetic source data.

Run: python -m unittest test_scripts.test_generation_netcdf -v
Model construction and diffusion sampling are stubbed; checkpoint loading,
batching, physical conversion, aggregation, metadata and file writing are real.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
import xarray as xr
import yaml

from data_adapters.norcp.adapter import NorCPDataset
from stride_core.configs.config_compiler import ConfigCompiler
from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.generation.generator import GenerationRunConfig, Generator
from stride_core.training.ema import EMA
from test_scripts.test_generation_ema import TinyModel
from test_scripts.test_generation_physical import make_batch


class SyntheticNorCPDataset(NorCPDataset):
    """Keep NorCP's real metadata method, with prebuilt model-space samples."""

    def __init__(self, samples, source_path):
        self.samples = samples
        self.timestamps = [sample["meta"]["timestamp"] for sample in samples]
        self._sample_lookup = {ts: {"target_files": {"prcp": source_path}} for ts in self.timestamps}
        self.target_variable = "prcp"
        self.target_source = "NORCP_HR"
        self.target_time_offsets = {"prcp": -3.}
        self.hr_crop_spec = None
        self.spatial_shuffle = False

    def __getitem__(self, index):
        return self.samples[index]


class NetCDFGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base_path = self.root / "generation_base.yaml"
        self.base_path.write_text(yaml.safe_dump({"generation": {"outputs": {
            "save_physical": True, "save_members": True, "save_ensemble_mean": True,
            "save_pmm": True, "storage_mode": "per_member_bundle",
        }}}))
        self.physical_members = np.array([
            [[[0., 5.], [0., 1.]]], [[[20., 5.], [8., 1.]]],
        ], dtype="f4")
        batch, _, self.transform = make_batch("norcp")
        template = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                template[key] = value[0]
            elif isinstance(value, dict):
                template[key] = {name: entries[0] for name, entries in value.items()}
            else:
                template[key] = value
        template["cond_static"] = None
        samples = []
        for index in range(4):
            sample = deepcopy(template)
            timestamp = datetime(2024, 2, 29) + timedelta(hours=6 * index)
            sample["meta"].update(timestamp=timestamp.isoformat(), date="20240229", sample_index=index)
            samples.append(sample)
        source_path = self.root / "source.nc"
        xr.Dataset(
            data_vars={
                "pr": (("time", "y", "x"), np.zeros((4, 2, 2), dtype="f4"), {
                    "units": "kg m-2 s-1", "standard_name": "precipitation_flux", "cell_methods": "time: mean",
                }),
                "time_bnds": (("time", "bnds"), [[0., 6.], [6., 12.], [12., 18.], [18., 24.]]),
            },
            coords={
                "time": ("time", [3., 9., 15., 21.], {
                    "units": "hours since 2024-02-29 00:00:00", "calendar": "standard", "bounds": "time_bnds",
                }),
                "x": ("x", [1000., 2000.], {"units": "m"}),
                "y": ("y", [2000., 1000.], {"units": "m"}),
                "lat": (("y", "x"), [[56., 56.], [55., 55.]]),
                "lon": (("y", "x"), [[10., 11.], [10., 11.]]),
            },
        ).to_netcdf(source_path)
        self.dataset = SyntheticNorCPDataset(samples, source_path)
        self.checkpoint_path = self.root / "synthetic.pt"
        model = TinyModel()
        ema = EMA(model)
        with torch.no_grad():
            model.weight.fill_(7.)
        torch.save({"model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict()}, self.checkpoint_path)

    def config(self, output_format="netcdf", *, outputs=None, data=None, max_cases=3, checkpoint_path=None):
        return GenerationRunConfig.from_dict({"generation_run": {
            "run_name": "synthetic", "output_dir": str(self.root / output_format),
            "paths": {"dataset_config": str(self.root / "dataset.yaml"),
                      "model_config": str(self.root / "model.yaml"), "generation_config": str(self.base_path)},
            "checkpoint": {"mode": "explicit", "path": str(checkpoint_path or self.checkpoint_path)},
            "device": {"accelerator": "cpu"},
            "data": {"batch_size": 2, "shuffle": False, **(data or {})},
            "sampling": {"ensemble_size": 2}, "limits": {"max_cases": max_cases},
            "outputs": {"output_format": output_format, **(outputs or {})},
        }}, config_path=self.root / "generation_run.yaml")

    def generator(self, output_format="netcdf", **options):
        cfg = self.config(output_format, **options)
        with patch("stride_core.generation.generator.AdapterConfig.from_yaml", return_value=SimpleNamespace(static_variables=())), \
             patch.object(Generator, "_build_adapter", return_value=SimpleNamespace(build_dataset=lambda: self.dataset)), \
             patch("stride_core.generation.generator.build_model", side_effect=lambda *_: TinyModel()):
            return Generator(cfg)

    def sample(self, model, batch, **kwargs):
        member_index = self.sample_calls % 2
        self.sample_calls += 1
        return torch.from_numpy(np.stack([
            self.transform.forward(self.physical_members[member_index] + 10 * index)
            for index in batch["meta"]["sample_index"]
        ]))

    def run_generator(self, generator):
        self.sample_calls = 0
        with patch("stride_core.generation.generator.generate_batch", side_effect=self.sample):
            generator.run()

    def test_netcdf_only_matches_npz_physical_products_across_batches(self):
        npz_generator = self.generator("npz")
        self.run_generator(npz_generator)
        npz_metadata = json.loads(npz_generator.metadata_path.read_text())
        self.assertIs(npz_metadata["use_ema_weights"], True)
        self.assertEqual(npz_metadata["weight_source"], "ema")
        # Exercise fallback through the NetCDF output path with EMA still requested.
        raw_checkpoint = self.root / "raw.pt"
        torch.save({"model_state_dict": TinyModel().state_dict()}, raw_checkpoint)
        with self.assertLogs("stride_core.generation.generator", level="WARNING"):
            generator = self.generator(outputs={"save_metadata": False}, checkpoint_path=raw_checkpoint)
        # Even with NPZ sidecars disabled, essential NetCDF metadata is present.
        with patch("stride_core.generation.generator.build_case_output_dir", side_effect=AssertionError("NPZ path used")):
            self.run_generator(generator)
        self.assertFalse(generator.samples_dir.exists())
        self.assertEqual(list(generator.output_dir.rglob("*.npz")), [])
        self.assertEqual(sorted(path.name for path in generator.output_dir.rglob("*.json")),
                         ["generation_metadata.json", "netcdf_manifest.json"])
        self.assertEqual(len(list(generator.output_dir.rglob("*.nc"))), 3)
        manifest = json.loads((generator.output_dir / "netcdf/netcdf_manifest.json").read_text())
        self.assertEqual(manifest["timesteps"], 3)  # max_cases cuts the second batch in half.
        run_metadata = json.loads(generator.metadata_path.read_text())
        self.assertIs(run_metadata["use_ema_weights"], True)
        self.assertEqual(run_metadata["weight_source"], "raw")
        self.assertEqual(run_metadata["output_format"], "netcdf")
        self.assertEqual(run_metadata["storage_mode"], "temporal_stack")
        loader = EvaluationDataLoader(SimpleNamespace(
            paths=SimpleNamespace(generation_output_dir=npz_generator.output_dir),
            data=SimpleNamespace(use_case_limit=None),
        )) # type: ignore
        cases = list(loader)
        for product in ("ensemble_members", "ensemble_mean", "pmm"):
            with xr.open_dataset(generator.output_dir / "netcdf" / f"{product}.nc") as output:
                for index, case in enumerate(cases):
                    np.testing.assert_allclose(output.prcp[index], loader.get_forecast_array(case.get_product(product)))
                self.assertEqual(output.sizes["time"], 3)
                self.assertEqual(output.prcp.units, "mm day-1")
                self.assertEqual(output.stride_alignment_offset_hours, -3.)
                self.assertEqual(output.use_ema_weights, 1)
                self.assertEqual(output.weight_source, "raw")
                self.assertEqual(output.checkpoint_path, str(raw_checkpoint))
                np.testing.assert_array_equal(output.time, np.array([
                    "2024-02-29T03:00:00", "2024-02-29T09:00:00", "2024-02-29T15:00:00",
                ], dtype="datetime64[ns]"))
                self.assertEqual(output.time_bounds.shape, (3, 2))

    def test_no_case_limit_exports_the_full_split(self):
        generator = self.generator(max_cases=None)
        self.run_generator(generator)
        with xr.open_dataset(generator.output_dir / "netcdf/ensemble_members.nc") as output:
            self.assertEqual(output.sizes["time"], 4)
            self.assertEqual(output.weight_source, "ema")

    def test_sampling_failure_leaves_only_incomplete_output(self):
        generator = self.generator()
        self.sample_calls = 0

        def fail_on_second_batch(*args, **kwargs):
            if self.sample_calls == 2:
                raise RuntimeError("sampling failed")
            return self.sample(*args, **kwargs)

        with patch("stride_core.generation.generator.generate_batch", side_effect=fail_on_second_batch):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                generator.run()
        self.assertFalse((generator.output_dir / "netcdf").exists())
        self.assertTrue((generator.output_dir / "netcdf.incomplete").is_dir())
        self.assertEqual(list(generator.output_dir.rglob("*.npz")), [])
        self.assertEqual(list(generator.output_dir.rglob("netcdf_manifest.json")), [])

    def test_changed_grid_cannot_be_published_as_one_fixed_grid(self):
        generator = self.generator()
        original = self.dataset.get_output_metadata

        def metadata(index):
            result = original(index)
            return replace(result, grid=replace(result.grid, x=result.grid.x + 1)) if index else result

        with patch.object(self.dataset, "get_output_metadata", side_effect=metadata):
            with self.assertRaisesRegex(ValueError, "coordinates changed"):
                self.run_generator(generator)
        self.assertFalse((generator.output_dir / "netcdf").exists())

    def test_existing_npz_is_preserved_and_blocks_a_mixed_run(self):
        generator = self.generator()
        generator.samples_dir.mkdir()
        sentinel = generator.samples_dir / "existing.npz"
        np.savez(sentinel, existing=[1., 2.])
        before = sentinel.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "new output directory"):
            self.run_generator(generator)
        self.assertEqual(self.sample_calls, 0)
        self.assertEqual(sentinel.read_bytes(), before)

    def test_completed_netcdf_is_not_overwritten(self):
        generator = self.generator()
        self.run_generator(generator)
        before = generator.metadata_path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_generator(generator)
        self.assertEqual(generator.metadata_path.read_bytes(), before)
        self.assertEqual(self.sample_calls, 0)

    def test_unsupported_adapter_and_empty_split_fail_before_sampling(self):
        generator = self.generator()
        generator.dataset = []
        with self.assertRaisesRegex(ValueError, "NorCP adapter only"):
            self.run_generator(generator)
        generator.dataset = self.dataset
        self.dataset.timestamps = []
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            self.run_generator(generator)
        self.assertEqual(self.sample_calls, 0)

    def test_configuration_requires_chronological_physical_three_product_export(self):
        for key in ("save_physical", "save_members", "save_ensemble_mean", "save_pmm"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.config(outputs={key: False})
        with self.assertRaisesRegex(ValueError, "shuffle"):
            self.config(data={"shuffle": True})
        self.assertEqual(self.config(outputs={"storage_mode": "ignored"}).outputs.storage_mode, "temporal_stack")
        # Restrictions belong to NetCDF; legacy NPZ options stay available.
        cfg = self.config("npz", outputs={"save_physical": False, "save_pmm": False}, data={"shuffle": True})
        self.assertFalse(cfg.outputs.save_physical)
        self.assertTrue(cfg.data.shuffle)

    def test_pipeline_rejects_netcdf_with_evaluation_but_allows_export(self):
        path = Path(__file__).resolve().parents[1] / "configs/experiments/pipeline_norcp_test.yaml"
        for output_format, evaluation in (("netcdf", True), ("netcdf", False), ("npz", True)):
            with self.subTest(output_format=output_format, evaluation=evaluation):
                payload = yaml.safe_load(path.read_text())
                payload["experiment"]["output_root"] = str(self.root / f"compile_{output_format}_{evaluation}")
                payload["stages"]["evaluation"] = evaluation
                payload["generation"]["overrides"]["generation_run"]["outputs"] = {"output_format": output_format}
                compiler = ConfigCompiler(ExperimentConfig.from_dict(payload, config_path=path))
                if output_format == "netcdf" and evaluation:
                    with self.assertRaisesRegex(ValueError, "stages.evaluation=false"):
                        compiler.compile()
                else:
                    compiled = compiler.compile()
                    resolved = GenerationRunConfig.from_yaml(compiled.generation_run_config_path)
                    self.assertEqual(resolved.outputs.output_format, output_format)


if __name__ == "__main__":
    unittest.main()
