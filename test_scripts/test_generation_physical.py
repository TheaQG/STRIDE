"""Physical-output regressions using real adapter transforms and synthetic fields.

Run: python -m unittest test_scripts.test_generation_physical -v
No datasets, checkpoints, or model sampling are needed.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from data_adapters.danra_era5_small import transforms as danra
from data_adapters.norcp import transforms as norcp
from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.generation.generation_utils import (
    _inverse_transform_single_tensor,
    build_preview_payload,
    inverse_transform_batch_tensors,
    move_batch_to_device,
)
from stride_core.generation.output_saving import save_member, save_member_bundle
from stride_core.training.data import stride_collate_fn


def transform_metadata(transform):
    if isinstance(transform, norcp.BaseTransform):
        return transform.metadata().to_dict()
    return asdict(transform.get_metadata())


def make_batch(adapter, *, apply_transforms=True, batch_size=1):
    if adapter == "norcp":
        rain = norcp.LogZScoreTransform(
            variable="prcp", source="NORCP_HR",
            stats={"log_mean": -0.5, "log_std": 2.0, "epsilon": 0.01},
        )
        temperature = norcp.ZScoreTransform(
            variable="temp", source="NORCP_LR", stats={"mean": 10.0, "std": 5.0},
        )
        land = norcp.IdentityTransform(variable="lsm", source="NORCP_STATIC")
        elevation = norcp.ZScoreTransform(
            variable="topo", source="NORCP_STATIC", stats={"mean": 250.0, "std": 100.0},
        )
    else:
        rain = danra.LogZScoreTransform(mean_log=0.5, std_log=2.0, eps=0.125)
        temperature = danra.ZScoreTransform(mean=10.0, std=5.0, eps=0.125)
        land = danra.IdentityTransform()
        elevation = danra.ZScoreTransform(mean=250.0, std=100.0, eps=0.125)

    physical = {
        "target": np.array([[[0.0, 0.2], [3.0, 100.0]]], dtype=np.float32),
        "cond_dynamic": np.array(
            [[[0.0, 0.5], [4.0, 20.0]], [[-5.0, 0.0], [15.0, 30.0]]],
            dtype=np.float32,
        ),
        "cond_static": np.array(
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 100.0], [500.0, 1000.0]]],
            dtype=np.float32,
        ),
    }
    sample = {
        "target": torch.from_numpy(
            rain.forward(physical["target"]) if apply_transforms else physical["target"].copy()
        ),
        "cond_dynamic": torch.from_numpy(np.stack([
            rain.forward(physical["cond_dynamic"][0]),
            temperature.forward(physical["cond_dynamic"][1]),
        ]) if apply_transforms else physical["cond_dynamic"].copy()),
        "cond_static": torch.from_numpy(np.stack([
            land.forward(physical["cond_static"][0]),
            elevation.forward(physical["cond_static"][1]),
        ]) if apply_transforms else physical["cond_static"].copy()),
        "meta": {
            "date": "20130101",
            "target_var": "prcp",
            "cond_dynamic_vars": ["prcp", "temp"],
            "cond_static_vars": ["lsm", "topo"],
            "apply_transforms": apply_transforms,
            "transform_metadata": {
                "target": (
                    {"prcp": transform_metadata(rain)}
                    if adapter == "norcp" else transform_metadata(rain)
                ),
                "dynamic": {
                    "prcp": transform_metadata(rain), "temp": transform_metadata(temperature),
                },
                "static": {
                    "lsm": transform_metadata(land), "topo": transform_metadata(elevation),
                },
            },
        },
    }
    if adapter == "norcp":
        sample["meta"]["target_variables"] = ["prcp"]
    return stride_collate_fn([sample] * batch_size), physical, rain


class GenerationPhysicalTests(unittest.TestCase):
    def test_inverse_formulas_match_adapter_implementations(self):
        transforms = [
            norcp.IdentityTransform(variable="lsm", source="test"),
            norcp.ZScoreTransform(variable="temp", source="test", stats={"mean": 7, "std": 3}),
            norcp.LogZScoreTransform(variable="prcp", source="test", stats={"log_mean": -1, "log_std": 2}),
            norcp.LogZScoreTransform(variable="prcp", source="test", stats={"log_mean": -1, "log_std": 2, "epsilon": 0.05}),
            danra.IdentityTransform(),
            danra.ZScoreTransform(mean=7, std=3, eps=0.125),
            danra.LogZScoreTransform(mean_log=1, std_log=2),
            danra.LogZScoreTransform(mean_log=1, std_log=2, clip_min=0.25, eps=0.125),
        ]
        # Include values below the precipitation floor, not just round-trip inputs.
        values = np.array([-30, -1, 0, 2], dtype=np.float32).reshape(1, 1, 2, 2)
        for transform in transforms:
            with self.subTest(metadata=transform_metadata(transform)):
                actual = _inverse_transform_single_tensor(
                    torch.from_numpy(values.copy()), transform_metadata(transform)
                )
                np.testing.assert_allclose(actual.numpy(), transform.inverse(values), rtol=2e-6, atol=1e-6)

    def test_collated_batch_round_trip_and_no_input_mutation(self):
        for adapter in ("norcp", "danra"):
            with self.subTest(adapter=adapter):
                batch, physical, rain = make_batch(adapter, batch_size=2)
                originals = {key: batch[key].clone() for key in physical}
                generated = batch["target"].clone() + 0.4
                generated_before = generated.clone()
                payload = build_preview_payload(batch, generated)
                for key, expected in physical.items():
                    np.testing.assert_allclose(
                        payload[f"{key}_physical"].numpy(),
                        np.stack([expected] * 2), rtol=2e-6, atol=1e-6,
                    )
                    torch.testing.assert_close(batch[key], originals[key], rtol=0, atol=0)
                    self.assertIs(payload[key], batch[key])
                np.testing.assert_allclose(
                    payload["generated_physical"].numpy(), rain.inverse(generated.numpy()),
                    rtol=2e-6, atol=1e-6,
                )
                torch.testing.assert_close(generated, generated_before, rtol=0, atol=0)
                self.assertIs(payload["generated"], generated)

    def test_norcp_target_channel_order_comes_from_metadata(self):
        batch, _, rain = make_batch("norcp")
        meta = batch["meta"]
        meta["target_variables"] = [["temp", "prcp"]]
        cfg = meta["transform_metadata"][0]
        cfg["target"]["temp"] = cfg["dynamic"]["temp"]
        batch["target"] = torch.cat([batch["cond_dynamic"][:, 1:2], batch["target"]], dim=1)
        actual = inverse_transform_batch_tensors(batch, batch["target"])["generated"]
        expected = np.concatenate([
            batch["target"][:, :1].numpy() * 5 + 10,
            rain.inverse(batch["target"][:, 1:].numpy()),
        ], axis=1)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=1e-6)

    def test_disabled_transforms_and_missing_metadata_are_noops(self):
        for adapter in ("norcp", "danra"):
            for missing_metadata in (False, True):
                with self.subTest(adapter=adapter, missing_metadata=missing_metadata):
                    batch, _, _ = make_batch(adapter, apply_transforms=False)
                    if missing_metadata:
                        batch["meta"] = {}
                    generated = batch["target"].clone()
                    actual = inverse_transform_batch_tensors(batch, generated)
                    for key in ("target", "cond_dynamic", "cond_static"):
                        self.assertIs(actual[key], batch[key])
                    self.assertIs(actual["generated"], generated)

    def test_norcp_single_target_falls_back_to_target_var(self):
        batch, physical, _ = make_batch("norcp")
        del batch["meta"]["target_variables"]
        actual = inverse_transform_batch_tensors(batch, batch["target"])["generated"]
        np.testing.assert_allclose(actual.numpy(), physical["target"][None], rtol=2e-6, atol=1e-6)

    def test_optional_static_fields_are_supported(self):
        for adapter in ("norcp", "danra"):
            with self.subTest(adapter=adapter):
                batch, _, _ = make_batch(adapter)
                batch["cond_static"] = None
                actual = inverse_transform_batch_tensors(batch, batch["target"])
                self.assertIsNone(actual["cond_static"])

    def test_missing_stat_and_unknown_transform_do_not_silently_use_defaults(self):
        for cfg in (
            {"name": "zscore", "stats": {"std": 2}},
            {"transform_name": "unknown", "parameters": {}},
            {"unrecognized_schema": {}},
        ):
            with self.subTest(cfg=cfg), self.assertRaises((KeyError, ValueError)):
                _inverse_transform_single_tensor(torch.ones(1, 1, 2, 2), cfg)

    def test_missing_target_channel_metadata_is_rejected(self):
        for names in (["missing_variable"], ["prcp", "temp"], []):
            with self.subTest(names=names):
                batch, _, _ = make_batch("norcp")
                batch["meta"]["target_variables"] = [names]
                with self.assertRaises((KeyError, ValueError)):
                    inverse_transform_batch_tensors(batch, batch["target"])

    def test_cpu_dtype_device_and_shape_are_preserved(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                batch, _, _ = make_batch("norcp")
                for key in ("target", "cond_dynamic", "cond_static"):
                    batch[key] = batch[key].to(dtype=dtype)
                result = inverse_transform_batch_tensors(batch, batch["target"])
                for key, actual in result.items():
                    reference = batch["target"] if key == "generated" else batch[key]
                    self.assertEqual(
                        (actual.dtype, actual.device, actual.shape),
                        (reference.dtype, reference.device, reference.shape),
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm GPU not available")
    def test_gpu_conversion_matches_cpu(self):
        for adapter in ("norcp", "danra"):
            with self.subTest(adapter=adapter):
                batch, _, _ = make_batch(adapter)
                expected = inverse_transform_batch_tensors(batch, batch["target"])
                gpu_batch = move_batch_to_device(deepcopy(batch), torch.device("cuda:0"))
                result = inverse_transform_batch_tensors(gpu_batch, gpu_batch["target"])
                for key, actual in result.items():
                    self.assertEqual(actual.device, torch.device("cuda:0"))
                    torch.testing.assert_close(actual.cpu(), expected[key], rtol=1e-5, atol=1e-6)

    def test_bundled_and_individual_members_agree_for_evaluation(self):
        for adapter in ("norcp", "danra"):
            for save_physical in (False, True):
                with self.subTest(adapter=adapter, save_physical=save_physical), TemporaryDirectory() as temp_dir:
                    batch, _, rain = make_batch(adapter)
                    members = [batch["target"].clone(), batch["target"].clone() + 0.4]
                    expected = np.concatenate([
                        rain.inverse(member.numpy()) if save_physical else member.numpy()
                        for member in members
                    ], axis=0)
                    for mode in ("per_member", "per_member_bundle"):
                        run_dir = Path(temp_dir) / mode
                        output_dir = run_dir / "samples" / "20130101"
                        kwargs = dict(
                            case_output_dir=output_dir, checkpoint_path="unused.pt",
                            save_physical=save_physical, save_metadata=True,
                        )
                        if mode == "per_member":
                            for index, member in enumerate(members):
                                path, _ = save_member(batch, member, member_idx=index, **kwargs)
                                with np.load(path) as arrays:
                                    np.testing.assert_array_equal(arrays["generated"], member.numpy())
                        else:
                            save_member_bundle(batch, members, **kwargs)
                        loader = EvaluationDataLoader(SimpleNamespace(
                            paths=SimpleNamespace(generation_output_dir=run_dir),
                            data=SimpleNamespace(use_case_limit=None),
                        ))
                        product = next(iter(loader)).get_product("ensemble_members")
                        np.testing.assert_allclose(product.arrays["generated_members"], expected, rtol=2e-6, atol=1e-6)
                        np.testing.assert_allclose(loader.get_forecast_array(product), expected[:, 0], rtol=2e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
