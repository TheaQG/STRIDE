"""Regression tests for aggregate -> NPZ -> evaluation, without model sampling.

Run: python -m unittest test_scripts.test_generation_aggregates -v
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.generation.generator import Generator
from stride_core.generation.output_saving import save_aggregate
from stride_core.generation.pmm import pmm_from_ensemble
from test_scripts.test_generation_physical import make_batch


class GenerationAggregateTests(unittest.TestCase):
    def setUp(self):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.output_dir = Path(temp_dir.name)
        # Physical means are [10, 5, 4, 1]. Log-space averaging changes their
        # spatial ranking for BOTH adapters, so PMM must change as well.
        self.physical_members = np.array([
            [[[0.0, 5.0], [0.0, 1.0]]],
            [[[20.0, 5.0], [8.0, 1.0]]],
        ], dtype=np.float32)

    def _run_generation(
        self, adapter, *, save_physical=True, mask=None,
        apply_transforms=True, ensemble_size=2, storage_mode="per_member_bundle",
        save_members=True, save_mean=True, save_pmm=True,
    ):
        batch, physical_batch, transform = make_batch(adapter, apply_transforms=apply_transforms)
        if mask is None:
            batch["cond_static"] = None
        else:
            batch["cond_static"][:, 0] = torch.as_tensor(mask, dtype=torch.float32)
        members = [
            torch.from_numpy(transform.forward(member[None]) if apply_transforms else member[None].copy())
            for member in self.physical_members[:ensemble_size]
        ]
        originals = [member.clone() for member in members]

        # Exercise the real generation orchestration and output readers; replace
        # only checkpoint/model setup and the expensive diffusion sampler.
        generator = Generator.__new__(Generator)
        generator.samples_dir = self.output_dir / "samples"
        generator.checkpoint_path = Path("unused.pt")
        generator.model = None
        generator.device = torch.device("cpu")
        generator.adapter_cfg = SimpleNamespace(static_variables=("lsm", "topo"))
        generator.cfg = SimpleNamespace(
            sampling=SimpleNamespace(ensemble_size=ensemble_size, use_fixed_seed=False),
            paths=SimpleNamespace(generation_config_path=Path("unused.yaml")),
            outputs=SimpleNamespace(
                save_members=save_members, save_ensemble_mean=save_mean, save_pmm=save_pmm,
                save_physical=save_physical, save_metadata=True, storage_mode=storage_mode,
            ),
        )
        with patch("stride_core.generation.generator.generate_batch", side_effect=members):
            count = generator._generate_for_batch(batch, batch_idx=0, remaining_cases=None)
        self.assertEqual(count, 1)
        for member, original in zip(members, originals):
            torch.testing.assert_close(member, original, rtol=0, atol=0)

        loader = EvaluationDataLoader(SimpleNamespace(
            paths=SimpleNamespace(generation_output_dir=self.output_dir),
            data=SimpleNamespace(use_case_limit=None),
        ))
        return loader, next(iter(loader)), members, physical_batch

    def test_physical_mean_and_pmm_use_physical_members(self):
        run_root = self.output_dir
        for adapter in ("norcp", "danra"):
            for mode in ("per_member", "per_member_bundle"):
                with self.subTest(adapter=adapter, storage_mode=mode):
                    self.output_dir = run_root / adapter / mode
                    loader, case, members, physical_batch = self._run_generation(adapter, storage_mode=mode)
                    mean = case.get_product("ensemble_mean")
                    pmm = case.get_product("pmm")
                    np.testing.assert_allclose(
                        loader.get_forecast_array(mean), [[10, 5], [4, 1]], rtol=2e-6, atol=1e-6,
                    )
                    # Existing PMM quantile selection, calculated by hand for
                    # this 2-member, 4-pixel example (pooled indices 6,4,3,1).
                    np.testing.assert_allclose(
                        loader.get_forecast_array(pmm), [[8, 5], [1, 0]], rtol=2e-6, atol=1e-6,
                    )
                    ensemble = loader.get_forecast_array(case.get_product("ensemble_members"))
                    np.testing.assert_allclose(loader.get_forecast_array(mean), ensemble.mean(axis=0))
                    np.testing.assert_allclose(
                        loader.get_target_array(mean), physical_batch["target"][0], rtol=2e-6, atol=1e-6,
                    )

                    # Preserve legacy model-space arrays, not forward-transforms
                    # of the NEW physical aggregates (which are not equivalent).
                    raw_ensemble = torch.cat(members, dim=0)[:, 0]
                    np.testing.assert_array_equal(mean.arrays["generated"], torch.stack(members).mean(dim=0).numpy())
                    np.testing.assert_array_equal(pmm.arrays["generated"], pmm_from_ensemble(raw_ensemble).numpy())
                    for product in (mean, pmm):
                        self.assertEqual(product.metadata["aggregation_spaces"], {
                            "generated": "model", "generated_physical": "physical",
                        })

    def test_raw_output_mode_is_unchanged(self):
        run_root = self.output_dir
        for adapter in ("norcp", "danra"):
            with self.subTest(adapter=adapter):
                self.output_dir = run_root / adapter
                loader, case, members, _ = self._run_generation(adapter, save_physical=False)
                expected = {
                    "ensemble_mean": torch.stack(members).mean(dim=0),
                    "pmm": pmm_from_ensemble(torch.cat(members, dim=0)[:, 0]),
                }
                for name, raw in expected.items():
                    product = case.get_product(name)
                    self.assertNotIn("generated_physical", product.arrays)
                    np.testing.assert_array_equal(loader.get_forecast_array(product), raw.numpy()[0, 0])
                    self.assertEqual(product.metadata["aggregation_spaces"], {"generated": "model"})

    def test_pmm_mask_uses_physical_mean_outside_valid_pixels(self):
        run_root = self.output_dir
        for adapter in ("norcp", "danra"):
            for mask, expected in (
                ([[True, True], [False, False]], [[5, 5], [4, 1]]),
                ([[False, False], [False, False]], [[10, 5], [4, 1]]),
            ):
                with self.subTest(adapter=adapter, mask=mask):
                    self.output_dir = run_root / adapter / str(mask[0][0])
                    loader, case, _, _ = self._run_generation(adapter, mask=mask)
                    np.testing.assert_allclose(
                        loader.get_forecast_array(case.get_product("pmm")), expected, rtol=2e-6, atol=1e-6,
                    )

    def test_disabled_transforms_and_single_member_outputs(self):
        run_root = self.output_dir
        for adapter in ("norcp", "danra"):
            for apply_transforms, ensemble_size in ((False, 2), (True, 1)):
                with self.subTest(adapter=adapter, apply_transforms=apply_transforms, ensemble_size=ensemble_size):
                    self.output_dir = run_root / adapter / str(ensemble_size)
                    loader, case, _, _ = self._run_generation(
                        adapter, apply_transforms=apply_transforms, ensemble_size=ensemble_size,
                    )
                    expected_mean = self.physical_members[:ensemble_size, 0].mean(axis=0)
                    expected_pmm = self.physical_members[0, 0] if ensemble_size == 1 else [[8, 5], [1, 0]]
                    np.testing.assert_allclose(loader.get_forecast_array(case.get_product("ensemble_mean")), expected_mean, atol=1e-6)
                    np.testing.assert_allclose(loader.get_forecast_array(case.get_product("pmm")), expected_pmm, atol=1e-6)

    def test_aggregates_do_not_require_saving_members_or_each_other(self):
        run_root = self.output_dir
        for name in ("ensemble_mean", "pmm"):
            with self.subTest(product=name):
                self.output_dir = run_root / name
                loader, case, _, _ = self._run_generation(
                    "norcp", save_members=False, save_mean=name == "ensemble_mean", save_pmm=name == "pmm",
                )
                self.assertEqual(set(case.products), {name})
                expected = [[10, 5], [4, 1]] if name == "ensemble_mean" else [[8, 5], [1, 0]]
                np.testing.assert_allclose(loader.get_forecast_array(case.get_product(name)), expected, atol=1e-6)

    def test_writer_requires_explicit_physical_aggregate(self):
        batch, _, _ = make_batch("norcp")
        with self.assertRaisesRegex(ValueError, "generated_physical"):
            save_aggregate(
                batch, batch["target"], case_output_dir=self.output_dir, name="ensemble_mean",
                checkpoint_path="unused.pt", save_physical=True, save_metadata=True,
            )
        self.assertFalse((self.output_dir / "ensemble_mean.npz").exists())

    def test_writer_validates_physical_shape_and_output_mode(self):
        batch, _, _ = make_batch("norcp")
        for save_physical, shape in ((True, (1, 1, 3, 3)), (False, (1, 1, 2, 2))):
            with self.subTest(save_physical=save_physical, shape=shape):
                with self.assertRaisesRegex(ValueError, "generated_physical"):
                    save_aggregate(
                        batch, batch["target"], generated_physical=torch.ones(shape),
                        case_output_dir=self.output_dir, name="ensemble_mean",
                        checkpoint_path="unused.pt", save_physical=save_physical, save_metadata=False,
                    )
                self.assertFalse((self.output_dir / "ensemble_mean.npz").exists())


if __name__ == "__main__":
    unittest.main()
