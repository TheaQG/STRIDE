"""CPU-only output regressions; no dataset, checkpoint, or sampling run needed.

Run from the repository root:
    python -m unittest test_scripts.test_generation_timestamps -v
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.generation.generator import Generator
from stride_core.generation.output_saving import case_output_dir
from stride_core.training.data import stride_collate_fn


class GenerationTimestampTests(unittest.TestCase):
    def setUp(self):
        temporary_dir = TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.output_dir = Path(temporary_dir.name)

    def test_timestamp_directory_retains_seconds_and_fraction(self):
        for timestamp in (
            "2013-01-01T06:00:00",
            "2013-01-01T06:00:01",
            "2013-01-01T06:00:01.123456",
        ):
            with self.subTest(timestamp=timestamp):
                path = case_output_dir(
                    self.output_dir,
                    date="20130101",
                    timestamp=timestamp,
                    fallback_name="unused",
                )
                self.assertEqual(path.name, timestamp.replace(":", "-"))
                self.assertTrue(path.is_dir())

    def test_date_only_and_missing_metadata_keep_legacy_names(self):
        for date, expected in (("20130101", "20130101"), (None, "case_0000")):
            with self.subTest(date=date):
                path = case_output_dir(
                    self.output_dir, date=date, fallback_name="case_0000"
                )
                self.assertEqual(path.name, expected)

    def _generate_cases(self, *, storage_mode, timestamps, dates, remaining_cases=None):
        samples = []
        for index, date in enumerate(dates):
            meta = {"date": date, "target_var": "prcp"}
            if timestamps is not None:
                meta["timestamp"] = timestamps[index]
            samples.append({
                "target": torch.full((1, 2, 2), float(index + 1)),
                "cond_dynamic": torch.zeros(1, 2, 2),
                "meta": meta,
            })
        batch = stride_collate_fn(samples)

        # Exercise the real batch slicing, product writers and evaluation reader;
        # only replace expensive model sampling with deterministic CPU tensors.
        generator = Generator.__new__(Generator)
        generator.samples_dir = self.output_dir / "samples"
        generator.checkpoint_path = Path("unused.pt")
        generator.model = None
        generator.device = torch.device("cpu")
        generator.cfg = SimpleNamespace(
            sampling=SimpleNamespace(ensemble_size=2, use_fixed_seed=False),
            paths=SimpleNamespace(generation_config_path=Path("unused.yaml")),
            outputs=SimpleNamespace(
                save_members=True,
                save_ensemble_mean=True,
                save_pmm=True,
                save_physical=False,
                save_metadata=True,
                storage_mode=storage_mode,
            ),
        )
        with patch(
            "stride_core.generation.generator.generate_batch",
            return_value=batch["target"].clone(),
        ):
            count = generator._generate_for_batch(
                batch, batch_idx=0, remaining_cases=remaining_cases
            )

        loader = EvaluationDataLoader(SimpleNamespace(
            paths=SimpleNamespace(generation_output_dir=self.output_dir),
            data=SimpleNamespace(use_case_limit=None),
        ))
        return count, loader, list(loader)

    def test_subdaily_cases_survive_in_both_storage_modes(self):
        timestamps = ["2013-01-01T00:00:00", "2013-01-01T06:00:00"]
        run_root = self.output_dir
        for storage_mode in ("per_member", "per_member_bundle"):
            with self.subTest(storage_mode=storage_mode):
                # Each mode gets a fresh run directory, avoiding mixed artifacts.
                self.output_dir = run_root / storage_mode
                count, loader, cases = self._generate_cases(
                    storage_mode=storage_mode,
                    timestamps=timestamps,
                    dates=["20130101", "20130101"],
                )
                self.assertEqual(count, 2)
                self.assertEqual(len(cases), 2)
                for index, case in enumerate(cases):
                    self.assertEqual(case.case_id, timestamps[index].replace(":", "-"))
                    self.assertEqual(case.date, "20130101")
                    self.assertEqual(
                        set(case.products), {"ensemble_members", "ensemble_mean", "pmm"}
                    )
                    for product in case.products.values():
                        self.assertEqual(product.metadata["timestamp"], timestamps[index])
                        np.testing.assert_array_equal(
                            loader.get_forecast_array(product),
                            np.full_like(loader.get_forecast_array(product), index + 1),
                        )
                    for metadata_path in case.case_dir.glob("*.json"):
                        metadata = json.loads(metadata_path.read_text())
                        self.assertEqual(metadata["timestamp"], timestamps[index])

    def test_partial_batch_keeps_selected_timestamps(self):
        timestamps = [
            "2013-01-01T00:00:00",
            "2013-01-01T06:00:00",
            "2013-01-01T12:00:00",
        ]
        count, _, cases = self._generate_cases(
            storage_mode="per_member_bundle",
            timestamps=timestamps,
            dates=["20130101"] * 3,
            remaining_cases=2,
        )
        self.assertEqual(count, 2)
        self.assertEqual(
            [case.case_id for case in cases],
            [timestamp.replace(":", "-") for timestamp in timestamps[:2]],
        )
        self.assertEqual(
            [case.get_product("pmm").metadata["timestamp"] for case in cases],
            timestamps[:2],
        )

    def test_date_only_generation_keeps_legacy_layout(self):
        _, _, cases = self._generate_cases(
            storage_mode="per_member_bundle",
            timestamps=None,
            dates=["20130101", "20130102"],
        )
        self.assertEqual([case.case_id for case in cases], ["20130101", "20130102"])


if __name__ == "__main__":
    unittest.main()
