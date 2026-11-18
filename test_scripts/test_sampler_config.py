"""CPU-only tests for EDM configuration parsing."""

import unittest

from stride_core.generation.generation_utils import GenerationConfig


class SamplerConfigTests(unittest.TestCase):
    def test_canonical_sampler_base(self):
        config = GenerationConfig.from_dict({"generation": {
            "sampler": {"name": "edm", "num_steps": 56, "S_churn": 2, "S_min": 40, "S_max": 80},
            "output": {"return_intermediates": True},
        }})
        self.assertEqual(config.to_sampler_kwargs(), {
            "num_steps": 56, "sigma_min": 0.002, "sigma_max": 80., "rho": 7.,
            "S_churn": 2., "S_min": 40., "S_max": 80., "S_noise": 1.,
            "return_intermediates": True,
        })

    def test_legacy_flat_and_standalone_formats(self):
        fields = {"num_steps": "8", "S_max": "INF", "return_intermediates": True}
        for config in (fields, {"generation": fields}, {"sampler": fields}):
            with self.subTest(config=config):
                result = GenerationConfig.from_dict(config)
                self.assertEqual(result.num_steps, 8)
                self.assertEqual(result.S_max, float("inf"))
                self.assertTrue(result.return_intermediates)

    def test_historical_nulls_and_equivalent_duplicates(self):
        config = GenerationConfig.from_dict({"generation": {
            "num_steps": "9", "rho": 7, "return_intermediates": False,
            "sampler": {"num_steps": None, "sigma_min": None, "sigma_max": None,
                        "rho": "7.0", "S_max": None},
            "output": {"return_intermediates": False},
        }})
        self.assertEqual(config.num_steps, 9)
        self.assertEqual(config.sigma_min, 0.002)
        self.assertEqual(config.sigma_max, 80.)
        self.assertEqual(config.S_max, float("inf"))
        self.assertEqual(GenerationConfig.from_dict({}).num_steps, 18)

    def test_conflicts_and_misplaced_parameters(self):
        for config, message in (
            ({"num_steps": 12, "sampler": {"num_steps": 13}}, "num_steps"),
            ({"return_intermediates": True, "output": {"return_intermediates": False}}, "return_intermediates"),
            ({"sampling": {"num_steps": 12}}, "generation.sampler"),
            ({"generation_run": {"sampler": {"num_steps": 12}}}, "generation.sampler"),
            ({"generation": {}, "num_steps": 12}, "generation.sampler"),
            ({"sampler": {"ensemble_size": 3}}, "generation_run.sampling"),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                GenerationConfig.from_dict(config)

    def test_invalid_types_ranges_and_names(self):
        for config, message in (
            ({"sampler": []}, "sampler"),
            ({"output": []}, "output"),
            ({"sampler": {"name": "ddim"}}, "name"),
            ({"generation": {"name": "ddim", "num_steps": 9}}, "name"),
            ({"name": "ddim", "num_steps": 9}, "name"),
            ({"sampler": {"num_steps": 1.5}}, "num_steps"),
            ({"sampler": {"num_steps": True}}, "num_steps"),
            ({"sampler": {"sigma_min": 90}}, "sigma_max"),
            ({"sampler": {"rho": float("nan")}}, "rho"),
            ({"sampler": {"S_churn": -1}}, "S_churn"),
            ({"sampler": {"S_min": 90, "S_max": 80}}, "S_max"),
            ({"sampler": {"S_noise": 0}}, "S_noise"),
            ({"output": {"return_intermediates": "false"}}, "return_intermediates"),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                GenerationConfig.from_dict(config)
