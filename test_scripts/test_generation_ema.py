"""CPU-only generation checkpoint tests using the real STRIDE checkpoint writer."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from stride_core.generation.generator import Generator
from stride_core.training.checkpoints import save_checkpoint
from stride_core.training.ema import EMA


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2., 4.]))
        self.frozen = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.register_buffer("running", torch.tensor(3.))
        self.register_buffer("counter", torch.tensor(0))
        self.register_buffer("scratch", torch.tensor(4.), persistent=False)


class GenerationEMATests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.raw = TinyModel()
        self.ema = EMA(self.raw, decay=0.5)
        with torch.no_grad():
            self.raw.weight.fill_(8.)
            self.raw.frozen.fill_(9.)
            self.raw.running.fill_(10.)
            self.raw.counter.fill_(11)
            self.raw.scratch.fill_(12.)
        self.ema.update(self.raw)
        # Change the live buffer values after EMA captured them.
        self.raw.running.fill_(13.)
        self.raw.scratch.fill_(14.)
        self.path = save_checkpoint(
            self.directory, model=self.raw, optimizer=torch.optim.SGD(self.raw.parameters(), lr=0.1),
            ema=self.ema, epoch=1, global_step=1,
        )["primary"]

    def load(self, *, use_ema=True, payload=None, model=None):
        path = self.path
        if payload is not None:
            path = self.directory / "synthetic.pt"
            torch.save(payload, path)
        generator = Generator.__new__(Generator)
        generator.cfg = SimpleNamespace(checkpoint=SimpleNamespace(use_ema_weights=use_ema)) # type: ignore
        generator.device = torch.device("cpu")
        generator.model = TinyModel() if model is None else model
        generator._load_checkpoint_into_model(path)
        return generator

    def assert_weights(self, generator, expected, source):
        torch.testing.assert_close(generator.model.state_dict(), expected.state_dict())
        self.assertEqual(generator.weight_source, source)

    def test_current_checkpoint_matches_validation_clone(self):
        expected = self.ema.clone_model(self.raw)
        generator = self.load()
        self.assert_weights(generator, expected, "ema")
        torch.testing.assert_close(generator.model.scratch, expected.scratch)

    def test_raw_weights_when_ema_disabled(self):
        self.assert_weights(self.load(use_ema=False), self.raw, "raw")

    def test_untracked_floating_buffer_stays_raw(self):
        ema = EMA(self.raw, track_buffers=False)
        self.raw.running.fill_(15.)
        generator = self.load(payload={
            "model_state_dict": self.raw.state_dict(), "ema_state_dict": ema.state_dict(),
        })
        self.assert_weights(generator, self.raw, "ema")

    def test_legacy_flat_and_nested_shadow_payloads_with_parallel_prefixes(self):
        expected = self.ema.clone_model(self.raw)

        def prefixed(state):
            return {f"module.module.{key}": value for key, value in state.items()}

        shadows = {
            "shadow_params": prefixed(self.ema.shadow_params),
            "shadow_buffers": prefixed(self.ema.shadow_buffers),
        }
        for payload in (
            {"ema_state_dict": prefixed(expected.state_dict())},
            {"ema": {"shadow_params": prefixed(expected.state_dict())}},
            {"ema": shadows, "model": prefixed(self.raw.state_dict())},
            {"ema_state_dict": {**self.ema.state_dict(), **shadows},
             "model_state_dict": prefixed(self.raw.state_dict())},
        ):
            with self.subTest(keys=list(payload)):
                self.assert_weights(self.load(payload=payload), expected, "ema")

    def test_complete_legacy_ema_does_not_require_compatible_raw_weights(self):
        expected = self.ema.clone_model(self.raw)
        state = {f"module.{key}": value for key, value in expected.state_dict().items()}
        for ema_payload in (
            {"ema_state_dict": state},
            {"ema": {"shadow_params": state}},
        ):
            with self.subTest(key=next(iter(ema_payload))):
                generator = self.load(payload={
                    "model_state_dict": {"weight": torch.zeros(1)},
                    **ema_payload,
                })
                self.assert_weights(generator, expected, "ema")

    def test_current_ema_rejects_invalid_shadows_before_loading_raw_weights(self):
        for changes, message in (
            ({"shadow_params": {"frozen": torch.tensor(2.)},
              "tracked_param_names": ["frozen"]}, "frozen"),
            ({"shadow_buffers": {"counter": torch.tensor(2)},
              "tracked_buffer_names": ["counter"]}, "counter"),
            ({"shadow_params": {"weight": torch.tensor([1, 2])}}, "dtype"),
            ({"track_buffers": False}, "running"),
            ({"tracked_param_names": []}, "tracked_param_names"),
            ({"tracked_buffer_names": ["running", "running", "scratch"]}, "tracked_buffer_names"),
        ):
            with self.subTest(changes=changes):
                self.assert_invalid_current_ema({**self.ema.state_dict(), **changes}, message)

    def test_current_ema_requires_both_tracked_name_lists(self):
        for key in ("tracked_param_names", "tracked_buffer_names"):
            with self.subTest(key=key):
                state = self.ema.state_dict()
                del state[key]
                self.assert_invalid_current_ema(state, key)

    def assert_invalid_current_ema(self, state, message):
        model = TinyModel()
        with self.assertRaisesRegex(ValueError, message):
            self.load(payload={
                "model_state_dict": self.raw.state_dict(), "ema_state_dict": state,
            }, model=model)
        untouched = TinyModel()
        torch.testing.assert_close(model.state_dict(), untouched.state_dict())
        torch.testing.assert_close(model.scratch, untouched.scratch)

    def test_missing_ema_warns_and_records_raw_fallback(self):
        for extra in ({}, {"ema_state_dict": None}, {"ema": None}):
            with self.subTest(extra=extra), self.assertLogs(
                "stride_core.generation.generator", level="WARNING",
            ) as logs:
                generator = self.load(payload={"model": self.raw.state_dict(), **extra})
                self.assert_weights(generator, self.raw, "raw")
                self.assertRegex(logs.output[0], "EMA.*raw")

    def test_malformed_ema_is_rejected(self):
        for extra, message in (
            ({"ema_state_dict": []}, "ema_state_dict"),
            ({"ema_state_dict": {}}, "ema_state_dict"),
            ({"ema_state_dict": {**self.ema.state_dict(), "shadow_params": {"typo": self.raw.weight}}}, "typo"),
            ({"ema_state_dict": {**self.ema.state_dict(), "shadow_params": {"weight": torch.zeros(1)}}}, "weight"),
            ({"ema": {"shadow_params": {}, "shadow_buffers": {"running": "bad"}}}, "running"),
            ({"ema_state_dict": {"weight": torch.zeros(1)}}, "ema_state_dict"),
            ({"ema_state_dict": {**self.ema.state_dict(), "shadow_buffers": {}}}, "shadow_buffers"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.load(payload={"model_state_dict": self.raw.state_dict(), **extra})
