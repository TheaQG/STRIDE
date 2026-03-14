"""
Smoke test for STRIDE training with RainGate enabled.

This script starts from the experiment-driven config structure, writes a tiny
smoke version of the experiment, swaps in a temporary model config with
RainGate enabled, compiles the experiment, and runs a short training job.

The goal is structural validation that:
- the model config parses with nested RainGate settings
- the compiled training config resolves correctly
- Trainer constructs EDMUNet + EDMLoss with RainGate enabled
- a short fit() run succeeds
- checkpoints are written

This is not a scientific validation test.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stride_core.pipeline.config_compiler import ConfigCompiler
from stride_core.pipeline.experiment_config import ExperimentConfig
from stride_core.training.trainer import Trainer


EXPERIMENT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "experiments" / "train_generate_evaluate_test.yaml"
)
BASE_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "edm_small.yaml"
SMOKE_RUN_PARENT = REPO_ROOT / "runs" / "smoke_tests"
SMOKE_EXPERIMENT_NAME = "smoke_test_training_raingate"


def print_section(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def _abs_from_root(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def build_smoke_model_config(base_model_config_path: Path, temp_dir: Path) -> Path:
    if not base_model_config_path.exists():
        raise FileNotFoundError(
            f"Base model config does not exist: {base_model_config_path}"
        )

    with open(base_model_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected model config to load into a dict, got {type(payload)}"
        )

    model_cfg = payload.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Expected top-level 'model' section to be a dict")

    rain_gate_cfg = model_cfg.setdefault("rain_gate", {})
    if not isinstance(rain_gate_cfg, dict):
        raise ValueError("Expected 'model.rain_gate' to be a dict")

    rain_gate_model_cfg = rain_gate_cfg.setdefault("model", {})
    if not isinstance(rain_gate_model_cfg, dict):
        raise ValueError("Expected 'model.rain_gate.model' to be a dict")
    rain_gate_model_cfg["enabled"] = True
    rain_gate_model_cfg["hidden_channels"] = 16
    rain_gate_model_cfg["num_blocks"] = 2
    rain_gate_model_cfg["input_mode"] = "cond"

    rain_gate_loss_cfg = rain_gate_cfg.setdefault("loss", {})
    if not isinstance(rain_gate_loss_cfg, dict):
        raise ValueError("Expected 'model.rain_gate.loss' to be a dict")
    rain_gate_loss_cfg["enabled"] = True
    rain_gate_loss_cfg["loss_weight"] = 0.05
    rain_gate_loss_cfg["wet_threshold_mm"] = 0.1
    rain_gate_loss_cfg["target_variable"] = "prcp"
    rain_gate_loss_cfg["use_loss_reweighting"] = False
    rain_gate_loss_cfg["reweight_detach"] = True
    rain_gate_loss_cfg["reweight_power"] = 1.0

    temp_model_config_path = temp_dir / base_model_config_path.name
    with open(temp_model_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return temp_model_config_path


def build_smoke_experiment_config(
    experiment_config_path: Path,
    smoke_model_config_path: Path,
) -> Path:
    if not experiment_config_path.exists():
        raise FileNotFoundError(
            f"Experiment config does not exist: {experiment_config_path}"
        )

    with open(experiment_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected experiment config to load into a dict, got {type(payload)}"
        )

    experiment_cfg = payload.get("experiment")
    stages_cfg = payload.get("stages")
    bases_cfg = payload.get("bases")

    if not isinstance(experiment_cfg, dict):
        raise ValueError("Expected top-level 'experiment' section to be a dict")
    if not isinstance(stages_cfg, dict):
        raise ValueError("Expected top-level 'stages' section to be a dict")
    if not isinstance(bases_cfg, dict):
        raise ValueError("Expected top-level 'bases' section to be a dict")

    experiment_cfg["name"] = SMOKE_EXPERIMENT_NAME
    experiment_cfg["output_root"] = str(SMOKE_RUN_PARENT.resolve())

    stages_cfg["training"] = True
    stages_cfg["generation"] = False
    stages_cfg["evaluation"] = False

    for key in ("training", "generation", "evaluation", "data"):
        if key in bases_cfg and bases_cfg.get(key) is not None:
            bases_cfg[key] = _abs_from_root(bases_cfg.get(key))
    bases_cfg["model"] = str(smoke_model_config_path.resolve())

    training_cfg = payload.setdefault("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Expected top-level 'training' section to be a dict")
    training_overrides = training_cfg.setdefault("overrides", {})
    if not isinstance(training_overrides, dict):
        raise ValueError("Expected 'training.overrides' to be a dict")

    training_overrides.update(
        {
            "training": {
                "loop": {
                    "max_epochs": 1,
                    "max_train_batches": 2,
                    "max_val_batches": 1,
                    "validate_every_n_epochs": 1,
                },
                "checkpointing": {
                    "enabled": True,
                    "save_every_n_epochs": 1,
                    "save_best": False,
                    "resume_from": None,
                },
                "data": {
                    "num_workers": 0,
                },
                "validation": {
                    "enabled": True,
                },
            }
        }
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="stride_training_raingate_smoke_"))
    temp_experiment_config_path = temp_dir / experiment_config_path.name
    with open(temp_experiment_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return temp_experiment_config_path


def main() -> None:
    print_section("STRIDE RainGate training smoke test")
    print(f"Experiment config: {EXPERIMENT_CONFIG_PATH}")
    print(f"Base model config: {BASE_MODEL_CONFIG_PATH}")

    temp_model_dir = Path(tempfile.mkdtemp(prefix="stride_raingate_model_cfg_"))
    smoke_model_config_path = build_smoke_model_config(
        BASE_MODEL_CONFIG_PATH,
        temp_model_dir,
    )

    smoke_experiment_config_path = build_smoke_experiment_config(
        EXPERIMENT_CONFIG_PATH,
        smoke_model_config_path,
    )

    print_section("Smoke configs written")
    print(f"Smoke model config:      {smoke_model_config_path}")
    print(f"Smoke experiment config: {smoke_experiment_config_path}")

    exp_cfg = ExperimentConfig.from_yaml(smoke_experiment_config_path)
    compiler = ConfigCompiler(exp_cfg)
    compiled = compiler.compile()

    experiment_root = compiled.experiment_root
    training_output_dir = experiment_root / "training"

    if experiment_root.exists():
        shutil.rmtree(experiment_root)
        compiled = compiler.compile()
        experiment_root = compiled.experiment_root
        training_output_dir = experiment_root / "training"

    print_section("Compiled smoke training config")
    print(f"Training config: {compiled.training_run_config_path}")
    print(f"Experiment root: {experiment_root}")

    print_section("Initializing trainer")
    trainer = Trainer(compiled.training_run_config_path)
    print("Trainer initialized successfully.")

    print_section("Checking RainGate wiring")
    print(f"RainGate model enabled: {trainer.model_spec.rain_gate_model.enabled}")
    print(f"RainGate loss enabled:  {trainer.model_spec.rain_gate_loss.enabled}")
    print(f"RainGate input mode:    {trainer.model_spec.rain_gate_model.input_mode}")
    print(f"RainGate loss weight:   {trainer.model_spec.rain_gate_loss.loss_weight}")
    print(
        f"RainGate wet threshold: {trainer.model_spec.rain_gate_loss.wet_threshold_mm} mm"
    )

    if not trainer.model_spec.rain_gate_model.enabled:
        raise RuntimeError("Expected RainGate model to be enabled in smoke test")
    if not trainer.model_spec.rain_gate_loss.enabled:
        raise RuntimeError("Expected RainGate loss to be enabled in smoke test")
    wrapped_model = getattr(trainer.model, "model", trainer.model)
    if getattr(wrapped_model, "rain_gate", None) is None:
        raise RuntimeError(
            "Expected built model to contain a RainGate module on the wrapped EDMUNet"
        )
    print(f"Wrapped model type:     {type(wrapped_model).__name__}")
    print(f"RainGate module type:   {type(wrapped_model.rain_gate).__name__}")
    if not trainer.loss_fn.rain_gate_enabled:
        raise RuntimeError("Expected EDMLoss RainGate supervision to be enabled")

    print_section("Running fit()")
    trainer.fit()
    print("Trainer fit() completed successfully.")

    checkpoint_dir = training_output_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "checkpoint_latest.pt"
    epoch_checkpoint = checkpoint_dir / "checkpoint_epoch_0001.pt"

    print_section("Checking expected outputs")
    print(f"Checkpoint dir exists: {checkpoint_dir.exists()}")
    print(f"Latest checkpoint exists: {latest_checkpoint.exists()}")
    print(f"Epoch checkpoint exists: {epoch_checkpoint.exists()}")

    if not checkpoint_dir.exists():
        raise RuntimeError(f"Checkpoint directory was not created: {checkpoint_dir}")
    if not latest_checkpoint.exists():
        raise RuntimeError(
            f"Expected latest checkpoint was not created: {latest_checkpoint}"
        )
    if not epoch_checkpoint.exists():
        raise RuntimeError(
            f"Expected epoch checkpoint was not created: {epoch_checkpoint}"
        )

    print_section("RainGate training smoke test completed successfully")


if __name__ == "__main__":
    main()