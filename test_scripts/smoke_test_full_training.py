from pathlib import Path
import shutil
import sys
import tempfile

# Allow direct execution via: python test_scripts/smoke_test_full_training.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from stride_core.configs.config_compiler import ConfigCompiler
from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.training.trainer import Trainer


EXPERIMENT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "experiments" / "train_generate_evaluate_test.yaml"
)
SMOKE_RUN_PARENT = REPO_ROOT / "runs" / "smoke_tests"
SMOKE_EXPERIMENT_NAME = "smoke_test_full_training"


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


def build_smoke_experiment_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
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

    # Training-only smoke test.
    stages_cfg["training"] = True
    stages_cfg["generation"] = False
    stages_cfg["evaluation"] = False

    for key in ("model", "training", "generation", "evaluation", "data"):
        if key in bases_cfg and bases_cfg.get(key) is not None:
            bases_cfg[key] = _abs_from_root(bases_cfg.get(key))

    training_cfg = payload.setdefault("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Expected top-level 'training' section to be a dict")
    training_overrides = training_cfg.setdefault("overrides", {})
    if not isinstance(training_overrides, dict):
        raise ValueError("Expected 'training.overrides' to be a dict")

    # Keep the smoke test tiny and deterministic.
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

    temp_dir = Path(tempfile.mkdtemp(prefix="stride_training_smoke_"))
    temp_config_path = temp_dir / config_path.name
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return temp_config_path


def main() -> None:
    print_section("STRIDE full trainer smoke test")
    print(f"Experiment config: {EXPERIMENT_CONFIG_PATH}")

    smoke_config_path = build_smoke_experiment_config(EXPERIMENT_CONFIG_PATH)
    print_section("Smoke experiment config written")
    print(f"Smoke config: {smoke_config_path}")

    cfg = ExperimentConfig.from_yaml(smoke_config_path)
    compiler = ConfigCompiler(cfg)
    compiled = compiler.compile()

    training_config_path = compiled.training_run_config_path
    smoke_output_dir = compiled.experiment_root / "training"

    print_section("Compiled smoke training config")
    print(f"Training config: {training_config_path}")
    print(f"Smoke output dir: {smoke_output_dir}")

    if smoke_output_dir.parent.exists():
        # Ensure old artifacts do not make the smoke test pass accidentally.
        shutil.rmtree(smoke_output_dir.parent)
        compiled = compiler.compile()
        training_config_path = compiled.training_run_config_path
        smoke_output_dir = compiled.experiment_root / "training"

    print_section("Initializing trainer")
    trainer = Trainer(training_config_path)
    print("Trainer initialized successfully.")

    print_section("Running fit()")
    trainer.fit()
    print("Trainer fit() completed successfully.")

    checkpoint_dir = smoke_output_dir / "checkpoints"
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

    print_section("Trainer smoke test completed successfully")


if __name__ == "__main__":
    main()