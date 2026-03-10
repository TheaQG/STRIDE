

from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/smoke_test_full_training.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from stride_core.training.trainer import Trainer



TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "training" / "train_edm_small.yaml"
SMOKE_ROOT = REPO_ROOT / "runs" / "smoke_tests" / "full_training"
SMOKE_CONFIG_PATH = SMOKE_ROOT / "smoke_train_config.yaml"
SMOKE_OUTPUT_DIR = SMOKE_ROOT / "run_output"


def print_section(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))



def main() -> None:
    print_section("STRIDE full trainer smoke test")
    print(f"Base training config: {TRAINING_CONFIG_PATH}")

    if not TRAINING_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Training config does not exist: {TRAINING_CONFIG_PATH}")

    with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected training config to load into a dict, got {type(cfg)}"
        )

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        raise ValueError("Expected top-level 'training' section to be a dict")

    run_cfg = training_cfg.setdefault("run", {})
    loop_cfg = training_cfg.setdefault("loop", {})
    checkpoint_cfg = training_cfg.setdefault("checkpointing", {})
    data_cfg = training_cfg.setdefault("data", {})
    validation_cfg = training_cfg.setdefault("validation", {})

    if not isinstance(run_cfg, dict):
        raise ValueError("Expected 'training.run' to be a dict")
    if not isinstance(loop_cfg, dict):
        raise ValueError("Expected 'training.loop' to be a dict")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("Expected 'training.checkpointing' to be a dict")
    if not isinstance(data_cfg, dict):
        raise ValueError("Expected 'training.data' to be a dict")
    if not isinstance(validation_cfg, dict):
        raise ValueError("Expected 'training.validation' to be a dict")

    configs_cfg = training_cfg.setdefault("configs", {})
    if not isinstance(configs_cfg, dict):
        raise ValueError("Expected 'training.configs' to be a dict")

    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    run_cfg["name"] = "smoke_test_full_training"
    run_cfg["output_dir"] = str(SMOKE_OUTPUT_DIR)

    # Make all referenced config paths absolute so the smoke-test config can live
    # outside the normal configs/ tree without breaking path resolution.
    dataset_config_raw = configs_cfg.get("dataset_config")
    model_config_raw = configs_cfg.get("model_config")
    generation_config_raw = configs_cfg.get("generation_config", None)

    if dataset_config_raw is None:
        raise KeyError("Missing required key 'training.configs.dataset_config'")
    if model_config_raw is None:
        raise KeyError("Missing required key 'training.configs.model_config'")

    configs_cfg["dataset_config"] = str((REPO_ROOT / dataset_config_raw).resolve())
    configs_cfg["model_config"] = str((REPO_ROOT / model_config_raw).resolve())
    if generation_config_raw is not None:
        configs_cfg["generation_config"] = str(
            (REPO_ROOT / generation_config_raw).resolve()
        )

    loop_cfg["max_epochs"] = 1
    loop_cfg["max_train_batches"] = 2
    loop_cfg["max_val_batches"] = 1
    loop_cfg["validate_every_n_epochs"] = 1

    checkpoint_cfg["enabled"] = True
    checkpoint_cfg["save_every_n_epochs"] = 1
    checkpoint_cfg["save_best"] = False
    checkpoint_cfg["resume_from"] = None

    data_cfg["num_workers"] = 0
    validation_cfg["enabled"] = True

    with open(SMOKE_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print_section("Smoke-test config written")
    print(f"Smoke config: {SMOKE_CONFIG_PATH}")
    print(f"Smoke output dir: {SMOKE_OUTPUT_DIR}")

    print_section("Initializing trainer")
    trainer = Trainer(SMOKE_CONFIG_PATH)
    print("Trainer initialized successfully.")

    print_section("Running fit()")
    trainer.fit()
    print("Trainer fit() completed successfully.")

    checkpoint_dir = SMOKE_OUTPUT_DIR / "checkpoints"
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