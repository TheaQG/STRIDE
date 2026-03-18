"""
Should test only four things:
1. Can I build train/valid loaders
2. Can I get one batch from train loader?
3. Can I run that batch through the model + loss?
4. Can I do a backward pass and optimizer step?
(5. Can I run a validation step?)

Conceptual overview:
built = build_training_data(train_config_path)
batch = next(iter(built.train_loader))

model = build_model(model_spec)
loss_fn = EDMLoss()
optimizer = build_optimizer(...)

batch = move_batch_to_device(batch, device)
model = model.to(device)

loss = loss_fn(model=model, batch=batch)
loss.backward()
optimizer.step()
"""
from collections.abc import Sized
from pathlib import Path
import argparse
import logging
import sys
from typing import Any

# Allow direct execution via: python test_scripts/smoke_test_training_batch.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from stride_core.configs.config_compiler import ConfigCompiler
from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.configs.model_config import ModelSpec
from stride_core.models.build_model import build_model
from stride_core.models.edm_loss import EDMLoss
from stride_core.training.data import build_training_data, describe_batch
from test_scripts.utils.validation import validate_model_data_contract

EXPERIMENT_CONFIG_DIR = REPO_ROOT / "configs" / "experiments"

def print_section(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))



def print_nested_dict(d: dict[str, Any], indent: int = 0) -> None:
    prefix = " " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_nested_dict(value, indent=indent + 2)
        else:
            print(f"{prefix}{key}: {value}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the STRIDE training batch smoke test for one experiment config, "
            "or for all experiment configs in a directory."
        )
    )
    parser.add_argument(
        "experiment_config",
        nargs="?",
        default=None,
        help=(
            "Optional path to a specific experiment YAML, or a directory of experiment YAMLs. "
            "If omitted, all YAML files under configs/experiments are used."
        ),
    )
    return parser.parse_args()



def discover_experiment_configs(path_arg: str | None = None) -> list[Path]:
    if path_arg is None:
        search_dir = EXPERIMENT_CONFIG_DIR
        if not search_dir.exists():
            raise FileNotFoundError(
                f"Experiment config directory does not exist: {search_dir}"
            )
        paths = sorted(search_dir.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(
                f"No experiment YAML files found in: {search_dir}"
            )
        return paths
    
    requested_path = Path(path_arg)
    if not requested_path.is_absolute():
        requested_path = (REPO_ROOT / requested_path).resolve()

    if requested_path.is_file():
        if requested_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                f"Expected a YAML experiment config file, got: {requested_path}"
            )
        return [requested_path]

    if requested_path.is_dir():
        paths = sorted(requested_path.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(
                f"No experiment YAML files found in: {requested_path}"
            )
        return paths

    raise FileNotFoundError(
        f"Experiment config path does not exist: {requested_path}"
    )


def safe_len(obj: Any) -> int | None:
    if isinstance(obj, Sized):
        return len(obj)
    return None



def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def _move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _move(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_move(v) for v in value]
        return value

    return {key: _move(value) for key, value in batch.items()}



def run_test_case(experiment_config_path: Path) -> None:
    print("\n----------------------------------------")
    print(f"Experiment config: {experiment_config_path}")
    print("----------------------------------------")

    exp_cfg = ExperimentConfig.from_yaml(experiment_config_path)
    compiler = ConfigCompiler(exp_cfg)
    compiled = compiler.compile()

    training_config_path = compiled.training_run_config_path
    model_config_path = compiled.model_config_path

    if training_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce a training run config for experiment: {experiment_config_path}"
        )
    if model_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce a model config for experiment: {experiment_config_path}"
        )

    print(f"Training config: {training_config_path}")
    print(f"Model config:    {model_config_path}")

    print_section("Building training data")
    built = build_training_data(training_config_path)

    train_dataset_len = safe_len(built.train_dataset)
    val_dataset_len = safe_len(built.val_dataset)
    print(
        f"Train dataset length: {train_dataset_len if train_dataset_len is not None else 'unknown (dataset is not Sized)'}"
    )
    print(
        f"Valid dataset length: {val_dataset_len if val_dataset_len is not None else 'unknown (dataset is not Sized)'}"
    )
    print(f"Train loader batches: {len(built.train_loader)}")
    print(f"Valid loader batches: {len(built.val_loader)}")

    print_section("Inspecting one training batch")
    batch = next(iter(built.train_loader))
    print_nested_dict(describe_batch(batch))

    print_section("Building model + loss")
    model_spec = ModelSpec.from_yaml(model_config_path)
    validate_model_data_contract(model_spec, batch)
    print("Validated training batch against compiled model contract.")

    model = build_model(model_spec)
    loss_fn = EDMLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = move_batch_to_device(batch, device)
    model.train()

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"Device:               {device}")
    print(f"Built model type:     {type(model).__name__}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print_section("Running one forward/loss/backward pass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    optimizer.zero_grad(set_to_none=True)

    loss_details = loss_fn(model=model, batch=batch, return_details=True)
    if not isinstance(loss_details, dict):
        raise RuntimeError("Expected loss_details to be a dictionary")

    loss = loss_details["loss"]
    if loss.ndim != 0:
        raise RuntimeError(
            f"Expected scalar reduced loss, got shape {tuple(loss.shape)}"
        )

    print(f"Loss: {loss.item():.6f}")
    print(f"Sigma mean: {loss_details['sigma'].mean().item():.6f}")
    print(f"Prediction shape: {tuple(loss_details['pred'].shape)}")
    print(f"Noisy target shape: {tuple(loss_details['x_noisy'].shape)}")

    loss.backward()

    grad_norm_sq = 0.0
    grad_param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += float(param.grad.norm().item() ** 2)
            grad_param_count += 1
    grad_norm = grad_norm_sq ** 0.5

    print(f"Parameters with gradients: {grad_param_count}")
    print(f"Global grad norm (approx): {grad_norm:.6f}")

    optimizer.step()
    print("Optimizer step completed.")

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print_section("STRIDE training batch smoke test")
    args = parse_args()
    experiment_config_paths = discover_experiment_configs(args.experiment_config)
    print(f"Discovered {len(experiment_config_paths)} experiment config(s) to test.")
    
    for experiment_config_path in experiment_config_paths:
        run_test_case(experiment_config_path)

    
    print_section("Training batch smoke test completed successfully")


if __name__ == "__main__":
    main()