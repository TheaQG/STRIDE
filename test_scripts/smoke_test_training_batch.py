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
import sys
from typing import Any

# Allow direct execution via: python test_scripts/smoke_test_training_batch.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from stride_core.configs.model_config import ModelSpec
from stride_core.models.build_model import build_model
from stride_core.models.edm_loss import EDMLoss
from stride_core.training.data import build_training_data, describe_batch


TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "training" / "train_edm_small.yaml"


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



def safe_len(obj: Any) -> int | None:
    if isinstance(obj, Sized):
        return len(obj)
    return None



def load_model_spec_from_training_config(training_config_path: Path) -> ModelSpec:
    with open(training_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected training config to load into a dict, got {type(cfg)}"
        )

    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, dict):
        raise ValueError("Expected top-level 'training' section to be a dict")

    configs_cfg = training_cfg.get("configs")
    if not isinstance(configs_cfg, dict):
        raise ValueError("Expected 'training.configs' section to be a dict")

    model_config_raw = configs_cfg.get("model_config")
    if model_config_raw is None:
        raise KeyError("Missing required key 'training.configs.model_config'")

    model_config_path = Path(model_config_raw)
    if not model_config_path.is_absolute():
        model_config_path = REPO_ROOT / model_config_path

    return ModelSpec.from_yaml(model_config_path)



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



def main() -> None:
    print_section("STRIDE training batch smoke test")
    print(f"Training config: {TRAINING_CONFIG_PATH}")

    print_section("Building train/valid data")
    built = build_training_data(TRAINING_CONFIG_PATH)

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
    model_spec = load_model_spec_from_training_config(TRAINING_CONFIG_PATH)
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

    print_section("Training batch smoke test completed successfully")


if __name__ == "__main__":
    main()