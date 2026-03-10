

from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/smoke_test_model.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from stride_core.configs.model_config import ModelSpec
from stride_core.models.build_model import build_model
from stride_core.models.edm_loss import EDMLoss


MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "edm_small.yaml"


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  min:   {tensor.min().item():.6f}")
    print(f"  max:   {tensor.max().item():.6f}")
    print(f"  mean:  {tensor.mean().item():.6f}")



def build_fake_batch(spec: ModelSpec, batch_size: int = 2) -> dict:
    target = torch.randn(
        batch_size,
        spec.out_channels,
        spec.target_height,
        spec.target_width,
        dtype=torch.float32,
    )
    cond_dynamic = torch.randn(
        batch_size,
        spec.in_dynamic_channels,
        spec.cond_height,
        spec.cond_width,
        dtype=torch.float32,
    )

    cond_static = None
    if spec.in_static_channels > 0:
        cond_static = torch.randn(
            batch_size,
            spec.in_static_channels,
            spec.cond_height,
            spec.cond_width,
            dtype=torch.float32,
        )

    batch = {
        "target": target,
        "cond_dynamic": cond_dynamic,
        "cond_static": cond_static,
        "meta": {},
    }
    return batch



def main() -> None:
    print("\n========================")
    print("STRIDE model smoke test")
    print("========================")

    print(f"\nLoading model config from: {MODEL_CONFIG_PATH}")
    spec = ModelSpec.from_yaml(MODEL_CONFIG_PATH)
    print("Loaded ModelSpec successfully.")
    print(spec)

    print("\nBuilding model...")
    model = build_model(spec)
    model.eval()
    print(f"Built model of type: {type(model).__name__}")

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch = build_fake_batch(spec, batch_size=2)
    print_tensor_info("target", batch["target"])
    print_tensor_info("cond_dynamic", batch["cond_dynamic"])
    if batch["cond_static"] is not None:
        print_tensor_info("cond_static", batch["cond_static"])

    sigma = torch.exp(torch.randn(batch["target"].shape[0], dtype=batch["target"].dtype) * 1.2 - 1.2)
    print_tensor_info("sigma", sigma)

    print("\nRunning forward pass...")
    with torch.no_grad():
        pred = model(
            x=batch["target"],
            sigma=sigma,
            cond_dynamic=batch["cond_dynamic"],
            cond_static=batch["cond_static"],
        )

    print_tensor_info("prediction", pred)

    expected_shape = batch["target"].shape
    if pred.shape != expected_shape:
        raise RuntimeError(
            f"Prediction shape mismatch: expected {tuple(expected_shape)}, got {tuple(pred.shape)}"
        )
    print("Forward pass shape check passed.")

    print("\nRunning EDM loss...")
    loss_fn = EDMLoss()
    with torch.no_grad():
        loss_details = loss_fn(
            model=model,
            batch=batch,
            sigma=sigma,
            return_details=True,
        )

    if not isinstance(loss_details, dict):
        raise RuntimeError("Expected loss_details to be a dictionary")

    print(f"Loss value: {loss_details['loss'].item():.6f}")
    print_tensor_info("per_sample_loss", loss_details["per_sample_loss"])
    print_tensor_info("x_noisy", loss_details["x_noisy"])

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()