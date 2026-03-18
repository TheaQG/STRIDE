"""
    Smoke test for model set up given an experiment config. 
    Using synthetic data.

    For one specific experiment config run:
        python test_scripts/smoke_test_model.py configs/experiments/example_experiment.yaml
    Or to run for all experiment configs in a directory:
        python test_scripts/smoke_test_model.py configs/experiments
    Default behaviour is to run for all experiment configs under configs/experiments:
        python test_scripts/smoke_test_model.py
    
"""

from pathlib import Path
import argparse
import sys

# Allow direct execution via: python test_scripts/smoke_test_model.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import torch

from stride_core.configs.config_compiler import ConfigCompiler
from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.configs.model_config import ModelSpec
from stride_core.models.build_model import build_model
from stride_core.models.edm_loss import EDMLoss
from test_scripts.utils.validation import validate_model_data_contract


EXPERIMENT_CONFIG_DIR = REPO_ROOT / "configs" / "experiments"



def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  min:   {tensor.min().item():.6f}")
    print(f"  max:   {tensor.max().item():.6f}")
    print(f"  mean:  {tensor.mean().item():.6f}")



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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the STRIDE model smoke test for one experiment config, "
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
        static_height = spec.target_height if spec.align_cond_to_target else spec.cond_height
        static_width = spec.target_width if spec.align_cond_to_target else spec.cond_width
        cond_static = torch.randn(
            batch_size,
            spec.in_static_channels,
            static_height,
            static_width,
            dtype=torch.float32,
        )

    time_features = None
    if spec.use_doy_film:
        time_features = torch.randn(batch_size, 2, dtype=torch.float32)

    return {
        "target": target,
        "cond_dynamic": cond_dynamic,
        "cond_static": cond_static,
        "meta": {},
        "time_features": time_features,
    }



def run_smoke_test(experiment_config_path: Path) -> None:
    print("\n========================")
    print("STRIDE model smoke test")
    print("========================")

    print(f"\nLoading experiment config from: {experiment_config_path}")
    exp_cfg = ExperimentConfig.from_yaml(experiment_config_path)
    compiler = ConfigCompiler(exp_cfg)
    compiled = compiler.compile()

    if compiled.model_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce a model config for experiment: {experiment_config_path}"
        )

    print(f"Compiled model config: {compiled.model_config_path}")
    spec = ModelSpec.from_yaml(compiled.model_config_path)
    print("Loaded compiled ModelSpec successfully.")
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
    validate_model_data_contract(spec, batch)
    print("Validated fake batch against compiled model contract.")

    print_tensor_info("target", batch["target"])
    print_tensor_info("cond_dynamic", batch["cond_dynamic"])
    if batch["cond_static"] is not None:
        print_tensor_info("cond_static", batch["cond_static"])
    if batch["time_features"] is not None:
        print_tensor_info("time_features", batch["time_features"])

    sigma = torch.exp(
        torch.randn(batch["target"].shape[0], dtype=batch["target"].dtype) * 1.2 - 1.2
    )
    print_tensor_info("sigma", sigma)

    print("\nRunning forward pass...")
    with torch.no_grad():
        pred = model(
            x=batch["target"],
            sigma=sigma,
            cond_dynamic=batch["cond_dynamic"],
            cond_static=batch["cond_static"],
            y=batch["time_features"],
        )

    print_tensor_info("prediction", pred)

    expected_shape = batch["target"].shape
    if pred.shape != expected_shape:
        raise RuntimeError(
            f"Prediction shape mismatch: expected {tuple(expected_shape)}, got {tuple(pred.shape)}"
        )
    if torch.isnan(pred).any():
        raise RuntimeError("Prediction contains NaNs")
    if torch.isinf(pred).any():
        raise RuntimeError("Prediction contains Infs")
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



def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    experiment_config_paths = discover_experiment_configs(args.experiment_config)
    print(f"Discovered {len(experiment_config_paths)} experiment config(s) to test.")
    for experiment_config_path in experiment_config_paths:
        run_smoke_test(experiment_config_path)


if __name__ == "__main__":
    main()