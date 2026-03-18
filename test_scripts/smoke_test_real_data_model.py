from pathlib import Path
import argparse
import logging
import sys
from typing import Any

# Allow direct execution via: python test_scripts/smoke_test_real_data_model.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch
import yaml

from stride_core.configs.adapter_config import AdapterConfig
from stride_core.configs.config_compiler import ConfigCompiler
from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.configs.model_config import ModelSpec
from stride_core.generation.edm_sampler import edm_sampler
from stride_core.models.build_model import build_model
from stride_core.models.edm_loss import EDMLoss

from test_scripts.utils.validation import validate_model_data_contract


EXPERIMENT_CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


def print_tensor_info(name: str, tensor: torch.Tensor | None) -> None:
    print(f"\n{name}:")
    if tensor is None:
        print("  None")
        return

    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  min:   {tensor.min().item():.6f}")
    print(f"  max:   {tensor.max().item():.6f}")
    print(f"  mean:  {tensor.mean().item():.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the STRIDE real-data smoke test for one experiment config, "
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


def _load_generation_kwargs(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected generation config to load into a dict, got {type(cfg)}"
        )

    if "generation" in cfg:
        generation_cfg = cfg["generation"]
        if not isinstance(generation_cfg, dict):
            raise ValueError("Expected 'generation' section to be a dict")
        sampler_cfg = generation_cfg.get("sampler", generation_cfg)
    else:
        sampler_cfg = cfg.get("sampler", cfg)

    if not isinstance(sampler_cfg, dict):
        raise ValueError("Expected sampler configuration to be a dict")

    kwargs = {
        "num_steps": int(sampler_cfg.get("num_steps", 18)),
        "sigma_min": float(sampler_cfg.get("sigma_min", 0.002)),
        "sigma_max": float(sampler_cfg.get("sigma_max", 80.0)),
        "rho": float(sampler_cfg.get("rho", 7.0)),
        "S_churn": float(sampler_cfg.get("S_churn", 0.0)),
        "S_min": float(sampler_cfg.get("S_min", 0.0)),
        "S_max": float(sampler_cfg.get("S_max", float("inf"))),
        "S_noise": float(sampler_cfg.get("S_noise", 1.0)),
        "return_intermediates": bool(sampler_cfg.get("return_intermediates", False)),
    }
    return kwargs


def build_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    target = sample["target"].unsqueeze(0)
    cond_dynamic = sample["cond_dynamic"].unsqueeze(0)
    cond_static = (
        sample["cond_static"].unsqueeze(0)
        if sample["cond_static"] is not None
        else None
    )

    meta = dict(sample["meta"])
    meta["region_info"] = sample["cond_coord"]

    time_features = None
    if sample.get("time_features") is not None:
        time_features = sample["time_features"].unsqueeze(0)

    return {
        "target": target,
        "cond_dynamic": cond_dynamic,
        "cond_static": cond_static,
        "meta": meta,
        "time_features": time_features,
    }


def _get_target_transform(dataset: Any, batch: dict[str, Any]) -> Any:
    if hasattr(dataset, "target_transform"):
        return dataset.target_transform

    if hasattr(dataset, "target_transforms"):
        meta = batch.get("meta", {})
        target_var = meta.get("target_var")
        if target_var is None:
            cfg = getattr(dataset, "cfg", None)
            if cfg is not None and hasattr(cfg, "target_variable"):
                target_var = cfg.target_variable
        if target_var is None:
            raise ValueError("Could not resolve target variable for inverse transform")
        return dataset.target_transforms[target_var]

    raise AttributeError("Dataset exposes neither 'target_transform' nor 'target_transforms'")



def _get_dynamic_variable_names(dataset: Any, batch: dict[str, Any]) -> list[str]:
    meta = batch.get("meta", {})
    names = meta.get("cond_dynamic_vars")
    if names is not None:
        return list(names)
    cfg = getattr(dataset, "cfg", None)
    if cfg is not None and hasattr(cfg, "dynamic_variables"):
        return list(cfg.dynamic_variables)
    raise ValueError("Could not resolve dynamic variable names for inverse transform")



def _get_static_variable_names(dataset: Any, batch: dict[str, Any]) -> list[str]:
    meta = batch.get("meta", {})
    names = meta.get("cond_static_vars")
    if names is not None:
        return list(names)
    cfg = getattr(dataset, "cfg", None)
    if cfg is not None and hasattr(cfg, "static_variables"):
        return list(cfg.static_variables)
    return []


# Helper function: inverse_transform_batch_for_plotting
def inverse_transform_batch_for_plotting(
    dataset: Any,
    batch: dict[str, Any],
    pred: torch.Tensor,
    generated: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    """
    Back-transform selected tensors from model space to physical space for
    plotting/debugging.

    Notes
    -----
    - Target / prediction / generated sample use the dataset target transform.
    - Dynamic channels are inverse-transformed channel-wise using the dataset
      dynamic transforms.
    - Static channels are inverse-transformed channel-wise using the dataset
      static transforms.
    """
    target_transform = _get_target_transform(dataset, batch)

    target_phys = torch.from_numpy(
        target_transform.inverse(batch["target"][0].detach().cpu().numpy())
    ).to(torch.float32)

    pred_phys = torch.from_numpy(
        target_transform.inverse(pred[0].detach().cpu().numpy())
    ).to(torch.float32)

    generated_phys = torch.from_numpy(
        target_transform.inverse(generated[0].detach().cpu().numpy())
    ).to(torch.float32)

    cond_dynamic_phys_channels: list[torch.Tensor] = []
    for idx, variable_name in enumerate(_get_dynamic_variable_names(dataset, batch)):
        channel_np = batch["cond_dynamic"][0, idx].detach().cpu().numpy()
        inv_np = dataset.dynamic_transforms[variable_name].inverse(channel_np)
        cond_dynamic_phys_channels.append(torch.from_numpy(inv_np).to(torch.float32))
    cond_dynamic_phys = torch.stack(cond_dynamic_phys_channels, dim=0)

    cond_static_phys = None
    if batch["cond_static"] is not None:
        cond_static_phys_channels: list[torch.Tensor] = []
        for idx, variable_name in enumerate(_get_static_variable_names(dataset, batch)):
            channel_np = batch["cond_static"][0, idx].detach().cpu().numpy()
            inv_np = dataset.static_transforms[variable_name].inverse(channel_np)
            cond_static_phys_channels.append(torch.from_numpy(inv_np).to(torch.float32))
        cond_static_phys = torch.stack(cond_static_phys_channels, dim=0)

    return {
        "target": target_phys,
        "pred": pred_phys,
        "generated": generated_phys,
        "cond_dynamic": cond_dynamic_phys,
        "cond_static": cond_static_phys,
    }


def plot_real_smoke_test(
    batch: dict[str, Any],
    pred: torch.Tensor,
    generated: torch.Tensor,
    physical: dict[str, torch.Tensor | None],
) -> None:
    target_model = batch["target"][0, 0].detach().cpu().numpy()
    cond_prcp_model = batch["cond_dynamic"][0, 0].detach().cpu().numpy()
    cond_temp_model = (
        batch["cond_dynamic"][0, 1].detach().cpu().numpy()
        if batch["cond_dynamic"].shape[1] > 1
        else None
    )
    pred_model = pred[0, 0].detach().cpu().numpy()
    gen_model = generated[0, 0].detach().cpu().numpy()

    target_phys_tensor = physical["target"]
    cond_dynamic_phys_tensor = physical["cond_dynamic"]
    pred_phys_tensor = physical["pred"]
    generated_phys_tensor = physical["generated"]

    if target_phys_tensor is None:
        raise ValueError("physical['target'] must not be None")
    if cond_dynamic_phys_tensor is None:
        raise ValueError("physical['cond_dynamic'] must not be None")
    if pred_phys_tensor is None:
        raise ValueError("physical['pred'] must not be None")
    if generated_phys_tensor is None:
        raise ValueError("physical['generated'] must not be None")

    target_phys = target_phys_tensor[0].detach().cpu().numpy()
    cond_prcp_phys = cond_dynamic_phys_tensor[0].detach().cpu().numpy()
    cond_temp_phys = (
        cond_dynamic_phys_tensor[1].detach().cpu().numpy()
        if cond_dynamic_phys_tensor.shape[0] > 1
        else None
    )
    pred_phys = pred_phys_tensor[0].detach().cpu().numpy()
    gen_phys = generated_phys_tensor[0].detach().cpu().numpy()

    cond_static_phys = physical["cond_static"]
    lsm_phys = None
    topo_phys = None
    if cond_static_phys is not None:
        if cond_static_phys.shape[0] >= 1:
            lsm_phys = cond_static_phys[0].detach().cpu().numpy()
        if cond_static_phys.shape[0] >= 2:
            topo_phys = cond_static_phys[1].detach().cpu().numpy()
        elif cond_static_phys.shape[0] == 1:
            topo_phys = cond_static_phys[0].detach().cpu().numpy()
            lsm_phys = None

    fig, axes = plt.subplots(2, 6, figsize=(24, 8), constrained_layout=True)

    top_fields = [
        (target_model, "target (model space)"),
        (cond_prcp_model, "cond_dynamic ch0 (model)"),
        (pred_model, "forward prediction (model)"),
        (gen_model, "sampled field (model)"),
    ]
    if cond_temp_model is not None:
        top_fields.insert(2, (cond_temp_model, "cond_dynamic ch1 (model)"))

    bottom_fields = [
        (target_phys, "target (physical)"),
        (cond_prcp_phys, "cond_dynamic ch0 (physical)"),
        (pred_phys, "forward prediction (physical)"),
        (gen_phys, "sampled field (physical)"),
    ]
    if cond_temp_phys is not None:
        bottom_fields.insert(2, (cond_temp_phys, "cond_dynamic ch1 (physical)"))

    for col, (field, title) in enumerate(top_fields):
        ax = axes[0, col]
        im = ax.imshow(field, origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, (field, title) in enumerate(bottom_fields):
        ax = axes[1, col]
        im = ax.imshow(field, origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if lsm_phys is not None:
        ax = axes[0, 5]
        im = ax.imshow(lsm_phys, origin="lower", cmap="gray")
        ax.set_title("LSM (physical)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        axes[0, 5].axis("off")

    if topo_phys is not None:
        ax = axes[1, 5]
        im = ax.imshow(topo_phys, origin="lower", cmap="terrain")
        ax.set_title("topography (physical)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        axes[1, 5].axis("off")

    fig.suptitle("STRIDE Real Data Smoke Test", fontsize=14)
    plt.show()



def run_test_case(experiment_config_path: Path) -> None:
    print("\n---------------------------------")
    print(f"Experiment config: {experiment_config_path}")
    print("---------------------------------")

    exp_cfg = ExperimentConfig.from_yaml(experiment_config_path)
    compiler = ConfigCompiler(exp_cfg)
    compiled = compiler.compile()

    dataset_config_path = compiled.data_config_path
    model_config_path = compiled.model_config_path
    generation_config_path = compiled.generation_base_config_path

    if dataset_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce a data config for experiment: {experiment_config_path}"
        )
    if model_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce a model config for experiment: {experiment_config_path}"
        )
    if generation_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce a generation base config for experiment: {experiment_config_path}"
        )

    print(f"Dataset config:    {dataset_config_path}")
    print(f"Model config:      {model_config_path}")
    print(f"Generation config: {generation_config_path}")

    adapter_cfg = AdapterConfig.from_yaml(dataset_config_path)
    model_spec = ModelSpec.from_yaml(model_config_path)
    generation_kwargs = _load_generation_kwargs(generation_config_path)

    print("\nLoaded compiled configs successfully.")
    print(adapter_cfg)
    print(model_spec)
    print(f"Generation kwargs: {generation_kwargs}")

    print("\nBuilding dataset...")
    target_source = getattr(adapter_cfg, "target_source", None)
    if target_source == "NORCP_HR" or getattr(adapter_cfg, "scenario_name", None) is not None:
        from data_adapters.norcp.adapter import NorCPAdapter
        adapter = NorCPAdapter(adapter_cfg)
    else:
        from data_adapters.danra_era5_small.adapter import DanraEra5SmallAdapter
        adapter = DanraEra5SmallAdapter(adapter_cfg)

    dataset = adapter.build_dataset()
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print("Loaded first real sample successfully.")
    print(f"Date: {sample['meta'].get('date')}")
    print(f"Split: {sample['meta'].get('split_name')}")
    print(f"Domain: {sample['meta'].get('domain_tag')}")

    batch = build_batch_from_sample(sample)
    validate_model_data_contract(model_spec, batch)
    print("Validated real batch against compiled model contract.")

    print_tensor_info("target", batch["target"])
    print_tensor_info("cond_dynamic", batch["cond_dynamic"])
    print_tensor_info("cond_static", batch["cond_static"])
    if batch.get("time_features") is not None:
        print_tensor_info("time_features", batch["time_features"])

    print("\nBuilding model...")
    model = build_model(model_spec)
    model.eval()
    print(f"Built model of type: {type(model).__name__}")

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    sigma = torch.exp(
        torch.randn(batch["target"].shape[0], dtype=batch["target"].dtype) * 1.2 - 1.2
    )
    print_tensor_info("sigma", sigma)

    print("\nRunning forward pass on real data...")
    with torch.no_grad():
        pred = model(
            x=batch["target"],
            sigma=sigma,
            cond_dynamic=batch["cond_dynamic"],
            cond_static=batch["cond_static"],
            y=batch.get("time_features"),
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

    print("\nRunning EDM loss on real data...")
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

    print("\nRunning EDM sampler on real conditioning fields...")
    with torch.no_grad():
        sample_out = edm_sampler(
            model=model,
            cond_dynamic=batch["cond_dynamic"],
            cond_static=batch["cond_static"],
            y=batch.get("time_features"),
            **generation_kwargs,
        )

    if isinstance(sample_out, dict):
        generated = sample_out["sample"]
        print("Sampler returned intermediates.")
        print(f"Trajectory length: {len(sample_out['trajectory'])}")
    else:
        generated = sample_out

    print_tensor_info("generated_sample", generated)

    if generated.shape != expected_shape:
        raise RuntimeError(
            f"Generated sample shape mismatch: expected {tuple(expected_shape)}, got {tuple(generated.shape)}"
        )
    if torch.isnan(generated).any():
        raise RuntimeError("Generated sample contains NaNs")
    if torch.isinf(generated).any():
        raise RuntimeError("Generated sample contains Infs")
    print("Sampler shape check passed.")

    print("\nBack-transforming fields to physical space for plotting...")
    physical = inverse_transform_batch_for_plotting(
        dataset=dataset,
        batch=batch,
        pred=pred,
        generated=generated,
    )
    print_tensor_info("target_physical", physical["target"])
    print_tensor_info("pred_physical", physical["pred"])
    print_tensor_info("generated_physical", physical["generated"])
    print_tensor_info("cond_dynamic_physical", physical["cond_dynamic"])
    print_tensor_info("cond_static_physical", physical["cond_static"])

    plot_real_smoke_test(batch, pred, generated, physical)

    print("\nReal-data smoke test completed successfully.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("\n=================================")
    print("STRIDE real-data model smoke test")
    print("=================================")

    args = parse_args()
    experiment_config_paths = discover_experiment_configs(args.experiment_config)
    print(f"Discovered {len(experiment_config_paths)} experiment config(s) to test.")

    for experiment_config_path in experiment_config_paths:
        run_test_case(experiment_config_path)

if __name__ == "__main__":
    main()