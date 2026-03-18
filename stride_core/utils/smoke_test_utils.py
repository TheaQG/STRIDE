from configs.model_config import ModelSpec
from typing import Any

def validate_model_data_contract(spec: ModelSpec, batch: dict[str, Any]) -> None:
    target = batch["target"]
    cond_dynamic = batch["cond_dynamic"]
    cond_static = batch["cond_static"]

    # --- Target ---
    if target.shape[1] != spec.out_channels:
        raise AssertionError(
            f"Target channels mismatch: model expects {spec.out_channels}, "
            f"got {target.shape[1]}"
        )

    # --- Dynamic ---
    if cond_dynamic.shape[1] != spec.in_dynamic_channels:
        raise AssertionError(
            f"Dynamic conditioning mismatch: model expects {spec.in_dynamic_channels}, "
            f"got {cond_dynamic.shape[1]}"
        )

    # --- Static ---
    if spec.in_static_channels == 0 and cond_static is not None:
        raise AssertionError("Model expects no static conditioning but batch provides it")

    if spec.in_static_channels > 0:
        if cond_static is None:
            raise AssertionError("Model expects static conditioning but batch has None")
        if cond_static.shape[1] != spec.in_static_channels:
            raise AssertionError(
                f"Static conditioning mismatch: model expects {spec.in_static_channels}, "
                f"got {cond_static.shape[1]}"
            )

    # --- Spatial alignment ---
    if not spec.align_cond_to_target:
        if cond_dynamic.shape[-2:] != (spec.cond_height, spec.cond_width):
            raise AssertionError("Dynamic conditioning resolution mismatch")