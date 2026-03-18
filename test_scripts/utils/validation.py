from typing import Any

from stride_core.configs.model_config import ModelSpec



def validate_model_data_contract(spec: ModelSpec, batch: dict[str, Any]) -> None:
    target = batch["target"]
    cond_dynamic = batch["cond_dynamic"]
    cond_static = batch["cond_static"]
    time_features = batch.get("time_features")

    # --- Target ---
    if target.shape[1] != spec.out_channels:
        raise AssertionError(
            f"Target channels mismatch: model expects {spec.out_channels}, "
            f"got {target.shape[1]}"
        )
    if target.shape[-2:] != (spec.target_height, spec.target_width):
        raise AssertionError(
            "Target spatial mismatch: "
            f"model expects {(spec.target_height, spec.target_width)}, "
            f"got {tuple(target.shape[-2:])}"
        )

    # --- Dynamic ---
    if cond_dynamic.shape[1] != spec.in_dynamic_channels:
        raise AssertionError(
            f"Dynamic conditioning mismatch: model expects {spec.in_dynamic_channels}, "
            f"got {cond_dynamic.shape[1]}"
        )
    if cond_dynamic.shape[-2:] != (spec.cond_height, spec.cond_width):
        raise AssertionError(
            "Dynamic conditioning spatial mismatch: "
            f"model expects {(spec.cond_height, spec.cond_width)}, "
            f"got {tuple(cond_dynamic.shape[-2:])}"
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
        expected_static_hw = (
            (spec.target_height, spec.target_width)
            if spec.align_cond_to_target
            else (spec.cond_height, spec.cond_width)
        )
        if cond_static.shape[-2:] != expected_static_hw:
            raise AssertionError(
                "Static conditioning spatial mismatch: "
                f"expected {expected_static_hw}, got {tuple(cond_static.shape[-2:])}"
            )

    # --- Time features ---
    if spec.use_doy_film:
        if time_features is None:
            raise AssertionError("Model expects DOY time features but batch has None")
        if time_features.ndim != 2 or time_features.shape[1] != 2:
            raise AssertionError(
                "time_features mismatch: expected shape [B, 2], "
                f"got {tuple(time_features.shape)}"
            )
        if time_features.shape[0] != target.shape[0]:
            raise AssertionError(
                "time_features batch mismatch: "
                f"target batch={target.shape[0]}, time_features batch={time_features.shape[0]}"
            )