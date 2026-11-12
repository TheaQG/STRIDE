"""Effective EDM settings shared by compilation, generation and previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationConfig:
    num_steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1.0
    return_intermediates: bool = False

    @classmethod
    def from_yaml(cls, generation_config_path: str | Path) -> "GenerationConfig":
        path = Path(generation_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Generation config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected generation config YAML to load into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(
        cls, cfg: dict[str, Any], *, fallback: GenerationConfig | None = None,
    ) -> GenerationConfig:
        cfg = _mapping(cfg, "generation config")
        gen_cfg = _mapping(cfg.get("generation", cfg), "generation")
        sampler = _mapping(gen_cfg.get("sampler", {}), "generation.sampler")
        output = _mapping(gen_cfg.get("output", {}), "generation.output")
        defaults = (fallback if fallback is not None else cls()).to_sampler_kwargs()
        edm_keys = defaults.keys() - {"return_intermediates"}

        if "generation" in cfg:
            _reject_misplaced(cfg, "config root", defaults.keys() | {"sampler"})
        if "generation_run" in cfg:
            cls.validate_run_placement(cfg["generation_run"])
        _reject_misplaced(output, "generation.output", edm_keys | {"sampler"})
        for section in sorted(_RUN_CONTROL_SECTIONS - {"output"}):
            _reject_misplaced(gen_cfg.get(section, {}), f"generation.{section}", defaults.keys() | {"sampler"})
        ensemble_keys = {"ensemble_size", "use_fixed_seed", "base_seed"} & sampler.keys()
        if ensemble_keys:
            raise ValueError(f"Move {', '.join(sorted(ensemble_keys))} from generation.sampler to generation_run.sampling")
        unknown = sampler.keys() - defaults.keys() - {"name"}
        if unknown:
            raise ValueError(f"Unknown generation.sampler keys: {', '.join(sorted(map(str, unknown)))}")
        for source, path in ((sampler, "generation.sampler"), (gen_cfg, "generation")):
            if source.get("name", "edm") != "edm":
                raise ValueError(f"{path}.name must be 'edm'")

        resolved = {}
        for key, default in defaults.items():
            if key == "return_intermediates":
                # The legacy smoke loader also accepted this flag inside sampler.
                sources = (output, sampler, gen_cfg)
            else:
                sources = (sampler, gen_cfg)
            values = [
                _parse_value(key, source[key]) for source in sources
                if key in source and not (source[key] is None and key in _LEGACY_NULL_FIELDS)
            ]
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"Conflicting sampler values for {key!r}. Keep one value under "
                    "generation.sampler (generation.output for return_intermediates)."
                )
            resolved[key] = values[0] if values else default

        if resolved["sigma_max"] < resolved["sigma_min"]:
            raise ValueError("generation.sampler.sigma_max must be >= sigma_min")
        if resolved["S_max"] < resolved["S_min"]:
            raise ValueError("generation.sampler.S_max must be >= S_min")
        return cls(**resolved)

    @classmethod
    def validate_experiment_overrides(cls, overrides: dict[str, Any]) -> None:
        path = "generation.overrides"
        cls.validate_run_placement(
            {key: value for key, value in overrides.items() if key not in {"generation", "generation_run"}},
            path,
        )
        cls.validate_run_placement(overrides.get("generation_run", {}), f"{path}.generation_run")
        path += ".generation"
        generation = _mapping(overrides.get("generation", {}), path)
        keys = cls().to_sampler_kwargs().keys() | {"name"}
        for section, settings, allowed in (
            (path, generation, keys),
            (f"{path}.sampler", generation.get("sampler", {}), keys),
            (f"{path}.output", generation.get("output", {}), {"return_intermediates"}),
        ):
            for key, value in _mapping(settings, section).items():
                if key in allowed and key != "S_max" and value is None:
                    raise ValueError(f"{section}.{key} must not be null in experiment overrides")

    @classmethod
    def validate_run_placement(cls, run: Any, path: str = "generation_run") -> None:
        """Catch EDM settings in run controls before they can be ignored."""
        run = _mapping(run, path)
        keys = cls().to_sampler_kwargs().keys() | {"sampler", "generation"}
        _reject_misplaced(run, path, keys)
        for section in sorted(_RUN_CONTROL_SECTIONS):
            _reject_misplaced(run.get(section, {}), f"{path}.{section}", keys)

    def to_sampler_kwargs(self) -> dict[str, Any]:
        return asdict(self)

    def to_metadata(self) -> dict[str, Any]:
        settings = self.to_sampler_kwargs()
        if self.S_max == float("inf"):
            settings["S_max"] = "inf"  # Valid JSON, and accepted by the parser.
        return settings


_RUN_CONTROL_SECTIONS = {
    "checkpoint", "configs", "data", "device", "limits", "output", "outputs", "paths", "sampling",
}
_LEGACY_NULL_FIELDS = {"num_steps", "sigma_min", "sigma_max", "rho"}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _reject_misplaced(section: Any, path: str, keys: set[str]) -> None:
    misplaced = keys & _mapping(section, path).keys()
    if misplaced:
        raise ValueError(
            f"Misplaced sampler settings in {path}: {', '.join(sorted(misplaced))}. "
            "Use generation.sampler for EDM settings and generation.output.return_intermediates."
        )


def _parse_value(key: str, value: Any) -> int | float | bool:
    if key == "return_intermediates":
        if not isinstance(value, bool):
            raise ValueError("generation.output.return_intermediates must be a boolean")
        return value
    path = f"generation.sampler.{key}"
    if key == "S_max" and value is None:
        return float("inf")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{path} must be numeric")
    try:
        if key == "num_steps":
            number = int(value)
            if isinstance(value, float) and number != value:
                raise ValueError
        else:
            number = float(value)
    except (ValueError, OverflowError) as exc:
        expected = "an integer" if key == "num_steps" else "numeric"
        raise ValueError(f"{path} must be {expected}") from exc
    if key == "S_max" and number == float("inf"):
        return number
    if not isfinite(number):
        raise ValueError(f"{path} must be finite")
    if key in {"S_churn", "S_min", "S_max"}:
        if number < 0:
            raise ValueError(f"{path} must be >= 0")
    elif number <= 0:
        raise ValueError(f"{path} must be > 0")
    return number
