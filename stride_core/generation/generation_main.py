"""
Entrypoint for STRIDE post-training generation.

This file mirrors the role of `training_main.py`: it should stay very thin and
only handle top-level orchestration:
- parse one generation-run config
- initialize `Generator`
- call `run()`

Generation modes are selected through config, not through separate entrypoints.
The current first milestone is the clean path:
    checkpoint -> requested split -> generated outputs on disk
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import yaml


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.generation.generator import Generator, GenerationRunConfig
from stride_core.utils.logging_utils import get_stage_log_file, setup_logging
logger = logging.getLogger(__name__)
def _resolve_experiment_root_from_generation_config(generation_config_path: Path) -> Path:
    """
    Infer the experiment root from a compiled generation run config.

    Expected layout:
        runs/<experiment>/compiled_configs/generation_run_resolved.yaml

    In fallback cases, use the parent directory of the configured generation
    output dir if available.
    """
    with open(generation_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected generation config to load into a dict, got {type(payload)}"
        )

    generation_cfg = payload.get("generation")
    if isinstance(generation_cfg, dict):
        run_cfg = generation_cfg.get("run", {})
        if isinstance(run_cfg, dict):
            output_dir = run_cfg.get("output_dir")
            if output_dir is not None:
                return Path(output_dir).resolve().parent

    return generation_config_path.resolve().parents[1]


DEFAULT_GENERATION_CONFIG = "configs/generation/generation_base.yaml"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STRIDE post-training generation from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_GENERATION_CONFIG,
        help=(
            "Path to generation-run YAML config. "
            f"Default: {DEFAULT_GENERATION_CONFIG}"
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    experiment_root = _resolve_experiment_root_from_generation_config(config_path)
    log_file = get_stage_log_file(experiment_root, "generation")
    setup_logging(log_file)

    line = "=" * len("STRIDE generation main")
    logger.info(f"\n{line}\nSTRIDE generation main\n{line}")
    logger.info(f"Repository root:   {Path(__file__).resolve().parents[2]}")
    logger.info(f"Generation config: {config_path}")
    logger.info(f"Experiment root:   {experiment_root}")
    logger.info(f"Generation log:    {log_file}")

    cfg = GenerationRunConfig.from_yaml(config_path)
    generator = Generator(cfg)
    generator.run()


if __name__ == "__main__":
    main()