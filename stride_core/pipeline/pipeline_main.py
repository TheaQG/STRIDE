"""
Thin CLI entrypoint for the STRIDE full experiment pipeline.

This mirrors the structure used by:

- training_main.py
- generation_main.py
- evaluation_main.py

Responsibilities
----------------
- Parse the experiment YAML config
- Initialize `ExperimentRunner` from `ExperimentConfig`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this script directly
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.pipeline.experiment_runner import ExperimentRunner


DEFAULT_PIPELINE_CONFIG = "configs/experiments/train_generate_evaluate_test.yaml"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full STRIDE experiment pipeline (training → generation → evaluation)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_PIPELINE_CONFIG,
        help=(
            "Path to pipeline experiment YAML config. "
            f"Default: {DEFAULT_PIPELINE_CONFIG}"
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    print("\n=============================")
    print("STRIDE full experiment runner")
    print("=============================")

    print(f"Experiment config: {config_path}")

    cfg = ExperimentConfig.from_yaml(config_path)

    runner = ExperimentRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()