"""
Thin orchestration wrapper for STRIDE evaluation.

Responsibilities
----------------
- parse one evaluation config
- initialize `Evaluator`
- call `run()`

This file should stay intentionally small, mirroring the role of
`training_main.py` and `generation_main.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.evaluation.evaluation_config import EvaluationRunConfig
from stride_core.evaluation.evaluator import Evaluator


DEFAULT_EVALUATION_CONFIG = "configs/evaluation/evaluation_base.yaml"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STRIDE evaluation from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_EVALUATION_CONFIG,
        help=(
            "Path to evaluation-run YAML config. "
            f"Default: {DEFAULT_EVALUATION_CONFIG}"
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()

    print("\n===================")
    print("STRIDE evaluation main")
    print("===================")
    print(f"Repository root:  {Path(__file__).resolve().parents[2]}")
    print(f"Evaluation config: {config_path}")

    cfg = EvaluationRunConfig.from_yaml(config_path)
    evaluator = Evaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()