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
import logging
from pathlib import Path
import sys

import yaml


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.evaluation.evaluation_config import EvaluationRunConfig
from stride_core.evaluation.evaluator import Evaluator
from stride_core.utils.logging_utils import get_stage_log_file, setup_logging

logger = logging.getLogger(__name__)
def _resolve_experiment_root_from_evaluation_config(evaluation_config_path: Path) -> Path:
    """
    Infer the experiment root from a compiled evaluation run config.

    Expected layout:
        runs/<experiment>/compiled_configs/evaluation_run_resolved.yaml

    In fallback cases, use the parent directory of the configured evaluation
    output dir if available.
    """
    with open(evaluation_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected evaluation config to load into a dict, got {type(payload)}"
        )

    evaluation_cfg = payload.get("evaluation")
    if isinstance(evaluation_cfg, dict):
        run_cfg = evaluation_cfg.get("run", {})
        if isinstance(run_cfg, dict):
            output_dir = run_cfg.get("output_dir")
            if output_dir is not None:
                return Path(output_dir).resolve().parent

    return evaluation_config_path.resolve().parents[1]


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
    experiment_root = _resolve_experiment_root_from_evaluation_config(config_path)
    log_file = get_stage_log_file(experiment_root, "evaluation")
    setup_logging(log_file)

    line = "=" * len("STRIDE evaluation main")
    logger.info(f"\n{line}\nSTRIDE evaluation main\n{line}")
    logger.info(f"Repository root:   {Path(__file__).resolve().parents[2]}")
    logger.info(f"Evaluation config: {config_path}")
    logger.info(f"Experiment root:   {experiment_root}")
    logger.info(f"Evaluation log:    {log_file}")

    cfg = EvaluationRunConfig.from_yaml(config_path)
    evaluator = Evaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()