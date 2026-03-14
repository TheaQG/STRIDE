"""
Thin launcher for the STRIDE full experiment pipeline.

This wrapper keeps the command-line interface to the full pipeline stable and
cluster-friendly. It adds a light validation / dry-run layer on top of the
actual pipeline entrypoint in `stride_core.pipeline.pipeline_main`.

Typical usage
-------------
Local:
    python cli/launch_pipeline.py --config configs/experiments/train_generate_evaluate_test.yaml

Dry run:
    python cli/launch_pipeline.py --config configs/experiments/train_generate_evaluate_test.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
import sys


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.pipeline.experiment_config import ExperimentConfig
from stride_core.pipeline.experiment_runner import ExperimentRunner
from stride_core.utils.logging_utils import setup_logging

DEFAULT_EXPERIMENT_CONFIG = "configs/experiments/train_generate_evaluate_test.yaml"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the STRIDE full pipeline in a cluster-friendly way."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help=(
            "Path to experiment YAML config. "
            f"Default: {DEFAULT_EXPERIMENT_CONFIG}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the pipeline config and stage config paths without running any stage.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------



def _resolve_base_config_path(raw_path: Path | None) -> Path | None:
    if raw_path is None:
        return None
    return raw_path.resolve()



def _format_header(title: str) -> str:
    line = "=" * len(title)
    return f"\n{line}\n{title}\n{line}"



def _launch_summary_lines(cfg: ExperimentConfig) -> list[str]:
    return [
        _format_header("STRIDE pipeline launcher"),
        f"Experiment name: {cfg.meta.name}",
        f"Experiment config: {cfg.config_path}",
        f"Repository root:  {Path(__file__).resolve().parents[1]}",
        f"Host:             {socket.gethostname()}",
        f"Python:           {sys.executable}",
        f"CWD:              {Path.cwd()}",
    ]



def _validate_stage(stage_name: str, enabled: bool, config_path: Path | None) -> str:
    if not enabled:
        return f"[{stage_name}] disabled"
    if config_path is None:
        raise ValueError(f"Stage '{stage_name}' is enabled but has no config path.")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Stage '{stage_name}' config does not exist: {config_path}"
        )
    return f"[{stage_name}] enabled -> {config_path}"


def _validate_base(name: str, path: Path | None) -> str:
    if path is None:
        return f"[{name}] not provided"
    if not path.exists():
        raise FileNotFoundError(f"Base config '{name}' does not exist: {path}")
    return f"[{name}] -> {path}"



def _run_dry_validation(cfg: ExperimentConfig) -> None:
    lines = [
        _format_header("Pipeline dry run"),
        _validate_base("model", _resolve_base_config_path(cfg.bases.model_config_path)),
        _validate_base("training", _resolve_base_config_path(cfg.bases.training_config_path)),
        _validate_base("generation", _resolve_base_config_path(cfg.bases.generation_config_path)),
        _validate_base("evaluation", _resolve_base_config_path(cfg.bases.evaluation_config_path)),
        _validate_base("data", _resolve_base_config_path(cfg.bases.data_config_path)),
        f"[training] enabled={cfg.stages.training}",
        f"[generation] enabled={cfg.stages.generation}",
        f"[evaluation] enabled={cfg.stages.evaluation}",
        "\nDry run successful. Experiment config and base config paths were resolved correctly.",
    ]
    print("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    cfg = ExperimentConfig.from_yaml(config_path)

    if args.dry_run:
        print("\n".join(_launch_summary_lines(cfg)))
        _run_dry_validation(cfg)
        return

    if cfg.meta.output_root is None:
        raise ValueError("cfg.meta.output_root is None. Please specify 'output_root' in your experiment config.")
    experiment_root = (cfg.meta.output_root / cfg.meta.name).resolve()
    log_file = experiment_root / "logs" / "pipeline.log"
    setup_logging(log_file)

    logger = logging.getLogger(__name__)
    for line in _launch_summary_lines(cfg):
        logger.info(line)

    logger.info("Launching STRIDE pipeline stages.")
    runner = ExperimentRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
