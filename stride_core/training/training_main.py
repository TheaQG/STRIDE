from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import torch.distributed as dist
import yaml


# Allow direct execution via:
# python stride_core/training/training_main.py --config configs/training/train_edm_small.yaml
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[2]

sys.path = [p for p in sys.path if Path(p).resolve() != THIS_DIR]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stride_core.training.trainer import Trainer
from stride_core.utils.distributed import (
    cleanup,
    initialize,
    is_distributed,
    is_main_process,
    local_rank,
    rank,
    world_size,
)
from stride_core.utils.logging_utils import get_stage_log_file, setup_logging

logger = logging.getLogger(__name__)


DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "training" / "training_base.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run STRIDE training from a training YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_TRAINING_CONFIG),
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help="Node-local process rank supplied by a distributed launcher.",
    )
    return parser


def resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Training config does not exist: {path}")

    return path


def _resolve_experiment_root_from_training_config(training_config_path: Path) -> Path:
    """
    Infer the experiment root from a compiled training run config.

    Expected layout:
        runs/<experiment>/compiled_configs/training_run_resolved.yaml

    In fallback cases, use the parent directory of the configured training
    output dir if available.
    """
    with open(training_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected training config to load into a dict, got {type(payload)}"
        )

    training_cfg = payload.get("training")
    if isinstance(training_cfg, dict):
        run_cfg = training_cfg.get("run", {})
        if isinstance(run_cfg, dict):
            output_dir = run_cfg.get("output_dir")
            if output_dir is not None:
                return Path(output_dir).resolve().parent

    return training_config_path.resolve().parents[1]


def _read_distributed_runtime_options(
    training_config_path: Path,
) -> tuple[str | bool, str]:
    with open(training_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected training config to load into a dict, got {type(payload)}"
        )

    training_cfg = payload.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Expected 'training' section to be a dict")

    distributed_cfg = training_cfg.get("distributed", {})
    if not isinstance(distributed_cfg, dict):
        raise ValueError("Expected 'training.distributed' to be a dict")

    enabled = distributed_cfg.get("enabled", "auto")
    backend = str(distributed_cfg.get("backend", "nccl"))
    return enabled, backend


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    training_config_path = resolve_config_path(args.config)
    if args.local_rank is not None:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

    distributed_enabled, distributed_backend = _read_distributed_runtime_options(
        training_config_path
    )

    try:
        initialize(enabled=distributed_enabled, backend=distributed_backend)

        experiment_root = _resolve_experiment_root_from_training_config(
            training_config_path
        )
        shared_log_file = get_stage_log_file(experiment_root, "training")
        log_file = shared_log_file
        if not is_main_process():
            log_file = shared_log_file.with_name(
                f"{shared_log_file.stem}_rank_{rank():04d}{shared_log_file.suffix}"
            )
        setup_logging(log_file)

        line = "=" * len("STRIDE training main")
        logger.info(f"\n{line}\nSTRIDE training main\n{line}")
        logger.info(f"Repository root: {REPO_ROOT}")
        logger.info(f"Training config: {training_config_path}")
        logger.info(f"Experiment root: {experiment_root}")
        logger.info(f"Training log file: {log_file}")
        logger.info(
            "Distributed runtime: "
            f"enabled={is_distributed()} rank={rank()} "
            f"local_rank={local_rank()} world_size={world_size()} "
            f"backend={dist.get_backend() if dist.is_initialized() else None}"
        )

        trainer = Trainer(training_config_path)
        trainer.fit()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
