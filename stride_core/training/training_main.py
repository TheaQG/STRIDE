from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Allow direct execution via:
# python stride_core/training/training_main.py --config configs/training/train_edm_small.yaml
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[2]

sys.path = [p for p in sys.path if Path(p).resolve() != THIS_DIR]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stride_core.training.trainer import Trainer


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
    return parser


def resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Training config does not exist: {path}")

    return path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    training_config_path = resolve_config_path(args.config)

    print("\n===================")
    print("STRIDE training main")
    print("===================")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Training config: {training_config_path}")

    trainer = Trainer(training_config_path)
    trainer.fit()


if __name__ == "__main__":
    main()
