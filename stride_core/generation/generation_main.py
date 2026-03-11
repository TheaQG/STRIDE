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
from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.generation.generator import Generator, GenerationRunConfig


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

    print("\n======================")
    print("STRIDE generation main")
    print("======================")
    print(f"Repository root:   {Path(__file__).resolve().parents[2]}")
    print(f"Generation config: {config_path}")

    cfg = GenerationRunConfig.from_yaml(config_path)
    generator = Generator(cfg)
    generator.run()


if __name__ == "__main__":
    main()