"""
Experiment runner for the STRIDE full pipeline.

This runner executes a full experiment defined by a single
`ExperimentConfig`:

    training -> generation -> evaluation

The runner first compiles the experiment into fully resolved stage
configs using `ConfigCompiler`, then executes the existing stage
entrypoints as subprocesses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys
import time

from stride_core.pipeline.experiment_config import ExperimentConfig
from stride_core.pipeline.config_compiler import ConfigCompiler


# -----------------------------------------------------------------------------
# Typed summaries
# -----------------------------------------------------------------------------


@dataclass
class StageRunSummary:
    name: str
    enabled: bool
    config_path: str | None
    command: list[str] | None
    return_code: int | None
    duration_seconds: float | None
    success: bool
    skipped: bool
    stdout_path: str | None
    stderr_path: str | None


@dataclass
class ExperimentSummary:
    experiment_name: str
    experiment_config_path: str
    repo_root: str
    started_at_utc: str
    finished_at_utc: str | None
    total_duration_seconds: float | None
    success: bool
    stages: dict[str, StageRunSummary]


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


class ExperimentRunner:
    """Run a STRIDE experiment defined by `ExperimentConfig`."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.repo_root = self._infer_repo_root()

        self.logs_dir = (
            self.repo_root / "runs" / "pipeline_logs" / self.cfg.meta.name
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # compile configs
        compiler = ConfigCompiler(cfg)
        self.compiled = compiler.compile()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self) -> ExperimentSummary:
        start_ts = time.time()
        started_at = self._utc_now_iso()

        self._print_header("STRIDE full experiment")
        print(f"Experiment name: {self.cfg.meta.name}")
        print(f"Experiment config: {self.cfg.config_path}")
        print(f"Repository root: {self.repo_root}")
        print(f"Logs dir: {self.logs_dir}")

        stage_summaries: dict[str, StageRunSummary] = {}
        overall_success = True

        stage_plan = [
            ("training", self.cfg.stages.training, self.compiled.training_run_config_path),
            ("generation", self.cfg.stages.generation, self.compiled.generation_run_config_path),
            ("evaluation", self.cfg.stages.evaluation, self.compiled.evaluation_run_config_path),
        ]

        try:
            for stage_name, enabled, config_path in stage_plan:
                summary = self._run_stage(stage_name, enabled, config_path)
                stage_summaries[stage_name] = summary

                if not summary.success and not summary.skipped:
                    overall_success = False
                    break

        finally:
            finished_at = self._utc_now_iso()
            total_duration = time.time() - start_ts

            experiment_summary = ExperimentSummary(
                experiment_name=self.cfg.meta.name,
                experiment_config_path=str(self.cfg.config_path)
                if self.cfg.config_path is not None
                else "",
                repo_root=str(self.repo_root),
                started_at_utc=started_at,
                finished_at_utc=finished_at,
                total_duration_seconds=total_duration,
                success=overall_success,
                stages=stage_summaries,
            )

            self._save_experiment_summary(experiment_summary)

        if overall_success:
            print("\nFull pipeline completed successfully.")
        else:
            print("\nFull pipeline stopped because a stage failed.")

        return experiment_summary

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage_name: str,
        enabled: bool,
        config_path: Path,
    ) -> StageRunSummary:

        if not enabled:
            print(f"\n[{stage_name}] Skipped (disabled in experiment config).")

            return StageRunSummary(
                name=stage_name,
                enabled=False,
                config_path=str(config_path),
                command=None,
                return_code=None,
                duration_seconds=None,
                success=True,
                skipped=True,
                stdout_path=None,
                stderr_path=None,
            )

        stage_script = self._stage_script_path(stage_name)

        stdout_path = self.logs_dir / f"{stage_name}.stdout.log"
        stderr_path = self.logs_dir / f"{stage_name}.stderr.log"

        command = [sys.executable, str(stage_script), "--config", str(config_path)]

        self._print_header(f"Running stage: {stage_name}")
        print(f"Config: {config_path}")
        print(f"Script: {stage_script}")
        print(f"Stdout: {stdout_path}")
        print(f"Stderr: {stderr_path}")

        start_ts = time.time()

        with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_f:

            process = subprocess.run(
                command,
                cwd=str(self.repo_root),
                env=self._build_subprocess_env(),
                stdout=stdout_f,
                stderr=stderr_f,
                check=False,
                text=True,
            )

        duration = time.time() - start_ts
        success = process.returncode == 0

        print(
            f"[{stage_name}] return_code={process.returncode} duration={duration:.2f}s success={success}"
        )

        if not success:
            print(f"[{stage_name}] See logs for details:")
            print(f"  stdout: {stdout_path}")
            print(f"  stderr: {stderr_path}")

        return StageRunSummary(
            name=stage_name,
            enabled=True,
            config_path=str(config_path),
            command=command,
            return_code=int(process.returncode),
            duration_seconds=duration,
            success=success,
            skipped=False,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _stage_script_path(self, stage_name: str) -> Path:
        mapping = {
            "training": self.repo_root
            / "stride_core"
            / "training"
            / "training_main.py",
            "generation": self.repo_root
            / "stride_core"
            / "generation"
            / "generation_main.py",
            "evaluation": self.repo_root
            / "stride_core"
            / "evaluation"
            / "evaluation_main.py",
        }

        if stage_name not in mapping:
            raise KeyError(f"Unknown stage: {stage_name}")

        return mapping[stage_name]

    def _build_subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)

        repo_str = str(self.repo_root)
        existing = env.get("PYTHONPATH", "")

        if existing:
            if repo_str not in existing.split(os.pathsep):
                env["PYTHONPATH"] = repo_str + os.pathsep + existing
        else:
            env["PYTHONPATH"] = repo_str

        return env

    def _save_experiment_summary(self, summary: ExperimentSummary) -> None:
        summary_path = self.logs_dir / "experiment_summary.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._to_serializable(summary), f, indent=2)

        print(f"\nSaved experiment summary: {summary_path}")

    def _to_serializable(self, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return {
                key: self._to_serializable(val) for key, val in asdict(value).items()
            }

        if isinstance(value, dict):
            return {key: self._to_serializable(val) for key, val in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]

        return value

    def _print_header(self, title: str) -> None:
        line = "=" * len(title)
        print(f"\n{line}\n{title}\n{line}")

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()