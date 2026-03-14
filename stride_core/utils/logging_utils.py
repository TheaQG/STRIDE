import logging
from pathlib import Path
import sys


def setup_logging(
    log_file: Path,
    *,
    level: int = logging.INFO,
    logger_name: str | None = None,
    reset_handlers: bool = True,
) -> logging.Logger:
    """
    Configure logging for STRIDE.

    This sets up logging to both stdout and a file, typically used for:

        runs/<experiment>/logs/
            pipeline.log
            training.log
            generation.log
            evaluation.log

    Parameters
    ----------
    log_file:
        Path to the log file.
    level:
        Logging level (default: INFO).
    logger_name:
        Optional logger name. If None, the root logger is configured.
    reset_handlers:
        If True, existing handlers are cleared before configuring logging.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """

    log_file = Path(log_file).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Prevent logs from being duplicated in parent loggers
    logger.propagate = False

    if reset_handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    # Use stdout rather than the default stderr so pipeline-captured stage logs
    # land in `<stage>.stdout.log` instead of `<stage>.stderr.log`.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_stage_log_file(experiment_root: Path, stage: str) -> Path:
    """
    Resolve the log file path for a given stage.

    Parameters
    ----------
    experiment_root:
        Root directory of the experiment.
    stage:
        One of: "pipeline", "training", "generation", "evaluation".

    Returns
    -------
    Path
        Path to the stage log file.
    """

    logs_dir = Path(experiment_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir / f"{stage}.log"