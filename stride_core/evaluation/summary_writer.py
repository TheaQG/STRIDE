"""
Writers for compact STRIDE evaluation summary tables.

This module is intentionally thin: it takes already-extracted summary rows and
writes them to simple artifact formats such as CSV and Markdown.

Design goals
------------
- keep file writing separate from summary extraction
- accept plain dictionaries so callers do not need to depend on dataclasses
- produce stable, readable column ordering
- stay dependency-free
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv


SUMMARY_COLUMNS = [
    "family",
    "metric",
    "display_name",
    "value",
    "direction",
    "ideal_value",
    "product",
    "notes",
]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path



def _normalize_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in summary_rows:
        normalized_row = {column: row.get(column) for column in SUMMARY_COLUMNS}
        normalized.append(normalized_row)
    return normalized



def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)



def _markdown_escape(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ")


# -----------------------------------------------------------------------------
# Public writers
# -----------------------------------------------------------------------------


def write_summary_csv(
    summary_rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    filename: str = "summary_metrics.csv",
) -> Path:
    """
    Write summary metrics to CSV.

    Parameters
    ----------
    summary_rows:
        Plain-dict summary rows, typically from
        ``summary_metrics_as_dicts(...)``.
    output_dir:
        Directory where the CSV file will be written.
    filename:
        Output filename.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    out_dir = _ensure_output_dir(output_dir)
    out_path = out_dir / filename
    rows = _normalize_summary_rows(summary_rows)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(value) for key, value in row.items()})

    return out_path



def write_summary_markdown(
    summary_rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    filename: str = "summary_metrics.md",
    title: str = "# STRIDE Evaluation Summary",
) -> Path:
    """
    Write summary metrics as a Markdown table.

    Parameters
    ----------
    summary_rows:
        Plain-dict summary rows, typically from
        ``summary_metrics_as_dicts(...)``.
    output_dir:
        Directory where the Markdown file will be written.
    filename:
        Output filename.
    title:
        Optional Markdown heading written above the table.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    out_dir = _ensure_output_dir(output_dir)
    out_path = out_dir / filename
    rows = _normalize_summary_rows(summary_rows)

    header = "| " + " | ".join(SUMMARY_COLUMNS) + " |"
    divider = "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |"

    lines = [title, "", header, divider]
    for row in rows:
        formatted = [_markdown_escape(_format_cell(row[column])) for column in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join(formatted) + " |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return out_path
