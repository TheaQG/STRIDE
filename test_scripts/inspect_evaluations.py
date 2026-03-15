"""
Create a synthetic evaluation output and run the STRIDE evaluation plotting /
summary-writing stack on it.

Purpose
-------
This script is a fast inspection tool for evaluation development. It does *not*
run a model and it does *not* require real generation outputs. Instead, it
constructs a fake `evaluation_metrics.json`-like payload in the same broad shape
that `Evaluator` produces, then calls the family plotters and summary writer.

Why this design
---------------
Your current plotting layer consumes computed metric outputs, not raw model
samples. So for plot-development smoke tests, the cleanest path is to synthesize
post-metric evaluation payloads directly rather than trying to fake the full
model generation directory structure.

Outputs
-------
The script writes into:
    runs/smoke_tests/inspect_evaluations/

including:
    evaluation_metrics_fake.json
    summary_metrics.csv
    summary_metrics.md
    figures/probabilistic/*.png
    figures/spatial/*.png

Usage
-----
From repo root:
    python test_scripts/inspect_evaluations.py
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stride_core.evaluation.plots.plot_climatology import plot_climatology
from stride_core.evaluation.plots.plot_probabilistic import plot_probabilistic
from stride_core.evaluation.plots.plot_spatial import plot_spatial
from stride_core.evaluation.plots.plot_temporal import plot_temporal
from stride_core.evaluation.summary import summary_metrics_as_dicts
from stride_core.evaluation.summary_writer import (
    write_summary_csv,
    write_summary_markdown,
)
from stride_core.evaluation.targets import summarize_target_routing


OUTPUT_DIR = REPO_ROOT / "runs" / "smoke_tests" / "inspect_evaluations"
FIGURES_DIR = OUTPUT_DIR / "figures"


def print_section(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


class _DummyPlottingCfg:
    def __init__(self) -> None:
        self.enabled = True
        self.families = {}
        self.save_quicklooks = False


class _DummyCfg:
    def __init__(self) -> None:
        self.plotting = _DummyPlottingCfg()


CFG = _DummyCfg()


# -----------------------------------------------------------------------------
# Synthetic metric payload builders
# -----------------------------------------------------------------------------


def _make_psd_payload(case_index: int) -> dict:
    scales = np.array([4, 8, 16, 32, 64, 128], dtype=float)
    target = np.array([14.0, 7.2, 3.5, 1.8, 0.95, 0.55], dtype=float)

    # Four synthetic members with small structured deviations around the target.
    base_offsets = np.array([-0.08, -0.03, 0.04, 0.09], dtype=float)
    member_curves = []
    for member_idx, offset in enumerate(base_offsets):
        slope_mod = 1.0 + 0.04 * case_index - 0.02 * member_idx
        member_curves.append(target * slope_mod * (1.0 + offset))

    member_curves = np.asarray(member_curves, dtype=float)
    forecast_mean = np.mean(member_curves, axis=0)

    forecast_slope = -1.82 + 0.03 * case_index
    target_slope = -1.95

    return {
        "scales_km": scales.tolist(),
        "forecast_psd": forecast_mean.tolist(),
        "target_psd": target.tolist(),
        "forecast_member_psd": member_curves.tolist(),
        "forecast_slope": float(forecast_slope),
        "target_slope": float(target_slope),
        "slope_difference": float(abs(forecast_slope - target_slope)),
    }



def _make_iss_payload(case_index: int) -> dict:
    thresholds = np.array([1.0, 5.0, 10.0, 20.0], dtype=float)
    scales = np.array([5.0, 10.0, 20.0, 40.0], dtype=float)
    base = np.array(
        [
            [0.82, 0.78, 0.70, 0.63],
            [0.76, 0.72, 0.65, 0.58],
            [0.70, 0.67, 0.60, 0.54],
            [0.64, 0.60, 0.55, 0.49],
        ],
        dtype=float,
    )
    matrix = base - 0.02 * case_index

    return {
        "thresholds_mm": thresholds.tolist(),
        "scales_km": scales.tolist(),
        "iss_matrix": matrix.tolist(),
        "iss_mean": float(np.mean(matrix)),
    }



def _make_sal_payload(case_index: int) -> dict:
    s_val = -0.18 + 0.05 * case_index
    a_val = 0.10 - 0.03 * case_index
    l_val = 0.14 + 0.02 * case_index
    return {
        "S": float(s_val),
        "A": float(a_val),
        "L": float(l_val),
        "aggregate": float(np.mean(np.abs([s_val, a_val, l_val]))),
    }



def _make_probabilistic_payloads(case_index: int) -> dict:
    rng = np.random.default_rng(100 + case_index)

    pit_values = np.clip(rng.beta(2.2 + 0.15 * case_index, 2.0, size=700), 0.0, 1.0)
    rank_counts = np.array([78, 86, 94, 97, 99, 95, 90, 84], dtype=float) - 2.0 * case_index

    probs = np.linspace(0.05, 0.95, 10)
    obs = np.clip(probs + 0.05 * np.sin(np.linspace(0, np.pi, probs.size)) - 0.01 * case_index, 0.0, 1.0)

    spread = np.array([0.4, 0.8, 1.2, 1.6, 2.0], dtype=float)
    skill = np.array([0.48, 0.86, 1.18, 1.55, 1.92], dtype=float) + 0.03 * case_index

    return {
        "crps": {"crps": float(0.36 + 0.04 * case_index)},
        "crps_decomposition": {
            "crps": float(0.36 + 0.04 * case_index),
            "reliability": float(0.09 + 0.01 * case_index),
            "resolution": float(0.12 + 0.01 * case_index),
            "uncertainty": float(0.48),
        },
        "spread_skill": {
            "spread": spread.tolist(),
            "skill": skill.tolist(),
            "mean_spread": float(np.mean(spread)),
            "mean_skill": float(np.mean(skill)),
        },
        "pit_histogram": {
            "pit_values": pit_values.tolist(),
            "ks_statistic": float(0.06 + 0.01 * case_index),
        },
        "rank_histogram": {
            "rank_counts": rank_counts.tolist(),
        },
        "reliability_diagram": {
            "forecast_probabilities": probs.tolist(),
            "observed_frequencies": obs.tolist(),
            "score": float(np.mean(np.abs(obs - probs))),
        },
    }



def _make_climatology_payloads(case_index: int) -> dict:
    q_levels = [0.95, 0.99, 0.999, 0.9999]
    q_target = np.array([12.0, 22.0, 38.0, 61.0], dtype=float)
    q_forecast = q_target * (0.96 + 0.01 * case_index)

    rng = np.random.default_rng(200 + case_index)
    target_samples = rng.gamma(shape=1.8, scale=3.2, size=1500)
    forecast_samples = target_samples * (0.94 + 0.02 * case_index)

    quantile_axis = np.linspace(0.01, 0.99, 60)
    target_quantiles = np.quantile(target_samples, quantile_axis)
    forecast_quantiles = np.quantile(forecast_samples, quantile_axis)

    return {
        "histogram_comparison": {
            "forecast_values": forecast_samples.tolist(),
            "target_values": target_samples.tolist(),
            "wasserstein": float(0.18 + 0.02 * case_index),
            "distance": float(0.18 + 0.02 * case_index),
        },
        "qq_plot": {
            "forecast_quantiles": forecast_quantiles.tolist(),
            "target_quantiles": target_quantiles.tolist(),
        },
        "extremes": {
            str(q): {"forecast": float(f), "target": float(t)}
            for q, f, t in zip(q_levels, q_forecast, q_target)
        },
        "annual_precipitation_sum": {
            "forecast_mean": float(842.0 + 11.0 * case_index),
            "target_mean": float(860.0),
            "forecast_std": float(118.0 + 3.0 * case_index),
            "target_std": float(125.0),
            "mean_bias": float((842.0 + 11.0 * case_index) - 860.0),
            "std_ratio": float((118.0 + 3.0 * case_index) / 125.0),
        },
    }



def _make_temporal_payloads(case_index: int) -> dict:
    lags = np.array([1, 3, 7, 14], dtype=float)
    autocorr = np.array([0.62, 0.43, 0.24, 0.11], dtype=float) - 0.02 * case_index

    rng = np.random.default_rng(300 + case_index)
    wet_spell_lengths = rng.integers(low=1, high=9, size=180)
    dry_spell_lengths = rng.integers(low=1, high=12, size=220)

    return {
        "lag_autocorrelation": {
            "lags": lags.tolist(),
            "autocorrelation": autocorr.tolist(),
            "value": float(np.mean(np.abs(autocorr))),
        },
        "wet_spell_length": {
            "spell_lengths": wet_spell_lengths.tolist(),
        },
        "dry_spell_length": {
            "spell_lengths": dry_spell_lengths.tolist(),
        },
    }



def build_fake_metrics(num_cases: int = 3) -> dict:
    metrics: dict = {
        "target_routing": summarize_target_routing(),
        "summary": [],
        "probabilistic": {"case_level": {}, "aggregate": {}},
        "spatial": {"case_level": {}, "aggregate": {}},
        "climatology": {"case_level": {}, "aggregate": {}},
        "temporal": {"case_level": {}, "aggregate": {}},
        "sigma_star": {},
    }

    for case_index in range(num_cases):
        case_id = f"fake_case_{case_index:03d}"

        probabilistic = _make_probabilistic_payloads(case_index)
        spatial = {
            "psd": _make_psd_payload(case_index),
            "iss": _make_iss_payload(case_index),
            "sal": _make_sal_payload(case_index),
        }
        climatology = _make_climatology_payloads(case_index)
        temporal = _make_temporal_payloads(case_index)

        metrics["probabilistic"]["case_level"][case_id] = {
            "forecast_shape": (12, 128, 128),
            "target_shape": (128, 128),
            "metric_products": {
                "crps": "ensemble_members",
                "crps_decomposition": "ensemble_members",
                "spread_skill": "ensemble_members",
                "pit_histogram": "ensemble_members",
                "rank_histogram": "ensemble_members",
                "reliability_diagram": "ensemble_members",
            },
            **probabilistic,
        }

        metrics["spatial"]["case_level"][case_id] = {
            "forecast_shape": (12, 128, 128),
            "target_shape": (128, 128),
            "metric_products": {
                "psd": "ensemble_members",
                "iss": "pmm",
                "sal": "pmm",
            },
            **spatial,
        }

        metrics["climatology"]["case_level"][case_id] = {
            "forecast_shape": (128, 128),
            "target_shape": (128, 128),
            "metric_products": {
                "histogram_comparison": "pmm",
                "qq_plot": "pmm",
                "extremes": "pmm",
                "annual_precipitation_sum": "pmm",
            },
            **climatology,
        }

        metrics["temporal"]["case_level"][case_id] = {
            "forecast_shape": (128, 128),
            "target_shape": (128, 128),
            "metric_products": {
                "lag_autocorrelation": "pmm",
                "wet_spell_length": "pmm",
                "dry_spell_length": "pmm",
            },
            **temporal,
        }

    metrics["summary"] = summary_metrics_as_dicts(metrics)
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print_section("STRIDE evaluation inspection script")
    print(f"Repo root:   {REPO_ROOT}")
    print(f"Output dir:  {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Building synthetic metrics payload")
    metrics = build_fake_metrics(num_cases=3)
    print(f"Built fake metrics for {len(metrics['probabilistic']['case_level'])} cases")

    print_section("Writing fake evaluation artifacts")
    metrics_json_path = OUTPUT_DIR / "evaluation_metrics_fake.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved fake metrics JSON: {metrics_json_path}")

    summary_rows = metrics.get("summary", [])
    csv_path = write_summary_csv(summary_rows, OUTPUT_DIR)
    md_path = write_summary_markdown(summary_rows, OUTPUT_DIR)
    print(f"Saved summary CSV:      {csv_path}")
    print(f"Saved summary Markdown: {md_path}")

    print_section("Generating plots")
    probabilistic_paths = plot_probabilistic(
        metrics=metrics,
        output_dir=FIGURES_DIR / "probabilistic",
        cfg=CFG,
    )
    spatial_paths = plot_spatial(
        metrics=metrics,
        output_dir=FIGURES_DIR / "spatial",
        cfg=CFG,
    )
    climatology_paths = plot_climatology(
        metrics=metrics,
        output_dir=FIGURES_DIR / "climatology",
        cfg=CFG,
    )
    temporal_paths = plot_temporal(
        metrics=metrics,
        output_dir=FIGURES_DIR / "temporal",
        cfg=CFG,
    )

    for path in probabilistic_paths:
        print(f"Saved probabilistic figure: {path}")
    for path in spatial_paths:
        print(f"Saved spatial figure:       {path}")
    for path in climatology_paths:
        print(f"Saved climatology figure:  {path}")
    for path in temporal_paths:
        print(f"Saved temporal figure:     {path}")

    print_section("Inspection run completed successfully")


if __name__ == "__main__":
    main()