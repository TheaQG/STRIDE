

# STRIDE Evaluation Framework

The evaluation pipeline assesses the quality of generated ensembles.

Entry point:

```
stride_core/evaluation/evaluation_main.py
```

---

# Evaluation philosophy

STRIDE evaluates **ensembles rather than deterministic forecasts**.

This allows analysis of:

- uncertainty
- probabilistic calibration
- spatial realism

---

# Metric families

Metrics are grouped into four families.

## Probabilistic metrics

Evaluate ensemble calibration.

Examples:

- CRPS
- PIT histogram
- reliability diagram
- spread‑skill relationship

---

## Spatial metrics

Evaluate spatial realism of generated fields.

Examples:

- Power Spectral Density
- PSD slope
- Intensity‑Scale Skill Score
- SAL

---

## Climatology metrics

Evaluate distributional agreement with observations.

Examples:

- pixel value distributions
- extreme precipitation
- seasonal accumulations

---

## Temporal metrics

Evaluate temporal characteristics.

Examples:

- lag autocorrelation
- wet spell length
- dry spell length

---

# STRIDE Evaluation Framework

The STRIDE evaluation pipeline assesses the quality of **generated ensembles** produced by the diffusion model.

Evaluation is implemented as a standalone pipeline stage that can be run independently or as part of the full experiment pipeline.

Entry point:

```
stride_core/evaluation/evaluation_main.py
```

The evaluation stage loads generated samples, computes a large set of diagnostics, produces plots, and writes summary tables for quick inspection of experiment performance.

---

# Evaluation philosophy

STRIDE evaluates **ensembles rather than deterministic forecasts**.

This allows analysis of:

- probabilistic skill
- calibration
- spatial realism
- extreme behaviour
- climatological consistency

Instead of evaluating only a single generated sample, STRIDE evaluates **the full generated ensemble** and compares it against observations or reference data.

Many metrics therefore operate on:

```
forecast ensemble
vs
reference target
```

This design ensures that the model is evaluated both as:

- a probabilistic model
- a spatial generator

---

# Evaluation pipeline structure

The evaluation system is organized into several components:

```
stride_core/evaluation/

    evaluation_main.py
        Entry point

    evaluator.py
        Main evaluation orchestrator

    data_loading.py
        Loads generated ensembles and targets

    metrics/
        Implementations of all evaluation metrics

    plots/
        Visualization of evaluation diagnostics

    targets.py
        Defines evaluation targets

    summary.py
        Extracts key scalar metrics

    summary_writer.py
        Writes evaluation summaries
```

The evaluator performs the following steps:

1. Load generated ensemble data
2. Load reference targets
3. Compute metrics
4. Generate diagnostic plots
5. Write summary tables

Outputs are written to:

```
runs/<experiment>/evaluation/
```

Typical output structure:

```
evaluation/

    metrics/
        evaluation_metrics.json

    plots/
        probabilistic/
        spatial/
        climatology/
        temporal/

    summary/
        summary_metrics.csv
        summary_metrics.md
```

---

# Metric families

Metrics are grouped into four families.

This structure mirrors the scientific questions we want the model to answer.

---

# Probabilistic metrics

Evaluate **ensemble calibration and probabilistic skill**.

These metrics answer:

- Is the ensemble well calibrated?
- Does ensemble spread correspond to forecast error?

Implemented metrics include:

- CRPS
- CRPS decomposition
- PIT histogram
- rank histogram
- reliability diagram
- spread‑skill relationship

Generated plots include:

```
plots/probabilistic/

    pit_histogram.png
    rank_histogram.png
    spread_skill.png
    reliability_diagram.png
```

---

# Spatial metrics

Evaluate **spatial realism and multiscale structure** of generated fields.

These metrics answer:

- Does the model generate realistic spatial structures?
- Are precipitation systems spatially coherent?
- Are spatial scales reproduced correctly?

Implemented metrics include:

### Power Spectral Density (PSD)

Evaluates multiscale spatial structure.

In STRIDE the PSD is **ensemble-aware**:

1. PSD is computed for each ensemble member
2. The ensemble mean PSD is computed
3. Ensemble spread is estimated

This allows analysis of:

- structural realism
- ensemble structural uncertainty

### PSD slope

Measures scaling behaviour of precipitation fields.

### Intensity‑Scale Skill Score (ISS)

Evaluates how well precipitation intensity is reproduced across spatial scales.

### SAL

Structure–Amplitude–Location diagnostic for precipitation events.

Spatial plots are written to:

```
plots/spatial/

    psd.png
    iss.png
    sal_components.png
```

---

# Climatology metrics

Evaluate **distributional agreement** between generated data and observations.

These metrics answer:

- Does the model reproduce the precipitation distribution?
- Are extremes represented correctly?
- Is mean climatology preserved?

Implemented metrics include:

- pixel value distributions
- histogram comparison
- QQ plots
- extreme precipitation diagnostics
- wet‑day frequency
- annual precipitation totals
- seasonal accumulations

Generated plots include:

```
plots/climatology/

    pixel_distribution.png
    qq_plot.png
    extremes.png
    annual_precipitation.png
```

---

# Temporal metrics

Evaluate **temporal behaviour** of generated precipitation.

These metrics answer:

- Does the model reproduce persistence structures?
- Are wet and dry spells realistic?

Implemented metrics include:

- lag autocorrelation
- wet spell length distribution
- dry spell length distribution

Generated plots include:

```
plots/temporal/

    lag_autocorrelation.png
    wet_spell_length.png
    dry_spell_length.png
```

---

# Summary metrics

To allow rapid experiment comparison, STRIDE extracts a set of **key scalar diagnostics** from the full evaluation output.

These are written to:

```
summary/summary_metrics.csv
summary/summary_metrics.md
```

The summary focuses on the most scientifically informative diagnostics:

| Metric | Interpretation |
|------|------|
| PSD | multiscale spatial structure |
| ISS | spatial coherence of events |
| Extremes | representation of heavy precipitation |
| Wasserstein distance | overall distribution shift |
| Annual precipitation mean | climatological bias |
| Annual precipitation std | spatial variability |
| CRPS | probabilistic forecast skill |
| PIT KS statistic | calibration |

These summary tables allow quick comparison between experiments.

---

# Ensemble handling

A key design principle of STRIDE evaluation is that **metrics operate on ensembles rather than deterministic fields**.

Where relevant, metrics are computed as:

```
metric(forecast ensemble, reference target)
```

Examples:

- CRPS uses the full ensemble
- PSD averages spectra across ensemble members
- calibration diagnostics use ensemble distributions

This design ensures that evaluation reflects the model's **generative nature**.

---

# Configuration

Evaluation behaviour is controlled through:

```
configs/evaluation/evaluation_base.yaml
```

This config enables or disables metric families and plotting.

Example:

```
evaluation:
  probabilistic:
    crps: true
    pit_histogram: true

  spatial:
    psd: true
    iss: true

  climatology:
    qq_plot: true

  temporal:
    lag_autocorrelation: true
```

This allows experiments to enable only the diagnostics relevant for a given study.

---

# Typical workflow

Evaluation is typically run after generation:

```
training → generation → evaluation
```

However it can also be run independently when generated samples already exist.

Example:

```
python stride_core/evaluation/evaluation_main.py \
    --config compiled_configs/evaluation_resolved.yaml
```

---

# Future extensions

Planned extensions include:

- event‑based extreme analysis
- storyline‑based extreme evaluation
- out‑of‑distribution testing
- ensemble SAL diagnostics

These features will further strengthen evaluation of generative climate models.