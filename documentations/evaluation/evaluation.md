

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
