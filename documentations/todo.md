# STRIDE Development Roadmap

This document tracks upcoming development tasks and priorities for the STRIDE framework.

The goal is to stabilize the full training → generation → evaluation pipeline and then expand the modeling capabilities and supported datasets.


Immediate ToDo:

- Change print() calls to proper logging throughout the pipeline
- Implement spatial shuffling augmentation during training
- Implement evaluation plotting and metrics summary writer
- Begin implementation of NorCP data adapter

---

# Phase A — Stabilize the full STRIDE pipeline

Priority: **highest**

Before adding new modeling components, ensure the full experiment pipeline is robust and reproducible.

Tasks:

- Ensure the full pipeline runs end‑to‑end:
  - training
  - generation
  - evaluation
- Ensure compiled configs are written correctly
- Ensure all paths resolve correctly from compiled configs
- Ensure logging outputs (stdout/stderr) capture training progress
- Verify experiment outputs are stored in

```
runs/<experiment>/
```

Expected structure:

```
runs/<experiment>/

  compiled_configs/

  training/
      checkpoints/
      history.json

  generation/
      samples/

  evaluation/
      evaluation_metrics.json
```

Once this phase is stable, the core pipeline architecture should not change significantly.

---

# Phase B — Restore core modeling features

Several modeling features were implemented during the pipeline refactor. This phase documents what is already available and what remains to be added.

Recommended order:

## Implemented features

The following model features are now implemented and stable:

- Day‑Of‑Year extraction in the data adapter
- DOY cyclic encoding (`sin(DOY)`, `cos(DOY)`)
- FiLM‑based seasonal conditioning inside the diffusion model
- Configurable conditioning variables per experiment
- RainGate auxiliary precipitation module

These components are now part of the core STRIDE architecture.

---

## 1. Spatial shuffling augmentation

Add spatial shuffling as a training augmentation.

Goals:

- improve robustness
- prevent spatial overfitting

Should be configurable via training config.

---

## 2. Larger context encoder

Add a larger‑scale context encoder for LR inputs.

Motivation:

- capture larger spatial structures
- improve downscaling realism

This will likely modify the conditioning pathway of the model.

---

# Phase C — New dataset adapter: HCLIM NorCP

This is now the primary dataset development target for STRIDE.

Add a new data adapter for NorCP HCLIM data.

Directory:

```
data_adapters/hclim_norcp/
```

---

## Dataset families

- ECMWF‑ERAINT
  - historical ERA‑Interim driven simulation
  - 1998–2018

- ICHEC‑EC‑EARTH_HIST
  - historical EC‑EARTH simulation
  - 1986–2005

- ICHEC‑EC‑EARTH_RCP45_MC
  - future EC‑EARTH scenario
  - 2041–2060

---

## Spatial grids

Low‑resolution grid (conditions):

```
17 × 23
```

High‑resolution grid (target):

```
68 × 92
```

Downscaling factor:

```
4×
```

The domain is rectangular and must be supported throughout the pipeline.

---

## Target variables (3 km)

- pr (precipitation)
- tas (near‑surface temperature)
- orog (orography)

---

## Conditioning variables (12 km)

Single‑level:

- pr
- tas
- orog

Pressure‑level variables:

- hus
- ta
- ua
- va
- zg

Levels:

```
500
700
850
950
1000
```

These should expand to channels in the dataset.

---

## Temporal resolution

Datasets may exist at:

- 3 hr
- 6 hr

Initial adapter implementation should support **one frequency per experiment**.

---

## Adapter requirements

The adapter should:

- build file indices from directory structure
- match HR and LR files by timestamp
- support optional static fields
- support configurable variable selection

---

# Phase D — Evaluation improvements

Future evaluation improvements include:

- add plotting functions
- implement evaluation masking
- remove "forecast" terminology (use "downscaling" instead)

Plotting should include:

- spatial maps
- reliability diagrams
- PSD comparisons

---

# Important validation tasks

Before running NorCP experiments:

Verify precipitation units.

Some files report:

```
kg m‑2 s‑1
```

Confirm whether this represents:

- instantaneous flux
- accumulated precipitation
- mean over time interval

This affects preprocessing and evaluation thresholds.

---

# Long‑term research directions

Once the system is stable, STRIDE can support:

- multi‑conditioning experiments
- scale‑aware downscaling
- climate scenario generation
- hydrological impact modeling

The goal is a flexible generative framework for climate downscaling across datasets and scenarios.
