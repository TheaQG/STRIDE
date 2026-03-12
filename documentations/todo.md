# STRIDE Development Roadmap

This document tracks upcoming development tasks and priorities for the STRIDE framework.

The goal is to stabilize the full training → generation → evaluation pipeline and then expand the modeling capabilities and supported datasets.


Immediate ToDo:
- Change print()'s to logging calls in the pipeline
- DOY output from dataset
  - Step 1: Add DOY to dataset batch (adapter.py and features.py)
  - Step 2: Thread DOY through training (trainer.py)
  - Step 3: Thread DOY through generation (generator.py)
  - Step 4: Implement actual embedding + FiLM in the model (edm_unet.py). Update based on old implementation
  - Step 5: Expose config knobs in model config YAML and model construction plumbing ()
- FiLM conditioning of DOY in the model
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

Several modeling features were temporarily removed during the pipeline refactor and should now be re‑implemented.

Recommended order:

## 1. Dates in dataset / dataloader

Add explicit timestamp handling in datasets.

Goals:

- return timestamps with each sample
- allow temporal conditioning
- allow correct metadata for generated samples
- enable temporal evaluation metrics

Dataset output should include:

```
{
  "target",
  "dynamic_conditions",
  "static_conditions",
  "timestamp"
}
```

---

## 2. Conditioning variable selection per experiment

Experiments should be able to control which conditioning variables are used.

Add configuration support in experiment configs such as:

```
data:
  conditioning:
    dynamic_variables:
      - pr
      - tas
      - zg
    static_variables:
      - orog
```

This allows experiments to explore different conditioning setups without modifying adapters.

---

## 3. FiLM conditioning of day‑of‑year (DOY)

Once timestamps are available, implement DOY conditioning in the model.

Approach:

- compute DOY from timestamps
- embed DOY
- inject via FiLM modulation into the model

Purpose:

- capture seasonal precipitation patterns

---

## 4. Larger context encoder

Add a larger‑scale context encoder for LR inputs.

Motivation:

- capture larger spatial structures
- improve downscaling realism

This will likely modify the conditioning pathway of the model.

---

## 5. Spatial shuffling augmentation

Add spatial shuffling as a training augmentation.

Goals:

- improve robustness
- prevent spatial overfitting

Should be configurable via training config.

---

## 6. RainGate

Implement RainGate for precipitation generation.

Potential benefits:

- improved precipitation intermittency
- better extreme precipitation modeling

Implementation will likely affect the output head and/or post‑processing.

---

# Phase C — New dataset adapter: HCLIM NorCP

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
