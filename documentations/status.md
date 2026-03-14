# STRIDE Project Status

## Project Overview

STRIDE is a research framework for **generative downscaling of climate fields** using **Elucidated Diffusion Models (EDM)**.

Current primary application:

- **High-resolution target:** DANRA
- **Low-resolution conditioning:** ERA5
- **Task:** probabilistic downscaling of precipitation fields

The system is designed to support:

- reproducible experiments
- modular pipelines
- dataset adapters
- configurable conditioning strategies

The full pipeline currently runs:

training → generation → evaluation

through a single experiment configuration.

---

# Repository Architecture

The repository is organized into the following core modules:

```
stride_core/
    training/
    generation/
    evaluation/
    pipeline/

configs/
    training/
    generation/
    evaluation/
    datasets/
    experiments/

data_adapters/
    danra_era5_small/

runs/
    <experiment_name>/
```

---

# Experiment System

Experiments are controlled via YAML configuration files located in:

```
configs/experiments/
```

Example experiment:

```
configs/experiments/train_generate_evaluate_test.yaml
```

Each experiment defines:

```
experiment:
  name: <experiment_name>
  output_root: runs/
  seed: 42
```

The **config compiler** generates resolved configs in:

```
runs/<experiment_name>/compiled_configs/
```

Typical resolved files:

```
model_resolved.yaml
data_resolved.yaml
training_run_resolved.yaml
generation_base_resolved.yaml
generation_run_resolved.yaml
evaluation_base_resolved.yaml
evaluation_run_resolved.yaml
```

These ensure that experiments are **fully reproducible**.

---

# Pipeline Execution

Full experiments are launched via:

```
cli/launch_pipeline.py
```

or

```
bash/run_pipeline.sh
```

Pipeline stages:

1. training
2. generation
3. evaluation

Outputs are stored in:

```
runs/<experiment_name>/
    compiled_configs/
    training/
    generation/
    evaluation/
```

---

# Orchestration and Change Impact Guide

This section explains **where to modify the codebase when different parts of the system change**. It serves as a quick reference for maintaining the pipeline.

## Pipeline Flow Overview

```
Experiment config (configs/experiments/*.yaml)
        ↓
config_compiler.py
        ↓
compiled_configs/ (runs/<experiment>/compiled_configs)
        ↓
training_main.py
        ↓
generation_main.py
        ↓
evaluation_main.py
```

This diagram summarizes how configuration changes propagate through the STRIDE pipeline.

## Experiment Directory Layout

Each experiment produces a structured output directory:

```
runs/<experiment_name>/
    compiled_configs/
        model_resolved.yaml
        data_resolved.yaml
        training_run_resolved.yaml
        generation_base_resolved.yaml
        generation_run_resolved.yaml
        evaluation_base_resolved.yaml
        evaluation_run_resolved.yaml

    training/
        checkpoints/
        logs/

    generation/
        samples/
            case_xxxx/
                member_000.npz
                member_001.npz
                ...

    evaluation/
        metrics/
        figures/
```

This directory structure ensures that every experiment stores:

- the **exact resolved configuration used**
- the **training artifacts (checkpoints, logs)**
- the **generated ensemble samples**
- the **evaluation outputs and metrics**

As a result, experiments are fully reproducible and easy to inspect after completion.

## 1. Experiment Configuration Changes

If you modify experiment configs in:

```
configs/experiments/
```

Key orchestration files involved:

```
stride_core/pipeline/config_compiler.py
stride_core/pipeline/experiment_runner.py
cli/launch_pipeline.py
```

Responsibilities:

- **config_compiler.py** — builds resolved configs for all pipeline stages
- **experiment_runner.py** — orchestrates training → generation → evaluation
- **launch_pipeline.py** — CLI entrypoint that launches the pipeline

Typical changes handled here:

- experiment output directory structure
- wiring base configs into resolved configs
- stage enable/disable logic

---

## 2. Base Configuration Changes

If modifying base configs in:

```
configs/training/
configs/generation/
configs/evaluation/
configs/datasets/
configs/models/
```

Relevant code:

```
stride_core/pipeline/config_compiler.py
```

This file:

- loads base configs
- merges experiment overrides
- produces the `*_resolved.yaml` configs used during runtime

Any structural change in base configs should be verified here.

---

## 3. Data Adapter Changes

If upstream data format, variables, or preprocessing change:

Primary location:

```
data_adapters/<adapter_name>/
```

For the current DANRA–ERA5 adapter:

```
data_adapters/danra_era5_small/
```

Important files:

```
adapter.py
features.py
paths.py
splits.py
regions.py
transforms.py
unit_conversion.py
statistics/
```

Typical responsibilities:

- **paths.py** — dataset indexing
- **splits.py** — train/val/test splits
- **features.py** — variable loading and stacking
- **regions.py** — spatial cropping
- **adapter.py** — construction of final dataset samples

If new conditioning variables or metadata are introduced (e.g. DOY), they are typically implemented in:

```
adapter.py
features.py
```

---

## 4. Model Conditioning Changes

If modifying how the model receives conditioning variables:

Relevant components:

```
stride_core/training/trainer.py
stride_core/generation/generator.py
models/score_unet.py (or equivalent model implementation)
```

Typical modifications include:

- passing new conditioning tensors through the batch
- modifying model forward inputs
- adding FiLM or embedding layers

Example upcoming change:

```
Day-of-year (DOY) FiLM conditioning
```

---

## 5. Training Pipeline Changes

Core training orchestration lives in:

```
stride_core/training/training_main.py
stride_core/training/trainer.py
```

Update these if modifying:

- training loops
- logging
- checkpoint handling
- batch structure

---

## 6. Generation Pipeline Changes

Generation is controlled by:

```
stride_core/generation/generation_main.py
stride_core/generation/generator.py
```

Additional helpers:

```
output_saving.py
pmm.py
training_preview.py
```

Modify these if changing:

- ensemble sampling
- checkpoint loading
- generation output structure

---

## 7. Evaluation Pipeline Changes

Evaluation orchestration lives in:

```
stride_core/evaluation/evaluation_main.py
stride_core/evaluation/evaluator.py
```

Supporting modules:

```
data_loading.py
metrics/
plots/
saving.py
utils/
```

Modify these when:

- adding metrics
- changing evaluation datasets
- modifying output formats

---

## 8. Pipeline Debugging Strategy

When debugging pipeline issues, inspect in this order:

1. **experiment config** (`configs/experiments/...`)
2. **compiled configs** (`runs/<experiment>/compiled_configs/`)
3. **config compiler logic**
4. **stage entrypoints** (training/generation/evaluation)

This approach typically identifies path or configuration mismatches quickly.

---

# Training

Training uses an **Elucidated Diffusion Model (EDM)** for conditional generation.

Model inputs currently include:

- `target` — high-resolution precipitation field
- `cond_dynamic` — configurable dynamic atmospheric predictors
- `cond_static` — configurable static geographic fields
- `time_features` — Day‑Of‑Year cyclic encoding (`sin(DOY)`, `cos(DOY)`)
- `cond_coord` — spatial metadata

Additional model conditioning mechanisms:

- **Day‑Of‑Year FiLM conditioning** — seasonal context modulates intermediate model features
- **RainGate auxiliary module** — predicts precipitation occurrence probability and provides an auxiliary training signal

Training is controlled by:

```
stride_core/training/training_main.py
```

with configuration-driven setup.

---

# Generation

Generation loads trained checkpoints and produces ensemble samples.

Outputs are stored as:

```
runs/<experiment_name>/generation/samples/
    case_xxxx/
        member_000.npz
        member_001.npz
        ...
```

Optional derived outputs:

- ensemble mean
- probability-matched mean (PMM)

Generation is controlled by:

```
stride_core/generation/generation_main.py
```

---

# Evaluation

Evaluation compares generated ensembles with ground truth.

Metrics include:

### Probabilistic

- CRPS
- CRPS decomposition
- PIT histogram
- Rank histogram
- reliability diagrams

### Spatial

- Power spectrum (PSD)
- ISS
- SAL

### Climatology

- pixel distributions
- extreme quantiles
- wet-day frequency
- seasonal accumulations

### Temporal

- autocorrelation
- wet/dry spell statistics

Evaluation runs through:

```
stride_core/evaluation/evaluation_main.py
```

---

# Data Adapter System

Data loading is handled through **dataset adapters**.

Current adapter:

```
data_adapters/danra_era5_small/
```

Modules include:

```
adapter.py
features.py
paths.py
splits.py
regions.py
transforms.py
unit_conversion.py
statistics/
```

Responsibilities:

- **paths.py** — builds date-to-file mappings
- **splits.py** — dataset splits via date manifests
- **features.py** — loads and stacks variables
- **regions.py** — spatial cropping
- **transforms.py** — normalization / transforms
- **unit_conversion.py** — unit standardization
- **adapter.py** — constructs dataset samples

Dataset sample structure:

```
{
    "target": tensor[C_out, H, W],
    "cond_dynamic": tensor[C_dyn, H, W],
    "cond_static": tensor[C_static, H, W] | None,
    "time_features": tensor[2],
    "cond_coord": dict,
    "meta": dict
}
```

---

# Statistics Pipeline

Offline statistics are computed for normalization.

Script:

```
data_adapters/danra_era5_small/run_statistics.py
```

Statistics are stored by:

- split
- domain
- variable
- transform

These are used during dataset loading.

---

# Spatial Setup

Full DANRA domain:

```
589 × 789
```

Current evaluation crop:

```
128 × 128
```

Shuffle patch used during training:

```
180 × 180
```

Cropping is handled in `regions.py`.

---

# Current Conditioning Variables

### Target

- DANRA precipitation

### Dynamic conditions (configurable)

Examples currently available:

- ERA5 precipitation
- ERA5 temperature
- CAPE
- mean sea level pressure
- geopotential height (multiple pressure levels)
- water vapour flux
- potential evaporation

Dynamic variables are **selected per experiment via configuration**.

### Static conditions (configurable)

- land-sea mask
- topography

---

# Implemented Model Features

The following conditioning and architectural features are currently implemented:

- **Day‑Of‑Year encoding** provided by the data adapter
- **FiLM‑based seasonal conditioning** inside the diffusion model
- **RainGate auxiliary precipitation module**
- **Configurable dynamic and static conditioning variables**

# Planned Model Improvements

Upcoming work:

1. **Spatial shuffling augmentation during training**
2. **Evaluation plotting and metrics summary writer**
3. **NorCP data adapter**
4. **Larger context encoder**
5. **Support for non‑square model domains** (required for NorCP grids)

Do not modify any other parts of the file.