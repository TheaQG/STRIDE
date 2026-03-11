

# STRIDE Orchestration

This document describes how STRIDE orchestrates full experiments including training, generation, and evaluation.

The orchestration system is designed for:

- reproducibility
- experiment management
- cluster execution (e.g. LUMI)
- clear separation between experiment definition and implementation

---

# Pipeline overview

A STRIDE experiment runs the following pipeline:

experiment config  
→ config compiler  
→ compiled configs  
→ training  
→ generation  
→ evaluation

Concretely:

```
configs/experiments/*.yaml
        ↓
ExperimentRunner
        ↓
ConfigCompiler
        ↓
runs/<experiment>/compiled_configs/
        ↓
training_main.py
        ↓
generation_main.py
        ↓
evaluation_main.py
```

Each stage is fully reproducible because the compiler writes all resolved configuration files to disk.

---

# Configuration hierarchy

STRIDE uses a three‑layer configuration structure.

## 1. Experiment configuration

Location:

```
configs/experiments/
```

Example:

```
train_generate_evaluate_test.yaml
```

This file defines:

- experiment name
- base configs to use
- stage enable/disable flags
- stage‑specific overrides

Example structure:

```
experiment:
  name: train_generate_evaluate_test

bases:
  model: configs/models/edm_small.yaml
  data: configs/datasets/danra_era5_small.yaml
  training: configs/training/training_base.yaml
  generation: configs/generation/generation_base.yaml
  evaluation: configs/evaluation/evaluation_base.yaml
```

---

## 2. Base configs

Base configs define reusable defaults.

They live in:

```
configs/models/
configs/training/
configs/generation/
configs/evaluation/
configs/datasets/
```

Examples:

```
training_base.yaml
generation_base.yaml
evaluation_base.yaml
```

These contain:

- algorithm settings
- metric selections
- model architecture parameters

They **do not contain experiment‑specific paths**.

---

## 3. Compiled configs

When an experiment starts, the `ConfigCompiler` produces fully resolved configs.

These are written to:

```
runs/<experiment>/compiled_configs/
```

Typical contents:

```
model_resolved.yaml
data_resolved.yaml
training_run_resolved.yaml
generation_run_resolved.yaml
evaluation_run_resolved.yaml
generation_base_resolved.yaml
evaluation_base_resolved.yaml
compiled_manifest.json
```

These files contain:

- resolved file paths
- merged base + run configuration
- experiment‑specific overrides

They serve as the **exact provenance record** for the experiment.

---

# Running experiments

## Full pipeline

```
python cli/launch_pipeline.py \
  --config configs/experiments/train_generate_evaluate_test.yaml
```

Stages run sequentially:

1. training
2. generation
3. evaluation

---

## Individual stages

Stages can also be executed directly.

Training:

```
python stride_core/training/training_main.py --config <training_config>
```

Generation:

```
python stride_core/generation/generation_main.py --config <generation_config>
```

Evaluation:

```
python stride_core/evaluation/evaluation_main.py --config <evaluation_config>
```

---

# Experiment output structure

All experiment artifacts live under:

```
runs/<experiment>/
```

Example:

```
runs/train_generate_evaluate_test/
```

Structure:

```
runs/<experiment>/

  compiled_configs/

  training/
      checkpoints/
      training_history.json

  generation/
      samples/

  evaluation/
      evaluation_metrics.json
```

This layout ensures every experiment is fully self‑contained.

---

# Design philosophy

The orchestration layer follows several principles:

1. **Reproducibility**  
   All resolved configs are stored.

2. **Separation of concerns**  
   Experiments define configuration, not implementation.

3. **Cluster‑friendly execution**  
   Stages run independently and can be scheduled.

4. **Extensibility**  
   New datasets, models, or metrics can be added without modifying the orchestration system.

---

