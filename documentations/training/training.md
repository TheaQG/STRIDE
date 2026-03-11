

# STRIDE Training Pipeline

This document describes how models are trained in STRIDE.

Training is implemented in:

```
stride_core/training/
```

Main entrypoint:

```
training_main.py
```

---

# Training overview

The training stage performs:

1. dataset construction
2. model initialization
3. score‑based diffusion training
4. checkpointing
5. training statistics logging

Pipeline:

```
training_main.py
        ↓
Trainer
        ↓
build_training_data()
        ↓
Diffusion training loop
```

---

# Dataset construction

Training datasets are built through the **data adapter system**.

Function:

```
build_training_data()
```

Steps:

1. load dataset configuration
2. initialize adapter
3. build datasets
4. create dataloaders

Outputs:

```
train_loader
val_loader
```

---

# Diffusion objective

Training follows the **Elucidated Diffusion Model (EDM)** framework.

The model learns a score function:

```
∇x log p(x)
```

Noise levels are sampled from a continuous sigma distribution.

Training steps:

1. sample noise level σ
2. perturb data
3. predict score
4. compute score matching loss

---

# Training loop

For each batch:

```
sample sigma
add noise
predict score
compute loss
backpropagation
optimizer step
```

Loss values are aggregated and periodically logged.

---

# EMA weights

An **Exponential Moving Average (EMA)** of model parameters is maintained.

Purpose:

- improves sampling quality
- stabilizes evaluation

EMA weights are stored in checkpoints.

---

# Checkpoints

Checkpoints are written to:

```
runs/<experiment>/training/checkpoints/
```

Typical files:

```
checkpoint_latest.pt
checkpoint_best.pt
```

The "best" checkpoint is selected based on validation loss.

---

# Training outputs

Training artifacts include:

```
training_history.json
checkpoints/
```

These files allow generation and evaluation to reuse the trained model.

---
