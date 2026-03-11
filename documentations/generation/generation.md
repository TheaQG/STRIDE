

# STRIDE Generation Pipeline

The generation stage produces high‑resolution ensemble samples from a trained diffusion model.

Entry point:

```
stride_core/generation/generation_main.py
```

---

# Generation overview

Pipeline:

```
generation_main.py
        ↓
Generator
        ↓
load model + checkpoint
        ↓
run EDM sampler
        ↓
save ensemble members
```

---

# Ensemble generation

Generation produces ensembles of stochastic realizations.

Key parameters:

```
ensemble_size
base_seed
use_fixed_seed
```

Each ensemble member is generated independently.

---

# Sampling algorithm

STRIDE uses the **EDM sampling algorithm**.

Key parameters:

```
num_steps
sigma_min
sigma_max
rho
```

These define the noise schedule used during reverse diffusion.

---

# Outputs

Generated samples are written to:

```
runs/<experiment>/generation/
```

Structure:

```
samples/
   <date>/
      member_0000.npz
      member_0001.npz
      ensemble_mean.npz
      pmm.npz
```

---

# Storage format

Each `.npz` file contains:

```
generated
generated_physical
target_physical
cond_dynamic_physical
cond_static_physical
```

This ensures evaluation has access to both forecasts and conditions.
