# ERA5 → DANRA Small Dataset Adapter

Data side of ERA5→DANRA downscaling for the **small test dataset** used in STRIDE development.

This adapter implements the complete data pipeline for the initial STRIDE experiments and is used to validate the full system before scaling to the full dataset.

---

# Canonical Sample Definition

For this dataset, **one sample corresponds to one date**.

Each sample contains:

- **Target (HR)**
  - One high‑resolution target variable

- **Dynamic conditioning variables (LR)**
  - One or more low‑resolution atmospheric variables
  - Same timestamp as the target

- **Optional static variables**
  - Land–sea mask
  - Topography

- **Fixed spatial domain**

Current limitations:

- No context branch yet
- No multi‑target output yet

---

# Adapter Output Contract

The adapter returns batches with the following structure:

```python
batch = {
    "target":       [B, 1, H_hr, W_hr],
    "cond_dynamic": [B, C_dyn, H_lr, W_lr],
    "cond_static":  [B, C_static, H_hr, W_hr] or None,
    "cond_coord":   {...},
    "meta":         {...},
}
```

---

# Dataset Description

Each sample corresponds to **one date**.

For each date the adapter loads:

### Target (HR)

- DANRA precipitation

### Dynamic conditioning variables (LR)

- ERA5 precipitation
- ERA5 temperature

### Static conditioning variables

- Land‑sea mask
- Topography

All variables are **spatially co‑located**.

Dataset spatial configuration:

```
Full domain:     589 × 789
Target crop:     128 × 128 (Denmark region)
Shuffle region:  180 × 180 (potential random crop region)
```

---

# Implemented Pipeline

The following data pipeline components are currently implemented.

---

## File Indexing

**File:** `paths.py`

Builds a **date → file path index** linking the required files for each sample.

Example structure:

```
date
 ├── target (DANRA)
 └── cond_dynamic
      ├── prcp (ERA5)
      └── temp (ERA5)
```

Only dates available in **all variables simultaneously** are used.

---

## Data Loading

**File:** `features.py`

Responsibilities:

- Load `.npz` files
- Stack dynamic variables
- Load static variables
- Ensure consistent dtype and shape

Returned arrays:

```
target        (H, W)
cond_dynamic  (C_dyn, H, W)
cond_static   (C_static, H, W)
```

---

## Unit Conversions

**File:** `unit_conversions.py`

Ensures all physical variables use consistent units.

Example:

```
ERA5 precipitation  → mm/day
DANRA precipitation → mm/day
```

---

## Static Alignment

Static fields are aligned with dynamic variables.

Topography is masked using the land‑sea mask to avoid numerical artefacts over ocean:

```python
topography = topography * lsm
```

This guarantees **zero elevation over ocean pixels**.

---

## Spatial Cropping

**File:** `regions.py`

Supports extracting spatial subregions from the full domain.

Implemented domains:

```
full_589x789

dk_128x128

shuffle_180x180
```

Cropping is currently **deterministic**.

---

# Planned / Missing Functionality

The following features are planned but not yet implemented.

## Random Spatial Shuffle (Training Augmentation)

Instead of always cropping the same 128×128 region:

```
sample random 128×128 crop
inside 180×180 shuffle region
```

This improves training stability.

---

## Larger Conditioning Context

Support conditioning on a **larger LR spatial domain** than the HR output.

Example:

```
LR conditioning: 256×256
HR output:       128×128
```

This will require a **context encoder** in the model.

---

## Temporal Stacking

Support conditioning on multiple time steps.

Example:

```
t-2
t-1
t
```

These would be stacked as additional input channels.

---

## Day‑of‑Year Conditioning

Add seasonal information using cyclic encoding:

```
sin(2π * doy / 365)
cos(2π * doy / 365)
```

These will likely be injected through **FiLM conditioning**.

---

## Multi‑Target Output

Support predicting multiple HR variables simultaneously.

Example:

```
precipitation

temperature

potential evaporation
```

---

# Status

The **entire data pipeline for the small dataset is implemented and verified**.

Remaining work mainly concerns:

- Model implementation
- Training pipeline
- Evaluation pipeline

---
