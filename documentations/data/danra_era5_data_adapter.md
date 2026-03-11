

# DANRA‑ERA5 Data Adapter

This adapter provides training and evaluation datasets combining:

- DANRA high‑resolution fields
- ERA5 large‑scale predictors

---

# Data sources

## DANRA

High‑resolution regional climate reanalysis used as target data.

Typical variables:

```
precipitation
temperature
```

---

## ERA5

Coarser resolution atmospheric predictors used for conditioning.

Examples:

```
precipitation
temperature
pressure
CAPE
```

---

# Dataset structure

Each dataset sample contains:

```
target
conditions_dynamic
conditions_static
```

Dynamic conditions vary per timestep.

Static conditions include:

```
topography
land‑sea mask
```

---

# Dataset splits

Splits are defined by a manifest file:

```
split_manifest.json
```

Typical splits:

```
train
validation
test
```

---
