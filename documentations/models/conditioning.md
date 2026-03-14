# Conditioning in STRIDE

The STRIDE diffusion model is **conditionally generated**. High‑resolution climate fields are generated given a set of large‑scale predictors and auxiliary conditioning signals.

Conditioning information provides the model with the large‑scale atmospheric state and geographic constraints needed to produce physically consistent fine‑scale structure.

---

# Types of conditioning

STRIDE currently supports three primary conditioning sources:

1. **Dynamic atmospheric predictors**
2. **Static geographic predictors**
3. **Temporal conditioning (Day‑Of‑Year)**

These signals are provided by the data adapter and integrated into the model through channel concatenation and FiLM modulation.

---

# Dynamic conditioning variables

Dynamic predictors represent the **large‑scale atmospheric state** at the same timestep as the target field.

Typical examples include:

```
ERA5 precipitation
ERA5 temperature
CAPE
mean sea level pressure
geopotential height
water vapour flux
potential evaporation
```

Dynamic variables are configurable per experiment through the dataset configuration.

Example configuration:

```
data:
  conditioning:
    dynamic:
      variables: [prcp, temp, z_pl_500, ewvf, nwvf]
```

Each selected variable becomes an **input channel** to the model.

This allows experiments to explore different predictor sets without modifying the codebase.

---

# Static conditioning variables

Static predictors describe geographic properties that do not change over time.

Examples include:

```
topography
land‑sea mask
```

These variables provide information about:

- orographic effects
- coastline boundaries
- terrain‑driven precipitation processes

Example configuration:

```
data:
  conditioning:
    static:
      variables: [lsm, topo]
```

Static fields are loaded once per domain and reused across all samples.

---

# Temporal conditioning (Day‑Of‑Year)

Seasonal information is included through **Day‑Of‑Year (DOY) encoding**.

The data adapter extracts the date for each sample and converts it into a cyclic representation:

```
sin(DOY)
cos(DOY)
```

These values form the vector:

```
time_features = [sin(DOY), cos(DOY)]
```

This representation preserves seasonal continuity (e.g. December 31 and January 1 remain close in feature space).

---

# FiLM conditioning

Temporal features are injected into the network using **Feature‑wise Linear Modulation (FiLM)**.

FiLM layers modify intermediate feature maps using learned scale and shift parameters:

```
y = gamma(condition) * x + beta(condition)
```

Where:

- `x` is an intermediate feature map
- `gamma` and `beta` are learned functions of the conditioning vector

For DOY conditioning, the FiLM parameters are generated from the DOY embedding.

This allows the model to adjust its internal representations depending on the **seasonal state of the atmosphere**.

---

# RainGate conditioning head

STRIDE optionally includes a **RainGate** auxiliary module.

RainGate predicts the **probability of precipitation occurrence** (wet vs dry) at each pixel.

The module:

- receives conditioning inputs (typically dynamic predictors)
- outputs a precipitation probability map

RainGate can be used for:

- auxiliary supervision during training
- improving precipitation intermittency
- guiding the model in dry vs wet regimes

RainGate behaviour is controlled entirely through the **model configuration**.

---

# Input tensor structure

During training the model receives the following inputs:

```
x_noisy      (noisy target field)
cond_dynamic (dynamic predictors)
cond_static  (static predictors)
time_features (optional DOY encoding)
```

Dynamic and static fields are concatenated with the noisy target field before entering the U‑Net.

Temporal conditioning is applied through FiLM layers.

---

# Design philosophy

The conditioning system is designed to be:

- **configurable** — experiments control which predictors are used
- **dataset‑agnostic** — adapters define variable availability
- **extensible** — new conditioning sources can be added without rewriting the model

This flexibility is essential for exploring different climate downscaling configurations and datasets.
