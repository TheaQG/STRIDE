

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
cond_dynamic
cond_static
time_features
meta
```

Where:

- **target**: High‑resolution DANRA field used as the prediction target.
- **cond_dynamic**: Dynamic ERA5 predictors for the same timestep as the target.
- **cond_static**: Static geographic predictors.
- **time_features**: Encoded temporal information derived from the sample date.
- **meta**: Metadata describing the sample (date, variables, crop information, etc.).

Dynamic conditions vary per timestep.

Static conditions typically include:

```
topography
land‑sea mask
```

---

# Temporal features (Day‑Of‑Year)

The adapter extracts the **day of year (DOY)** from the dataset timestamp and encodes it as a cyclic representation.

For each sample:

```
doy_sin = sin(2π * DOY / 365)
doy_cos = cos(2π * DOY / 365)
```

These two values form the `time_features` vector:

```
time_features = [sin(DOY), cos(DOY)]
```

This encoding preserves the cyclical nature of the seasonal cycle (e.g. December 31 and January 1 remain close in feature space).

The DOY information is passed to the model and used for **FiLM conditioning**, allowing the network to adapt its feature processing based on seasonal context.

Metadata fields related to this feature include:

```
meta.day_of_year
meta.doy_sin_cos
```

This design allows seasonal information to influence the generative process without introducing discontinuities in the temporal representation.

---

# Model‑side conditioning features

The STRIDE model can optionally use additional conditioning mechanisms built on top of the adapter outputs.

## Day‑of‑Year FiLM conditioning

The `time_features` vector produced by the adapter can be passed to the model where it is embedded and applied through **FiLM (Feature‑wise Linear Modulation)** layers.

This allows the network to adapt its internal activations based on the seasonal state of the atmosphere.

## RainGate

An optional **RainGate module** can be enabled in the model configuration. This auxiliary head predicts the probability of precipitation occurrence (wet vs dry) and can:

- act as an auxiliary supervision signal during training
- improve precipitation realism in generated samples

RainGate uses the same conditioning inputs as the main model (typically the dynamic ERA5 predictors).

These conditioning mechanisms are controlled entirely through the **model configuration** and do not require changes to the data adapter.

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
