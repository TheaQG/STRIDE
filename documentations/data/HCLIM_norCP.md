

# HCLIM NorCP Dataset (Planned Adapter)

## Overview

The **HCLIM NorCP dataset** will be integrated into STRIDE as a new data adapter to support experiments with regional climate model simulations over Northern Europe.

The dataset originates from the **NorCP (Nordic Convection-Permitting Climate Projections)** project and contains high‑resolution climate simulations produced with the **HCLIM climate modeling system**.

This dataset differs from the current DANRA–ERA5 setup in several ways:

- it is **model output rather than observational data**
- it contains **multi-variable atmospheric fields**
- it supports **future climate scenario experiments**

The goal of integrating this dataset is to enable **generative downscaling experiments across multiple climate model regimes**, including historical and future projections.

---

# Dataset Structure

Three primary simulation datasets are available.

## 1. ECMWF‑ERAINT

Historical simulation driven by ERA‑Interim reanalysis.

```
Period: 1998 – 2018
Temporal resolution: 6‑hourly
Purpose: evaluation and historical comparison
```

---

## 2. ICHEC‑EC‑EARTH_HIST

Historical climate simulation driven by the EC‑EARTH global climate model.

```
Period: 1986 – 2005
Temporal resolution:
    • 3‑hourly
    • 6‑hourly
```

This dataset allows comparison between **reanalysis‑driven simulations and GCM‑driven simulations**.

---

## 3. ICHEC‑EC‑EARTH_RCP45_MC

Future climate projection using the **RCP4.5 scenario**.

```
Period: 2041 – 2060
Temporal resolution:
    • 3‑hourly
    • 6‑hourly
```

This dataset will enable **future climate downscaling experiments**.

Note:

```
The future simulation does not include orography fields.
```

---

# Spatial Resolution

The NorCP simulations provide two spatial resolutions:

## Low‑resolution fields (conditioning input)

```
Resolution: 12 km
Grid size: 17 × 23
```

These fields correspond to the **driving regional climate model (RCM)**.


## High‑resolution fields (target)

```
Resolution: 3 km
Grid size: 68 × 92
```

These correspond to the **convection‑permitting climate model (CPRCM)**.

The spatial relationship is:

```
1 coarse cell (12 km) = 4 × 4 high‑resolution cells (3 km)
```

The domain is **rectangular and non‑square**.

---

# Variables

## High‑Resolution Target Variables (3 km)

These are candidate **downscaling targets**.

```
pr   — precipitation
      units: kg m‑2 s‑1

tas  — near‑surface air temperature
      units: K

orog — orography
      units: meters
```

---

## Low‑Resolution Conditioning Variables (12 km)

These fields will act as **conditioning inputs to the generative model**.

### Surface variables

```
pr   — precipitation

tas  — near‑surface air temperature

orog — orography
```

---

### Pressure level variables

Pressure levels:

```
500
700
850
950
1000 hPa
```

Available variables:

```
hus* — specific humidity

ta*  — air temperature

ua*  — eastward wind

va*  — northward wind

zg*  — geopotential height
```

Example variable names:

```
hus500
ua850
zg700
```

Units follow CF‑convention metadata in the NetCDF files.

---

# File Metadata

The NetCDF files follow **CF‑1.7 conventions** and use a **Lambert Conformal Conic projection**.

Key metadata fields include:

```
grid_mapping_name = lambert_conformal_conic
standard_parallel
longitude_of_central_meridian
latitude_of_projection_origin
```

Coordinates provided:

```
time
lat
lon
x
y
```

Time encoding:

```
time: units = "days since 1949‑12‑01"
calendar = "standard"
```

---

# Intended Adapter Design

A new STRIDE adapter will likely be implemented under:

```
data_adapters/hclim_norcp/
```

Proposed modules:

```
adapter.py
features.py
paths.py
splits.py
regions.py
transforms.py
statistics/
```

The structure will follow the same design as the existing adapter:

```
data_adapters/danra_era5_small/
```

---

# Planned Use in STRIDE

The NorCP dataset enables several new experiment types:

### 1. Climate model downscaling

```
12 km -> 3 km generative downscaling
```

---

### 2. Multi‑variable conditioning

The larger set of atmospheric fields allows conditioning on:

```
temperature
humidity
winds
geopotential height
```

---

### 3. Future climate experiments

The RCP4.5 simulations allow evaluation of **model robustness under climate change scenarios**.

---

# Integration Challenges

Several aspects differ from the DANRA–ERA5 dataset and must be handled in the adapter.

## 1. Multi‑level variables

Pressure‑level variables require careful stacking and ordering.

## 2. Temporal resolution

The dataset contains both:

```
3‑hourly
6‑hourly
```

Adapters must support configurable temporal aggregation.

## 3. Missing variables

Some datasets lack certain fields (e.g. orography in future runs).

The adapter should gracefully handle missing static variables.

## 4. Non‑square spatial domain

The domain is not square, so cropping utilities must support arbitrary shapes.

---

# Long‑Term Goal

Integrating NorCP will allow STRIDE to support:

```
ERA5 -> DANRA
RCM -> CPRCM
GCM -> RCM -> CPRCM
```

This enables experiments on **multi‑scale generative climate downscaling across different model hierarchies**.
