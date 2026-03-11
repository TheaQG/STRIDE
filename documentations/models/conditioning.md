

# Conditioning in STRIDE

The diffusion model is conditioned on large‑scale atmospheric variables.

---

# Conditioning variables

Examples:

```
ERA5 precipitation
ERA5 temperature
CAPE
mean sea level pressure
```

These fields provide large‑scale context for high‑resolution generation.

---

# Static conditioning

Static geographic information can also be included.

Examples:

```
topography
land‑sea mask
```

These fields are constant for a given spatial domain.

---

# Conditioning mechanism

Conditioning fields are concatenated as additional input channels to the model.

The U‑Net therefore receives both:

```
noisy target field
conditioning variables
```

This enables the model to learn mappings from large‑scale predictors to fine‑scale structure.
