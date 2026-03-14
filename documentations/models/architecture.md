# STRIDE Model Architecture

STRIDE uses a **diffusion‑based generative model** for climate downscaling.

The model is based on the **Elucidated Diffusion Model (EDM)** formulation and uses a conditional **U‑Net** backbone to predict denoised fields during the reverse diffusion process.

---

# Core model

The architecture follows the **Elucidated Diffusion Model (EDM)** framework:

- noise is added to the target field during training
- the model predicts the denoised field
- the sampler iteratively removes noise to generate high‑resolution samples

The model is **conditional**, meaning that high‑resolution fields are generated conditioned on:

- dynamic large‑scale predictors (ERA5)
- static geographic features
- optional temporal conditioning (Day‑Of‑Year)

---

# U‑Net backbone

The main network is a **U‑Net** consisting of:

Encoder path

- convolutional residual blocks
- spatial downsampling
- optional attention layers

Decoder path

- upsampling layers
- skip connections from encoder blocks
- residual blocks

Skip connections allow high‑resolution spatial information to propagate directly to later layers.

Optional **self‑attention layers** allow long‑range spatial interactions.

---

# Diffusion noise embedding

The diffusion noise level (sigma) is embedded using **sinusoidal embeddings**.

These embeddings are passed through a small MLP and injected into the residual blocks of the U‑Net.

This allows the model to condition its predictions on the current diffusion timestep.

---

# Conditioning inputs

The model receives three main types of inputs:

```
target_noisy
cond_dynamic
cond_static
```

Where:

- **target_noisy** - noisy version of the high‑resolution target field
- **cond_dynamic** - dynamic large‑scale predictors (ERA5 variables)
- **cond_static** - static geographic predictors (e.g. topography, land‑sea mask)

These conditioning fields are concatenated and processed jointly by the network.

---


# Day‑Of‑Year conditioning (FiLM)

Seasonal information is provided through **Day‑Of‑Year (DOY)** features produced by the data adapter.

The adapter encodes DOY as a cyclic representation:

```
sin(DOY)
cos(DOY)
```

These values form a `time_features` vector that is passed to the model.

Inside the network, the DOY vector is embedded and applied using **FiLM (Feature‑wise Linear Modulation)**.

FiLM layers modify intermediate feature maps using learned scale and shift parameters:

```
y = gamma(DOY) * x + beta(DOY)
```

This allows the model to adapt its internal representation based on **seasonal context**.

---

# RainGate module

STRIDE optionally includes an auxiliary **RainGate** module.

RainGate predicts the **probability of precipitation occurrence** (wet vs dry) from the conditioning inputs.

The module:

- receives conditioning inputs (typically dynamic predictors)
- produces a per‑pixel precipitation probability map

RainGate can be used for:

- auxiliary supervision during training
- improved modeling of precipitation intermittency

The RainGate loss is combined with the diffusion loss using a configurable weight.

RainGate behavior is controlled entirely through the **model configuration**.

---

# Model configuration

The architecture is fully controlled through the model configuration file.

Key configurable components include:

- number of U‑Net channels
- attention layers
- FiLM embeddings
- RainGate module
- temporal conditioning options

This design allows experiments to modify the architecture without changing the model code.

---

# Summary

The STRIDE architecture combines:

- EDM diffusion training
- conditional U‑Net backbone
- FiLM‑based seasonal conditioning
- optional RainGate auxiliary prediction

This provides a flexible framework for **probabilistic climate downscaling** while incorporating both physical predictors and seasonal context.
