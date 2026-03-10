STRIDE - Stochastic Transport and Reconstruction for Integrated Downscaling Emulation


---

# Phase 1: STRIDE v1 Scope

## Keep now
- EDM loss
- EDM sampler
- EDM-preconditioned UNet
- Context encoder + FiLM variable-label embeddings
- RainGate
- Scaling via offline global stats
- Random spatial shuffle
- Anchored input/output regions
- Temporal stacking 
- Dynamics/static conditioning split
- Evaluation families
- Ensemble generation
- Probability Matched Mean (PMM)
- Quicklooks
- Variable utilities
- Unified naming

## Postpone
- Multi-target output
- Larger-domain context + co-located joint conditioning at the same time
- Probabilistic evaluation family in full detail
 
## Retire
- SBGM/DDPM legacy 
- Residual prediction
- Classifier-free guidance (CFG)
- Dual LR scaling
- Probably SDF-weighting unless strong result later forces it back in





# Phase 2: Contracts

## A: Data batch contract
What a batch looks like when haded to the model trainer

batch = {
    "target": Tensor,                # [B, C_out, H_hr, W_hr]
    "cond_dynamic": Tensor,          # [B, T, C_dyn, H_lr, W_lr] or [B, C_dyn, H_lr, W_lr] if no temporal stacking
    "cond_static": Tensor | None,    # [B, C_static, H_hr, W_hr] or possibly HR/larger context form
    "cond_coord": dict | None,       # metadata for anchored regions
    "meta": {
        "timestamps": ...,
        "target_vars": ...,
        "cond_vars": ...,
        "domain_info": ...,
        "scaling_info": ...,
        ...
    }
}

Optionally later add "cond_context_dynamic", "cond_context_static"

## B: Transform contract
Every adapter should expose:
- forward_target
- inverse_target
- forward_conditioning
- inverse_conditioning (if needed, e.g. for LR comparison metrics)
- Stat loading from offline files

Each transform should be described by metadata, so there is no hidden logic.

## C: Model input contract
Model should NOT know anything about "topography" or "LSM" or "temperature" or "precipitation". It should receive:
- dynamic conditioning channels
- Static conditioning channels
- Temporal conditioning
- Optional context branch
- Optional FiLM variable-label embeddings

Input channel accounting must be computed outside the model or passed through a clean config object.

## D: Evaluation contract
Generation output must always save enough to evaluate later without rerunning model inference.
At minimum save:
- Generated samples/ensemble
- PMM if computed
- Conditioning used
- Target if available (for test set)
- Metadata including date, variable names, domain info, scaling state, region



# Phase 3 - Minimal vertical slice implementation
Use the small dataset to build the first complete path:
- One data adapter
- One transform pipeline
- One training loop
- One generation run
- One evaluation run

### Minimal v1 feature set: 
#### Data
- One HR target variable
- One or more LR dynamic variables
- Optional statics
- Fixed co-located domain
- Optional random shuffle
- One scaling method first: log-z-score
#### Model
- EDM-preconditioned UNet
- Native LR conditioning path
- No large-context branch yet (unless already trivial)
- RainGate optional flag
#### Training
- EDM loss
- EMA
- One or two sanity monitoring metrics
#### Generation
- Ensemble generation
- PMM
- Quicklook dates
#### Evaluation
- Minimal family subset
    - Dates
    - Distributions
    - Extremes
    - Probabilistic
    - Spatial
    - Scale


# Phase 4 - Porting old code (by functionality, not by file)
### Porting order:
1. Variable metadata utilities
2. Offline stats/scaling machinery
3. Data region selection/shuffle logic
4. EDM model core
5. RainGate
6. Training loop + EMA
7. Generation
8. Evaluation metrics
9. Plotting

Ask: 
- Is it data-agnostic?
- If not, can it be parameterised?
- If not, does it belong in the adapter instead?
- If not: retire it.

# Phase 5 - build the data system properly
Rich data requirements -> complexity and technical debt -> explicit structure and contracts

Five components:
## 1. paths.py
- File discovery
- Naming conventions
- Splits
- Roots

## 2. regions.py
- Anchored box definitions
- HR/LR crop logic
- Random spatial shuffle logic
- Larger context regions
- Coordinate bookkeeping

## 3. transforms.py
- Scaling/inverse-scaling
- BoxCox, z-score, log-z-score, min-max, etc
- Stat loading from offline files

## 4. features.py
- Temporal stacking
- Seasonality/day-of-year encoding
- Static/dynamic assembly
- Variable ordering

## 5. adapter.py
- Tying it together into datasets/dataloaders
- Returning the contract batch


# Phase 6 - Build model/config boundary (carefully)
Model should be configurable to a number of things (high complexity):
- HR target size
- LR conditioning size
- Temporal stack length
- Context encoder on/off
- Static channels on/off
- RainGate on/off
- Target variables count
- Variable embeddings/FiLM

Model should not infer these from tensors internally. Should be passed as a clear config object, e.g.:
`
ModelSpec(
    in_dynamic_channels=...,
    in_static_channels=...,
    out_channels=...,
    cond_lr_shape=...,
    target_hr_shape=...,
    temporal_steps=...,
    use_context_encoder=...,
    use_rain_gate=...,
    use_film_vars=...,
    use_film_doy=...,
    ...
)
`

Adapter then computes channels counts and the config layer assembles the spec.


# Phase 7 - Rebuild evaluation as composable families

`
evaluation/
    eval_runner.py
    registry.py
    families/
        dates.py
        distributions.py
        extremes.py
        probabilistic.py
        spatial.py
        scale.py
        features.py
        temporal.py
    metrics/
    plots/
`

Each family should expose `compute(...)`, `plot(...)`, `compute_and_plot(...)`.

Eval_runner.py should only orchestrate according to a config, e.g.:
- selected families
- mode = minimal/metrics/plots/full


# Phase 8 - Unified metadata

Make one module for:
- Variable canonical names
- Display names
- Units
- Colour maps
- Plotting ranges
- Variable groups

Make a specified config:
`
VariableSpec(
    key="prcp",
    long_name="Precipitation",
    data_name="tp",
    units="mm day-1",
    cmap="precip_cmap",
    is_positive_definite=True,
    transform_default="log-zscore",
)
`

