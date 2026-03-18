# STRIDE – Quickstart & Onboarding

This document gets you from **zero → successful smoke run** on STRIDE.

---

## 1. What STRIDE does

STRIDE is a config-driven pipeline for **climate downscaling with diffusion models (EDM)**.

Supported workflows:
- Training
- Generation
- Evaluation

Supported datasets (current):
- NorCP (cropped)
- DANRA / ERA5 (small test dataset)

Main entrypoint:
```
python cli/launch_pipeline.py --config <experiment.yaml>
```

Target runtime:
- LUMI (primary)
- Local (for dry-run + smoke tests)

---

## 2. Minimal repo structure

You only need to understand:

- `configs/` → all configuration files
- `stride_core/` → training, generation, evaluation logic
- `data_adapters/` → dataset-specific handling
- `cli/` → launch scripts
- `documentations/` → docs (this file)

---

## 3. Environment requirements

STRIDE is expected to run inside a **Singularity container** on HPC.

### Required Python packages

Make sure your container (or overlay) contains:

- torch
- numpy
- pandas
- xarray
- netCDF4
- zarr
- dask
- omegaconf
- pyyaml

If something fails with `ModuleNotFoundError`, your container is missing packages.

---

## 4. Data setup

### NorCP

Expected root:
```
/scratch/<project>/<user>/Data/NorCP/cropped
```

⚠️ Watch out for:
- accidental nested folders like `cropped/cropped/`

---

### ⚠️ Using larger (non-cropped) domains

If you want to use a **larger spatial domain** than the provided `cropped` dataset, you must update both **data paths and model configuration**.

#### 1. Update dataset path

Point your dataset config to the correct directory, e.g.:

```
/scratch/<project>/<user>/Data/NorCP/<your_domain>
```

instead of:
```
.../NorCP/cropped
```

---

#### 2. Update spatial dimensions

The model and data pipeline assume specific spatial shapes.

You must update (depending on config setup):

- target resolution (HR)
- conditioning resolution (LR)
- number of channels (if different variables)

Typical places to update:
- dataset config (`configs/datasets/...`)
- model config (`configs/models/...`)

---

#### 3. Be aware of non-square inputs

STRIDE supports **non-square inputs** (e.g. 92×68), but:

- model architecture must be compatible
- downsampling/upsampling factors must divide spatial dimensions cleanly

If shapes are incompatible, you will get runtime errors in the model.

---

#### 4. Recompute splits and statistics

Splits and statistics are **domain-specific**.

If you change domain:
- you MUST rerun split generation
- you MUST recompute statistics

---

#### Summary

Switching domain requires updating:
- data path
- spatial dimensions (HR + LR)
- possibly model config
- splits + statistics

---

### DANRA / ERA5 (small)

Expected root:
```
/scratch/<project>/<user>/Data/Data_DiffMod_small
```

### Required files

Each dataset must have:
- train/val/test split
- statistics (computed on train split)

These are typically stored under:
```
data_adapters/.../saved/
```

---

### ⚠️ Required preprocessing (splits + statistics)

Before you can run STRIDE, you **must first generate dataset splits and statistics**.

The pipeline will fail or behave incorrectly if these are missing.

#### Step 1: Generate splits

Example (NorCP):

```
python -m data_adapters.norcp.launch_splits \
  --root-dir /path/to/NorCP/cropped \
  --scenario-name ECMWF-ERAINT \
  --val-start 2010-01-01T00:00:00 \
  --val-end 2012-12-31T18:00:00 \
  --test-start 2013-01-01T00:00:00 \
  --test-end 2018-12-31T18:00:00
```

This creates a split manifest under:
```
data_adapters/norcp/saved/splits/
```

---

#### Step 2: Compute statistics (on TRAIN split only)

Example:

```
python -m data_adapters.norcp.launch_statistics \
  --root-dir /path/to/NorCP/cropped \
  --scenario-name ECMWF-ERAINT \
  --split-manifest-path <path-to-generated-split-json> \
  --split-name train \
  --domain-tag full_domain \
  --backend cdo \
  --include-static
```

This computes normalization statistics and saves them under:
```
data_adapters/.../saved/statistics/
```

---

#### Important notes

- Statistics must be computed **only on the training split**
- The exact `split-manifest-path` comes from the split generation step
- Both splits and statistics must exist before running the pipeline

---

---

## 5. Config system (important)

STRIDE uses layered configs:

- dataset config → full dataset definition
- model config → architecture
- training config → training behavior
- generation config → sampling behavior
- evaluation config → evaluation logic
- experiment config → ties everything together

The **config compiler** resolves all configs into:
```
runs/<experiment>/<experiment>/compiled_configs/
```

---

## 6. How to run

### Dry run (always first)

```
python cli/launch_pipeline.py \
  --config configs/experiments/pipeline_norcp.yaml \
  --mode dry-run
```

Checks:
- config resolution
- paths
- basic validation

---

### Smoke test (recommended)

Use reduced batches/epochs:

```
python cli/launch_pipeline.py \
  --config configs/experiments/pipeline_norcp.yaml \
  --mode smoke
```

This should:
- run a few training steps
- generate samples
- run evaluation

---

### Full run

```
python cli/launch_pipeline.py \
  --config configs/experiments/pipeline_norcp.yaml
```

---

## 7. HPC (LUMI) usage

Typical workflow:

1. Copy data to `/scratch/...`
2. Ensure container + overlay are available
3. Submit SLURM job

Important environment variables:

```
ROOT_DIR
DATA_DIR
RUN_ROOT
```

---

## 8. Performance tips

### Data loading

In `training_base.yaml`:

```
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor_train: 4
prefetch_factor_val: 2
```

⚠️ Too many workers (e.g. 50+) can slow things down.

---

### GPU usage

Check with:
```
nvidia-smi
```

If you only see activity on one GPU:
→ you are running single-GPU

---

### Mixed precision

Enabled automatically in trainer:
- AMP (`torch.cuda.amp`)
- cuDNN benchmark

---

## 9. Validation checklist

Before running full experiments:

- [ ] Dry-run succeeds
- [ ] Dataset path exists
- [ ] No missing Python packages
- [ ] Smoke test completes
- [ ] Outputs appear in `runs/`

---

## 10. Common issues

### Missing packages

Error:
```
ModuleNotFoundError
```
Fix:
- update container or overlay

---

### Wrong data path

Error:
```
path does not exist
```
Fix:
- check `/scratch/...` paths

---

### Nested directories

Example:
```
cropped/cropped/
```
Fix:
- move files one level up

---

### Slow training

Common causes:
- too many workers
- no GPU utilization
- data loading bottleneck

---

## 11. Known limitations

- Further GPU optimization may require additional setup (DDP)
- Performance still under active optimization
- Some configs are experimental

---

## 12. Recommended onboarding flow

1. Pull repo
2. Verify container / overlay
3. Verify dataset path
4. Run dry-run
5. Run smoke test
6. Inspect compiled configs
7. Run full experiment

---

If something breaks, start from **dry-run** and work forward.
