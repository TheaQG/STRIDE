

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

## NetCDF-only export (NorCP)

Set `output_format: netcdf` to write NetCDF instead of NPZ. In an experiment
configuration, merge these settings into the existing sections:

```yaml
stages:
  generation: true
  evaluation: false  # The built-in evaluator still reads NPZ.

generation:
  overrides:
    generation_run:
      data:
        shuffle: false
      outputs:
        output_format: netcdf
        save_physical: true
        save_members: true
        save_ensemble_mean: true
        save_pmm: true
      limits:
        max_cases: 3  # Small first check; null exports the full selected split.
```

For a standalone generation-run YAML, use the same `data`, `outputs`, and
`limits` entries directly under `generation_run`. The reusable generation base
uses these entries under `generation`. NPZ remains the default.

Run the usual pipeline command with your experiment config, or run generation
alone with a resolved generation-run config and an existing checkpoint:

```bash
python stride_core/generation/generation_main.py --config /path/to/generation_run.yaml
```

Use a new output directory/experiment name for an export. To reuse a trained
checkpoint in a new experiment, set `stages.training: false` and
`generation.overrides.generation_run.checkpoint` to
`{mode: explicit, path: /absolute/path/to/checkpoint_best.pt, use_ema_weights: true}`.
Keep its model and dataset/transform configuration consistent with that checkpoint.

Completed output under the generation output directory is:

```text
generation_metadata.json
netcdf/
    ensemble_members.nc  # variable(time, member, y, x)
    ensemble_mean.nc     # variable(time, y, x)
    pmm.nc               # variable(time, y, x)
    netcdf_manifest.json
```

No NPZ files, per-case directories, or per-case JSON sidecars are written. The
small run metadata and completion manifest are retained; scientific metadata
are embedded in each NetCDF file. `storage_mode` is ignored in NetCDF mode.
`save_metadata` controls NPZ sidecars, not essential NetCDF metadata.

The first NetCDF version writes all three products, in physical units, using
the NorCP adapter's full domain or one fixed crop. Random spatial crops and
non-Gregorian NorCP indexing are not supported. Generation remains single-process.
Each case is appended as it completes, so the full temporal stack is not held
in memory. On failure, files remain under `netcdf.incomplete/`; completed output
is published as `netcdf/`. Existing NPZ or NetCDF output is never deleted to
make room for a new NetCDF run.

NorCP precipitation is exported as a rate in `mm day-1`, not an accumulation.
Members are inverse-transformed before computing the mean and PMM. Output time
is the source target's representative time (e.g. 03:00 for a 00:00–06:00 mean);
source time bounds are retained when available. STRIDE's aligned sample time
is recovered as `time + stride_alignment_offset_hours` (a global attribute).

These three files contain forecasts and their coordinates, not reference targets
or conditioning fields. Built-in NetCDF evaluation is a separate follow-up step;
the pipeline rejects simultaneous NetCDF generation and evaluation for now.
