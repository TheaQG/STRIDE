

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

EDM settings belong in `generation.sampler` in the reusable generation YAML.
The compiler merges the sampler base, then the generation base, then the
experiment override at `generation.overrides.generation.sampler`. Full
generation and training previews both use those resolved settings.

Ensemble size and seeding belong in `generation_run.sampling` in a standalone
run YAML, or `generation.overrides.generation_run.sampling` in an experiment.
These values override the corresponding defaults in `generation.sampling`.
A fixed seed is computed as `base_seed + 10000 * batch_idx + member_idx`, using
zero-based indices. Preview sampling uses the training process's RNG.

Legacy flat EDM keys under `generation` or in an unwrapped sampler YAML still
work. An unwrapped YAML can also use a top-level `sampler` block. Duplicate flat
and nested settings must agree. Historical null placeholders for `num_steps`,
`sigma_min`, `sigma_max`, and `rho` are treated as unspecified when reading existing YAML. New experiment overrides
reject explicit null values, except for `S_max: null` to request an unbounded
churn limit. A legacy run YAML with a copied sampler block must agree with
its referenced generation config.

Set `generation.output.return_intermediates: true` to ask EDM to collect its
trajectory. The sampler accumulates one CPU tensor for the initial state and
one per step, transferring each from the accelerator when sampling on one.
Peak trajectory memory is proportional to the number of steps and the size
of the generated batch. The `generate_batch` wrapper returns only the final
sample, and full generation and previews do not persist the trajectory.

The effective settings are recorded as `sampler` in the generation log, run JSON,
NetCDF global attributes, and each preview's JSON sidecar. Full generation
records the ensemble and seed settings separately as `sampling`. NetCDF stores
both mappings as JSON strings. An unbounded `S_max` is written as `"inf"`.

The current sampler base explicitly selects 56 steps and `S_churn: 2.0`, with
churn active between sigma 40 and 80. Older STRIDE versions ignored this nested
configuration. To replay the old effective sampler settings with the corrected
parser, explicitly set `num_steps: 18`, `S_churn: 0.0`, `S_min: 0.0`, and
`S_max: "inf"`. Matching sampler settings alone does not reproduce a run
without the original weights, input order, and RNG state.

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
