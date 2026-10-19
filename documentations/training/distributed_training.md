# Single-node distributed training on LUMI

STRIDE supports single-node PyTorch `DistributedDataParallel` (DDP) training while preserving the existing CPU and single-GPU paths. Generation and evaluation remain single-process.

This guide describes the supported scope, how to launch NorCP jobs on LUMI, how batch size changes under DDP, and what was learned during the initial rollout.

## Supported scope

The current implementation supports:

- CPU training with one process;
- single-GPU training with one process;
- single-node DDP training with one process per GPU;
- distributed train and validation samplers;
- loss aggregation across ranks;
- rank-zero logging, history, monitoring, previews, and checkpoint writes;
- checkpoints that can be loaded by CPU, single-GPU, or DDP processes.

The following remain deliberately single-process:

- generation;
- evaluation;
- the top-level pipeline runner.

The pipeline runner starts one DDP subprocess for training and waits for it to finish. It then starts generation and evaluation once each. Do not launch the full pipeline itself once per GPU.

## Relevant files

- `stride_core/utils/distributed.py`: distributed runtime helpers.
- `stride_core/training/training_main.py`: process-group setup and rank-aware logging.
- `stride_core/training/data.py`: distributed samplers.
- `stride_core/training/trainer.py`: rank-local devices, DDP wrapping, metric reduction, and rank-zero side effects.
- `stride_core/training/checkpoints.py`: portable model state dictionaries.
- `stride_core/pipeline/experiment_runner.py`: DDP training subprocess and single-process downstream stages.
- `configs/experiments/pipeline_norcp_ddp.yaml`: full NorCP DDP experiment.
- `configs/experiments/pipeline_norcp_ddp_test.yaml`: shorter DDP performance/pipeline test.
- `bash/hpc/run_norcp.sh`: full LUMI job.
- `bash/hpc/run_norcp_test.sh`: LUMI smoke and test job.

## Runtime configuration

The base training configuration contains:

```yaml
training:
  distributed:
    enabled: auto
    backend: nccl
    sync_batchnorm: false
    find_unused_parameters: false
```

`enabled: auto` initializes DDP only when a multi-process launcher is detected. A normal CPU or single-GPU invocation therefore retains its existing one-process behavior.

On LUMI, PyTorch continues to use the backend name `nccl`; ROCm supplies RCCL underneath it.

`STRIDE_TRAIN_NPROC` controls how many training processes the pipeline runner launches. It defaults to `1`. The LUMI scripts default to `8`.

## LUMI launch commands

Run commands from the repository root. Replace `<project>` with the Slurm project used for the allocation.

Before submitting a normal run, remove any temporary verbose RCCL diagnostics inherited from the login shell:

```bash
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
```

### Configuration and full-pipeline smoke test

```bash
sbatch --account=<project> \
  --export=ALL,MODE=smoke,STRIDE_TRAIN_NPROC=8 \
  bash/hpc/run_norcp_test.sh
```

Smoke mode intentionally limits work. It checks that DDP training starts and that generation and evaluation subsequently run once.

### Five-epoch DDP throughput/pipeline test

```bash
sbatch --account=<project> \
  --export=ALL,MODE=full,STRIDE_TRAIN_NPROC=8 \
  bash/hpc/run_norcp_test.sh
```

This uses `configs/experiments/pipeline_norcp_ddp_test.yaml` by default.

### Full NorCP experiment

```bash
sbatch --account=<project> \
  --export=ALL,STRIDE_TRAIN_NPROC=8 \
  bash/hpc/run_norcp.sh
```

This uses `configs/experiments/pipeline_norcp_ddp.yaml` by default. Check the requested wall time against a measured epoch duration before launching a long experiment.

The full config currently retains batch size 2 per rank, global batch size 16,
to preserve the earlier optimization batch. It is conservative rather than
throughput-optimized. Do not assume that the batch-8 test result applies to the
full config until the validated settings are deliberately transferred.

## LUMI RCCL settings

The LUMI scripts export these values outside and inside the container:

```bash
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
NCCL_NET_GDR_LEVEL=PHB
```

The interface setting is important. During initial testing, RCCL sometimes selected a management-network interface. Jobs then failed around the first synchronization with symptoms such as:

- a rank terminating with `SIGSEGV`;
- `ncclInternalError`;
- `socketProgress: Connection closed by remote peer`;
- another rank failing in `dist.barrier()`.

These failures were communication failures, not evidence that the project had exhausted storage hours. Explicitly selecting LUMI's HSN interfaces allowed all eight ranks to reach `Init COMPLETE` and enter training.

See the current [LUMI PyTorch documentation](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/) when changing containers or ROCm versions. RCCL behavior must be revalidated when the software environment changes.

## Batch size and expected scaling

The configured training batch size is the **per-rank** batch size:

```text
global batch size = per-rank batch size x world size
```

For eight ranks:

| Per-rank batch | Global batch | Approximate train steps in the current NorCP split |
| ---: | ---: | ---: |
| 2 | 16 | 1,096 per rank |
| 8 | 64 | 274 per rank |
| 16 | 128 | 137 per rank |

Increasing the number of GPUs while preserving global batch size does not guarantee a speedup. With eight GPUs and batch size 2 per rank, each GPU performs little model work before every gradient synchronization. For this relatively small model, communication and fixed per-step overhead can dominate.

Increasing the per-rank batch size can improve throughput because it:

- gives each GPU more work;
- uses more available VRAM;
- reduces optimizer steps per data epoch.

It also changes optimization behavior. A global batch of 64 is not numerically equivalent to a global batch of 16. Learning rate and epoch count may need separate tuning after throughput has been measured.

### Initial observed performance

The first successful eight-GPU pilot used batch size 2 per rank, global batch size 16. On the tested LUMI node and container it produced:

- approximately 9 minutes 38 seconds for a steady-state training epoch;
- approximately 2 minutes 13 seconds for validation;
- approximately 30 global samples per second;
- roughly 2-3% reported VRAM allocation per GPU.

These numbers are a diagnostic baseline, not a performance guarantee. Node state, container/ROCm version, filesystem load, model configuration, and data configuration can change them.

The DDP test configuration now uses batch size 8 per rank, global batch size 64, as a throughput pilot. It should report approximately 274 training steps per rank. Record measured results before changing another performance variable.

## RainGate and unused parameters

The current NorCP DDP configurations enable the RainGate model while disabling its auxiliary loss. RainGate parameters are consequently present in the model but are not connected to the returned training loss.

The NorCP DDP experiment configs therefore override:

```yaml
training:
  overrides:
    distributed:
      find_unused_parameters: true
```

Removing this setting without changing how RainGate participates in the loss can make DDP fail because not every parameter receives a gradient. Unused-parameter detection has overhead, so RainGate behavior should be optimized separately rather than mixed into DDP correctness changes.

## Monitoring progress

Primary training progress is written by rank zero to:

```text
<compiled-experiment-root>/logs/training.log
```

The DDP test config enables:

```yaml
training:
  overrides:
    logging:
      print_step_progress: true
      print_every_n_steps: 50
```

Example:

```text
[train] epoch=0001 step=000050 batch=0050/0274 loss=...
```

Timestamps on successive lines can be used to estimate steps per second and epoch duration. Without step progress, the log can remain unchanged between the first-batch overview and the end-of-epoch summary.

The startup line `Train batches` is the number of batches processed by each rank. The epoch-summary `num_batches` currently reports the sum of batch contributions across ranks, so it is larger by approximately the world size.

To inspect GPU activity in a running Slurm allocation:

```bash
srun --jobid=<job-id> --overlap -N1 -n1 -c1 \
  rocm-smi --showuse --showmemuse
```

All eight GPUs should have model memory allocated and should show activity across repeated samples. A single instantaneous 0% utilization reading is not by itself a failure.

## Log volume

`NCCL_DEBUG=INFO` is useful while diagnosing RCCL initialization, but it produces hundreds of topology and channel messages from all ranks in `training.stdout.log`. It does not represent repeated STRIDE training output.

After communication has been validated, unset `NCCL_DEBUG` and `NCCL_DEBUG_SUBSYS` for normal runs. Keep compact rank-zero step progress enabled when measuring performance.

## Checkpoint and stage behavior

Only rank zero writes shared checkpoints. Model state is saved without the DDP wrapper prefix, allowing a DDP checkpoint to be loaded for one-process generation.

After DDP training exits successfully, the pipeline launches generation once and evaluation once. A successful end-to-end test should therefore show:

1. one distributed training stage;
2. one generation stage;
3. one evaluation stage;
4. a checkpoint usable by the single-process generation stage.

## Validation checklist

Before relying on a new container or performance configuration, verify:

- all requested ranks report the expected `world_size`;
- each rank uses a different local GPU;
- all ranks complete RCCL initialization;
- training progress advances and losses remain finite;
- reduced training and validation losses are logged by rank zero;
- only rank zero writes checkpoints, history, previews, and monitoring files;
- generation and evaluation each run once after training;
- the DDP checkpoint loads in the single-process generation stage;
- CPU and single-GPU smoke tests still work.
