#!/bin/bash
#SBATCH --job-name=stride_norcp_test
#SBATCH --output=logs/slurm_%x_%j.log
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00

set -eo pipefail

# -----------------------------------------------------------------------------
# STRIDE NorCP test run on LUMI (inside container)
#
# Default behaviour:
#   1) run config compiler dry-run
#   2) run the lightweight full-pipeline smoke test
#
# Example submit:
# sbatch --account=project_465002493 \
#   --export=ALL,ROOT_DIR=/scratch/project_465002493/$USER/Code/STRIDE \
#   bash/hpc/run_norcp_test.sh
#
# Optional overrides:
#   CONFIG_REL=configs/experiments/pipeline_norcp.yaml
#   MODE=smoke            # smoke | dry-run | full
#   CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif
#   OVERLAY_IMG=/scratch/project_465002493/containers/overlays/my_overlay.img
#   STRIDE_RUNS=/scratch/project_465002493/$USER/runs/STRIDE
#   NORCP_DATA_DIR=/scratch/project_465002493/$USER/Data/NorCP/cropped
# -----------------------------------------------------------------------------

# --- User/site configuration (override when submitting) ---
ACCOUNT="${SLURM_JOB_ACCOUNT:-${ACCOUNT:-project_465002493}}"
USER_BASE="/scratch/${ACCOUNT}/${USER}"

ROOT_DIR="${ROOT_DIR:-${USER_BASE}/Code/STRIDE}"
STRIDE_RUNS="${STRIDE_RUNS:-${USER_BASE}/runs/STRIDE}"
DATA_BASE="${DATA_BASE:-${USER_BASE}/Data}"
NORCP_DATA_DIR="${NORCP_DATA_DIR:-${DATA_BASE}/NorCP/cropped}"

CONTAINER="${CONTAINER:-/scratch/${ACCOUNT}/containers/images/my_torch_container_with_plotting.sif}"
OVERLAY_IMG="${OVERLAY_IMG:-/scratch/${ACCOUNT}/containers/overlays/my_overlay.img}"
USE_OVERLAY="${USE_OVERLAY:-1}"

CONFIG_REL="${CONFIG_REL:-configs/experiments/pipeline_norcp.yaml}"
MODE="${MODE:-smoke}"
RUN_NAME="${RUN_NAME:-stride_norcp_test}"
RUN_ROOT="${RUN_ROOT:-${STRIDE_RUNS}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

export ACCOUNT USER_BASE ROOT_DIR STRIDE_RUNS DATA_BASE NORCP_DATA_DIR
export CONTAINER OVERLAY_IMG USE_OVERLAY CONFIG_REL MODE RUN_NAME RUN_ROOT LOG_DIR

set -u

# --- Modules ---
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

# --- Create log dirs ---
mkdir -p logs
mkdir -p "${LOG_DIR}"

# --- Threading caps ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- MIOpen workaround ---
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
MIOPEN_DB_DIR="${SCRATCH}/${USER}/miopen_db_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_DB_DIR}"
export MIOPEN_USER_DB_PATH="${MIOPEN_DB_DIR}/userdb.sql"
export MIOPEN_SYSTEM_DB_PATH="${MIOPEN_DB_DIR}/systemdb.sql"

# --- Derived paths ---
CFG="${ROOT_DIR}/${CONFIG_REL}"

# --- Export PYTHONPATH ---
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# --- Container command setup ---
SINGULARITY_ARGS=()
if [ "${USE_OVERLAY}" = "1" ]; then
  if [ ! -f "${OVERLAY_IMG}" ]; then
    echo "[ERROR] USE_OVERLAY=1 but overlay image not found: ${OVERLAY_IMG}" >&2
    echo "[ERROR] Build/update the overlay first before running STRIDE on LUMI." >&2
    exit 1
  fi
  SINGULARITY_ARGS+=(--overlay "${OVERLAY_IMG}:ro")
fi

# --- Info ---
echo "[INFO] ROOT_DIR        = ${ROOT_DIR}"
echo "[INFO] NORCP_DATA_DIR  = ${NORCP_DATA_DIR}"
echo "[INFO] STRIDE_RUNS     = ${STRIDE_RUNS}"
echo "[INFO] RUN_ROOT        = ${RUN_ROOT}"
echo "[INFO] LOG_DIR         = ${LOG_DIR}"
echo "[INFO] CONTAINER       = ${CONTAINER}"
echo "[INFO] OVERLAY_IMG     = ${OVERLAY_IMG}"
echo "[INFO] USE_OVERLAY     = ${USE_OVERLAY}"
echo "[INFO] CONFIG          = ${CFG}"
echo "[INFO] MODE            = ${MODE}"

if [ ! -f "${CFG}" ]; then
  echo "[ERROR] Config not found: ${CFG}" >&2
  exit 1
fi

if [ ! -d "${ROOT_DIR}" ]; then
  echo "[ERROR] ROOT_DIR not found: ${ROOT_DIR}" >&2
  exit 1
fi

if [ ! -d "${NORCP_DATA_DIR}" ]; then
  echo "[WARNING] NORCP_DATA_DIR does not exist: ${NORCP_DATA_DIR}"
  echo "[WARNING] The dataset config must point to a valid NorCP directory on LUMI."
fi

echo "[INFO] Running STRIDE NorCP test inside container..."

srun singularity exec "${SINGULARITY_ARGS[@]}" "${CONTAINER}" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  export ROOT_DIR='${ROOT_DIR}'
  export STRIDE_RUNS='${STRIDE_RUNS}'
  export NORCP_DATA_DIR='${NORCP_DATA_DIR}'
  export RUN_ROOT='${RUN_ROOT}'
  export LOG_DIR='${LOG_DIR}'
  cd '${ROOT_DIR}'

  BASE_PYTHON=\"\$(command -v python)\"

  if [ -z \"\${BASE_PYTHON}\" ]; then
    echo '[ERROR] Could not resolve python inside container.' >&2
    exit 1
  fi

  echo '[INFO] Python executable:'
  \"\${BASE_PYTHON}\" - <<'PY'
import sys
print(sys.executable)
PY

  echo '[INFO] Preflight imports...'
  \"\${BASE_PYTHON}\" - <<'PY'
import importlib
modules = ['yaml', 'numpy', 'torch', 'pandas']
missing = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f'[OK] import {name}')
    except Exception as exc:
        print(f'[MISSING] import {name}: {exc}')
        missing.append(name)
if missing:
    raise SystemExit(
        'Missing required Python packages in container/overlay: ' + ', '.join(missing)
    )
PY

  echo '[INFO] Running config dry-run...'
  \"\${BASE_PYTHON}\" cli/launch_pipeline.py --config '${CFG}' --dry-run

  case '${MODE}' in
    dry-run)
      echo '[INFO] MODE=dry-run -> stopping after config validation.'
      ;;
    smoke)
      echo '[INFO] MODE=smoke -> running lightweight full-pipeline smoke test.'
      \"\${BASE_PYTHON}\" test_scripts/smoke_test_full_pipeline.py '${CFG}'
      ;;
    full)
      echo '[INFO] MODE=full -> running full pipeline launcher.'
      \"\${BASE_PYTHON}\" cli/launch_pipeline.py --config '${CFG}'
      ;;
    *)
      echo '[ERROR] Unsupported MODE=${MODE}. Use one of: dry-run, smoke, full.' >&2
      exit 1
      ;;
  esac
"

echo "[INFO] Done."
