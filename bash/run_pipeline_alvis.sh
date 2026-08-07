#!/bin/bash 
#SBATCH -A NAISS2025-1-11  -p alvis
#SBATCH -N 1 
###SBATCH --gpus-per-node=A40:1
#SBATCH --gpus-per-node=A100:1 
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00
#SBATCH -J stride-pipeline
#SBATCH --chdir=/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/log/log_stride/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch bash/run_pipeline.sh <pipeline_config.yaml> [--dry-run]"
  exit 1
fi

PIPELINE_CONFIG="$1"
shift || true

EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo
echo "========================="
echo "STRIDE Slurm pipeline run"
echo "========================="
echo "Repository root: $REPO_ROOT"
echo "Pipeline config: $PIPELINE_CONFIG"
echo "Python: $(command -v python || true)"
echo "Working dir: $(pwd)"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

current_date_time="`date`";
echo The run starts from $current_date_time
echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

#export HDF5_USE_FILE_LOCKING=FALSE
#export TF_GPU_ALLOCATOR=cuda_malloc_async
##export CUDA_VISIBLE_DEVICES=1 
#export TF_DETERMINISTIC_OPS=0
#export TF_FORCE_GPU_ALLOW_GROWTH=true
#ecinteractive -g

DOMAIN='norcp'
#DOMAIN='TestDomain'
VARIABLE='tas'

echo 'domain is' ${DOMAIN}
set -exu 

module --force purge
#module load virtualenv/20.26.2-GCCcore-13.3.0
#module load Python/3.12.3-GCCcore-13.3.0
#module load netcdf4-python/1.7.1.post2-foss-2024a
module load virtualenv/20.23.1-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load netcdf4-python/1.6.4-foss-2023a
module load zarr/2.17.1-foss-2023a
module load xarray/2023.9.0-gfbf-2023a
module load PyYAML/6.0-GCCcore-12.3.0
module load dask/2023.9.2-foss-2023a
source $HOME/venvs/stride/bin/activate

cd $HOME/STRIDE
python cli/launch_pipeline.py --config "$PIPELINE_CONFIG" "${EXTRA_ARGS[@]}"

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0

