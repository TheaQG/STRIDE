#!/bin/bash 
#SBATCH -N 1
###SBATCH --nodes 1

#SBATCH -t 15:00:00 
#SBATCH -J STRIDE
#SBATCH --chdir=/ec/res4/scratch/smf/log/log_stride
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH --gpus=2
#SBATCH --qos=ng
##BATCH --mem-per-cpu=4G

###SBATCH --ntasks=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --account=spsewang

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch bash/run_pipeline.sh <pipeline_config.yaml> [--dry-run]"
  exit 1
fi

STRIDE_RUNS=$SCRATCH/HCLIMAI/STRIDE_RUNS
export STRIDE_RUNS

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
#echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

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
ENV_NAME='sdm'
#interactive -p gpu --gpus 1 -A NAISS2026-4-912-gpu
#module load python3/3.13.16-01
#python3 -m venv --system-site-packages stride $PERM/venvs/stride
source $PERM/venvs/${ENV_NAME}/bin/activate
#pip install pyyaml (pyyaml-6.0.3)
#pip install torch (torch-2.14.1)
#pip install numpy (numpy-2.3.5)
#pip install pandas (pandas-2.3.3)
#pip install xarray (xarray-2026.9.0)
#pip install matplotlib (matplotlib-3.10.9)
#pip install netcdf4 (netcdf4-1.7.4)

cd $HOME/Scripts/STRIDE
python cli/launch_pipeline.py --config "$PIPELINE_CONFIG" "${EXTRA_ARGS[@]}"

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0
#sbatch bash/run_pipeline_atos.sh  configs/experiments/pipeline_norcp_atos.yaml

