

#!/bin/bash
#SBATCH --job-name=stride-pipeline
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

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

python cli/launch_pipeline.py --config "$PIPELINE_CONFIG" "${EXTRA_ARGS[@]}"