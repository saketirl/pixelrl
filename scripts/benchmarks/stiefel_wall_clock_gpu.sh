#!/bin/bash
#SBATCH --job-name=stiefel-clock
#SBATCH --output=slurm_logs/stiefel_wall_clock_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/scripts/benchmarks/stiefel_wall_clock.py" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"
cd "${REPO_ROOT}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

uv run python scripts/benchmarks/stiefel_wall_clock.py --iters 200 --warmup 10
