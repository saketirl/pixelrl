#!/bin/bash
#SBATCH --job-name=maze-humanoid-4algos-ar4
#SBATCH --output=slurm_logs/maze_humanoid_umaze_4algos_6seeds_ar4_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "${REPO_ROOT}/scripts/maze/ant_umaze_4algos_6seeds_ar4.sh" ]]; then
  REPO_ROOT="/home/guests/arjun/pixelrl"
fi

WANDB_PROJECT="${1:-pixel-maze}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
ENV_NAME="${4:-humanoid_u_maze}"
BACKEND="${5:-spring}"
ACTION_REPEAT="${6:-4}"
RUNS_PER_GPU="${7:-8}"
JAX_MEM_FRACTION="${8:-0.11}"
START_STAGGER_SECONDS="${9:-15}"

bash "${REPO_ROOT}/scripts/maze/ant_umaze_4algos_6seeds_ar4.sh" \
  "${WANDB_PROJECT}" \
  "${WANDB_ENTITY}" \
  "${TOTAL_TIMESTEPS}" \
  "${ENV_NAME}" \
  "${BACKEND}" \
  "${ACTION_REPEAT}" \
  "${RUNS_PER_GPU}" \
  "${JAX_MEM_FRACTION}" \
  "${START_STAGGER_SECONDS}"
