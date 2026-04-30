#!/bin/bash
#SBATCH --job-name=goal-humanoid-cratecnn-s0
#SBATCH --output=slurm_logs/goal_humanoid_crate_cnn_meanbound_boundedstd_crate_heads_seed0_stiefel_adam_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/scripts/goal/crate_cnn_meanbound_boundedstd_crate_heads_goal_seed0_stiefel_adam.sh" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"
ENV_NAME="${5:-humanoid_goal}"
BACKEND="${6:-spring}"
ACTION_REPEAT="${7:-4}"
JAX_MEM_FRACTION="${8:-0.45}"

bash "${REPO_ROOT}/scripts/goal/crate_cnn_meanbound_boundedstd_crate_heads_goal_seed0_stiefel_adam.sh" \
  "${WANDB_PROJECT}" \
  "${WANDB_ENTITY}" \
  "${TOTAL_TIMESTEPS}" \
  "${SEED}" \
  "${ENV_NAME}" \
  "${BACKEND}" \
  "${ACTION_REPEAT}" \
  "${JAX_MEM_FRACTION}"
