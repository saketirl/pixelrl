#!/bin/bash
#SBATCH --job-name=ppo-le-jepa-sweep
#SBATCH --output=slurm_logs/le_jepa_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

export PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH:-}"
WANDB_KEY_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

# Usage:
#   sbatch scripts/ppo_jax_pixel/le_jepa.sh [env_name] [seed]
# Override sweep size/concurrency at submit time, e.g.:
#   sbatch --array=0-7%8 --time=08:00:00 scripts/ppo_jax_pixel/le_jepa.sh walker2d 0
ENV_NAME=${1:-"halfcheetah"}
SEED=${2:-0}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# LeJEPA-only sweep for 8 H100s with JEPA controls fixed:
# - jepa-lambda fixed at 1e-4.
# - jepa-ema-tau fixed at 0.998.
# Sweep only LeJEPA-specific hyperparameters.
JEPA_LAMBDA=1e-4
JEPA_EMA_TAU=0.998
SIGREG_WEIGHTS=(0.5 1.0)
SIGREG_NUM_SLICES=(32 64)
PROJ_DIMS=(128 256)
JEPA_WARMUP_UPDATES=50
JEPA_RAMPUP_UPDATES=200

NUM_SIGREG=${#SIGREG_WEIGHTS[@]}
NUM_SLICES=${#SIGREG_NUM_SLICES[@]}
NUM_PROJ_DIMS=${#PROJ_DIMS[@]}
NUM_CONFIGS=$((NUM_SIGREG * NUM_SLICES * NUM_PROJ_DIMS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

SIGREG_IDX=$((TASK_ID % NUM_SIGREG))
SLICES_IDX=$(((TASK_ID / NUM_SIGREG) % NUM_SLICES))
PROJ_IDX=$((TASK_ID / (NUM_SIGREG * NUM_SLICES)))

SIGREG_WEIGHT=${SIGREG_WEIGHTS[$SIGREG_IDX]}
SIGREG_SLICE_COUNT=${SIGREG_NUM_SLICES[$SLICES_IDX]}
PROJ_DIM=${PROJ_DIMS[$PROJ_IDX]}

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} on ${ENV_NAME}"
echo "seed=${SEED}, jepa_lambda=${JEPA_LAMBDA}, jepa_ema_tau=${JEPA_EMA_TAU}, sigreg_weight=${SIGREG_WEIGHT}, sigreg_num_slices=${SIGREG_SLICE_COUNT}, proj_dim=${PROJ_DIM}, warmup=${JEPA_WARMUP_UPDATES}, rampup=${JEPA_RAMPUP_UPDATES}"

uv run ppo_pixelbrax_jax2_muon.py \
  --env-name "${ENV_NAME}" \
  --backend spring \
  --n-envs 128 \
  --track \
  --hw 84 \
  --total-timesteps 5000000 \
  --num-steps 10 \
  --num-minibatches 32 \
  --update-epochs 4 \
  --encoder-lr 3e-4 \
  --heads-muon-lr 0.001 \
  --heads-adam-lr 3e-4 \
  --jepa-heads-lr 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --seed 1 \
  --log-interval 1 \
  --frame-stack 4 \
  --action-repeat 4 \
  --anneal-lr \
  --jepa-mode le_jepa \
  --jepa-lambda "${JEPA_LAMBDA}" \
  --jepa-warmup-updates "${JEPA_WARMUP_UPDATES}" \
  --jepa-rampup-updates "${JEPA_RAMPUP_UPDATES}" \
  --jepa-ema-tau "${JEPA_EMA_TAU}" \
  --sigreg-weight "${SIGREG_WEIGHT}" \
  --sigreg-num-slices "${SIGREG_SLICE_COUNT}" \
  --proj-dim "${PROJ_DIM}"
