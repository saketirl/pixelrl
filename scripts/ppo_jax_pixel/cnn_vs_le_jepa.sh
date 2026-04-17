#!/bin/bash
#SBATCH --job-name=cnn-vs-le-jepa
#SBATCH --output=slurm_logs/cnn_vs_le_jepa_%A_%a.out
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
#   sbatch scripts/ppo_jax_pixel/cnn_vs_le_jepa.sh [env_name] [wandb_project] [wandb_entity]
# Example:
#   sbatch scripts/ppo_jax_pixel/cnn_vs_le_jepa.sh halfcheetah benchmark my_entity
ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# 8 total jobs = 2 methods x 4 seeds to fully utilize 8 GPUs.
SEEDS=(0 1 2 3)
METHODS=("none" "le_jepa")

NUM_SEEDS=${#SEEDS[@]}
NUM_METHODS=${#METHODS[@]}
NUM_CONFIGS=$((NUM_SEEDS * NUM_METHODS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

METHOD_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

JEPA_MODE="${METHODS[$METHOD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Group all 8 runs under one comparison label in WandB.
GROUP_NAME="${ENV_NAME}_cnn_vs_le_jepa_5m_${SLURM_JOB_ID:-local}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

# Tags make filtering/aggregation easier in the UI.
if [[ "${JEPA_MODE}" == "none" ]]; then
  export WANDB_TAGS="cnn_vs_le_jepa,method_cnn,seed_${SEED},env_${ENV_NAME}"
else
  export WANDB_TAGS="cnn_vs_le_jepa,method_le_jepa,seed_${SEED},env_${ENV_NAME}"
fi

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} jepa_mode=${JEPA_MODE} seed=${SEED} group=${GROUP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend spring
  --n-envs 128
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --hw 84
  --total-timesteps 20000000
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --encoder-lr 3e-4
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
  --jepa-heads-lr 3e-4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --max-grad-norm 0.5
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --anneal-lr
  --jepa-mode "${JEPA_MODE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${JEPA_MODE}" == "le_jepa" ]]; then
  uv run ppo_pixelbrax_jax2_muon.py \
    "${COMMON_ARGS[@]}" \
    --jepa-lambda 1e-4 \
    --jepa-warmup-updates 50 \
    --jepa-rampup-updates 200 \
    --jepa-ema-tau 0.998 \
    --sigreg-weight 1.0 \
    --sigreg-num-slices 64 \
    --proj-dim 128
else
  uv run ppo_pixelbrax_jax2_muon.py "${COMMON_ARGS[@]}"
fi
