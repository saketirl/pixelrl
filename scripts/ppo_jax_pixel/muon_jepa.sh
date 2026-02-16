#!/bin/bash
#SBATCH --job-name=ppo-muon-le-jepa
#SBATCH --output=slurm_logs/ppo_muon_le_jepa_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

export PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH:-}"
export WANDB_API_KEY=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

# Usage:
#   sbatch scripts/ppo/muon_jepa [env_name] [wandb_project] [wandb_entity]
# Example:
#   sbatch scripts/ppo/muon_jepa halfcheetah benchmark my_entity
ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# 4 jobs = 4 seeds.
SEEDS=(0 1 2 3)
NUM_SEEDS=${#SEEDS[@]}

if (( TASK_ID < 0 || TASK_ID >= NUM_SEEDS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_SEEDS - 1))."
  exit 1
fi

SEED="${SEEDS[$TASK_ID]}"

# Fixed LeJEPA settings.
JEPA_LAMBDA=1e-4
JEPA_WARMUP_UPDATES=50
JEPA_RAMPUP_UPDATES=200
JEPA_EMA_TAU=0.998
SIGREG_WEIGHT=1.0
SIGREG_NUM_SLICES=64
PROJ_DIM=128

GROUP_NAME="${ENV_NAME}_ppo_muon_le_jepa_10m_${SLURM_JOB_ID:-local}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="ppo_muon_le_jepa,method_le_jepa,encoder_cnn,seed_${SEED},env_${ENV_NAME}"

echo "Running TASK_ID=${TASK_ID}/${NUM_SEEDS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend spring
  --n-envs 128
  --hw 84
  --total-timesteps 10000000
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --encoder-lr 3e-4
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --max-grad-norm 0.05
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --encoder-type cnn
  --anneal-lr
  --track
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --encoder-tanh-scale 0.5
  --wandb-project-name "${WANDB_PROJECT}"
  --jepa-mode le_jepa
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

uv run ppo_pixelbrax_jax2_muon.py \
  "${COMMON_ARGS[@]}" \
  --jepa-lambda "${JEPA_LAMBDA}" \
  --jepa-warmup-updates "${JEPA_WARMUP_UPDATES}" \
  --jepa-rampup-updates "${JEPA_RAMPUP_UPDATES}" \
  --jepa-ema-tau "${JEPA_EMA_TAU}" \
  --sigreg-weight "${SIGREG_WEIGHT}" \
  --sigreg-num-slices "${SIGREG_NUM_SLICES}" \
  --proj-dim "${PROJ_DIM}"
