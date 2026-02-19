#!/bin/bash
#SBATCH --job-name=ppo-muon-cnn-vs-mlp
#SBATCH --output=slurm_logs/ppo_muon_cnn_vs_mlp_%A_%a.out
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
#   sbatch scripts/ppo_jax_pixel/ppo_muon_cnn_vs_mlp.sh [env_name] [wandb_project] [wandb_entity]
# Example:
#   sbatch scripts/ppo_jax_pixel/ppo_muon_cnn_vs_mlp.sh halfcheetah benchmark my_entity
ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# 8 total jobs = 2 encoder configs x 4 seeds.
SEEDS=(0 1 2 3)
ENCODER_TYPES=("cnn" "mlp")

NUM_SEEDS=${#SEEDS[@]}
NUM_ENCODERS=${#ENCODER_TYPES[@]}
NUM_CONFIGS=$((NUM_SEEDS * NUM_ENCODERS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

ENCODER_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

ENCODER_TYPE="${ENCODER_TYPES[$ENCODER_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

GROUP_NAME="${ENV_NAME}_ppo_muon_cnn_vs_mlp_10m_${SLURM_JOB_ID:-local}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="ppo_muon_cnn_vs_mlp,encoder_${ENCODER_TYPE},seed_${SEED},env_${ENV_NAME}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} encoder_type=${ENCODER_TYPE} seed=${SEED} group=${GROUP_NAME}"

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
  --encoder-type "${ENCODER_TYPE}"
  --anneal-lr
  --track
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --encoder-tanh-scale 0.5
  --wandb-project-name "${WANDB_PROJECT}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

uv run ppo_pixelbrax_jax2_muon.py "${COMMON_ARGS[@]}"
