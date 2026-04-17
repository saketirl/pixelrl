#!/bin/bash
#SBATCH --job-name=vit-muon
#SBATCH --output=slurm_logs/vit_muon_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Usage:
#   sbatch scripts/ppo_jax_pixel/vit_muon.sh [env_name] [wandb_project] [wandb_entity]
# Example:
#   sbatch scripts/ppo_jax_pixel/vit_muon.sh halfcheetah benchmark my_entity
ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

mkdir -p "${REPO_ROOT}/slurm_logs"

if [[ -d "${REPO_ROOT}/pixelbrax/brax/brax" ]]; then
  BRAX_PYTHONPATH="${REPO_ROOT}/pixelbrax/brax"
else
  BRAX_PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"
fi
export PYTHONPATH="${BRAX_PYTHONPATH}:${PYTHONPATH:-}"
WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

SEEDS=(0 1 2 3)
NUM_SEEDS=${#SEEDS[@]}

if (( TASK_ID < 0 || TASK_ID >= NUM_SEEDS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_SEEDS - 1))."
  exit 1
fi

SEED="${SEEDS[$TASK_ID]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_muon_20m_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_muon,encoder_vit,method_ppo_muon,jepa_none,seed_${SEED},env_${ENV_NAME},patch14,hidden192,layers4,stemtrue,clsfalse,outtanhfalse"

echo "Running TASK_ID=${TASK_ID}/${NUM_SEEDS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME} encoder_type=vit jepa_mode=none"

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
  --jepa-mode none
  --encoder-type vit
  --encoder-tanh-scale 0.25
  --vit-patch-size 14
  --vit-hidden-size 192
  --vit-mlp-dim 768
  --vit-num-heads 3
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-use-conv-stem
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --exp-name ppo_muon_vit
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
