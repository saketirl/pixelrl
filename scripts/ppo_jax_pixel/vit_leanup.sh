#!/bin/bash
#SBATCH --job-name=vit-leanup
#SBATCH --output=slurm_logs/vit_leanup_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# Single-run sense check using the best ViT+heads-MUON hyperparameters from
# vit_heads_muon_multienv.sh, fixed to halfcheetah and one seed.
#
# Usage:
#   sbatch scripts/ppo_jax_pixel/vit_leanup.sh [wandb_project] [wandb_entity] [total_timesteps] [seed]
# Example:
#   sbatch scripts/ppo_jax_pixel/vit_leanup.sh benchmark my_entity 10000000 0

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"
ENV_NAME="halfcheetah"

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

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="vit_leanup_${ENV_NAME}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_leanup,encoder_vit,heads_muon_true,env_${ENV_NAME},seed_${SEED}"

echo "Running env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Using worktree repo root: ${REPO_ROOT}"

EXP_NAME="ppo_vit_leanup_${ENV_NAME}_s${SEED}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend spring
  --n-envs 128
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --hw 84
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --encoder-lr 5e-5
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
  --weight-decay 1e-4
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
  --encoder-warmup-updates 500
  --encoder-type vit
  --encoder-tanh-scale 0.25
  --vit-patch-size 21
  --vit-hidden-size 128
  --vit-mlp-dim 512
  --vit-num-heads 8
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --vit-use-conv-stem
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --no-vit-qk-stiefel
  --use-heads-muon
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
