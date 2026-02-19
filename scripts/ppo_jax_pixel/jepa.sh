#!/bin/bash
#SBATCH --job-name=ppo-jepa-sweep
#SBATCH --output=slurm_logs/jepa_%A_%a.out
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
#   sbatch scripts/ppo_jax_pixel/jepa.sh [env_name] [seed]
# Override sweep size/concurrency at submit time, e.g.:
#   sbatch --array=0-7%8 --time=08:00:00 scripts/ppo_jax_pixel/jepa.sh walker2d 0
ENV_NAME=${1:-"halfcheetah"}
SEED=${2:-0}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Sensible first-pass JEPA sweep for 8 H100s:
# - jepa-lambda has the biggest effect on aux-loss strength.
# - jepa-ema-tau controls target stability.
LAMBDAS=(1e-4 3e-4 1e-3 3e-3)
EMA_TAUS=(0.995 0.998)
JEPA_WARMUP_UPDATES=50
JEPA_RAMPUP_UPDATES=200

NUM_LAMBDAS=${#LAMBDAS[@]}
NUM_TAUS=${#EMA_TAUS[@]}
NUM_CONFIGS=$((NUM_LAMBDAS * NUM_TAUS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

LAMBDA_IDX=$((TASK_ID % NUM_LAMBDAS))
TAU_IDX=$((TASK_ID / NUM_LAMBDAS))

JEPA_LAMBDA=${LAMBDAS[$LAMBDA_IDX]}
JEPA_EMA_TAU=${EMA_TAUS[$TAU_IDX]}

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} on ${ENV_NAME}"
echo "seed=${SEED}, jepa_lambda=${JEPA_LAMBDA}, jepa_ema_tau=${JEPA_EMA_TAU}, warmup=${JEPA_WARMUP_UPDATES}, rampup=${JEPA_RAMPUP_UPDATES}"

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
  --jepa-mode jepa \
  --jepa-lambda "${JEPA_LAMBDA}" \
  --jepa-warmup-updates "${JEPA_WARMUP_UPDATES}" \
  --jepa-rampup-updates "${JEPA_RAMPUP_UPDATES}" \
  --jepa-ema-tau "${JEPA_EMA_TAU}"
