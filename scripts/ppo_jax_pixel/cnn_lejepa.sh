#!/bin/bash
#SBATCH --job-name=cnn-lejepa
#SBATCH --output=slurm_logs/cnn_lejepa_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH:-}"
WANDB_KEY_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

ENV_NAME="${1:-halfcheetah}"
SEED="${2:-0}"
WANDB_PROJECT="${3:-benchmark}"
WANDB_ENTITY="${4:-}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend spring
  --encoder-type cnn
  --n-envs 128
  --hw 84
  --total-timesteps 5000000
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --encoder-lr 3e-4
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
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
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --lejepa
  --lejepa-lambda 0.1
  --lejepa-proj-dim 128
  --lejepa-num-views 2
  --lejepa-num-slices 64
  --lejepa-t-points 17
  --lejepa-t-min -5.0
  --lejepa-t-max 5.0
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

uv run python ppo_pixelbrax.py "${COMMON_ARGS[@]}"
