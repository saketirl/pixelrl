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

# Environment (change as needed: halfcheetah, walker2d, hopper, ant, humanoid)
ENV_NAME=${1:-"halfcheetah"}
SEED=${2:-0}

# uv run ppo_pixelbrax_jax.py \
#   --env-name ${ENV_NAME} \
#   --backend spring \
#   --n-envs 512 \
#   --hw 84 \
#   --total-timesteps 10000000 \
#   --num-steps 256 \
#   --num-minibatches 32 \
#   --update-epochs 4 \
#   --learning-rate 3e-5 \
#   --gamma 0.99 \
#   --gae-lambda 0.95 \
#   --clip-eps 0.2 \
#   --ent-coef 0.01 \
#   --vf-coef 0.5 \
#   --max-grad-norm 0.5 \
#   --seed ${SEED} \
#   --log-interval 1 \
#   --anneal-lr \
#   --track

# uv run ppo_pixelbrax_jax2.py \
#   --env-name ${ENV_NAME} \
#   --backend spring \
#   --n-envs 128 \
#   --hw 84 \
#   --total-timesteps 10000000 \
#   --num-steps 128 \
#   --num-minibatches 16 \
#   --update-epochs 10 \
#   --learning-rate 2e-4 \
#   --gamma 0.99 \
#   --gae-lambda 0.95 \
#   --clip-eps 0.1 \
#   --ent-coef 0.0 \
#   --vf-coef 0.5 \
#   --max-grad-norm 0.5 \
#   --seed ${SEED} \
#   --log-interval 1 \
#   --anneal-lr \
#   --track

uv run ppo_pixelbrax_jax2.py \
  --env-name ${ENV_NAME} \
  --backend spring \
  --n-envs 128 \
  --hw 84 \
  --total-timesteps 10000000 \
  --num-steps 10 \
  --num-minibatches 32 \
  --update-epochs 4 \
  --learning-rate 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --seed ${SEED} \
  --log-interval 1 \
  --frame-stack 4 \
  --action-repeat 4 \
  --anneal-lr \
  --track \
