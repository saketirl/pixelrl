#!/bin/bash
#SBATCH --job-name=ppo-multienv
#SBATCH --output=slurm_logs/ppo_multienv_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Match ppo.sh defaults, but run an array sweep:
#   env in {hopper, swimmer, pusher, inverted_pendulum}
#   condition in {baseline, muon}
#   seed in {0}
# Total = 4 x 2 x 1 = 8 runs

export PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ -z "${WANDB_API_KEY:-}" && -f "secrets/wandb_api_key.txt" ]]; then
  export WANDB_API_KEY="$(cat secrets/wandb_api_key.txt)"
fi

ENVS=(hopper swimmer pusher inverted_pendulum)
BACKENDS=(positional generalized generalized generalized)
CONDITIONS=(baseline muon)
SEEDS=(0)

NUM_ENVS=${#ENVS[@]}
NUM_CONDITIONS=${#CONDITIONS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_CONDITIONS * NUM_SEEDS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
COND_IDX=$(((TASK_ID / NUM_SEEDS) % NUM_CONDITIONS))
ENV_IDX=$((TASK_ID / (NUM_SEEDS * NUM_CONDITIONS)))

ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"
CONDITION="${CONDITIONS[$COND_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} backend=${BACKEND} condition=${CONDITION} seed=${SEED}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs 128
  --hw 84
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
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
  --anneal-lr
  --track
)

if [[ -n "${WANDB_PROJECT}" ]]; then
  COMMON_ARGS+=(--wandb-project-name "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${CONDITION}" == "baseline" ]]; then
  uv run ppo_pixelbrax_jax2.py \
    "${COMMON_ARGS[@]}" \
    --learning-rate 3e-4 \
    --exp-name "ppo_pixel_cnn_baseline"
else
  uv run ppo_pixelbrax_jax2_muon.py \
    "${COMMON_ARGS[@]}" \
    --debug-repr \
    --encoder-lr 3e-4 \
    --heads-adam-lr 3e-4 \
    --heads-muon-lr 0.001 \
    --muon-dual-lr 0.01 \
    --muon-dual-steps 5 \
    --actor-muon-max-grad-norm 100 \
    --critic-muon-max-grad-norm 1 \
    --encoder-tanh-scale 0.5 \
    --exp-name "ppo_pixel_cnn_muon"
fi
