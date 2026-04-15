#!/bin/bash
#SBATCH --job-name=cnn-crate-adam-ha45
#SBATCH --output=slurm_logs/cnn_crate_adam_ha45_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# CNN encoder with CRATE heads and ADAM optimizer (head and trunk)
# on humanoid and ant environments, seeds 4-5.
#
# Fixed:
#   encoder_tanh_scale = 0.5
#   crate_step_size = 0.1
#   actor_grad_norm = 100
#   critic_grad_norm = 1
#
# 2 x 2 = 4 configs

WANDB_PROJECT="${1:-benchmark}"
TOTAL_TIMESTEPS="${2:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

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

ENVS=(humanoid ant)
BACKENDS=(spring spring)
SEEDS=(4 5)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_SEEDS))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
ENV_IDX=$((TASK_ID / NUM_SEEDS))

SEED="${SEEDS[$SEED_IDX]}"
ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"
ENCODER_SCALE="0.5"
CRATE_STEP_SIZE="0.1"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_crate_adam_ha45_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn_crate_adam,encoder_cnn,opt_adam,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},scale_${ENCODER_SCALE},crate,sweep_seeds45"

EXP_NAME="ppo_cnn_crate_adam_scale${ENCODER_SCALE}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} encoder=cnn encoder_scale=${ENCODER_SCALE} crate_step_size=${CRATE_STEP_SIZE} optimizer=adam seed=${SEED}"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs 128
  --track
  --debug-repr
  --wandb-project-name "${WANDB_PROJECT}"
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
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --anneal-lr
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --exp-name "${EXP_NAME}"
)

ARCH_ARGS=(
  --encoder-type cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --max-grad-norm 0.05
  --encoder-tanh-scale "${ENCODER_SCALE}"
  --use-crate-head
  --crate-step-size "${CRATE_STEP_SIZE}"
)

cd "${REPO_ROOT}"
uv run python -m wandb login 9fb4ba17a708de72496774b2e25d219f07de038d
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  --no-use-heads-muon