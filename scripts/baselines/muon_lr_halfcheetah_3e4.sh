#!/bin/bash
#SBATCH --job-name=cnn-muon-lr-hc
#SBATCH --output=slurm_logs/cnn_muon_lr_halfcheetah_3e4_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# Single 10M-step HalfCheetah run with Optax Muon head LR 3e-4.

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
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

ENV_NAME="halfcheetah"
BACKEND="spring"
ENCODER_TYPE="cnn"
MUON_LR="3e-4"
LR_TAG="3em4"

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="cnn_muon_lr_halfcheetah_3e4_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn_muon_lr_single,encoder_${ENCODER_TYPE},opt_muon,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},heads_muon_lr_${LR_TAG},halfcheetah,10m,neurips_baselines"

EXP_NAME="ppo_${ENCODER_TYPE}_muon_lr${LR_TAG}_${ENV_NAME}_b${BACKEND}_s${SEED}_10m"

echo "Running group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} encoder=${ENCODER_TYPE} opt=muon heads_muon_lr=${MUON_LR} seed=${SEED}"
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
  --muon-ns-steps 5
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

ARCH_ARGS=(
  --encoder-type "${ENCODER_TYPE}"
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr "${MUON_LR}"
  --max-grad-norm 0.05
  --encoder-tanh-scale 0.5
)

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  --heads-optimizer muon
