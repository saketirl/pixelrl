#!/bin/bash
#SBATCH --job-name=cnn-muon-lr-walk
#SBATCH --output=slurm_logs/cnn_muon_lr_walker2d_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Tune Optax Muon head LR on walker2d with one seed per LR.
#
# Swept:
#   heads_muon_lr in {3e-4, 1e-3, 3e-3, 1e-2}
#
# Fixed:
#   env      = walker2d
#   backend  = spring
#   seed     = 0 by default
#   encoder  = cnn
#
# 4 x 1 seed = 4 configs

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-1000000}"
SEED="${4:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

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

ENV_NAME="walker2d"
BACKEND="spring"
ENCODER_TYPE="cnn"
MUON_LRS=(3e-4 1e-3 3e-3 1e-2)

NUM_LRS=${#MUON_LRS[@]}
NUM_CONFIGS=${NUM_LRS}

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

MUON_LR="${MUON_LRS[$TASK_ID]}"
LR_TAG="${MUON_LR//./p}"
LR_TAG="${LR_TAG//-/m}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_muon_lr_walker2d_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn_muon_lr_sweep,encoder_${ENCODER_TYPE},opt_muon,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},heads_muon_lr_${LR_TAG},walker2d,1m,neurips_sweep"

EXP_NAME="ppo_${ENCODER_TYPE}_muon_lr${LR_TAG}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
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

#   --heads-muon-lr 3e-4 seems good