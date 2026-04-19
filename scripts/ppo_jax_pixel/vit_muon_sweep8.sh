#!/bin/bash
#SBATCH --job-name=vit-muon-sweep8
#SBATCH --output=slurm_logs/vit_muon_sweep8_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Usage:
#   sbatch scripts/ppo_jax_pixel/vit_muon_sweep8.sh [env_name] [wandb_project] [wandb_entity] [seed] [total_timesteps]
# Example:
#   sbatch scripts/ppo_jax_pixel/vit_muon_sweep8.sh halfcheetah benchmark my_entity 0 10000000
ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
SEED="${4:-0}"
TOTAL_TIMESTEPS="${5:-10000000}"
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

ENCODER_LRS=(1e-4 5e-5)
ENCODER_TANH_SCALES=(0.10 0.25)
VIT_PATCH_SIZES=(7 14)

NUM_LRS=${#ENCODER_LRS[@]}
NUM_TANH=${#ENCODER_TANH_SCALES[@]}
NUM_PATCH=${#VIT_PATCH_SIZES[@]}
NUM_CONFIGS=$((NUM_LRS * NUM_TANH * NUM_PATCH))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

LR_IDX=$((TASK_ID % NUM_LRS))
TANH_IDX=$(((TASK_ID / NUM_LRS) % NUM_TANH))
PATCH_IDX=$((TASK_ID / (NUM_LRS * NUM_TANH)))

ENCODER_LR="${ENCODER_LRS[$LR_IDX]}"
ENCODER_TANH_SCALE="${ENCODER_TANH_SCALES[$TANH_IDX]}"
VIT_PATCH_SIZE="${VIT_PATCH_SIZES[$PATCH_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_muon_sweep8_10m_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_muon_sweep8,encoder_vit,method_ppo_muon,jepa_none,seed_${SEED},env_${ENV_NAME},lr_${ENCODER_LR},tanh_${ENCODER_TANH_SCALE},patch_${VIT_PATCH_SIZE},weight_decay_1e-4"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME} encoder_type=vit jepa_mode=none"
echo "Config: encoder_lr=${ENCODER_LR} tanh_scale=${ENCODER_TANH_SCALE} vit_patch_size=${VIT_PATCH_SIZE} weight_decay=1e-4 total_timesteps=${TOTAL_TIMESTEPS}"

LR_TOKEN="${ENCODER_LR//./p}"
LR_TOKEN="${LR_TOKEN//-/m}"
TANH_TOKEN="${ENCODER_TANH_SCALE//./p}"
EXP_NAME="ppo_muon_vit_sweep8_t${TASK_ID}_lr${LR_TOKEN}_ts${TANH_TOKEN}_p${VIT_PATCH_SIZE}"

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
  --encoder-lr "${ENCODER_LR}"
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
  --jepa-heads-lr 3e-4
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
  --jepa-mode none
  --encoder-type vit
  --encoder-tanh-scale "${ENCODER_TANH_SCALE}"
  --vit-patch-size "${VIT_PATCH_SIZE}"
  --vit-hidden-size 192
  --vit-mlp-dim 768
  --vit-num-heads 3
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-use-cls-token
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
