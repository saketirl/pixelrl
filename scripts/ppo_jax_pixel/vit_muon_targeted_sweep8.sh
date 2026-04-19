#!/bin/bash
#SBATCH --job-name=vit-muon-targeted8
#SBATCH --output=slurm_logs/vit_muon_targeted8_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Usage:
#   sbatch scripts/ppo_jax_pixel/vit_muon_targeted_sweep8.sh [env_name] [wandb_project] [wandb_entity] [seed] [total_timesteps]
# Example:
#   sbatch scripts/ppo_jax_pixel/vit_muon_targeted_sweep8.sh halfcheetah benchmark my_entity 0 10000000
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

# 8 configs around current best:
#   encoder_lr=5e-5, tanh_scale=0.25, patch=14, hidden=192, mlp=768, heads=3, layers=4
# Swept targeted fixes:
#   1) conv stem off/on
#   2) CLS readout vs mean pool
#   3) output tanh on/off
USE_CONV_STEMS=(false true)
USE_CLS_TOKENS=(true false)
APPLY_OUTPUT_TANHS=(true false)

NUM_STEMS=${#USE_CONV_STEMS[@]}
NUM_CLS=${#USE_CLS_TOKENS[@]}
NUM_TANH=${#APPLY_OUTPUT_TANHS[@]}
NUM_CONFIGS=$((NUM_STEMS * NUM_CLS * NUM_TANH))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

STEM_IDX=$((TASK_ID % NUM_STEMS))
CLS_IDX=$(((TASK_ID / NUM_STEMS) % NUM_CLS))
TANH_IDX=$((TASK_ID / (NUM_STEMS * NUM_CLS)))

USE_CONV_STEM="${USE_CONV_STEMS[$STEM_IDX]}"
USE_CLS_TOKEN="${USE_CLS_TOKENS[$CLS_IDX]}"
APPLY_OUTPUT_TANH="${APPLY_OUTPUT_TANHS[$TANH_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_muon_targeted8_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_muon_targeted8,encoder_vit,method_ppo_muon,jepa_none,seed_${SEED},env_${ENV_NAME},lr_5e-5,tanhscale_0.25,patch_14,warmup_500,convstem_${USE_CONV_STEM},cls_${USE_CLS_TOKEN},outtanh_${APPLY_OUTPUT_TANH}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: conv_stem=${USE_CONV_STEM} use_cls_token=${USE_CLS_TOKEN} apply_output_tanh=${APPLY_OUTPUT_TANH}"

EXP_NAME="ppo_muon_vit_targeted8_t${TASK_ID}_stem${USE_CONV_STEM}_cls${USE_CLS_TOKEN}_ot${APPLY_OUTPUT_TANH}"

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
  --encoder-warmup-updates 500
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
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --exp-name "${EXP_NAME}"
)

if [[ "${USE_CONV_STEM}" == "true" ]]; then
  COMMON_ARGS+=(--vit-use-conv-stem)
fi

if [[ "${USE_CLS_TOKEN}" == "true" ]]; then
  COMMON_ARGS+=(--vit-use-cls-token)
else
  COMMON_ARGS+=(--no-vit-use-cls-token)
fi

if [[ "${APPLY_OUTPUT_TANH}" == "true" ]]; then
  COMMON_ARGS+=(--vit-apply-output-tanh)
else
  COMMON_ARGS+=(--no-vit-apply-output-tanh)
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
