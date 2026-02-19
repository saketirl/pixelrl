#!/bin/bash
#SBATCH --job-name=vit-lit-v2
#SBATCH --output=slurm_logs/vit_lit_v2_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Literature-based ViT sweep v2 - FASTER configs
#
# Key insight: The paper used NO conv stem with patch_size=12 on 84x84 → 49 patches
# We try middle ground: more patches than 9 but fewer than 36
#
# Strategy: Remove conv stem, use larger patches on raw 84x84
#   - patch_size=21 on 84x84 → 4x4 = 16 patches (faster, reasonable)
#   - patch_size=14 on 84x84 → 6x6 = 36 patches (no stem overhead)
#
# Also try: conv stem + fewer layers for speed

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

# Config 0: No stem, patch=21, 16 patches, 4 layers (balanced)
# Config 1: No stem, patch=14, 36 patches, 3 layers (more patches, fewer layers)
# Config 2: No stem, patch=12, 49 patches, 2 layers (paper patches, minimal layers)
# Config 3: Stem, patch=7, 36 patches, 2 layers (original approach, fewer layers)

USE_CONV_STEMS=(false false false true)
PATCH_SIZES=(21 14 12 7)
NUM_LAYERS_ARR=(4 3 2 2)
HIDDEN_SIZES=(128 128 128 128)
NUM_HEADS=(8 8 8 8)
MLP_DIMS=(512 512 512 512)

NUM_CONFIGS=${#PATCH_SIZES[@]}

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

USE_CONV_STEM="${USE_CONV_STEMS[$TASK_ID]}"
PATCH_SIZE="${PATCH_SIZES[$TASK_ID]}"
NUM_LAYERS="${NUM_LAYERS_ARR[$TASK_ID]}"
HIDDEN_SIZE="${HIDDEN_SIZES[$TASK_ID]}"
NUM_HEAD="${NUM_HEADS[$TASK_ID]}"
MLP_DIM="${MLP_DIMS[$TASK_ID]}"

# Calculate expected patches
if [[ "${USE_CONV_STEM}" == "true" ]]; then
  GRID_SIZE=$((42 / PATCH_SIZE))
else
  GRID_SIZE=$((84 / PATCH_SIZE))
fi
NUM_PATCHES=$((GRID_SIZE * GRID_SIZE))

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_lit_v2_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_lit_v2,encoder_vit,method_ppo_muon,seed_${SEED},env_${ENV_NAME},patch_${PATCH_SIZE},layers_${NUM_LAYERS},patches_${NUM_PATCHES},stem_${USE_CONV_STEM}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: stem=${USE_CONV_STEM} patch=${PATCH_SIZE} layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} heads=${NUM_HEAD} -> ${NUM_PATCHES} patches"

EXP_NAME="ppo_vit_litv2_t${TASK_ID}_p${PATCH_SIZE}_l${NUM_LAYERS}_${NUM_PATCHES}pat"

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
  --vit-patch-size "${PATCH_SIZE}"
  --vit-hidden-size "${HIDDEN_SIZE}"
  --vit-mlp-dim "${MLP_DIM}"
  --vit-num-heads "${NUM_HEAD}"
  --vit-num-layers "${NUM_LAYERS}"
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --exp-name "${EXP_NAME}"
)

if [[ "${USE_CONV_STEM}" == "true" ]]; then
  COMMON_ARGS+=(--vit-use-conv-stem)
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
