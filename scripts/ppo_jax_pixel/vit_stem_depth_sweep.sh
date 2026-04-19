#!/bin/bash
#SBATCH --job-name=vit-stem-depth
#SBATCH --output=slurm_logs/vit_stem_depth_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Conv stem x depth ablation
#
# Motivation: previous v2 sweep did not explicitly pass --no-vit-use-conv-stem,
# so stem may have been on (default=True) for all configs. This sweep cleanly
# ablates stem on/off.
#
# Fixed (best config from vit_lit_v2):
#   patch_size=21, hidden=128, heads=8, mlp=512, no qk_stiefel
#   no cls token, no output tanh
#
# With stem (84->42): 42/21 = 2x2 =  4 patches
# Without stem (84):  84/21 = 4x4 = 16 patches
#
# Swept:
#   conv_stem in {true, false}
#   depth     in {3, 4, 5, 6}
#
# 2x4 = 8 configs

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

DEPTHS=(3 4 5 6)
USE_CONV_STEMS=(false true)

NUM_DEPTHS=${#DEPTHS[@]}        # 4
NUM_STEMS=${#USE_CONV_STEMS[@]} # 2
NUM_CONFIGS=$((NUM_DEPTHS * NUM_STEMS)) # 8

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

DEPTH_IDX=$((TASK_ID % NUM_DEPTHS))
STEM_IDX=$((TASK_ID / NUM_DEPTHS))

NUM_LAYERS="${DEPTHS[$DEPTH_IDX]}"
USE_CONV_STEM="${USE_CONV_STEMS[$STEM_IDX]}"

if [[ "${USE_CONV_STEM}" == "true" ]]; then
  NUM_PATCHES=4   # 42/21 = 2x2
else
  NUM_PATCHES=16  # 84/21 = 4x4
fi

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_stem_depth_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_stem_depth,encoder_vit,seed_${SEED},env_${ENV_NAME},stem_${USE_CONV_STEM},layers_${NUM_LAYERS},patches_${NUM_PATCHES}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: conv_stem=${USE_CONV_STEM} layers=${NUM_LAYERS} -> ${NUM_PATCHES} patches"

EXP_NAME="ppo_vit_stem_t${TASK_ID}_stem${USE_CONV_STEM}_l${NUM_LAYERS}_${NUM_PATCHES}pat"

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
  --vit-patch-size 21
  --vit-hidden-size 128
  --vit-mlp-dim 512
  --vit-num-heads 8
  --vit-num-layers "${NUM_LAYERS}"
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --no-vit-qk-stiefel
  --exp-name "${EXP_NAME}"
)

if [[ "${USE_CONV_STEM}" == "true" ]]; then
  COMMON_ARGS+=(--vit-use-conv-stem)
else
  COMMON_ARGS+=(--no-vit-use-conv-stem)
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
