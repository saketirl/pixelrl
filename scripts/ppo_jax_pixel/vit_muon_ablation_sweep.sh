#!/bin/bash
#SBATCH --job-name=vit-muon-abl
#SBATCH --output=slurm_logs/vit_muon_abl_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Q/K Stiefel Muon ablation sweep
#
# Fixed (best config from vit_stem_depth sweep):
#   patch_size=21, conv_stem=True, hidden=128, heads=8, mlp=512
#   no cls token, no output tanh
#
# With stem: 84->42, 42/21 = 2x2 = 4 patches
#
# Swept:
#   stiefel_lr in {none, 1e-4, 1e-3, 1e-2}  (none = no Q/K Stiefel)
#   depth      in {4, 6}                     (top two from stem sweep)
#
# 4x2 = 8 configs

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

# 4x2 grid: rows=stiefel_lr (none,1e-4,1e-3,1e-2), cols=depth (4,6)
STIEFEL_LRS=(none 1e-4 1e-3 1e-2)
DEPTHS=(4 6)

NUM_LRS=${#STIEFEL_LRS[@]}    # 4
NUM_DEPTHS=${#DEPTHS[@]}      # 2
NUM_CONFIGS=$((NUM_LRS * NUM_DEPTHS)) # 8

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

LR_IDX=$((TASK_ID % NUM_LRS))
DEPTH_IDX=$((TASK_ID / NUM_LRS))

STIEFEL_LR="${STIEFEL_LRS[$LR_IDX]}"
NUM_LAYERS="${DEPTHS[$DEPTH_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_muon_abl_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_muon_abl,encoder_vit,seed_${SEED},env_${ENV_NAME},layers_${NUM_LAYERS},stiefel_lr_${STIEFEL_LR}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: layers=${NUM_LAYERS} stiefel_lr=${STIEFEL_LR}"

LR_TAG="${STIEFEL_LR//./p}"
EXP_NAME="ppo_vit_muon_t${TASK_ID}_l${NUM_LAYERS}_slr${LR_TAG}"

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
  --vit-use-conv-stem
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --exp-name "${EXP_NAME}"
)

if [[ "${STIEFEL_LR}" == "none" ]]; then
  COMMON_ARGS+=(--no-vit-qk-stiefel)
else
  COMMON_ARGS+=(
    --vit-qk-stiefel
    --vit-qk-stiefel-lr "${STIEFEL_LR}"
    --vit-qk-stiefel-dual-lr 0.01
    --vit-qk-stiefel-dual-steps 5
    --vit-qk-stiefel-msign-steps 5
    --vit-qk-stiefel-max-grad-norm 0.5
  )
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
