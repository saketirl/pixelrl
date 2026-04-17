#!/bin/bash
#SBATCH --job-name=vit-depth-muon
#SBATCH --output=slurm_logs/vit_depth_muon_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Depth x Muon ablation sweep
#
# Fixed (best config from vit_lit_v2):
#   patch_size=21, no conv stem, hidden=128, heads=8, mlp=512
#   no cls token, no output tanh
#
# Swept:
#   depth  in {3, 4, 5, 6}
#   muon   in {none, qk_stiefel @ lr=1e-3}
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

# 2x4 grid: rows=muon, cols=depth
DEPTHS=(3 4 5 6)
USE_QK_STIEFEL=(false true)

NUM_DEPTHS=${#DEPTHS[@]}       # 4
NUM_MUON=${#USE_QK_STIEFEL[@]} # 2
NUM_CONFIGS=$((NUM_DEPTHS * NUM_MUON)) # 8

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

DEPTH_IDX=$((TASK_ID % NUM_DEPTHS))
MUON_IDX=$((TASK_ID / NUM_DEPTHS))

NUM_LAYERS="${DEPTHS[$DEPTH_IDX]}"
QK_STIEFEL="${USE_QK_STIEFEL[$MUON_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_depth_muon_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_depth_muon,encoder_vit,seed_${SEED},env_${ENV_NAME},layers_${NUM_LAYERS},qk_stiefel_${QK_STIEFEL}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: layers=${NUM_LAYERS} qk_stiefel=${QK_STIEFEL}"

EXP_NAME="ppo_vit_depth_t${TASK_ID}_l${NUM_LAYERS}_qk${QK_STIEFEL}"

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
  # Fixed best ViT config (no stem, 16 patches)
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
  --no-vit-use-conv-stem
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --exp-name "${EXP_NAME}"
)

if [[ "${QK_STIEFEL}" == "true" ]]; then
  COMMON_ARGS+=(
    --vit-qk-stiefel
    --vit-qk-stiefel-lr 1e-3
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
