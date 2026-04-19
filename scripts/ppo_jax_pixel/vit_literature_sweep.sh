#!/bin/bash
#SBATCH --job-name=vit-literature
#SBATCH --output=slurm_logs/vit_literature_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Literature-based ViT sweep for RL from pixels
# Based on: "Evaluating Vision Transformer Methods for Deep RL from Pixels" (arxiv 2204.04905)
#
# Key changes from previous sweeps:
#   1. Smaller patch size (7 instead of 14) -> more patches (36 vs 9)
#   2. More attention heads (6 or 8 instead of 3)
#   3. Smaller hidden dim (128 instead of 192) to match literature
#
# Usage:
#   sbatch scripts/ppo_jax_pixel/vit_literature_sweep.sh [env_name] [wandb_project] [wandb_entity] [seed] [total_timesteps]
# Example:
#   sbatch scripts/ppo_jax_pixel/vit_literature_sweep.sh halfcheetah benchmark my_entity 0 10000000

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

# Literature-based configurations:
# All use: patch_size=7, conv_stem=true, cls_token=false, output_tanh=false
#
# Config 0: hidden=128, heads=8, mlp=512  (closest to paper: 128 dim, 8 heads)
# Config 1: hidden=128, heads=4, mlp=512  (smaller heads variant)
# Config 2: hidden=192, heads=6, mlp=768  (your hidden size, more heads)
# Config 3: hidden=192, heads=8, mlp=768  (your hidden size, paper heads)

HIDDEN_SIZES=(128 128 192 192)
NUM_HEADS=(8 4 6 8)
MLP_DIMS=(512 512 768 768)

NUM_CONFIGS=${#HIDDEN_SIZES[@]}

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

HIDDEN_SIZE="${HIDDEN_SIZES[$TASK_ID]}"
NUM_HEAD="${NUM_HEADS[$TASK_ID]}"
MLP_DIM="${MLP_DIMS[$TASK_ID]}"

# Fixed from literature: smaller patches = more tokens
PATCH_SIZE=7

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_literature_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_literature,encoder_vit,method_ppo_muon,jepa_none,seed_${SEED},env_${ENV_NAME},patch_${PATCH_SIZE},hidden_${HIDDEN_SIZE},heads_${NUM_HEAD}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: patch_size=${PATCH_SIZE} hidden_size=${HIDDEN_SIZE} num_heads=${NUM_HEAD} mlp_dim=${MLP_DIM}"

EXP_NAME="ppo_vit_lit_t${TASK_ID}_p${PATCH_SIZE}_h${HIDDEN_SIZE}_heads${NUM_HEAD}"

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
  # Literature-based ViT config
  --vit-patch-size "${PATCH_SIZE}"
  --vit-hidden-size "${HIDDEN_SIZE}"
  --vit-mlp-dim "${MLP_DIM}"
  --vit-num-heads "${NUM_HEAD}"
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  # Best settings from previous sweep
  --vit-use-conv-stem
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
