#!/bin/bash
#SBATCH --job-name=vit-heads-muon
#SBATCH --output=slurm_logs/vit_heads_muon_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Actor/critic Muon ablation: 2 conditions x 4 seeds = 8 runs
#
# Fixed (best config from vit_stem_depth sweep):
#   patch_size=21, conv_stem=True -> 4 patches, hidden=128, heads=8,
#   mlp=512, layers=4, no cls token, no output tanh, no qk_stiefel
#
# Swept:
#   use_heads_muon in {true, false}  (Muon vs Adam for actor/critic matrices)
#   seed           in {0, 1, 2, 3}
#
# 2x4 = 8 configs

ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
TOTAL_TIMESTEPS="${4:-10000000}"
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

SEEDS=(0 1 2 3)
USE_HEADS_MUONS=(true false)

NUM_SEEDS=${#SEEDS[@]}       # 4
NUM_MUON=${#USE_HEADS_MUONS[@]} # 2
NUM_CONFIGS=$((NUM_SEEDS * NUM_MUON)) # 8

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
MUON_IDX=$((TASK_ID / NUM_SEEDS))

SEED="${SEEDS[$SEED_IDX]}"
USE_HEADS_MUON="${USE_HEADS_MUONS[$MUON_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="${ENV_NAME}_vit_heads_muon_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_heads_muon_abl,encoder_vit,seed_${SEED},env_${ENV_NAME},heads_muon_${USE_HEADS_MUON}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Config: use_heads_muon=${USE_HEADS_MUON} seed=${SEED}"

EXP_NAME="ppo_vit_heads_muon_t${TASK_ID}_muon${USE_HEADS_MUON}_s${SEED}"

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
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --vit-use-conv-stem
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --no-vit-qk-stiefel
  --exp-name "${EXP_NAME}"
)

if [[ "${USE_HEADS_MUON}" == "true" ]]; then
  COMMON_ARGS+=(--use-heads-muon)
else
  COMMON_ARGS+=(--no-use-heads-muon)
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
