#!/bin/bash
#SBATCH --job-name=vit-debug-sweep
#SBATCH --output=slurm_logs/vit_debug_sweep_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# ViT control/exploration sweep on hard environments.
#
# Fixed:
#   env            in {ant, humanoid}
#   use_heads_muon = true
#   use_conv_stem  = true
#   patch_size     = 14
#   vit_layers     = 4
#   seed           = 0
#
# Swept:
#   action_repeat  in {2, 4}
#   ent_coef       in {0.01, 0.02}
#
# 2 x 2 x 2 x 1 = 8 configs

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

REPO_ROOT=""
ROOT_CANDIDATES=(
  "${SLURM_SUBMIT_DIR:-}"
  "${PWD}"
  "${PWD}/.worktrees/vit-muon"
  "/home/guests/arjun/pixelrl/.worktrees/vit-muon"
)

for CANDIDATE in "${ROOT_CANDIDATES[@]}"; do
  if [[ -n "${CANDIDATE}" && -f "${CANDIDATE}/ppo_pixelbrax_jax2_muon.py" && -f "${CANDIDATE}/encoders.py" ]]; then
    REPO_ROOT="${CANDIDATE}"
    break
  fi
done

if [[ -z "${REPO_ROOT}" ]]; then
  echo "Failed to resolve REPO_ROOT with ppo_pixelbrax_jax2_muon.py + encoders.py" >&2
  echo "Tried: ${ROOT_CANDIDATES[*]}" >&2
  exit 1
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

ENVS=(ant humanoid)
BACKENDS=(spring spring)
ACTION_REPEATS=(2 4)
ENT_COEFS=(0.01 0.02)
SEEDS=(0)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_ACTION_REPEATS=${#ACTION_REPEATS[@]}
NUM_ENT_COEFS=${#ENT_COEFS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_ACTION_REPEATS * NUM_ENT_COEFS * NUM_SEEDS))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
ENT_IDX=$(((TASK_ID / NUM_SEEDS) % NUM_ENT_COEFS))
ACTION_REPEAT_IDX=$(((TASK_ID / (NUM_SEEDS * NUM_ENT_COEFS)) % NUM_ACTION_REPEATS))
ENV_IDX=$((TASK_ID / (NUM_SEEDS * NUM_ENT_COEFS * NUM_ACTION_REPEATS)))

SEED="${SEEDS[$SEED_IDX]}"
ENT_COEF="${ENT_COEFS[$ENT_IDX]}"
ACTION_REPEAT="${ACTION_REPEATS[$ACTION_REPEAT_IDX]}"
ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"
ENT_TOKEN="${ENT_COEF//./p}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="vit_debug_sweep_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_debug_sweep,encoder_vit,opt_muon,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},patch_14,layers_4,convstem_true,action_repeat_${ACTION_REPEAT},ent_${ENT_TOKEN}"

EXP_NAME="ppo_vit_debug_${ENV_NAME}_ar${ACTION_REPEAT}_ent${ENT_TOKEN}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} action_repeat=${ACTION_REPEAT} ent_coef=${ENT_COEF} patch=14 layers=4 conv_stem=true heads_muon=true seed=${SEED}"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
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
  --weight-decay 1e-4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef "${ENT_COEF}"
  --vf-coef 0.5
  --max-grad-norm 0.5
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat "${ACTION_REPEAT}"
  --anneal-lr
  --encoder-warmup-updates 500
  --encoder-type vit
  --encoder-tanh-scale 0.25
  --vit-patch-size 14
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
  --use-heads-muon
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
