#!/bin/bash
#SBATCH --job-name=vit-crate-4env
#SBATCH --output=slurm_logs/vit_create_4env_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# ViT encoder with CRATE heads; ablate head optimizer (MUON vs Adam)
# across 4 PixelBrax environments with 1 seed.
#
# Swept:
#   env       in {halfcheetah, walker2d, ant, humanoid}
#   head_opt  in {muon, adam}  (use_heads_muon true/false)
#   seed      in {0}
#
# Fixed:
#   encoder_type = vit
#   encoder_tanh_scale = 0.5
#   crate_step_size = 0.1
#
# 4 x 2 x 1 = 8 configs

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

ENVS=(halfcheetah walker2d ant humanoid)
BACKENDS=(spring spring spring spring)
OPT_CONDITIONS=(muon adam)
SEEDS=(0)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_OPTS=${#OPT_CONDITIONS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_OPTS * NUM_SEEDS))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
OPT_IDX=$(((TASK_ID / NUM_SEEDS) % NUM_OPTS))
ENV_IDX=$((TASK_ID / (NUM_SEEDS * NUM_OPTS)))

SEED="${SEEDS[$SEED_IDX]}"
OPT_CONDITION="${OPT_CONDITIONS[$OPT_IDX]}"
ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="vit_create_4env_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="vit_crate,encoder_vit,head_crate,opt_${OPT_CONDITION},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},sweep1seed_4env"

EXP_NAME="ppo_vit_create_${OPT_CONDITION}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} encoder=vit head=crate opt=${OPT_CONDITION} seed=${SEED}"
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
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

ARCH_ARGS=(
  --encoder-type vit
  --encoder-lr 5e-5
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --weight-decay 1e-4
  --max-grad-norm 0.5
  --encoder-warmup-updates 500
  --encoder-tanh-scale 0.5
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
  --use-crate-head
  --crate-step-size 0.1
)

OPT_ARGS=()
if [[ "${OPT_CONDITION}" == "muon" ]]; then
  OPT_ARGS+=(--use-heads-muon)
else
  OPT_ARGS+=(--no-use-heads-muon)
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  "${OPT_ARGS[@]}"
