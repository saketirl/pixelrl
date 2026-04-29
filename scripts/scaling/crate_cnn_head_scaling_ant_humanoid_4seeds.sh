#!/bin/bash
#SBATCH --job-name=scale-crate-heads-ant-hum
#SBATCH --output=slurm_logs/scaling_crate_cnn_heads_ant_humanoid_4seeds_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-55%4

set -euo pipefail

# Scaling experiment based on:
#   scripts/staging/crate_cnn_meanbound_boundedstd_crate_stiefel_allenv_4seeds.sh
#
# Paper-style width-vs-depth comparison for CRATE actor/critic heads:
#   env    in {ant, humanoid}
#   seed   in {0, 1, 2, 3}
#   config in {
#     baseline/depth: width=256,  layers=1,
#     width:    width=512,  layers=1,
#     width:    width=1024, layers=1,
#     width:    width=2048, layers=1,
#     depth:    width=256,  layers=8,
#     depth:    width=256,  layers=16,
#     depth:    width=256,  layers=32
#   }
#
# 2 x 4 x 7 = 56 configs

WANDB_PROJECT="${1:-encoder}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"

if [[ -d "${REPO_ROOT}/pixelbrax/brax/brax" ]]; then
  BRAX_PYTHONPATH="${REPO_ROOT}/pixelbrax/brax"
else
  BRAX_PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"
fi
export PYTHONPATH="${BRAX_PYTHONPATH}:${PYTHONPATH:-}"

ALT_REPO_ROOT="${REPO_ROOT}"
if [[ "${REPO_ROOT}" == *"/.worktrees/"* ]]; then
  ALT_REPO_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
fi

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ ! -f "${WANDB_KEY_FILE}" && -f "${ALT_REPO_ROOT}/secrets/wandb_api_key.txt" ]]; then
  WANDB_KEY_FILE="${ALT_REPO_ROOT}/secrets/wandb_api_key.txt"
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

# Respect caller overrides, but use a smaller default for packed/concurrent runs.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.22}"

ENVS=(ant humanoid)
BACKENDS=(spring spring)
SEEDS=(0 1 2 3)
SCALE_AXES=(baseline width width width depth depth depth)
HEAD_WIDTHS=(256 512 1024 2048 256 256 256)
HEAD_CRATE_LAYERS=(1 1 1 1 8 16 32)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_SCALES=${#HEAD_WIDTHS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_SEEDS * NUM_SCALES))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( NUM_SCALES != ${#HEAD_CRATE_LAYERS[@]} || NUM_SCALES != ${#SCALE_AXES[@]} )); then
  echo "Scaling array length mismatch." >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
SCALE_IDX=$(((TASK_ID / NUM_SEEDS) % NUM_SCALES))
ENV_IDX=$((TASK_ID / (NUM_SEEDS * NUM_SCALES)))

SEED="${SEEDS[$SEED_IDX]}"
ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"
SCALE_AXIS="${SCALE_AXES[$SCALE_IDX]}"
HEAD_WIDTH="${HEAD_WIDTHS[$SCALE_IDX]}"
HEAD_CRATE_LAYER_COUNT="${HEAD_CRATE_LAYERS[$SCALE_IDX]}"

BASE_HEADS_STIEFEL_LR="0.001"
BASE_HEADS_ADAM_LR="0.0003"
LR_SCALE="$(awk -v width="${HEAD_WIDTH}" -v layers="${HEAD_CRATE_LAYER_COUNT}" 'BEGIN { printf "%.12g", (256.0 / width) * sqrt(layers) }')"
HEADS_STIEFEL_LR="$(awk -v base="${BASE_HEADS_STIEFEL_LR}" -v scale="${LR_SCALE}" 'BEGIN { printf "%.12g", base * scale }')"
HEADS_ADAM_LR="$(awk -v base="${BASE_HEADS_ADAM_LR}" -v scale="${LR_SCALE}" 'BEGIN { printf "%.12g", base * scale }')"

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="scaling_crate_cnn_heads_ant_humanoid_4seeds_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="scaling,crate_cnn_head_scaling,scale_axis_${SCALE_AXIS},head_width_${HEAD_WIDTH},head_crate_layers_${HEAD_CRATE_LAYER_COUNT},lr_scale_${LR_SCALE},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},encoder_crate_cnn,actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},bounded_global_logstd,actor_logstd_init_${ACTOR_LOGSTD_INIT},actor_logstd_min_${ACTOR_LOGSTD_MIN},actor_logstd_max_${ACTOR_LOGSTD_MAX},opt_stiefel,headarch_crate,sigreg_off,encoder_crate_step_${ENCODER_CRATE_STEP_SIZE},head_crate_step_${CRATE_STEP_SIZE},ant_humanoid,seeds0123"

EXP_NAME="ppo_scaling_cratecnn_heads_${SCALE_AXIS}_w${HEAD_WIDTH}_d${HEAD_CRATE_LAYER_COUNT}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} scale_axis=${SCALE_AXIS} head_width=${HEAD_WIDTH} head_crate_layers=${HEAD_CRATE_LAYER_COUNT} lr_scale=${LR_SCALE}"
echo "Head LRs: heads_stiefel_lr=${HEADS_STIEFEL_LR} heads_adam_lr=${HEADS_ADAM_LR} encoder_lr=3e-4"
echo "Runtime: XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
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
  --stiefel-dual-lr 0.01
  --stiefel-dual-steps 5
  --actor-stiefel-max-grad-norm 100
  --critic-stiefel-max-grad-norm 1
  --actor-mean-tanh
  --actor-mean-scale "${ACTOR_MEAN_SCALE}"
  --bounded-global-logstd
  --actor-logstd-init="${ACTOR_LOGSTD_INIT}"
  --actor-logstd-min="${ACTOR_LOGSTD_MIN}"
  --actor-logstd-max="${ACTOR_LOGSTD_MAX}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

ARCH_ARGS=(
  --encoder-type crate_cnn
  --encoder-lr 3e-4
  --heads-adam-lr "${HEADS_ADAM_LR}"
  --heads-stiefel-lr "${HEADS_STIEFEL_LR}"
  --max-grad-norm 0.05
  --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
  --use-crate-head
  --crate-step-size "${CRATE_STEP_SIZE}"
  --head-hidden-dim "${HEAD_WIDTH}"
  --head-crate-layers "${HEAD_CRATE_LAYER_COUNT}"
  --sigreg-mode off
)

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  --heads-optimizer stiefel
