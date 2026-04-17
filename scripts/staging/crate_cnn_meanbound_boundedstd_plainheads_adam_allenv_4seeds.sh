#!/bin/bash
#SBATCH --job-name=stg-cratecnn-boundstd-plain-adam-allenv4
#SBATCH --output=slurm_logs/staging_crate_cnn_meanbound_boundedstd_plainheads_adam_allenv_4seeds_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%8

set -euo pipefail

# Plain-head Adam ablation of the successful staging run.
# Keeps the CRATECNN encoder and policy-bound setup, but does not pass
# --use-crate-head, so actor/critic use the standard MLP heads.
#
# Swept:
#   env  in {halfcheetah, walker2d, ant, humanoid, reacher, swimmer, pusher, hopper, inverted_pendulum}
#   seed in {0, 1, 2, 3}
#
# 9 x 4 = 36 configs

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

ENVS=(halfcheetah walker2d ant humanoid reacher swimmer pusher hopper inverted_pendulum)
BACKENDS=(spring spring spring spring generalized generalized generalized positional generalized)
SEEDS=(0 1 2 3)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_SEEDS))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
ENV_IDX=$((TASK_ID / NUM_SEEDS))

SEED="${SEEDS[$SEED_IDX]}"
ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"

ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="staging_crate_cnn_meanbound_boundedstd_plainheads_adam_allenv4_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="staging,crate_cnn_meanbound_boundedstd,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},encoder_crate_cnn,actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},bounded_global_logstd,actor_logstd_init_${ACTOR_LOGSTD_INIT},actor_logstd_min_${ACTOR_LOGSTD_MIN},actor_logstd_max_${ACTOR_LOGSTD_MAX},opt_adam,headarch_plain,sigreg_off,encoder_crate_step_${ENCODER_CRATE_STEP_SIZE},allenv9,seeds0123,ablate_crate_head"

EXP_NAME="ppo_staging_cratecnn_meanbound_boundedstd_plainheads_adam_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} encoder=crate_cnn encoder_crate_step_size=${ENCODER_CRATE_STEP_SIZE} actor_mean_tanh=true actor_mean_scale=${ACTOR_MEAN_SCALE} bounded_global_logstd=true actor_logstd_init=${ACTOR_LOGSTD_INIT} actor_logstd_min=${ACTOR_LOGSTD_MIN} actor_logstd_max=${ACTOR_LOGSTD_MAX} crate_head=false heads_stiefel=false sigreg=off seed=${SEED}"
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
  --heads-adam-lr 3e-4
  --heads-stiefel-lr 0.001
  --max-grad-norm 0.05
  --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
  --sigreg-mode off
)

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  --no-use-heads-stiefel
