#!/bin/bash
#SBATCH --job-name=goal-cratecnn-crateheads-s0
#SBATCH --output=slurm_logs/goal_crate_cnn_meanbound_boundedstd_crate_heads_seed0_stiefel_adam_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# One-GPU, one-seed goal-reaching launcher adapted from the maze CRATE-CNN
# launcher.  Runs the same CRATE-CNN + CRATE-head config twice concurrently:
#   1. actor/critic heads on manifold Stiefel
#   2. actor/critic heads on Adam

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"
ENV_NAME="${5:-ant_goal}"
BACKEND="${6:-spring}"
ACTION_REPEAT="${7:-4}"
JAX_MEM_FRACTION="${8:-0.45}"

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
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="goal_crate_cnn_meanbound_boundedstd_crate_heads_ar${ACTION_REPEAT}_${ENV_NAME}_seed${SEED}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

COMMON_TAGS="goal,pixel_goal,goal_aware_tracking_camera,muted_textured_floor,dense_distance_reward,action_repeat_${ACTION_REPEAT},crate_cnn_meanbound_boundedstd,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},encoder_crate_cnn,actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},bounded_global_logstd,actor_logstd_init_${ACTOR_LOGSTD_INIT},actor_logstd_min_${ACTOR_LOGSTD_MIN},actor_logstd_max_${ACTOR_LOGSTD_MAX},headarch_crate,sigreg_off,encoder_crate_step_${ENCODER_CRATE_STEP_SIZE},head_crate_step_${CRATE_STEP_SIZE},single_seed,concurrent_gpu,mem_fraction_${JAX_MEM_FRACTION}"

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
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat "${ACTION_REPEAT}"
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
  --use-crate-head
  --crate-step-size "${CRATE_STEP_SIZE}"
  --sigreg-mode off
)

run_variant() {
  local heads_optimizer="$1"
  local exp_name="ppo_goal_cratecnn_meanbound_boundedstd_crate_${heads_optimizer}_ar${ACTION_REPEAT}_${ENV_NAME}_b${BACKEND}_s${SEED}"

  export WANDB_TAGS="${COMMON_TAGS},opt_${heads_optimizer}"

  echo "Running ${heads_optimizer} variant in group=${GROUP_NAME}"
  echo "Config: env=${ENV_NAME} backend=${BACKEND} action_repeat=${ACTION_REPEAT} encoder=crate_cnn crate_head=true heads_optimizer=${heads_optimizer} seed=${SEED} wandb_project=${WANDB_PROJECT} jax_mem_fraction=${JAX_MEM_FRACTION}"
  echo "Exp: ${exp_name}"

  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --heads-optimizer "${heads_optimizer}" \
    --exp-name "${exp_name}"
}

cd "${REPO_ROOT}"
run_variant stiefel &
STIEFEL_PID=$!

run_variant adam &
ADAM_PID=$!

STIEFEL_STATUS=0
ADAM_STATUS=0
wait "${STIEFEL_PID}" || STIEFEL_STATUS=$?
wait "${ADAM_PID}" || ADAM_STATUS=$?

if (( STIEFEL_STATUS != 0 || ADAM_STATUS != 0 )); then
  echo "One or more variants failed: stiefel=${STIEFEL_STATUS} adam=${ADAM_STATUS}" >&2
  exit 1
fi
