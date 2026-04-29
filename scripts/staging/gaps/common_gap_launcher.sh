#!/bin/bash

run_staging_gap_task() {
  if [[ -z "${CONFIG_NAME:-}" || -z "${HEADARCH:-}" || -z "${HEADS_OPTIMIZER:-}" ]]; then
    echo "CONFIG_NAME, HEADARCH, and HEADS_OPTIMIZER must be set before sourcing common_gap_launcher.sh" >&2
    exit 1
  fi

  WANDB_PROJECT="${1:-encoder}"
  WANDB_ENTITY="${2:-}"
  TOTAL_TIMESTEPS="${3:-10000000}"
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

  if (( ${#ENVS[@]} == 0 )); then
    echo "No staging3 gaps to run for ${CONFIG_NAME}."
    exit 0
  fi

  NUM_CONFIGS=${#ENVS[@]}
  if (( ${#BACKENDS[@]} != NUM_CONFIGS || ${#SEEDS[@]} != NUM_CONFIGS )); then
    echo "ENVS/BACKENDS/SEEDS length mismatch: ${#ENVS[@]} ${#BACKENDS[@]} ${#SEEDS[@]}" >&2
    exit 1
  fi

  if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
    echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
    exit 1
  fi

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

  ENV_NAME="${ENVS[$TASK_ID]}"
  BACKEND="${BACKENDS[$TASK_ID]}"
  SEED="${SEEDS[$TASK_ID]}"

  CRATE_STEP_SIZE="0.1"
  ENCODER_CRATE_STEP_SIZE="0.1"
  ACTOR_MEAN_SCALE="1.0"
  ACTOR_LOGSTD_INIT="-2.0"
  ACTOR_LOGSTD_MIN="-5.0"
  ACTOR_LOGSTD_MAX="-1.4"

  GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  GROUP_NAME="staging3_crate_cnn_meanbound_boundedstd_${CONFIG_NAME}_gaps_${GROUP_ID}"
  export WANDB_RUN_GROUP="${GROUP_NAME}"

  HEAD_TAG="headarch_${HEADARCH}"
  OPT_TAG="opt_${HEADS_OPTIMIZER}"
  EXTRA_TAGS="staging,staging3,crate_cnn_meanbound_boundedstd,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},encoder_crate_cnn,actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},bounded_global_logstd,actor_logstd_init_${ACTOR_LOGSTD_INIT},actor_logstd_min_${ACTOR_LOGSTD_MIN},actor_logstd_max_${ACTOR_LOGSTD_MAX},${OPT_TAG},${HEAD_TAG},sigreg_off,encoder_crate_step_${ENCODER_CRATE_STEP_SIZE},allenv9,gap_fill"
  if [[ "${HEADARCH}" == "crate" ]]; then
    EXTRA_TAGS+=",head_crate_step_${CRATE_STEP_SIZE}"
  else
    EXTRA_TAGS+=",ablate_crate_head"
  fi
  export WANDB_TAGS="${EXTRA_TAGS}"

  EXP_NAME="ppo_staging3_cratecnn_meanbound_boundedstd_${CONFIG_NAME}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

  echo "Running staging3 gap TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
  echo "Config: env=${ENV_NAME} backend=${BACKEND} headarch=${HEADARCH} heads_optimizer=${HEADS_OPTIMIZER} seed=${SEED}"
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

  if [[ "${HEADARCH}" == "crate" ]]; then
    ARCH_ARGS+=(--use-crate-head --crate-step-size "${CRATE_STEP_SIZE}")
  fi

  cd "${REPO_ROOT}"
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --heads-optimizer "${HEADS_OPTIMIZER}"
}
