#!/bin/bash

set -euo pipefail

# Env-only search for a Slippery Humanoid configuration that learns and can
# expose loss of plasticity as task count increases. The pilot/validate waves
# keep PPO/update settings fixed; the lr wave changes only the PPO learning
# rate while keeping the default-friction control schedule.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
WAVE="${WAVE:-pilot}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_brax.py" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"

if [[ -d "${REPO_ROOT}/pixelbrax/brax/brax" ]]; then
  BRAX_PYTHONPATH="${REPO_ROOT}/pixelbrax/brax"
else
  BRAX_PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"
fi
export PYTHONPATH="${BRAX_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

ENV_NAME="humanoid"
BACKEND="${BACKEND:-spring}"
ACTOR_CRITIC_ACTIVATION="${ACTOR_CRITIC_ACTIVATION:-relu}"
HEADS_OPTIMIZER="${HEADS_OPTIMIZER:-adam}"
N_ENVS="${N_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAM_EPS="${ADAM_EPS:-1e-5}"
ACTOR_MEAN_SCALE="${ACTOR_MEAN_SCALE:-1.0}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:--2.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:--1.4}"
CRATE_STEP_SIZE="${CRATE_STEP_SIZE:-0.1}"
NETWORK_CRATE_STEP_SIZE="${NETWORK_CRATE_STEP_SIZE:-0.1}"
OBS_NORM_CLIP="${OBS_NORM_CLIP:-10.0}"
SCHEDULE_TAG="default_then_csv_schedule"

select_pilot_config() {
  case "${TASK_ID}" in
    0) RUN_KIND="default_control"; SEED=0; SLIPPERY_SCHEDULE_SEED=14; PHASE_EVERY_ENV_STEPS=20000000; NUM_PHASES=1 ;;
    1) RUN_KIND="default_control"; SEED=1; SLIPPERY_SCHEDULE_SEED=14; PHASE_EVERY_ENV_STEPS=20000000; NUM_PHASES=1 ;;
    2) RUN_KIND="two_phase"; SEED=0; SLIPPERY_SCHEDULE_SEED=14; PHASE_EVERY_ENV_STEPS=20000000; NUM_PHASES=2 ;;
    3) RUN_KIND="two_phase"; SEED=0; SLIPPERY_SCHEDULE_SEED=58; PHASE_EVERY_ENV_STEPS=20000000; NUM_PHASES=2 ;;
    4) RUN_KIND="four_phase"; SEED=0; SLIPPERY_SCHEDULE_SEED=14; PHASE_EVERY_ENV_STEPS=10000000; NUM_PHASES=4 ;;
    5) RUN_KIND="four_phase"; SEED=0; SLIPPERY_SCHEDULE_SEED=58; PHASE_EVERY_ENV_STEPS=10000000; NUM_PHASES=4 ;;
    6) RUN_KIND="eight_phase"; SEED=0; SLIPPERY_SCHEDULE_SEED=14; PHASE_EVERY_ENV_STEPS=4999936; NUM_PHASES=8 ;;
    7) RUN_KIND="eight_phase"; SEED=0; SLIPPERY_SCHEDULE_SEED=58; PHASE_EVERY_ENV_STEPS=4999936; NUM_PHASES=8 ;;
    *) echo "Invalid pilot TASK_ID=${TASK_ID}. Expected 0..7." >&2; exit 1 ;;
  esac
}

select_lr_config() {
  RUN_KIND="lr_default_control"
  SEED="${LR_SWEEP_SEED:-0}"
  SLIPPERY_SCHEDULE_SEED="${LR_SWEEP_SCHEDULE_SEED:-14}"
  PHASE_EVERY_ENV_STEPS="${LR_SWEEP_PHASE_ENV_STEPS:-20000000}"
  NUM_PHASES=1

  case "${TASK_ID}" in
    0) LEARNING_RATE="3e-5" ;;
    1) LEARNING_RATE="1e-4" ;;
    2) LEARNING_RATE="2e-4" ;;
    3) LEARNING_RATE="3e-4" ;;
    4) LEARNING_RATE="5e-4" ;;
    5) LEARNING_RATE="1e-3" ;;
    6) LEARNING_RATE="2e-3" ;;
    7) LEARNING_RATE="3e-3" ;;
    *) echo "Invalid lr TASK_ID=${TASK_ID}. Expected 0..7." >&2; exit 1 ;;
  esac
}

select_validate_config() {
  local best_schedule_seed="${BEST_SCHEDULE_SEED:-14}"
  local phase_env_steps="${VALIDATION_PHASE_ENV_STEPS:-4999936}"
  local counts_csv="${VALIDATION_PHASE_COUNTS:-1,4,8,20}"
  IFS=',' read -r -a phase_counts <<< "${counts_csv}"
  if (( ${#phase_counts[@]} != 4 )); then
    echo "VALIDATION_PHASE_COUNTS must contain exactly four comma-separated counts; got ${counts_csv}." >&2
    exit 1
  fi
  if (( TASK_ID < 0 || TASK_ID >= 8 )); then
    echo "Invalid validate TASK_ID=${TASK_ID}. Expected 0..7." >&2
    exit 1
  fi

  local seed_idx=$((TASK_ID % 2))
  local count_idx=$((TASK_ID / 2))
  RUN_KIND="validate_${phase_counts[$count_idx]}phase"
  SEED="${seed_idx}"
  SLIPPERY_SCHEDULE_SEED="${best_schedule_seed}"
  PHASE_EVERY_ENV_STEPS="${phase_env_steps}"
  NUM_PHASES="${phase_counts[$count_idx]}"
}

case "${WAVE}" in
  pilot) select_pilot_config ;;
  lr) select_lr_config ;;
  validate) select_validate_config ;;
  *) echo "Unsupported WAVE=${WAVE}. Expected pilot, lr, or validate." >&2; exit 1 ;;
esac

if (( PHASE_EVERY_ENV_STEPS < 1 )); then
  echo "PHASE_EVERY_ENV_STEPS must be >= 1; got ${PHASE_EVERY_ENV_STEPS}." >&2
  exit 1
fi
if (( NUM_PHASES < 1 )); then
  echo "NUM_PHASES must be >= 1; got ${NUM_PHASES}." >&2
  exit 1
fi

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
if (( PHASE_EVERY_ENV_STEPS % N_ENVS != 0 )); then
  echo "PHASE_EVERY_ENV_STEPS=${PHASE_EVERY_ENV_STEPS} must be divisible by num envs ${N_ENVS}." >&2
  exit 1
fi
if (( ROLLOUT_ENV_STEPS % NUM_MINIBATCHES != 0 )); then
  echo "batch=${ROLLOUT_ENV_STEPS} must be divisible by num minibatches ${NUM_MINIBATCHES}." >&2
  exit 1
fi

MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))
CHANGE_EVERY_POLICY_STEPS=$((PHASE_EVERY_ENV_STEPS / N_ENVS))
CHANGE_EVERY=$((CHANGE_EVERY_POLICY_STEPS * ACTION_REPEAT))
TOTAL_TIMESTEPS=$((PHASE_EVERY_ENV_STEPS * NUM_PHASES))
if (( TOTAL_TIMESTEPS < ROLLOUT_ENV_STEPS )); then
  echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} must be at least one rollout (${ROLLOUT_ENV_STEPS})." >&2
  exit 1
fi
if (( TOTAL_TIMESTEPS % ROLLOUT_ENV_STEPS != 0 )); then
  ALIGNED_TOTAL_TIMESTEPS=$(((TOTAL_TIMESTEPS / ROLLOUT_ENV_STEPS) * ROLLOUT_ENV_STEPS))
  echo "Aligning TOTAL_TIMESTEPS from ${TOTAL_TIMESTEPS} down to ${ALIGNED_TOTAL_TIMESTEPS} to fit rollout transitions ${ROLLOUT_ENV_STEPS}."
  TOTAL_TIMESTEPS="${ALIGNED_TOTAL_TIMESTEPS}"
fi

EFFECTIVE_ENV_STEPS_PER_ENV=$((TOTAL_TIMESTEPS / N_ENVS))
EFFECTIVE_PHASES=$(((EFFECTIVE_ENV_STEPS_PER_ENV + CHANGE_EVERY_POLICY_STEPS - 1) / CHANGE_EVERY_POLICY_STEPS))
GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
PHASE_MILLIONS=$((PHASE_EVERY_ENV_STEPS / 1000000))
GROUP_NAME="slippery_humanoid_envonly_${WAVE}_${GROUP_ID}"
LR_TAG="${LEARNING_RATE//./p}"
LR_TAG="${LR_TAG//-/_}"
EXP_NAME="ppo_brax_sliphum_${WAVE}_${RUN_KIND}_ph${NUM_PHASES}_phase${PHASE_MILLIONS}m_sched${SLIPPERY_SCHEDULE_SEED}_s${SEED}_lr${LR_TAG}_t${TASK_ID}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_humanoid_sweep,env_only,${WAVE},${RUN_KIND},brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},schedule_seed_${SLIPPERY_SCHEDULE_SEED},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${CHANGE_EVERY},phases_${EFFECTIVE_PHASES},requested_phases_${NUM_PHASES},action_repeat_${ACTION_REPEAT},actorcritic_${ACTOR_CRITIC_ACTIVATION},headopt_${HEADS_OPTIMIZER},${SCHEDULE_TAG},batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},reward_norm,vclip,lr_${LEARNING_RATE},repo_ppo"

echo "Running Slippery Humanoid env-only sweep group=${GROUP_NAME}"
echo "Config: wave=${WAVE} task_id=${TASK_ID} kind=${RUN_KIND} env=${ENV_NAME} backend=${BACKEND} seed=${SEED} schedule_seed=${SLIPPERY_SCHEDULE_SEED}"
echo "Schedule: phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} requested_phases=${NUM_PHASES} effective_phases=${EFFECTIVE_PHASES} change_every_policy_steps=${CHANGE_EVERY_POLICY_STEPS} change_every_wrapper_steps=${CHANGE_EVERY} total_timesteps=${TOTAL_TIMESTEPS} per_env_steps=${EFFECTIVE_ENV_STEPS_PER_ENV}"
echo "PPO fixed: batch=${ROLLOUT_ENV_STEPS} minibatch_size=${MINIBATCH_SIZE} lr=${LEARNING_RATE} anneal_lr=true epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} clip_eps=0.1 reward_normalize=true clip_vloss=true max_grad_norm=0.05"
echo "Policy: actor_mean_tanh=true actor_mean_scale=${ACTOR_MEAN_SCALE} bounded_global_logstd=true actor_logstd_init=${ACTOR_LOGSTD_INIT} actor_logstd_min=${ACTOR_LOGSTD_MIN} actor_logstd_max=${ACTOR_LOGSTD_MAX}"
echo "Obs: obs_normalize=true obs_norm_clip=${OBS_NORM_CLIP}"
echo "Network: use_crate_network=true network_crate_step_size=${NETWORK_CRATE_STEP_SIZE}"
echo "Heads: use_crate_head=true crate_step_size=${CRATE_STEP_SIZE} heads_optimizer=${HEADS_OPTIMIZER}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate "${LEARNING_RATE}"
  --adam-eps "${ADAM_EPS}"
  --anneal-lr
  --num-steps "${NUM_STEPS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --update-epochs "${UPDATE_EPOCHS}"
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --max-grad-norm 0.05
  --actor-critic-activation "${ACTOR_CRITIC_ACTIVATION}"
  --heads-optimizer "${HEADS_OPTIMIZER}"
  --actor-mean-tanh
  --actor-mean-scale "${ACTOR_MEAN_SCALE}"
  --bounded-global-logstd
  --actor-logstd-init "${ACTOR_LOGSTD_INIT}"
  --actor-logstd-min "${ACTOR_LOGSTD_MIN}"
  --actor-logstd-max "${ACTOR_LOGSTD_MAX}"
  --obs-normalize
  --obs-norm-clip "${OBS_NORM_CLIP}"
  --use-crate-network
  --network-crate-step-size "${NETWORK_CRATE_STEP_SIZE}"
  --use-crate-head
  --crate-step-size "${CRATE_STEP_SIZE}"
  --heads-stiefel-lr 0.001
  --stiefel-dual-lr 0.01
  --stiefel-dual-steps 5
  --stiefel-msign-steps 5
  --actor-stiefel-max-grad-norm 100
  --critic-stiefel-max-grad-norm 1
  --reward-normalize
  --seed "${SEED}"
  --log-interval 1
  --action-repeat "${ACTION_REPEAT}"
  --slippery
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry run command:'
  printf ' %q' uv run python "${REPO_ROOT}/ppo_brax.py" "${COMMON_ARGS[@]}"
  printf '\n'
  exit 0
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" "${COMMON_ARGS[@]}"
