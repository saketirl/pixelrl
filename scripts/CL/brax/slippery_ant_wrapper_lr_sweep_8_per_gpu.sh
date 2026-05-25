#!/bin/bash
#SBATCH --job-name=slippery-ant-lr8
#SBATCH --output=slurm_logs/slippery_ant_wrapper_lr8_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# State-observation Brax Ant launcher for an 8-way learning-rate sweep on one GPU.
# Baseline PPO/env defaults match slippery_ant_wrapper_lr1e4.sh; each child process
# varies only --learning-rate while sharing the same seed unless SEED is overridden.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-4999936}"
NUM_PHASES="${4:-20}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
SEED="${6:-${SEED:-0}}"
ACTOR_CRITIC_ACTIVATION="${7:-${ACTOR_CRITIC_ACTIVATION:-relu}}"
BACKEND="${8:-${BACKEND:-spring}}"
HEADS_OPTIMIZER="${9:-${HEADS_OPTIMIZER:-adam}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

ENV_NAME="ant"
N_ENVS="${N_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.05}"
LRS="${LRS:-1e-5 3e-5 5e-5 1e-4 2e-4 3e-4 5e-4 1e-3}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-${SEED}}"
DEFAULT_NUM_PHASES=20
SCHEDULE_DESC="default_then_csv"
SCHEDULE_TAG="default_then_csv_schedule"
if [[ -z "${NUM_PHASES}" ]]; then
  NUM_PHASES="${DEFAULT_NUM_PHASES}"
fi

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
if [[ -n "${TOTAL_TIMESTEPS_OVERRIDE}" ]]; then
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE}"
else
  TOTAL_TIMESTEPS=$((PHASE_EVERY_ENV_STEPS * NUM_PHASES))
fi
if (( TOTAL_TIMESTEPS < ROLLOUT_ENV_STEPS )); then
  echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} must be at least one rollout (${ROLLOUT_ENV_STEPS})." >&2
  exit 1
fi
if (( TOTAL_TIMESTEPS % ROLLOUT_ENV_STEPS != 0 )); then
  ALIGNED_TOTAL_TIMESTEPS=$(((TOTAL_TIMESTEPS / ROLLOUT_ENV_STEPS) * ROLLOUT_ENV_STEPS))
  echo "Aligning TOTAL_TIMESTEPS from ${TOTAL_TIMESTEPS} down to ${ALIGNED_TOTAL_TIMESTEPS} to fit rollout transitions ${ROLLOUT_ENV_STEPS}."
  TOTAL_TIMESTEPS="${ALIGNED_TOTAL_TIMESTEPS}"
fi

read -r -a LR_LIST <<< "${LRS}"
if (( ${#LR_LIST[@]} != 8 )); then
  echo "LRS must contain exactly 8 learning rates; got ${#LR_LIST[@]}: ${LRS}" >&2
  exit 1
fi

EFFECTIVE_ENV_STEPS_PER_ENV=$((TOTAL_TIMESTEPS / N_ENVS))
EFFECTIVE_PHASES=$(((EFFECTIVE_ENV_STEPS_PER_ENV + CHANGE_EVERY_POLICY_STEPS - 1) / CHANGE_EVERY_POLICY_STEPS))
DISTINCT_PHASES="${EFFECTIVE_PHASES}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="slippery_ant_wrapper_${BACKEND}_${DISTINCT_PHASES}phases_${ACTOR_CRITIC_ACTIVATION}_headopt_${HEADS_OPTIMIZER}_${GROUP_ID}_lr_sweep8_s${SEED}"

echo "Running 8-way SlipperyAnt LR sweep group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} action_repeat=${ACTION_REPEAT} phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} change_every_per_env_policy_steps=${CHANGE_EVERY_POLICY_STEPS} change_every_wrapper_steps=${CHANGE_EVERY} schedule=${SCHEDULE_DESC} schedule_seed=${SLIPPERY_SCHEDULE_SEED} total_timesteps=${TOTAL_TIMESTEPS} per_env_steps=${EFFECTIVE_ENV_STEPS_PER_ENV}"
echo "PPO per child: batch=${ROLLOUT_ENV_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} minibatch_size=${MINIBATCH_SIZE} clip_eps=0.1 reward_normalize=true clip_vloss=true max_grad_norm=0.05"
echo "Sweep LRs: ${LRS}"
echo "JAX: XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRACTION}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"

cd "${REPO_ROOT}"

PIDS=()
LOGS=()
for LR in "${LR_LIST[@]}"; do
  LR_TAG="${LR//./p}"
  LR_TAG="${LR_TAG//-/_}"
  EXP_NAME="ppo_slippery_brax_${ENV_NAME}_${BACKEND}_${DISTINCT_PHASES}phases_${ACTOR_CRITIC_ACTIVATION}_headopt_${HEADS_OPTIMIZER}_s${SEED}_lr${LR_TAG}"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/slippery_ant_wrapper_lr8_${GROUP_ID}_s${SEED}_lr${LR_TAG}.out"
  LOGS+=("${CHILD_LOG}")

  COMMON_ARGS=(
    --env-name "${ENV_NAME}"
    --backend "${BACKEND}"
    --n-envs "${N_ENVS}"
    --track
    --wandb-project-name "${WANDB_PROJECT}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --learning-rate "${LR}"
    --adam-eps 1e-5
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
  --slippery-ant
    --slippery-change-every "${CHANGE_EVERY}"
    --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
    --exp-name "${EXP_NAME}"
  )

  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi

  (
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
    export WANDB_RUN_GROUP="${GROUP_NAME}"
    export WANDB_TAGS="continual_rl,slippery_ant_wrapper,brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},lr_${LR},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${CHANGE_EVERY},phases_${DISTINCT_PHASES},action_repeat_${ACTION_REPEAT},actorcritic_${ACTOR_CRITIC_ACTIVATION},headopt_${HEADS_OPTIMIZER},wrapper_faithful,${SCHEDULE_TAG},batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},reward_norm,vclip,repo_ppo,lr_sweep8_per_gpu,jax_mem_fraction_${JAX_MEM_FRACTION}"
    echo "Starting lr=${LR} seed=${SEED} exp=${EXP_NAME} log=${CHILD_LOG}"
    uv run python "${REPO_ROOT}/ppo_brax.py" "${COMMON_ARGS[@]}"
  ) >"${CHILD_LOG}" 2>&1 &

  PID=$!
  PIDS+=("${PID}")
  echo "Launched lr=${LR} pid=${PID} log=${CHILD_LOG}"
done

STATUS=0
for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  CHILD_LOG="${LOGS[$i]}"
  if wait "${PID}"; then
    echo "Child pid=${PID} finished successfully log=${CHILD_LOG}"
  else
    CHILD_STATUS=$?
    echo "Child pid=${PID} failed with status=${CHILD_STATUS} log=${CHILD_LOG}" >&2
    STATUS="${CHILD_STATUS}"
  fi
done

exit "${STATUS}"
