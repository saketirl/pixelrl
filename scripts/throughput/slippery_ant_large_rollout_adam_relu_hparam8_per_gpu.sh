#!/bin/bash
#SBATCH --job-name=slip-ant-big-adam8
#SBATCH --output=slurm_logs/slippery_ant_big_adam8_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# One-GPU 8-way hyperparameter sweep for the fastest Adam/relu throughput shape.
# Fixed geometry:
#   N_ENVS=256, NUM_STEPS=40, batch=10240
# Goal:
#   recover reasonable SlipperyAnt learning by ~5M steps while preserving high
#   single-process throughput as much as possible.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-4999936}"
NUM_PHASES="${4:-1}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-4999936}"
SEED="${6:-${SEED:-0}}"
BACKEND="${7:-${BACKEND:-spring}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_slippery_brax_memfrac.py" ]]; then
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
N_ENVS="${N_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-40}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
ACTOR_CRITIC_ACTIVATION="relu"
HEADS_OPTIMIZER="adam"
JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.05}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-${SEED}}"
SCHEDULE_TAG="default_then_csv_schedule"

# Eight configs: primarily increase LR to compensate for fewer PPO updates from
# the 10240-transition rollout, with two higher-epoch candidates.
CONFIG_NAMES=(
  lr3e4_ep4
  lr5e4_ep4
  lr1e3_ep4
  lr2e3_ep4
  lr3e3_ep4
  lr5e3_ep4
  lr1e3_ep8
  lr2e3_ep8
)
CONFIG_LRS=(3e-4 5e-4 1e-3 2e-3 3e-3 5e-3 1e-3 2e-3)
CONFIG_UPDATE_EPOCHS=(4 4 4 4 4 4 8 8)

NUM_CONFIGS=${#CONFIG_NAMES[@]}
if (( NUM_CONFIGS != 8 )); then
  echo "Expected exactly 8 configs; got ${NUM_CONFIGS}." >&2
  exit 1
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

EFFECTIVE_ENV_STEPS_PER_ENV=$((TOTAL_TIMESTEPS / N_ENVS))
EFFECTIVE_PHASES=$(((EFFECTIVE_ENV_STEPS_PER_ENV + CHANGE_EVERY_POLICY_STEPS - 1) / CHANGE_EVERY_POLICY_STEPS))
GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="slippery_ant_large_rollout_adam_relu_hparam8_${GROUP_ID}"

echo "Running 8-way large-rollout Adam/relu hparam sweep group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} n_envs=${N_ENVS} num_steps=${NUM_STEPS} batch=${ROLLOUT_ENV_STEPS} minibatches=${NUM_MINIBATCHES} minibatch_size=${MINIBATCH_SIZE} action_repeat=${ACTION_REPEAT} phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} change_every_policy_steps=${CHANGE_EVERY_POLICY_STEPS} change_every_wrapper_steps=${CHANGE_EVERY} total_timesteps=${TOTAL_TIMESTEPS} per_env_steps=${EFFECTIVE_ENV_STEPS_PER_ENV}"
echo "Sweep configs: ${CONFIG_NAMES[*]}"
echo "LRS: ${CONFIG_LRS[*]}"
echo "Epochs: ${CONFIG_UPDATE_EPOCHS[*]}"
echo "JAX: XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRACTION}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"

cd "${REPO_ROOT}"

PIDS=()
LOGS=()
for IDX in "${!CONFIG_NAMES[@]}"; do
  CONFIG_NAME="${CONFIG_NAMES[$IDX]}"
  LR="${CONFIG_LRS[$IDX]}"
  UPDATE_EPOCHS="${CONFIG_UPDATE_EPOCHS[$IDX]}"
  EXP_NAME="ppo_slippery_big_adam_relu_env${N_ENVS}_steps${NUM_STEPS}_${CONFIG_NAME}_s${SEED}"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/slippery_ant_big_adam8_${GROUP_ID}_${CONFIG_NAME}.out"
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
    export WANDB_TAGS="continual_rl,slippery_ant_wrapper,throughput_hparam_sweep,large_rollout,adam,relu,config_${CONFIG_NAME},lr_${LR},nenvs_${N_ENVS},steps_${NUM_STEPS},batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${CHANGE_EVERY},phases_${EFFECTIVE_PHASES},action_repeat_${ACTION_REPEAT},wrapper_faithful,${SCHEDULE_TAG},reward_norm,vclip,target_5m_return_2000,jax_mem_fraction_${JAX_MEM_FRACTION}"
    echo "Starting config=${CONFIG_NAME} lr=${LR} epochs=${UPDATE_EPOCHS} exp=${EXP_NAME} log=${CHILD_LOG}"
    uv run python "${REPO_ROOT}/ppo_slippery_brax_memfrac.py" "${COMMON_ARGS[@]}"
  ) >"${CHILD_LOG}" 2>&1 &

  PID=$!
  PIDS+=("${PID}")
  echo "Launched config=${CONFIG_NAME} pid=${PID} log=${CHILD_LOG}"
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
