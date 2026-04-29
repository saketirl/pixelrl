#!/bin/bash
#SBATCH --job-name=px-slip-first8
#SBATCH --output=slurm_logs/pixelbrax_slippery_first8_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# First PixelBrax SlipperyAnt batch:
# - plain-head Adam CRATECNN
# - four seeds at LR 1e-4 and four seeds at LR 3e-4
# - eight concurrent child runs on one GPU

WANDB_PROJECT="${1:-continual_pixelbrax}"
WANDB_ENTITY="${2:-rl-power}"
TOTAL_TIMESTEPS="${3:-10000000}"
PHASE_EVERY_ENV_STEPS="${4:-4999936}"
JAX_MEM_FRACTION="${5:-${JAX_MEM_FRACTION:-0.10}}"
HEADS_OPTIMIZER_BATCH="${6:-adam}"
USE_CRATE_HEAD_BATCH="${7:-0}"
BATCH_LABEL="${8:-plain_adam}"

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

ENV_NAME="ant"
BACKEND="spring"
N_ENVS="${N_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"

RUN_NAMES=(
  "${BATCH_LABEL}_lr1e4_s0"
  "${BATCH_LABEL}_lr1e4_s1"
  "${BATCH_LABEL}_lr1e4_s2"
  "${BATCH_LABEL}_lr1e4_s3"
  "${BATCH_LABEL}_lr3e4_s0"
  "${BATCH_LABEL}_lr3e4_s1"
  "${BATCH_LABEL}_lr3e4_s2"
  "${BATCH_LABEL}_lr3e4_s3"
)
HEADS_OPTIMIZERS=(
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
  "${HEADS_OPTIMIZER_BATCH}"
)
USE_CRATE_HEADS=(
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
  "${USE_CRATE_HEAD_BATCH}"
)
LRS=(1e-4 1e-4 1e-4 1e-4 3e-4 3e-4 3e-4 3e-4)
SEEDS=(0 1 2 3 0 1 2 3)

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
if (( PHASE_EVERY_ENV_STEPS % N_ENVS != 0 )); then
  echo "PHASE_EVERY_ENV_STEPS=${PHASE_EVERY_ENV_STEPS} must be divisible by N_ENVS=${N_ENVS}." >&2
  exit 1
fi
if (( TOTAL_TIMESTEPS % ROLLOUT_ENV_STEPS != 0 )); then
  ALIGNED_TOTAL_TIMESTEPS=$(((TOTAL_TIMESTEPS / ROLLOUT_ENV_STEPS) * ROLLOUT_ENV_STEPS))
  if (( ALIGNED_TOTAL_TIMESTEPS < ROLLOUT_ENV_STEPS )); then
    ALIGNED_TOTAL_TIMESTEPS="${ROLLOUT_ENV_STEPS}"
  fi
  echo "Aligning TOTAL_TIMESTEPS from ${TOTAL_TIMESTEPS} to ${ALIGNED_TOTAL_TIMESTEPS}."
  TOTAL_TIMESTEPS="${ALIGNED_TOTAL_TIMESTEPS}"
fi

CHANGE_EVERY_POLICY_STEPS=$((PHASE_EVERY_ENV_STEPS / N_ENVS))
SLIPPERY_CHANGE_EVERY=$((CHANGE_EVERY_POLICY_STEPS * ACTION_REPEAT))
EFFECTIVE_ENV_STEPS_PER_ENV=$((TOTAL_TIMESTEPS / N_ENVS))
EFFECTIVE_PHASES=$(((EFFECTIVE_ENV_STEPS_PER_ENV + CHANGE_EVERY_POLICY_STEPS - 1) / CHANGE_EVERY_POLICY_STEPS))
MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))

ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"
ENCODER_CRATE_STEP_SIZE="0.1"

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="cl_pixelbrax_slippery_ant_${BATCH_LABEL}_lr_compare8_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

echo "PixelBrax SlipperyAnt first8 batch group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} total_timesteps=${TOTAL_TIMESTEPS} n_envs=${N_ENVS} num_steps=${NUM_STEPS} minibatches=${NUM_MINIBATCHES} minibatch_size=${MINIBATCH_SIZE} update_epochs=${UPDATE_EPOCHS}"
echo "Slippery: phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} slippery_change_every=${SLIPPERY_CHANGE_EVERY} effective_phases=${EFFECTIVE_PHASES} action_repeat=${ACTION_REPEAT}"
echo "Packing: runs=8 jax_mem_fraction=${JAX_MEM_FRACTION}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${GROUP_NAME}"

WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"

pids=()
for IDX in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$IDX]}"
  HEADS_OPTIMIZER="${HEADS_OPTIMIZERS[$IDX]}"
  USE_CRATE_HEAD="${USE_CRATE_HEADS[$IDX]}"
  LR="${LRS[$IDX]}"
  SEED="${SEEDS[$IDX]}"

  CRATE_HEAD_ARGS=()
  HEADARCH_TAG="plainheads"
  if (( USE_CRATE_HEAD == 1 )); then
    CRATE_HEAD_ARGS+=(--use-crate-head)
    HEADARCH_TAG="crateheads"
  fi

  EXP_NAME="ppo_pixelbrax_slippery_ant_${RUN_NAME}_first8"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/pixelbrax_slippery_first8_${GROUP_ID}_${RUN_NAME}.out"
  (
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
    export PYTHONUNBUFFERED=1
    export WANDB_RUN_GROUP="${GROUP_NAME}"
    export WANDB_TAGS="continual_rl,pixelbrax,slippery_ant,first8,config_${RUN_NAME},${HEADARCH_TAG},headopt_${HEADS_OPTIMIZER},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},lr_${LR},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${SLIPPERY_CHANGE_EVERY},phases_${EFFECTIVE_PHASES},action_repeat_${ACTION_REPEAT},crate_cnn,batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},jax_mem_fraction_${JAX_MEM_FRACTION},runs_per_gpu_8"
    uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
      --env-name "${ENV_NAME}" \
      --backend "${BACKEND}" \
      --n-envs "${N_ENVS}" \
      --track \
      --wandb-project-name "${WANDB_PROJECT}" \
      --hw 84 \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --num-steps "${NUM_STEPS}" \
      --num-minibatches "${NUM_MINIBATCHES}" \
      --update-epochs "${UPDATE_EPOCHS}" \
      --gamma 0.99 \
      --gae-lambda 0.95 \
      --clip-eps 0.1 \
      --ent-coef 0.0 \
      --vf-coef 0.5 \
      --seed "${SEED}" \
      --log-interval 1 \
      --frame-stack 4 \
      --action-repeat "${ACTION_REPEAT}" \
      --anneal-lr \
      --stiefel-dual-lr 0.01 \
      --stiefel-dual-steps 5 \
      --actor-stiefel-max-grad-norm 100 \
      --critic-stiefel-max-grad-norm 1 \
      --actor-mean-tanh \
      --actor-mean-scale "${ACTOR_MEAN_SCALE}" \
      --bounded-global-logstd \
      --actor-logstd-init="${ACTOR_LOGSTD_INIT}" \
      --actor-logstd-min="${ACTOR_LOGSTD_MIN}" \
      --actor-logstd-max="${ACTOR_LOGSTD_MAX}" \
      --encoder-type crate_cnn \
      --encoder-lr "${LR}" \
      --heads-adam-lr "${LR}" \
      --heads-stiefel-lr 0.001 \
      --max-grad-norm 0.05 \
      --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}" \
      --sigreg-mode off \
      --heads-optimizer "${HEADS_OPTIMIZER}" \
      "${CRATE_HEAD_ARGS[@]}" \
      --slippery-ant \
      --slippery-change-every "${SLIPPERY_CHANGE_EVERY}" \
      --slippery-schedule-seed "${SEED}" \
      --exp-name "${EXP_NAME}" \
      "${WANDB_ENTITY_ARGS[@]}"
  ) > "${CHILD_LOG}" 2>&1 &
  pids+=("$!")
  echo "Launched ${RUN_NAME} pid=${pids[-1]} log=${CHILD_LOG}"
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
