#!/bin/bash
#SBATCH --job-name=px-slip-stage
#SBATCH --output=slurm_logs/pixelbrax_slippery_staging_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%8

set -euo pipefail

# PixelBrax SlipperyAnt sweep over the four Ant staging configs and two LRs.
# RUNS_PER_GPU can be set from the packing probe result; array tasks whose
# grouped index exceeds the 32 logical runs exit immediately.

WANDB_PROJECT="${1:-continual_pixelbrax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-4999936}"
NUM_PHASES="${4:-20}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
RUNS_PER_GPU="${6:-${RUNS_PER_GPU:-1}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

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

CONFIG_NAMES=(plain_adam plain_stiefel crate_adam crate_stiefel)
CONFIG_HEADS_OPT=(adam stiefel adam stiefel)
CONFIG_USE_CRATE_HEAD=(0 0 1 1)
LRS=(1e-4 3e-4)
SEEDS=(0 1 2 3)

NUM_CONFIGS=${#CONFIG_NAMES[@]}
NUM_LRS=${#LRS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_RUNS=$((NUM_CONFIGS * NUM_LRS * NUM_SEEDS))

if (( RUNS_PER_GPU < 1 )); then
  echo "RUNS_PER_GPU must be >= 1; got ${RUNS_PER_GPU}." >&2
  exit 1
fi

START_IDX=$((TASK_ID * RUNS_PER_GPU))
if (( START_IDX >= TOTAL_RUNS )); then
  echo "TASK_ID=${TASK_ID} start=${START_IDX} >= TOTAL_RUNS=${TOTAL_RUNS}; nothing to do."
  exit 0
fi
END_IDX=$((START_IDX + RUNS_PER_GPU - 1))
if (( END_IDX >= TOTAL_RUNS )); then
  END_IDX=$((TOTAL_RUNS - 1))
fi

ENV_NAME="ant"
BACKEND="spring"
N_ENVS="${N_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))

if (( PHASE_EVERY_ENV_STEPS % N_ENVS != 0 )); then
  echo "PHASE_EVERY_ENV_STEPS=${PHASE_EVERY_ENV_STEPS} must be divisible by N_ENVS=${N_ENVS}." >&2
  exit 1
fi

if [[ -n "${TOTAL_TIMESTEPS_OVERRIDE}" ]]; then
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE}"
else
  TOTAL_TIMESTEPS=$((PHASE_EVERY_ENV_STEPS * NUM_PHASES))
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

case "${RUNS_PER_GPU}" in
  1) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.60}" ;;
  2) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.35}" ;;
  3|4) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.20}" ;;
  *) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.10}" ;;
esac

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cl_pixelbrax_slippery_ant_staging_lr_sweep_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

echo "PixelBrax SlipperyAnt staging sweep: logical runs ${START_IDX}..${END_IDX}/${TOTAL_RUNS} runs_per_gpu=${RUNS_PER_GPU} mem_fraction=${JAX_MEM_FRACTION}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} total_timesteps=${TOTAL_TIMESTEPS} phases=${EFFECTIVE_PHASES} n_envs=${N_ENVS} num_steps=${NUM_STEPS} minibatches=${NUM_MINIBATCHES} minibatch_size=${MINIBATCH_SIZE} update_epochs=${UPDATE_EPOCHS}"
echo "Slippery: phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} slippery_change_every=${SLIPPERY_CHANGE_EVERY}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${GROUP_NAME}"

cd "${REPO_ROOT}"

pids=()
for RUN_IDX in $(seq "${START_IDX}" "${END_IDX}"); do
  SEED_IDX=$((RUN_IDX % NUM_SEEDS))
  LR_IDX=$(((RUN_IDX / NUM_SEEDS) % NUM_LRS))
  CONFIG_IDX=$((RUN_IDX / (NUM_SEEDS * NUM_LRS)))

  SEED="${SEEDS[$SEED_IDX]}"
  LR="${LRS[$LR_IDX]}"
  CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"
  HEADS_OPTIMIZER="${CONFIG_HEADS_OPT[$CONFIG_IDX]}"
  USE_CRATE_HEAD="${CONFIG_USE_CRATE_HEAD[$CONFIG_IDX]}"

  EXP_NAME="ppo_pixelbrax_slippery_ant_${CONFIG_NAME}_lr${LR}_s${SEED}"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/pixelbrax_slippery_staging_${GROUP_ID}_run${RUN_IDX}_${CONFIG_NAME}_lr${LR}_s${SEED}.out"

  CRATE_HEAD_ARGS=()
  HEADARCH_TAG="plainheads"
  if (( USE_CRATE_HEAD == 1 )); then
    CRATE_HEAD_ARGS+=(--use-crate-head)
    HEADARCH_TAG="crateheads"
  fi

  WANDB_ENTITY_ARGS=()
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi

  (
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
    export WANDB_RUN_GROUP="${GROUP_NAME}"
    export WANDB_TAGS="continual_rl,pixelbrax,slippery_ant,staging_configs,config_${CONFIG_NAME},${HEADARCH_TAG},headopt_${HEADS_OPTIMIZER},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},lr_${LR},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${SLIPPERY_CHANGE_EVERY},phases_${EFFECTIVE_PHASES},action_repeat_${ACTION_REPEAT},crate_cnn,batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},jax_mem_fraction_${JAX_MEM_FRACTION},runs_per_gpu_${RUNS_PER_GPU}"
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
  echo "Launched run_idx=${RUN_IDX} config=${CONFIG_NAME} lr=${LR} seed=${SEED} pid=${pids[-1]} log=${CHILD_LOG}"
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
