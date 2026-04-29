#!/bin/bash
#SBATCH --job-name=px-slip-lr1e4x8
#SBATCH --output=slurm_logs/pixelbrax_slippery_lr1e4_8seeds_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# PixelBrax analogue of scripts/CL/brax/slippery_ant_wrapper_lr1e4.sh.
# Uses the Ant plain-head Adam CRATECNN staging config, LR=1e-4, and packs
# eight seeds onto one GPU.

WANDB_PROJECT="${1:-continual_pixelbrax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-4999936}"
NUM_PHASES="${4:-20}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
SEEDS="${6:-${SEEDS:-0 1 2 3 4 5 6 7}}"

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
LR="1e-4"
JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.10}"

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
if (( PHASE_EVERY_ENV_STEPS % N_ENVS != 0 )); then
  echo "PHASE_EVERY_ENV_STEPS=${PHASE_EVERY_ENV_STEPS} must be divisible by N_ENVS=${N_ENVS}." >&2
  exit 1
fi

CHANGE_EVERY_POLICY_STEPS=$((PHASE_EVERY_ENV_STEPS / N_ENVS))
SLIPPERY_CHANGE_EVERY=$((CHANGE_EVERY_POLICY_STEPS * ACTION_REPEAT))
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

EFFECTIVE_ENV_STEPS_PER_ENV=$((TOTAL_TIMESTEPS / N_ENVS))
EFFECTIVE_PHASES=$(((EFFECTIVE_ENV_STEPS_PER_ENV + CHANGE_EVERY_POLICY_STEPS - 1) / CHANGE_EVERY_POLICY_STEPS))
MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))

ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"
ENCODER_CRATE_STEP_SIZE="0.1"

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="cl_pixelbrax_slippery_ant_lr1e4_plain_adam_8seeds_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

echo "PixelBrax SlipperyAnt lr1e-4 8-seed replication group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seeds=${SEEDS} total_timesteps=${TOTAL_TIMESTEPS} n_envs=${N_ENVS} num_steps=${NUM_STEPS} minibatches=${NUM_MINIBATCHES} minibatch_size=${MINIBATCH_SIZE} update_epochs=${UPDATE_EPOCHS}"
echo "Slippery: phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} slippery_change_every=${SLIPPERY_CHANGE_EVERY} effective_phases=${EFFECTIVE_PHASES} action_repeat=${ACTION_REPEAT}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${GROUP_NAME}"

WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"

pids=()
for SEED in ${SEEDS}; do
  EXP_NAME="ppo_pixelbrax_slippery_ant_lr1e4_plain_adam_s${SEED}_8seeds_per_gpu"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/pixelbrax_slippery_lr1e4_8seeds_${GROUP_ID}_seed${SEED}.out"
  (
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
    export PYTHONUNBUFFERED=1
    export WANDB_RUN_GROUP="${GROUP_NAME}"
    export WANDB_TAGS="continual_rl,pixelbrax,slippery_ant,lr1e4,plainheads,adam,crate_cnn,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${SLIPPERY_CHANGE_EVERY},phases_${EFFECTIVE_PHASES},action_repeat_${ACTION_REPEAT},batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},jax_mem_fraction_${JAX_MEM_FRACTION},eight_seeds_per_gpu,wrapper_faithful"
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
      --heads-optimizer adam \
      --slippery-ant \
      --slippery-change-every "${SLIPPERY_CHANGE_EVERY}" \
      --slippery-schedule-seed "${SEED}" \
      --exp-name "${EXP_NAME}" \
      "${WANDB_ENTITY_ARGS[@]}"
  ) > "${CHILD_LOG}" 2>&1 &
  pids+=("$!")
  echo "Launched seed=${SEED} pid=${pids[-1]} log=${CHILD_LOG}"
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
