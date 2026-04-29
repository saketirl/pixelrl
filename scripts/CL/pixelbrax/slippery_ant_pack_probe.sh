#!/bin/bash
#SBATCH --job-name=px-slip-pack
#SBATCH --output=slurm_logs/pixelbrax_slippery_pack_probe_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%1

set -euo pipefail

# PixelBrax SlipperyAnt packing probe. Each array task tests one pack level by
# launching multiple short child runs on the same GPU.

WANDB_PROJECT="${1:-continual_pixelbrax}"
WANDB_ENTITY="${2:-rl-power}"
TOTAL_TIMESTEPS="${3:-12800}"
PHASE_EVERY_ENV_STEPS="${4:-4999936}"
SEED_BASE="${5:-0}"
LR="${6:-3e-4}"
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

PACK_LEVELS=(1 2 4 8 12 16)
MEM_FRACTIONS=(0.60 0.35 0.20 0.10 0.075 0.055)

if (( TASK_ID < 0 || TASK_ID >= ${#PACK_LEVELS[@]} )); then
  echo "Invalid TASK_ID=${TASK_ID}." >&2
  exit 1
fi

PACK_LEVEL="${PACK_LEVELS[$TASK_ID]}"
JAX_MEM_FRACTION="${MEM_FRACTIONS[$TASK_ID]}"

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
MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))

ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"
ENCODER_CRATE_STEP_SIZE="0.1"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cl_pixelbrax_slippery_ant_pack_probe_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

echo "PixelBrax SlipperyAnt pack probe: pack=${PACK_LEVEL} mem_fraction=${JAX_MEM_FRACTION} lr=${LR}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} total_timesteps=${TOTAL_TIMESTEPS} n_envs=${N_ENVS} num_steps=${NUM_STEPS} minibatches=${NUM_MINIBATCHES} minibatch_size=${MINIBATCH_SIZE} update_epochs=${UPDATE_EPOCHS}"
echo "Slippery: phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} slippery_change_every=${SLIPPERY_CHANGE_EVERY}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${GROUP_NAME}"

cd "${REPO_ROOT}"

WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

pids=()
for CHILD_IDX in $(seq 0 $((PACK_LEVEL - 1))); do
  SEED=$((SEED_BASE + CHILD_IDX))
  EXP_NAME="ppo_pixelbrax_slippery_ant_pack${PACK_LEVEL}_plain_adam_lr${LR}_s${SEED}"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/pixelbrax_slippery_pack_probe_${GROUP_ID}_pack${PACK_LEVEL}_child${CHILD_IDX}.out"
  (
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
    export PYTHONUNBUFFERED=1
    export WANDB_RUN_GROUP="${GROUP_NAME}"
    export WANDB_TAGS="continual_rl,pixelbrax,slippery_ant,pack_probe,pack_${PACK_LEVEL},child_${CHILD_IDX},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},lr_${LR},jax_mem_fraction_${JAX_MEM_FRACTION},plainheads,adam,crate_cnn,batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},action_repeat_${ACTION_REPEAT}"
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
  echo "Launched child=${CHILD_IDX} seed=${SEED} pid=${pids[-1]} log=${CHILD_LOG}"
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
