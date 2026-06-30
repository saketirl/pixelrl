#!/bin/bash
#SBATCH --job-name=ant-rlopt-lop-adam
#SBATCH --output=slurm_logs/ant_rlopt_lop_plain_adam_seed0_%j.out
#SBATCH --error=slurm_logs/ant_rlopt_lop_plain_adam_seed0_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=geforce3090

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_brax_rlopt_lop_adam.py" ]]; then
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
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/pixelrl-uv-cache-${USER:-user}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pixelrl-mpl-${USER:-user}}"
mkdir -p "${UV_CACHE_DIR}" "${MPLCONFIGDIR}"

CONDITION="${CONDITION:-rlopt_lop_plain_adam_clip_actions}"
SEED=0

ENV_NAME="${ENV_NAME:-ant}"
BACKEND="${BACKEND:-positional}"
N_ENVS="${N_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
CHANGE_EVERY="${CHANGE_EVERY:-2000000}"
NUM_STEPS="${NUM_STEPS:-2048}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
BASE_OPTIMIZER="${BASE_OPTIMIZER:-adam}"
HEADS_OPTIMIZER="${HEADS_OPTIMIZER:-adam}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_EPS="${CLIP_EPS:-0.2}"
ENT_COEF="${ENT_COEF:-0.0}"
VF_COEF="${VF_COEF:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1000000000}"
MAX_ACTION="${MAX_ACTION:-1.0}"
CLIP_ACTIONS="${CLIP_ACTIONS:-1}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
ACTOR_CRITIC_ACTIVATION="${ACTOR_CRITIC_ACTIVATION:-relu}"
NETWORK_ARCH="${NETWORK_ARCH:-lop_reference}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:-0.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:-2.0}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
if (( ROLLOUT_ENV_STEPS % NUM_MINIBATCHES != 0 )); then
  echo "batch=${ROLLOUT_ENV_STEPS} must be divisible by num minibatches ${NUM_MINIBATCHES}." >&2
  exit 1
fi

ALIGNED_TOTAL_TIMESTEPS=$(((TOTAL_TIMESTEPS / ROLLOUT_ENV_STEPS) * ROLLOUT_ENV_STEPS))
if (( ALIGNED_TOTAL_TIMESTEPS != TOTAL_TIMESTEPS )); then
  echo "Aligning TOTAL_TIMESTEPS from ${TOTAL_TIMESTEPS} down to ${ALIGNED_TOTAL_TIMESTEPS}."
  TOTAL_TIMESTEPS="${ALIGNED_TOTAL_TIMESTEPS}"
fi

MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))
WANDB_PROJECT="${WANDB_PROJECT:-continual_brax}"
WANDB_ENTITY="${WANDB_ENTITY:-enyan_zhang1-brown-university}"
WANDB_GROUP="${WANDB_GROUP:-slippery-ant-rlopt-lop-plain-adam}"
EXP_NAME="${EXP_NAME:-ppo_slippery_ant_${CONDITION}_seed${SEED}}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${REPO_ROOT}/secrets/wandb_api_key.txt}"

case "${CLIP_ACTIONS}" in
  1|true|True|TRUE|yes|Yes|YES)
    CLIP_ACTIONS_ENABLED=1
    CLIP_ACTIONS_TAG="clip_actions"
    ;;
  0|false|False|FALSE|no|No|NO)
    CLIP_ACTIONS_ENABLED=0
    CLIP_ACTIONS_TAG="no_clip_actions"
    ;;
  *)
    echo "Invalid CLIP_ACTIONS=${CLIP_ACTIONS}. Expected 0/1 or true/false." >&2
    exit 1
    ;;
esac

if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_TAGS="continual_rl,slippery_ant,${CONDITION},plain_adam,seed_${SEED},backend_${BACKEND},network_arch_${NETWORK_ARCH},actorcritic_${ACTOR_CRITIC_ACTIVATION},no_reward_norm,no_obs_norm,unbounded_global_logstd,no_actor_mean_tanh,${CLIP_ACTIONS_TAG},pixelrl_ppo_objective"

COMMAND=(
  uv run python "${REPO_ROOT}/ppo_brax_rlopt_lop_adam.py"
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate "${LEARNING_RATE}"
  --adam-eps "${ADAM_EPS}"
  --base-optimizer "${BASE_OPTIMIZER}"
  --heads-optimizer "${HEADS_OPTIMIZER}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-steps "${NUM_STEPS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --update-epochs "${UPDATE_EPOCHS}"
  --gamma "${GAMMA}"
  --gae-lambda "${GAE_LAMBDA}"
  --clip-eps "${CLIP_EPS}"
  --ent-coef "${ENT_COEF}"
  --vf-coef "${VF_COEF}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --max-action "${MAX_ACTION}"
  --actor-critic-activation "${ACTOR_CRITIC_ACTIVATION}"
  --network-arch "${NETWORK_ARCH}"
  --actor-logstd-init "${ACTOR_LOGSTD_INIT}"
  --actor-logstd-min "${ACTOR_LOGSTD_MIN}"
  --actor-logstd-max "${ACTOR_LOGSTD_MAX}"
  --seed "${SEED}"
  --log-interval "${LOG_INTERVAL}"
  --action-repeat "${ACTION_REPEAT}"
  --slippery-ant
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
  --exp-name "${EXP_NAME}"
)

if [[ "${CLIP_ACTIONS_ENABLED}" == "1" ]]; then
  COMMAND+=(--clip-actions)
fi

printf -v FINAL_COMMAND "%q " "${COMMAND[@]}"

cd "${REPO_ROOT}"
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Repo root: ${REPO_ROOT}"
  echo "Condition: ${CONDITION}"
  echo "Seed: ${SEED}"
  echo "Wandb project: ${WANDB_PROJECT}"
  echo "Wandb entity: ${WANDB_ENTITY}"
  echo "Wandb group: ${WANDB_GROUP}"
  echo "Experiment name: ${EXP_NAME}"
  echo "Total timesteps: ${TOTAL_TIMESTEPS}"
  echo "Change every: ${CHANGE_EVERY}"
  echo "Rollout env steps: ${ROLLOUT_ENV_STEPS}"
  echo "Minibatch size: ${MINIBATCH_SIZE}"
  echo "Reward normalize: 0"
  echo "Observation normalize: 0"
  echo "Clip actions: ${CLIP_ACTIONS_ENABLED}"
  echo "Final command: ${FINAL_COMMAND}"
  exit 0
fi

"${COMMAND[@]}"
