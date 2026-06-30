#!/bin/bash
#SBATCH --job-name=ant-rlopt-lop-o35
#SBATCH --output=slurm_logs/ant_rlopt_lop_original_config_seeds3_5_%A_%a.out
#SBATCH --error=slurm_logs/ant_rlopt_lop_original_config_seeds3_5_%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --array=0-5

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID > 5 )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..5 for seeds 3..5 Adam/Stiefel." >&2
  exit 1
fi

HEADS_STIEFEL_LR="${HEADS_STIEFEL_LR:-0.001}"

if (( TASK_ID < 3 )); then
  RUN_KIND="adam"
  CONDITION="rlopt_lop_original_ppo_config_adam"
  HEADS_OPTIMIZER="adam"
  LEARNING_RATE="${ADAM_LEARNING_RATE:-1e-4}"
  SEED=$((3 + TASK_ID))
else
  RUN_KIND="stiefel"
  CONDITION="rlopt_lop_original_ppo_config_stiefel"
  HEADS_OPTIMIZER="stiefel"
  LEARNING_RATE="${STIEFEL_BASE_LEARNING_RATE:-3e-5}"
  SEED=$((3 + TASK_ID - 3))
fi

ENV_NAME="${ENV_NAME:-ant}"
BACKEND="${BACKEND:-positional}"
N_ENVS="${N_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
CHANGE_EVERY="${CHANGE_EVERY:-2000000}"
NUM_STEPS="${NUM_STEPS:-2048}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
BASE_OPTIMIZER="${BASE_OPTIMIZER:-adam}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_EPS="${CLIP_EPS:-0.2}"
ENT_COEF="${ENT_COEF:-0.01}"
VF_COEF="${VF_COEF:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1000000000}"
MAX_ACTION="${MAX_ACTION:-1.0}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
ACTOR_CRITIC_ACTIVATION="${ACTOR_CRITIC_ACTIVATION:-relu}"
NETWORK_ARCH="${NETWORK_ARCH:-lop}"
ACTOR_MEAN_SCALE="${ACTOR_MEAN_SCALE:-1.0}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:--2.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:--1.4}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
STIEFEL_DUAL_LR="${STIEFEL_DUAL_LR:-0.01}"
STIEFEL_DUAL_STEPS="${STIEFEL_DUAL_STEPS:-5}"
STIEFEL_MSIGN_STEPS="${STIEFEL_MSIGN_STEPS:-5}"
ACTOR_STIEFEL_MAX_GRAD_NORM="${ACTOR_STIEFEL_MAX_GRAD_NORM:-100}"
CRITIC_STIEFEL_MAX_GRAD_NORM="${CRITIC_STIEFEL_MAX_GRAD_NORM:-1}"

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
WANDB_GROUP="${WANDB_GROUP:-slippery-ant-rlopt-lop-original-ppo-config-seed0-adam-vs-stiefel}"
EXP_NAME="${EXP_NAME:-ppo_slippery_ant_${CONDITION}_seed${SEED}}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${REPO_ROOT}/secrets/wandb_api_key.txt}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_TAGS="continual_rl,slippery_ant,${CONDITION},original_ppo_config,${RUN_KIND},seed_${SEED},backend_${BACKEND},network_arch_${NETWORK_ARCH},single_256_head,actorcritic_${ACTOR_CRITIC_ACTIVATION},no_reward_norm,no_obs_norm,bounded_global_logstd,actor_mean_tanh,clip_actions,pixelrl_ppo_objective"

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
  --heads-stiefel-lr "${HEADS_STIEFEL_LR}"
  --stiefel-dual-lr "${STIEFEL_DUAL_LR}"
  --stiefel-dual-steps "${STIEFEL_DUAL_STEPS}"
  --stiefel-msign-steps "${STIEFEL_MSIGN_STEPS}"
  --actor-stiefel-max-grad-norm "${ACTOR_STIEFEL_MAX_GRAD_NORM}"
  --critic-stiefel-max-grad-norm "${CRITIC_STIEFEL_MAX_GRAD_NORM}"
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
  --actor-mean-tanh
  --actor-mean-scale "${ACTOR_MEAN_SCALE}"
  --bounded-global-logstd
  --actor-logstd-init "${ACTOR_LOGSTD_INIT}"
  --actor-logstd-min "${ACTOR_LOGSTD_MIN}"
  --actor-logstd-max "${ACTOR_LOGSTD_MAX}"
  --seed "${SEED}"
  --log-interval "${LOG_INTERVAL}"
  --action-repeat "${ACTION_REPEAT}"
  --slippery-ant
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
  --clip-actions
  --exp-name "${EXP_NAME}"
)

printf -v FINAL_COMMAND "%q " "${COMMAND[@]}"

cd "${REPO_ROOT}"
unset LD_LIBRARY_PATH
unset VIRTUAL_ENV

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
  echo "Repo root: ${REPO_ROOT}"
  echo "Condition: ${CONDITION}"
  echo "Run kind: ${RUN_KIND}"
  echo "Task id: ${TASK_ID}"
  echo "Seed: ${SEED}"
  echo "Wandb project: ${WANDB_PROJECT}"
  echo "Wandb entity: ${WANDB_ENTITY}"
  echo "Wandb group: ${WANDB_GROUP}"
  echo "Experiment name: ${EXP_NAME}"
  echo "Total timesteps: ${TOTAL_TIMESTEPS}"
  echo "Change every: ${CHANGE_EVERY}"
  echo "Rollout env steps: ${ROLLOUT_ENV_STEPS}"
  echo "Minibatch size: ${MINIBATCH_SIZE}"
  echo "Network arch: ${NETWORK_ARCH}"
  echo "Reward normalize: 0"
  echo "Observation normalize: 0"
  echo "Actor mean tanh: 1"
  echo "Bounded global logstd: 1"
  echo "Clip actions: 1"
  echo "Base optimizer: ${BASE_OPTIMIZER}"
  echo "Heads optimizer: ${HEADS_OPTIMIZER}"
  echo "Learning rate: ${LEARNING_RATE}"
  echo "Heads Stiefel lr: ${HEADS_STIEFEL_LR}"
  echo "Final command: ${FINAL_COMMAND}"
  exit 0
fi

"${COMMAND[@]}"
