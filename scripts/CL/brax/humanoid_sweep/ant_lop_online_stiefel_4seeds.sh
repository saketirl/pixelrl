#!/bin/bash
#SBATCH --job-name=ant-lop-online
#SBATCH --output=slurm_logs/slippery_ant_lop_online_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Four-seed Adam vs online Stiefel comparison using the current successful
# lop-cadence Slippery Ant setup.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

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

if (( TASK_ID < 0 || TASK_ID >= 8 )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..7." >&2
  exit 1
fi

optimizer_idx=$((TASK_ID / 4))
seed_idx=$((TASK_ID % 4))

case "${optimizer_idx}" in
  0)
    HEADS_OPTIMIZER="${HEADS_OPTIMIZER:-adam}"
    RUN_KIND="adam"
    ;;
  1)
    HEADS_OPTIMIZER="${HEADS_OPTIMIZER:-stiefel_online}"
    RUN_KIND="${HEADS_OPTIMIZER}"
    ;;
  *)
    echo "Invalid optimizer index ${optimizer_idx}." >&2
    exit 1
    ;;
esac

ENV_NAME="${ENV_NAME:-ant}"
BACKEND="${BACKEND:-positional}"
SEED="${SEED_OFFSET:-0}"
SEED=$((SEED + seed_idx))
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-0}"

N_ENVS="${N_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-2048}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
CHANGE_EVERY="${CHANGE_EVERY:-2000000}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
BASE_OPTIMIZER="${BASE_OPTIMIZER:-adam}"
ACTOR_CRITIC_ACTIVATION="${ACTOR_CRITIC_ACTIVATION:-relu}"
NETWORK_ARCH="${NETWORK_ARCH:-lop}"
MAX_ACTION="${MAX_ACTION:-1.0}"
ACTOR_MEAN_SCALE="${ACTOR_MEAN_SCALE:-1.0}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:--2.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:--1.4}"
HEADS_STIEFEL_LR="${HEADS_STIEFEL_LR:-0.001}"
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
GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="slippery_ant_lop_${RUN_KIND}_ablation_${BACKEND}_${GROUP_ID}"
EXP_NAME="ppo_slippery_ant_lop_${RUN_KIND}_${BACKEND}_s${SEED}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_ant,lop_protocol,stiefel_ablation,policy_stable,network_arch_${NETWORK_ARCH},brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},schedule_seed_${SLIPPERY_SCHEDULE_SEED},${RUN_KIND},change_every_${CHANGE_EVERY},total_${TOTAL_TIMESTEPS},action_repeat_${ACTION_REPEAT},actorcritic_${ACTOR_CRITIC_ACTIVATION},baseopt_${BASE_OPTIMIZER},headopt_${HEADS_OPTIMIZER},weight_decay_${WEIGHT_DECAY},batch_${ROLLOUT_ENV_STEPS},num_steps_${NUM_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},max_action_${MAX_ACTION},clip_eps_0.2,ent_0.01,vf_1,no_reward_norm,no_obs_norm,no_crate,bounded_global_logstd,actor_mean_tanh,lr_${LEARNING_RATE},repo_ppo"

echo "Running SlipperyAnt lop Stiefel ablation group=${GROUP_NAME}"
echo "Config: task_id=${TASK_ID} run_kind=${RUN_KIND} env=${ENV_NAME} backend=${BACKEND} seed=${SEED} schedule_seed=${SLIPPERY_SCHEDULE_SEED} total_timesteps=${TOTAL_TIMESTEPS} change_every=${CHANGE_EVERY}"
echo "PPO: n_envs=${N_ENVS} rollout_batch=${ROLLOUT_ENV_STEPS} num_steps=${NUM_STEPS} minibatch_size=${MINIBATCH_SIZE} lr=${LEARNING_RATE} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} heads_optimizer=${HEADS_OPTIMIZER}"

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" \
  --env-name "${ENV_NAME}" \
  --backend "${BACKEND}" \
  --n-envs "${N_ENVS}" \
  --track \
  --wandb-project-name "${WANDB_PROJECT}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --adam-eps "${ADAM_EPS}" \
  --base-optimizer "${BASE_OPTIMIZER}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-steps "${NUM_STEPS}" \
  --num-minibatches "${NUM_MINIBATCHES}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --ent-coef 0.01 \
  --vf-coef 1.0 \
  --max-grad-norm 1000000000 \
  --max-action "${MAX_ACTION}" \
  --actor-critic-activation "${ACTOR_CRITIC_ACTIVATION}" \
  --network-arch "${NETWORK_ARCH}" \
  --heads-optimizer "${HEADS_OPTIMIZER}" \
  --heads-stiefel-lr "${HEADS_STIEFEL_LR}" \
  --stiefel-dual-lr "${STIEFEL_DUAL_LR}" \
  --stiefel-dual-steps "${STIEFEL_DUAL_STEPS}" \
  --stiefel-msign-steps "${STIEFEL_MSIGN_STEPS}" \
  --actor-stiefel-max-grad-norm "${ACTOR_STIEFEL_MAX_GRAD_NORM}" \
  --critic-stiefel-max-grad-norm "${CRITIC_STIEFEL_MAX_GRAD_NORM}" \
  --seed "${SEED}" \
  --log-interval 1 \
  --action-repeat "${ACTION_REPEAT}" \
  --slippery-ant \
  --slippery-change-every "${CHANGE_EVERY}" \
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}" \
  --actor-mean-tanh \
  --actor-mean-scale "${ACTOR_MEAN_SCALE}" \
  --bounded-global-logstd \
  --actor-logstd-init "${ACTOR_LOGSTD_INIT}" \
  --actor-logstd-min "${ACTOR_LOGSTD_MIN}" \
  --actor-logstd-max "${ACTOR_LOGSTD_MAX}" \
  --exp-name "${EXP_NAME}" \
  --wandb-entity "${WANDB_ENTITY}"
