#!/bin/bash
#SBATCH --job-name=ant-lop-protocol
#SBATCH --output=slurm_logs/slippery_ant_lop_protocol_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Pilot for matching the lop-jax Slippery Ant protocol:
#   total_steps=10M, num_envs=1, num_steps=2048, minibatches=128,
#   update_epochs=10, change_every=2M, 5 long friction phases.
# We run Adam with and without obs normalization because ppo_brax may need it.

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

if (( TASK_ID < 0 || TASK_ID >= 4 )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..3." >&2
  exit 1
fi

seed_idx=$((TASK_ID % 2))
variant_idx=$((TASK_ID / 2))

ENV_NAME="ant"
BACKEND="${BACKEND:-spring}"
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
HEADS_OPTIMIZER="${HEADS_OPTIMIZER:-adam}"
ACTOR_CRITIC_ACTIVATION="${ACTOR_CRITIC_ACTIVATION:-relu}"
NETWORK_ARCH="${NETWORK_ARCH:-ppo}"
OBS_NORM_CLIP="${OBS_NORM_CLIP:-10.0}"
MAX_ACTION="${MAX_ACTION:-1.0}"
POLICY_PROFILE="${POLICY_PROFILE:-lop}"
ACTOR_MEAN_SCALE="${ACTOR_MEAN_SCALE:-1.0}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:--2.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:--1.4}"

case "${variant_idx}" in
  0) RUN_KIND="lop_raw"; OBS_NORMALIZE=0 ;;
  1) RUN_KIND="lop_obsnorm"; OBS_NORMALIZE=1 ;;
  *) echo "Invalid variant index ${variant_idx}." >&2; exit 1 ;;
esac

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
GROUP_NAME="slippery_ant_lop_protocol_${POLICY_PROFILE}_${HEADS_OPTIMIZER}_${BACKEND}_${GROUP_ID}"
EXP_NAME="ppo_slippery_ant_lop_protocol_${POLICY_PROFILE}_${RUN_KIND}_${BACKEND}_s${SEED}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_ant,lop_protocol,policy_${POLICY_PROFILE},network_arch_${NETWORK_ARCH},brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},schedule_seed_${SLIPPERY_SCHEDULE_SEED},${RUN_KIND},change_every_${CHANGE_EVERY},total_${TOTAL_TIMESTEPS},action_repeat_${ACTION_REPEAT},actorcritic_${ACTOR_CRITIC_ACTIVATION},baseopt_${BASE_OPTIMIZER},headopt_${HEADS_OPTIMIZER},weight_decay_${WEIGHT_DECAY},batch_${ROLLOUT_ENV_STEPS},num_steps_${NUM_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},max_action_${MAX_ACTION},clip_eps_0.2,ent_0.01,vf_1,no_reward_norm,no_crate,lr_${LEARNING_RATE},repo_ppo"
if (( OBS_NORMALIZE == 1 )); then
  export WANDB_TAGS="${WANDB_TAGS},obs_norm"
else
  export WANDB_TAGS="${WANDB_TAGS},no_obs_norm"
fi

echo "Running SlipperyAnt lop protocol pilot group=${GROUP_NAME}"
echo "Config: task_id=${TASK_ID} run_kind=${RUN_KIND} env=${ENV_NAME} backend=${BACKEND} seed=${SEED} schedule_seed=${SLIPPERY_SCHEDULE_SEED} total_timesteps=${TOTAL_TIMESTEPS} change_every=${CHANGE_EVERY} action_repeat=${ACTION_REPEAT}"
echo "PPO: n_envs=${N_ENVS} rollout_batch=${ROLLOUT_ENV_STEPS} num_steps=${NUM_STEPS} minibatch_size=${MINIBATCH_SIZE} lr=${LEARNING_RATE} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} obs_normalize=${OBS_NORMALIZE} max_action=${MAX_ACTION} policy_profile=${POLICY_PROFILE} network_arch=${NETWORK_ARCH}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate "${LEARNING_RATE}"
  --adam-eps "${ADAM_EPS}"
  --base-optimizer "${BASE_OPTIMIZER}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-steps "${NUM_STEPS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --update-epochs "${UPDATE_EPOCHS}"
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.2
  --ent-coef 0.01
  --vf-coef 1.0
  --max-grad-norm 1000000000
  --max-action "${MAX_ACTION}"
  --actor-critic-activation "${ACTOR_CRITIC_ACTIVATION}"
  --network-arch "${NETWORK_ARCH}"
  --heads-optimizer "${HEADS_OPTIMIZER}"
  --seed "${SEED}"
  --log-interval 1
  --action-repeat "${ACTION_REPEAT}"
  --slippery-ant
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
  --exp-name "${EXP_NAME}"
)

if (( OBS_NORMALIZE == 1 )); then
  COMMON_ARGS+=(--obs-normalize --obs-norm-clip "${OBS_NORM_CLIP}")
fi

case "${POLICY_PROFILE}" in
  lop)
    ;;
  stable)
    COMMON_ARGS+=(
      --actor-mean-tanh
      --actor-mean-scale "${ACTOR_MEAN_SCALE}"
      --bounded-global-logstd
      --actor-logstd-init "${ACTOR_LOGSTD_INIT}"
      --actor-logstd-min "${ACTOR_LOGSTD_MIN}"
      --actor-logstd-max "${ACTOR_LOGSTD_MAX}"
    )
    ;;
  *)
    echo "Unsupported POLICY_PROFILE=${POLICY_PROFILE}. Expected lop or stable." >&2
    exit 1
    ;;
esac

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" "${COMMON_ARGS[@]}"
