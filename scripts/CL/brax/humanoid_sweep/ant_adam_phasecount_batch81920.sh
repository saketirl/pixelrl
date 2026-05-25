#!/bin/bash
#SBATCH --job-name=ant-adam-ph81920
#SBATCH --output=slurm_logs/slippery_ant_adam_phasecount81920_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Fixed-budget phase-count control for the large-batch Adam break candidate.
# Batch is fixed at 81920 and total steps are kept near 5M while task count
# varies over 1, 5, 10, and 20 phases.

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

ENV_NAME="ant"
BACKEND="${BACKEND:-spring}"
ACTOR_CRITIC_ACTIVATION="${ACTOR_CRITIC_ACTIVATION:-relu}"
BASE_OPTIMIZER="${BASE_OPTIMIZER:-adam}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HEADS_OPTIMIZER="adam"
N_ENVS="${N_ENVS:-128}"
NUM_STEPS=640
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAM_EPS="${ADAM_EPS:-1e-5}"
ACTOR_MEAN_SCALE="${ACTOR_MEAN_SCALE:-1.0}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:--2.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:--1.4}"
CRATE_STEP_SIZE="${CRATE_STEP_SIZE:-0.1}"
NETWORK_CRATE_STEP_SIZE="${NETWORK_CRATE_STEP_SIZE:-0.1}"
OBS_NORM_CLIP="${OBS_NORM_CLIP:-10.0}"

seed_idx=$((TASK_ID % 2))
phase_idx=$((TASK_ID / 2))
SEED="${seed_idx}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-${SEED}}"

case "${phase_idx}" in
  0) RUN_KIND="1phase"; NUM_PHASES=1; PHASE_EVERY_ENV_STEPS=4997120 ;;
  1) RUN_KIND="5phase"; NUM_PHASES=5; PHASE_EVERY_ENV_STEPS=999936 ;;
  2) RUN_KIND="10phase"; NUM_PHASES=10; PHASE_EVERY_ENV_STEPS=499968 ;;
  3) RUN_KIND="20phase"; NUM_PHASES=20; PHASE_EVERY_ENV_STEPS=249984 ;;
  *) echo "Invalid phase index ${phase_idx}." >&2; exit 1 ;;
esac

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
CHANGE_EVERY_POLICY_STEPS=$((PHASE_EVERY_ENV_STEPS / N_ENVS))
CHANGE_EVERY=$((CHANGE_EVERY_POLICY_STEPS * ACTION_REPEAT))
TOTAL_TIMESTEPS=$((PHASE_EVERY_ENV_STEPS * NUM_PHASES))
if (( TOTAL_TIMESTEPS % ROLLOUT_ENV_STEPS != 0 )); then
  ALIGNED_TOTAL_TIMESTEPS=$(((TOTAL_TIMESTEPS / ROLLOUT_ENV_STEPS) * ROLLOUT_ENV_STEPS))
  echo "Aligning TOTAL_TIMESTEPS from ${TOTAL_TIMESTEPS} down to ${ALIGNED_TOTAL_TIMESTEPS}."
  TOTAL_TIMESTEPS="${ALIGNED_TOTAL_TIMESTEPS}"
fi

MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))
GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="slippery_ant_adam_phasecount_batch81920_lr1e_4_${BACKEND}_${GROUP_ID}"
EXP_NAME="ppo_slippery_ant_adam_phasecount81920_${RUN_KIND}_${BACKEND}_s${SEED}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_ant_adam_phasecount,batch_81920,slippery_ant_wrapper,humanoid_ppo_recipe,brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},schedule_seed_${SLIPPERY_SCHEDULE_SEED},${RUN_KIND},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${CHANGE_EVERY},requested_phases_${NUM_PHASES},action_repeat_${ACTION_REPEAT},actorcritic_${ACTOR_CRITIC_ACTIVATION},baseopt_${BASE_OPTIMIZER},weight_decay_${WEIGHT_DECAY},headopt_${HEADS_OPTIMIZER},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},reward_norm,vclip,obs_norm,crate_network,crate_head,bounded_global_logstd,actor_mean_tanh,lr_${LEARNING_RATE},repo_ppo"

echo "Running SlipperyAnt Adam phase-count batch81920 group=${GROUP_NAME}"
echo "Config: task_id=${TASK_ID} run_kind=${RUN_KIND} env=${ENV_NAME} backend=${BACKEND} seed=${SEED} phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} requested_phases=${NUM_PHASES} change_every_wrapper_steps=${CHANGE_EVERY} total_timesteps=${TOTAL_TIMESTEPS}"
echo "PPO: rollout_batch=${ROLLOUT_ENV_STEPS} num_steps=${NUM_STEPS} minibatch_size=${MINIBATCH_SIZE} lr=${LEARNING_RATE} base_optimizer=${BASE_OPTIMIZER} weight_decay=${WEIGHT_DECAY} anneal_lr=true epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES}"

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
  --anneal-lr
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
  --actor-mean-tanh
  --actor-mean-scale "${ACTOR_MEAN_SCALE}"
  --bounded-global-logstd
  --actor-logstd-init "${ACTOR_LOGSTD_INIT}"
  --actor-logstd-min "${ACTOR_LOGSTD_MIN}"
  --actor-logstd-max "${ACTOR_LOGSTD_MAX}"
  --obs-normalize
  --obs-norm-clip "${OBS_NORM_CLIP}"
  --use-crate-network
  --network-crate-step-size "${NETWORK_CRATE_STEP_SIZE}"
  --use-crate-head
  --crate-step-size "${CRATE_STEP_SIZE}"
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

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" "${COMMON_ARGS[@]}"
