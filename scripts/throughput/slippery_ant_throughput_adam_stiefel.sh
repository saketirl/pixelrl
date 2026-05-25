#!/bin/bash
#SBATCH --job-name=slip-ant-throughput
#SBATCH --output=slurm_logs/slippery_ant_throughput_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%10

set -euo pipefail

# Single-process throughput sweep for SlipperyAnt state PPO.
# Each array task runs one vectorization/update-geometry config for either Adam
# or Stiefel heads. The target is high SPS while still reaching roughly 2000
# episodic return by the first 5M-step checkpoint.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-4999936}"
NUM_PHASES="${4:-1}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-4999936}"
SEED="${6:-${SEED:-0}}"
ACTOR_CRITIC_ACTIVATION="${7:-${ACTOR_CRITIC_ACTIVATION:-swish}}"
BACKEND="${8:-${BACKEND:-spring}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

ENV_NAME="ant"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-${SEED}}"
SCHEDULE_TAG="default_then_csv_schedule"

CONFIG_NAMES=(
  baseline
  safe_vectorized
  balanced_throughput
  large_rollout
  fewer_updates
)
CONFIG_N_ENVS=(128 256 256 256 256)
CONFIG_NUM_STEPS=(10 10 20 40 20)
CONFIG_NUM_MINIBATCHES=(32 32 32 32 16)
CONFIG_UPDATE_EPOCHS=(4 4 4 4 4)
OPTIMIZERS=(adam stiefel)

NUM_CONFIGS=${#CONFIG_NAMES[@]}
NUM_OPTS=${#OPTIMIZERS[@]}
NUM_TASKS=$((NUM_CONFIGS * NUM_OPTS))

if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_TASKS - 1))." >&2
  exit 1
fi

CONFIG_IDX=$((TASK_ID % NUM_CONFIGS))
OPT_IDX=$((TASK_ID / NUM_CONFIGS))

CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"
N_ENVS="${CONFIG_N_ENVS[$CONFIG_IDX]}"
NUM_STEPS="${CONFIG_NUM_STEPS[$CONFIG_IDX]}"
NUM_MINIBATCHES="${CONFIG_NUM_MINIBATCHES[$CONFIG_IDX]}"
UPDATE_EPOCHS="${CONFIG_UPDATE_EPOCHS[$CONFIG_IDX]}"
HEADS_OPTIMIZER="${OPTIMIZERS[$OPT_IDX]}"

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
GROUP_NAME="slippery_ant_throughput_${ACTOR_CRITIC_ACTIVATION}_adam_stiefel_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_ant_wrapper,throughput_sweep,opt_${HEADS_OPTIMIZER},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},nenvs_${N_ENVS},steps_${NUM_STEPS},batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},minibatch_size_${MINIBATCH_SIZE},activation_${ACTOR_CRITIC_ACTIVATION},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${CHANGE_EVERY},phases_${EFFECTIVE_PHASES},action_repeat_${ACTION_REPEAT},wrapper_faithful,${SCHEDULE_TAG},reward_norm,vclip,lr1e4,target_5m_return_2000"

EXP_NAME="ppo_slippery_throughput_${ENV_NAME}_${HEADS_OPTIMIZER}_${ACTOR_CRITIC_ACTIVATION}_env${N_ENVS}_steps${NUM_STEPS}_mb${NUM_MINIBATCHES}_ep${UPDATE_EPOCHS}_s${SEED}"

echo "Running SlipperyAnt throughput task=${TASK_ID}/${NUM_TASKS} group=${GROUP_NAME}"
echo "Config: name=${CONFIG_NAME} opt=${HEADS_OPTIMIZER} env=${ENV_NAME} backend=${BACKEND} seed=${SEED} activation=${ACTOR_CRITIC_ACTIVATION} action_repeat=${ACTION_REPEAT} phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} change_every_per_env_policy_steps=${CHANGE_EVERY_POLICY_STEPS} change_every_wrapper_steps=${CHANGE_EVERY} schedule_seed=${SLIPPERY_SCHEDULE_SEED} total_timesteps=${TOTAL_TIMESTEPS} per_env_steps=${EFFECTIVE_ENV_STEPS_PER_ENV}"
echo "PPO: n_envs=${N_ENVS} num_steps=${NUM_STEPS} batch=${ROLLOUT_ENV_STEPS} minibatch_size=${MINIBATCH_SIZE} lr=1e-4 epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} clip_eps=0.1 reward_normalize=true clip_vloss=true max_grad_norm=0.05"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate 1e-4
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
  --slippery-ant
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" \
  "${COMMON_ARGS[@]}"
