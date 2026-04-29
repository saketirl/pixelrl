#!/bin/bash
#SBATCH --job-name=slippery-ant-wrapper
#SBATCH --output=slurm_logs/slippery_ant_wrapper_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# State-observation Brax Ant launcher for configs/continual/slippery_ant_wrapper.py.
# This uses the wrapper's own per-env timestep friction schedule, starting with
# the environment's default friction before traversing the CSV-backed schedule
# selected by the seed row in configs/continual/frictions.csv.
#  - total experiment length: 10,000,000 * num_envs global steps
#  - if there are 5 tasks split evenly, task switches happen every 2,000,000 * num_envs global steps


WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-4999936}"
NUM_PHASES="${4:-20}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
SEED="${6:-${SEED:-0}}"
ACTOR_CRITIC_ACTIVATION="${7:-${ACTOR_CRITIC_ACTIVATION:-relu}}"
BACKEND="${8:-${BACKEND:-spring}}"
HEADS_OPTIMIZER="${9:-${HEADS_OPTIMIZER:-adam}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_slippery_brax.py" ]]; then
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
N_ENVS="${N_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SLIPPERY_SCHEDULE_SEED="${SLIPPERY_SCHEDULE_SEED:-${SEED}}"
DEFAULT_NUM_PHASES=20
SCHEDULE_DESC="default_then_csv"
SCHEDULE_TAG="default_then_csv_schedule"
if [[ -z "${NUM_PHASES}" ]]; then
  NUM_PHASES="${DEFAULT_NUM_PHASES}"
fi

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
DISTINCT_PHASES="${EFFECTIVE_PHASES}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="slippery_ant_wrapper_${BACKEND}_${DISTINCT_PHASES}phases_${ACTOR_CRITIC_ACTIVATION}_headopt_${HEADS_OPTIMIZER}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_ant_wrapper,brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},phase_every_${PHASE_EVERY_ENV_STEPS},change_every_per_env_${CHANGE_EVERY},phases_${DISTINCT_PHASES},action_repeat_${ACTION_REPEAT},actorcritic_${ACTOR_CRITIC_ACTIVATION},headopt_${HEADS_OPTIMIZER},wrapper_faithful,${SCHEDULE_TAG},batch_${ROLLOUT_ENV_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},reward_norm,vclip,repo_ppo"

EXP_NAME="ppo_slippery_brax_${ENV_NAME}_${BACKEND}_${DISTINCT_PHASES}phases_${ACTOR_CRITIC_ACTIVATION}_headopt_${HEADS_OPTIMIZER}_s${SEED}"

echo "Running SlipperyAnt wrapper group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} action_repeat=${ACTION_REPEAT} phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} change_every_per_env_policy_steps=${CHANGE_EVERY_POLICY_STEPS} change_every_wrapper_steps=${CHANGE_EVERY} schedule=${SCHEDULE_DESC} schedule_seed=${SLIPPERY_SCHEDULE_SEED} total_timesteps=${TOTAL_TIMESTEPS} per_env_steps=${EFFECTIVE_ENV_STEPS_PER_ENV}"
echo "PPO: batch=${ROLLOUT_ENV_STEPS} lr=3e-4 epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} clip_eps=0.1 reward_normalize=true clip_vloss=true max_grad_norm=0.05"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate 3e-4
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
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_slippery_brax.py" \
  "${COMMON_ARGS[@]}"
