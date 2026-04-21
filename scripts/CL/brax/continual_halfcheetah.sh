#!/bin/bash
#SBATCH --job-name=cl-brax-halfcheetah
#SBATCH --output=slurm_logs/cl_brax_halfcheetah_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# State-observation Brax continual HalfCheetah launcher. This uses the stronger
# HalfCheetah dynamics schedule to test whether task switches force nontrivial
# policy adaptation before adding pixels.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
SWITCH_EVERY_ENV_STEPS="${3:-1999360}"
NUM_TASKS="${4:-25}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
SEED="${6:-${SEED:-0}}"
ACTOR_CRITIC_ACTIVATION="${7:-${ACTOR_CRITIC_ACTIVATION:-swish}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

ENV_NAME="halfcheetah"
BACKEND="spring"

if (( NUM_TASKS < 1 )); then
  echo "NUM_TASKS must be >= 1; got ${NUM_TASKS}." >&2
  exit 1
fi

ROLLOUT_ENV_STEPS=$((128 * 10))
if (( SWITCH_EVERY_ENV_STEPS % ROLLOUT_ENV_STEPS != 0 )); then
  echo "SWITCH_EVERY_ENV_STEPS=${SWITCH_EVERY_ENV_STEPS} must be divisible by ${ROLLOUT_ENV_STEPS}." >&2
  exit 1
fi
if [[ -n "${TOTAL_TIMESTEPS_OVERRIDE}" ]]; then
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_OVERRIDE}"
else
  TOTAL_TIMESTEPS=$((SWITCH_EVERY_ENV_STEPS * NUM_TASKS))
fi
if (( TOTAL_TIMESTEPS % ROLLOUT_ENV_STEPS != 0 )); then
  echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} must be divisible by ${ROLLOUT_ENV_STEPS}." >&2
  exit 1
fi
EFFECTIVE_NUM_TASKS=$(((TOTAL_TIMESTEPS + SWITCH_EVERY_ENV_STEPS - 1) / SWITCH_EVERY_ENV_STEPS))

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cl_brax_halfcheetah_${EFFECTIVE_NUM_TASKS}tasks_${ACTOR_CRITIC_ACTIVATION}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,continual_dynamics,brax_state,halfcheetah,backend_${BACKEND},seed_${SEED},tasks_${EFFECTIVE_NUM_TASKS},switch_${SWITCH_EVERY_ENV_STEPS},actorcritic_${ACTOR_CRITIC_ACTIVATION},constant_lr,state_ppo_debug,strong_dynamics,morphology,structured_asymmetric,log_uniform"

EXP_NAME="ppo_brax_cl_halfcheetah_${EFFECTIVE_NUM_TASKS}tasks_${ACTOR_CRITIC_ACTIVATION}_constantlr_s${SEED}"

BASE_CONFIG="${REPO_ROOT}/configs/continual/halfcheetah_dynamics.yaml"
RUN_CONFIG="${REPO_ROOT}/slurm_logs/continual_brax_halfcheetah_${EFFECTIVE_NUM_TASKS}tasks_${GROUP_ID}.yaml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing continual dynamics config: ${BASE_CONFIG}" >&2
  exit 1
fi
sed "s/^switch_every_env_steps:.*/switch_every_env_steps: ${SWITCH_EVERY_ENV_STEPS}/" "${BASE_CONFIG}" > "${RUN_CONFIG}"

echo "Running Brax HalfCheetah continual debug group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} tasks=${EFFECTIVE_NUM_TASKS} switch_every_env_steps=${SWITCH_EVERY_ENV_STEPS} total_timesteps=${TOTAL_TIMESTEPS} actor_critic_activation=${ACTOR_CRITIC_ACTIVATION}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"
echo "Continual config: ${RUN_CONFIG}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs 128
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate 3e-4
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --max-grad-norm 0.05
  --actor-critic-activation "${ACTOR_CRITIC_ACTIVATION}"
  --seed "${SEED}"
  --log-interval 1
  --action-repeat 4
  --continual-dynamics-config "${RUN_CONFIG}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" \
  "${COMMON_ARGS[@]}"
