#!/bin/bash
#SBATCH --job-name=cl-brax-ant-dohare
#SBATCH --output=slurm_logs/cl_brax_ant_dohare_paper_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# State-observation Brax Ant launcher with PPO hyperparameters matched as
# closely as practical to Dohare 2024 SlipperyAnt-v3.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
SWITCH_EVERY_ENV_STEPS="${3:-2000896}"
NUM_TASKS="${4:-10}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
SEED="${6:-${SEED:-0}}"
ACTOR_CRITIC_ACTIVATION="${7:-${ACTOR_CRITIC_ACTIVATION:-relu}}"
BACKEND="${8:-${BACKEND:-generalized}}"
HEADS_OPTIMIZER="${9:-${HEADS_OPTIMIZER:-adam}}"

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

ENV_NAME="ant"
CONFIG_BASENAME="ant_friction_dohare"
N_ENVS=128
NUM_STEPS=16

if (( NUM_TASKS < 1 )); then
  echo "NUM_TASKS must be >= 1; got ${NUM_TASKS}." >&2
  exit 1
fi

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
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
GROUP_NAME="cl_brax_${ENV_NAME}_dohare_paper_${BACKEND}_${EFFECTIVE_NUM_TASKS}tasks_${ACTOR_CRITIC_ACTIVATION}_headopt_${HEADS_OPTIMIZER}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,continual_dynamics,brax_state,${ENV_NAME},backend_${BACKEND},seed_${SEED},tasks_${EFFECTIVE_NUM_TASKS},switch_${SWITCH_EVERY_ENV_STEPS},actorcritic_${ACTOR_CRITIC_ACTIVATION},headopt_${HEADS_OPTIMIZER},dohare_friction,shared_scalar,log_uniform,dohare_ppo_like,batch_2048,epochs_10,no_reward_norm,no_vclip"

EXP_NAME="ppo_brax_cl_${ENV_NAME}_friction_dohare_paper_${BACKEND}_${EFFECTIVE_NUM_TASKS}tasks_${ACTOR_CRITIC_ACTIVATION}_headopt_${HEADS_OPTIMIZER}_s${SEED}"

BASE_CONFIG="${REPO_ROOT}/configs/continual/${CONFIG_BASENAME}.yaml"
RUN_CONFIG="${REPO_ROOT}/slurm_logs/continual_brax_${CONFIG_BASENAME}_paper_${BACKEND}_${EFFECTIVE_NUM_TASKS}tasks_${GROUP_ID}.yaml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing continual dynamics config: ${BASE_CONFIG}" >&2
  exit 1
fi
sed \
  -e "s/^backend:.*/backend: ${BACKEND}/" \
  -e "s/^switch_every_env_steps:.*/switch_every_env_steps: ${SWITCH_EVERY_ENV_STEPS}/" \
  "${BASE_CONFIG}" > "${RUN_CONFIG}"

echo "Running Brax ${ENV_NAME} Dohare-matched continual friction group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} tasks=${EFFECTIVE_NUM_TASKS} switch_every_env_steps=${SWITCH_EVERY_ENV_STEPS} total_timesteps=${TOTAL_TIMESTEPS} actor_critic_activation=${ACTOR_CRITIC_ACTIVATION} heads_optimizer=${HEADS_OPTIMIZER}"
echo "PPO: batch=${ROLLOUT_ENV_STEPS} lr=1e-4 epochs=10 minibatches=16 clip_eps=0.2 reward_normalize=false clip_vloss=false"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"
echo "Continual config: ${RUN_CONFIG}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate 1e-4
  --adam-eps 1e-8
  --num-steps "${NUM_STEPS}"
  --num-minibatches 16
  --update-epochs 10
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.2
  --no-clip-vloss
  --ent-coef 0.0
  --vf-coef 2.0
  --max-grad-norm 1000000000
  --actor-critic-activation "${ACTOR_CRITIC_ACTIVATION}"
  --heads-optimizer "${HEADS_OPTIMIZER}"
  --heads-stiefel-lr 0.001
  --stiefel-dual-lr 0.01
  --stiefel-dual-steps 5
  --stiefel-msign-steps 5
  --actor-stiefel-max-grad-norm 100
  --critic-stiefel-max-grad-norm 1
  --no-reward-normalize
  --seed "${SEED}"
  --log-interval 1
  --action-repeat 1
  --continual-dynamics-config "${RUN_CONFIG}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" \
  "${COMMON_ARGS[@]}"
