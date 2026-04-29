#!/bin/bash
#SBATCH --job-name=cl-pixel-dohare-plain
#SBATCH --output=slurm_logs/cl_pixelbrax_dohare_plainheads_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%6

set -euo pipefail

# Continual PixelBrax launcher for Dohare-style log-uniform friction changes.
# Runs one seed for halfcheetah, walker2d, and hopper with the plain-head
# CRATECNN PPO configs from scripts/staging, sweeping Stiefel and Adam heads.

WANDB_PROJECT="${1:-continual_pixelbrax}"
WANDB_ENTITY="${2:-rl-power}"
SWITCH_EVERY_ENV_STEPS="${3:-1999360}"
NUM_TASKS="${4:-10}"
TOTAL_TIMESTEPS_OVERRIDE="${5:-}"
SEED="${6:-${SEED:-0}}"
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

ENVS=(halfcheetah walker2d hopper)
CONFIG_BASENAMES=(halfcheetah_friction_dohare walker2d_friction_dohare hopper_friction_dohare)
OPT_CONDITIONS=(stiefel adam)
BACKEND="generalized"

NUM_ENVS=${#ENVS[@]}
NUM_CONFIGS=${#CONFIG_BASENAMES[@]}
NUM_OPTS=${#OPT_CONDITIONS[@]}
NUM_RUNS=$((NUM_ENVS * NUM_OPTS))

if (( NUM_ENVS != NUM_CONFIGS )); then
  echo "ENVS/CONFIG_BASENAMES length mismatch: ${NUM_ENVS} vs ${NUM_CONFIGS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_RUNS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_RUNS - 1))." >&2
  exit 1
fi

if (( NUM_TASKS < 1 )); then
  echo "NUM_TASKS must be >= 1; got ${NUM_TASKS}." >&2
  exit 1
fi

N_ENVS=128
NUM_STEPS=10
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

OPT_IDX=$((TASK_ID % NUM_OPTS))
ENV_IDX=$((TASK_ID / NUM_OPTS))

ENV_NAME="${ENVS[$ENV_IDX]}"
CONFIG_BASENAME="${CONFIG_BASENAMES[$ENV_IDX]}"
OPT_CONDITION="${OPT_CONDITIONS[$OPT_IDX]}"

ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cl_pixelbrax_dohare_friction_plainheads_${BACKEND}_${EFFECTIVE_NUM_TASKS}tasks_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,continual_dynamics,pixelbrax,dohare_friction,shared_scalar,log_uniform,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},tasks_${EFFECTIVE_NUM_TASKS},switch_${SWITCH_EVERY_ENV_STEPS},encoder_crate_cnn,headarch_plain,opt_${OPT_CONDITION},actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},bounded_global_logstd,actor_logstd_init_${ACTOR_LOGSTD_INIT},actor_logstd_min_${ACTOR_LOGSTD_MIN},actor_logstd_max_${ACTOR_LOGSTD_MAX},sigreg_off"

EXP_NAME="ppo_pixelbrax_cl_dohare_friction_plainheads_${OPT_CONDITION}_${ENV_NAME}_${BACKEND}_${EFFECTIVE_NUM_TASKS}tasks_s${SEED}_t${TASK_ID}"

BASE_CONFIG="${REPO_ROOT}/configs/continual/${CONFIG_BASENAME}.yaml"
RUN_CONFIG="${REPO_ROOT}/slurm_logs/continual_pixelbrax_${CONFIG_BASENAME}_${BACKEND}_${EFFECTIVE_NUM_TASKS}tasks_${OPT_CONDITION}_s${SEED}_${GROUP_ID}_${TASK_ID}.yaml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing continual dynamics config: ${BASE_CONFIG}" >&2
  exit 1
fi
sed \
  -e "s/^backend:.*/backend: ${BACKEND}/" \
  -e "s/^switch_every_env_steps:.*/switch_every_env_steps: ${SWITCH_EVERY_ENV_STEPS}/" \
  "${BASE_CONFIG}" > "${RUN_CONFIG}"

echo "Running TASK_ID=${TASK_ID}/${NUM_RUNS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} opt=${OPT_CONDITION} tasks=${EFFECTIVE_NUM_TASKS} switch_every_env_steps=${SWITCH_EVERY_ENV_STEPS} total_timesteps=${TOTAL_TIMESTEPS}"
echo "PPO: encoder=crate_cnn headarch=plain heads_optimizer=${OPT_CONDITION} n_envs=${N_ENVS} num_steps=${NUM_STEPS} batch=${ROLLOUT_ENV_STEPS} anneal_lr=true"
echo "Policy bounds: actor_mean_tanh=true actor_mean_scale=${ACTOR_MEAN_SCALE} bounded_global_logstd=true actor_logstd_init=${ACTOR_LOGSTD_INIT} actor_logstd_min=${ACTOR_LOGSTD_MIN} actor_logstd_max=${ACTOR_LOGSTD_MAX}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"
echo "Continual config: ${RUN_CONFIG}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs "${N_ENVS}"
  --track
  --debug-repr
  --wandb-project-name "${WANDB_PROJECT}"
  --hw 84
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-steps "${NUM_STEPS}"
  --num-minibatches 32
  --update-epochs 4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --anneal-lr
  --stiefel-dual-lr 0.01
  --stiefel-dual-steps 5
  --actor-stiefel-max-grad-norm 100
  --critic-stiefel-max-grad-norm 1
  --actor-mean-tanh
  --actor-mean-scale "${ACTOR_MEAN_SCALE}"
  --bounded-global-logstd
  --actor-logstd-init="${ACTOR_LOGSTD_INIT}"
  --actor-logstd-min="${ACTOR_LOGSTD_MIN}"
  --actor-logstd-max="${ACTOR_LOGSTD_MAX}"
  --continual-dynamics-config "${RUN_CONFIG}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

ARCH_ARGS=(
  --encoder-type crate_cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-stiefel-lr 0.001
  --max-grad-norm 0.05
  --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
  --sigreg-mode off
)

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  --heads-optimizer "${OPT_CONDITION}"
