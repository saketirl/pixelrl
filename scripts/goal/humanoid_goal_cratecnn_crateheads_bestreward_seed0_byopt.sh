#!/bin/bash
#SBATCH --job-name=hum-goal-cratecnn-s0
#SBATCH --output=slurm_logs/humanoid_goal_cratecnn_crateheads_bestreward_seed0_byopt_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

set -euo pipefail

# Focused humanoid_goal CRATE-CNN + CRATE-head smoke run.
# Uses the current fixed humanoid goal position from pixelbrax/tasks/goal.py.
#
# Array task 0: Adam heads, seed 0 on one GPU.
# Array task 1: Stiefel heads, seed 0 on one GPU.

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
BACKEND="${4:-spring}"
ACTION_REPEAT="${5:-4}"
JAX_MEM_FRACTION="${6:-0.22}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

SEED=0
OPTIMIZER_BY_TASK=(adam stiefel)
if (( TASK_ID < 0 || TASK_ID >= ${#OPTIMIZER_BY_TASK[@]} )); then
  echo "Invalid TASK_ID=${TASK_ID}" >&2
  exit 1
fi
HEADS_OPTIMIZER="${OPTIMIZER_BY_TASK[$TASK_ID]}"

CONFIG_NAME="progress20_dist0p1_success50"
PROGRESS_SCALE="20"
DISTANCE_SCALE="0.1"
SUCCESS_REWARD="50"

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="humanoid_goal_cratecnn_crateheads_${CONFIG_NAME}_${HEADS_OPTIMIZER}_seed0_ar${ACTION_REPEAT}_${GROUP_ID}"
GIF_DIR="${REPO_ROOT}/outputs/goal_reward_sweep/${GROUP_NAME}"
mkdir -p "${GIF_DIR}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

COMMON_ARGS=(
  --env-name humanoid_goal
  --backend "${BACKEND}"
  --n-envs 128
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --hw 84
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --log-interval 1
  --frame-stack 4
  --action-repeat "${ACTION_REPEAT}"
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
  --goal-progress-reward-scale "${PROGRESS_SCALE}"
  --goal-distance-reward-scale "${DISTANCE_SCALE}"
  --goal-success-reward "${SUCCESS_REWARD}"
  --save-rollout-gif
  --rollout-gif-step 1000000
  --rollout-gif-dir "${GIF_DIR}"
  --rollout-gif-steps 250
  --rollout-gif-fps 20
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
  --use-crate-head
  --crate-step-size "${CRATE_STEP_SIZE}"
  --sigreg-mode off
)

EXP_NAME="ppo_goal_reward_${CONFIG_NAME}_cratecnn_crateheads_${HEADS_OPTIMIZER}_ar${ACTION_REPEAT}_humanoid_goal_b${BACKEND}_s${SEED}_goal2p5"
TAGS="goal,pixel_goal,humanoid_goal,reward_focus,${CONFIG_NAME},goal_dist_2p5,progress_${PROGRESS_SCALE},distance_${DISTANCE_SCALE},success_${SUCCESS_REWARD},action_repeat_${ACTION_REPEAT},env_humanoid_goal,backend_${BACKEND},seed_${SEED},encoder_crate_cnn,headarch_crate,opt_${HEADS_OPTIMIZER},one_seed,one_optimizer_per_gpu,mem_fraction_${JAX_MEM_FRACTION},gif_every_1m"

echo "Focused humanoid seed0 config=${CONFIG_NAME} optimizer=${HEADS_OPTIMIZER}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "GIF dir=${GIF_DIR}"
echo "Exp: ${EXP_NAME}"

cd "${REPO_ROOT}"

WANDB_TAGS="${TAGS}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  --seed "${SEED}" \
  --heads-optimizer "${HEADS_OPTIMIZER}" \
  --exp-name "${EXP_NAME}"
