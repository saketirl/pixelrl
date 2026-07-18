#!/bin/bash
#SBATCH --job-name=ant-goal-rerun9
#SBATCH --output=/home/guests/saket/pixelenvs/pixelrl/logs/ant_goal_bestreward_rerun9_%A_%a.out
#SBATCH --error=/home/guests/saket/pixelenvs/pixelrl/logs/ant_goal_bestreward_rerun9_%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

set -euo pipefail

# Reruns for 9 terminated ant_goal runs.
# Reward: 20 * progress - 0.1 * dist + healthy - ctrl + 50 * success
#
# Task 0: crate_cnn + crate + stiefel, seed 0
# Task 1: crate_cnn + crate + stiefel, seed 1
# Task 2: crate_cnn + crate + stiefel, seed 2
# Task 3: crate_cnn + crate + stiefel, seed 3
# Task 4: crate_cnn + crate + adam,    seed 3
# Task 5: cnn       + mlp   + stiefel, seed 0
# Task 6: cnn       + mlp   + stiefel, seed 1
# Task 7: cnn       + mlp   + stiefel, seed 2
# Task 8: cnn       + mlp   + stiefel, seed 3

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-saketirl}"
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
uv run wandb login

# Task → (encoder, head_arch, optimizer, seed)
ENCODER_BY_TASK=(crate_cnn crate_cnn crate_cnn crate_cnn crate_cnn cnn cnn cnn cnn)
HEADARCH_BY_TASK=(crate    crate    crate    crate    crate    mlp mlp mlp mlp)
OPTIMIZER_BY_TASK=(stiefel stiefel  stiefel  stiefel  adam     stiefel stiefel stiefel stiefel)
SEED_BY_TASK=(0 1 2 3 3 0 1 2 3)

if (( TASK_ID < 0 || TASK_ID >= ${#SEED_BY_TASK[@]} )); then
  echo "Invalid TASK_ID=${TASK_ID}" >&2
  exit 1
fi

ENCODER_TYPE="${ENCODER_BY_TASK[$TASK_ID]}"
HEAD_ARCH="${HEADARCH_BY_TASK[$TASK_ID]}"
HEADS_OPTIMIZER="${OPTIMIZER_BY_TASK[$TASK_ID]}"
SEED="${SEED_BY_TASK[$TASK_ID]}"

CONFIG_NAME="progress20_dist0p1_success50"
PROGRESS_SCALE="20"
DISTANCE_SCALE="0.1"
SUCCESS_REWARD="50"

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ENCODER_SCALE="0.5"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="ant_goal_${ENCODER_TYPE}_${HEAD_ARCH}heads_${CONFIG_NAME}_${HEADS_OPTIMIZER}_rerun9_ar${ACTION_REPEAT}_${GROUP_ID}"
GIF_DIR="${REPO_ROOT}/outputs/goal_reward_sweep/${GROUP_NAME}"
mkdir -p "${GIF_DIR}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_ENTITY="${WANDB_ENTITY}"

EXP_NAME="ppo_goal_reward_${CONFIG_NAME}_${ENCODER_TYPE}_${HEAD_ARCH}heads_${HEADS_OPTIMIZER}_ar${ACTION_REPEAT}_ant_goal_b${BACKEND}_s${SEED}"
TAGS="goal,pixel_goal,ant_goal,reward_focus,${CONFIG_NAME},progress_${PROGRESS_SCALE},distance_${DISTANCE_SCALE},success_${SUCCESS_REWARD},action_repeat_${ACTION_REPEAT},env_ant_goal,backend_${BACKEND},seed_${SEED},encoder_${ENCODER_TYPE},headarch_${HEAD_ARCH},opt_${HEADS_OPTIMIZER},episode_length_1000,mem_fraction_${JAX_MEM_FRACTION},gif_every_1m,rerun"

echo "encoder=${ENCODER_TYPE} heads=${HEAD_ARCH} optimizer=${HEADS_OPTIMIZER} seed=${SEED}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "GIF dir=${GIF_DIR}"
echo "Exp: ${EXP_NAME}"

cd "${REPO_ROOT}"

COMMON_ARGS=(
  --env-name ant_goal
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
  --seed "${SEED}"
  --heads-optimizer "${HEADS_OPTIMIZER}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${ENCODER_TYPE}" == "crate_cnn" ]]; then
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
else
  ARCH_ARGS=(
    --encoder-type cnn
    --encoder-lr 3e-4
    --heads-adam-lr 3e-4
    --heads-stiefel-lr 0.001
    --max-grad-norm 0.05
    --encoder-tanh-scale "${ENCODER_SCALE}"
  )
fi

WANDB_TAGS="${TAGS}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}"
