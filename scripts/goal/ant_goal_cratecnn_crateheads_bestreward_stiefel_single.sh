#!/bin/bash
#SBATCH --job-name=ant-goal-cratecnn-stiefel
#SBATCH --output=slurm_logs/ant_goal_cratecnn_crateheads_bestreward_stiefel_single_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1

set -euo pipefail

# ant_goal with CRATE-CNN encoder + CRATE heads, Stiefel optimizer, single seed.
# One process per GPU.
# Reward: 20 * progress - 0.1 * dist + healthy - ctrl + 50 * success

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"
BACKEND="${5:-spring}"
ACTION_REPEAT="${6:-4}"
JAX_MEM_FRACTION="${7:-0.9}"

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

HEADS_OPTIMIZER="stiefel"

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

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="ant_goal_cratecnn_crateheads_${CONFIG_NAME}_${HEADS_OPTIMIZER}_s${SEED}_ar${ACTION_REPEAT}_${GROUP_ID}"
GIF_DIR="${REPO_ROOT}/outputs/goal_reward_sweep/${GROUP_NAME}"
mkdir -p "${GIF_DIR}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

EXP_NAME="ppo_goal_reward_${CONFIG_NAME}_cratecnn_crateheads_${HEADS_OPTIMIZER}_ar${ACTION_REPEAT}_ant_goal_b${BACKEND}_s${SEED}"
TAGS="goal,pixel_goal,ant_goal,reward_focus,${CONFIG_NAME},progress_${PROGRESS_SCALE},distance_${DISTANCE_SCALE},success_${SUCCESS_REWARD},action_repeat_${ACTION_REPEAT},env_ant_goal,backend_${BACKEND},seed_${SEED},encoder_crate_cnn,headarch_crate,opt_${HEADS_OPTIMIZER},single_seed,one_per_gpu,mem_fraction_${JAX_MEM_FRACTION},gif_every_1m,episode_length_1000"

echo "Config=${CONFIG_NAME} optimizer=${HEADS_OPTIMIZER} seed=${SEED}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "GIF dir=${GIF_DIR}"
echo "Exp: ${EXP_NAME}"

cd "${REPO_ROOT}"

WANDB_TAGS="${TAGS}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  --env-name ant_goal \
  --backend "${BACKEND}" \
  --n-envs 128 \
  --track \
  --wandb-project-name "${WANDB_PROJECT}" \
  --hw 84 \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --num-steps 10 \
  --num-minibatches 32 \
  --update-epochs 4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --seed "${SEED}" \
  --log-interval 1 \
  --frame-stack 4 \
  --action-repeat "${ACTION_REPEAT}" \
  --anneal-lr \
  --stiefel-dual-lr 0.01 \
  --stiefel-dual-steps 5 \
  --actor-stiefel-max-grad-norm 100 \
  --critic-stiefel-max-grad-norm 1 \
  --actor-mean-tanh \
  --actor-mean-scale "${ACTOR_MEAN_SCALE}" \
  --bounded-global-logstd \
  --actor-logstd-init="${ACTOR_LOGSTD_INIT}" \
  --actor-logstd-min="${ACTOR_LOGSTD_MIN}" \
  --actor-logstd-max="${ACTOR_LOGSTD_MAX}" \
  --goal-progress-reward-scale "${PROGRESS_SCALE}" \
  --goal-distance-reward-scale "${DISTANCE_SCALE}" \
  --goal-success-reward "${SUCCESS_REWARD}" \
  --save-rollout-gif \
  --rollout-gif-step 1000000 \
  --rollout-gif-dir "${GIF_DIR}" \
  --rollout-gif-steps 250 \
  --rollout-gif-fps 20 \
  --encoder-type crate_cnn \
  --encoder-lr 3e-4 \
  --heads-adam-lr 3e-4 \
  --heads-stiefel-lr 0.001 \
  --max-grad-norm 0.05 \
  --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}" \
  --use-crate-head \
  --crate-step-size "${CRATE_STEP_SIZE}" \
  --sigreg-mode off \
  --heads-optimizer "${HEADS_OPTIMIZER}" \
  --exp-name "${EXP_NAME}"
