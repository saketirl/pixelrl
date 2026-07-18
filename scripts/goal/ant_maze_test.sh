#!/bin/bash

set -euo pipefail

GPU_ID="${1:-0}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-saketirl}"
TOTAL_TIMESTEPS="${4:-10000000}"
SEED="${5:-0}"
BACKEND="${6:-spring}"
ACTION_REPEAT="${7:-4}"
JAX_MEM_FRACTION="${8:-0.9}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SCRIPT_REPO_ROOT}"

if [[ -d "${REPO_ROOT}/pixelbrax/brax/brax" ]]; then
  BRAX_PYTHONPATH="${REPO_ROOT}/pixelbrax/brax"
else
  BRAX_PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"
fi
export PYTHONPATH="${BRAX_PYTHONPATH}:${PYTHONPATH:-}"

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi
uv run wandb login

HEADS_OPTIMIZER="stiefel"

CONFIG_NAME="dist_healthy1_ctrl0p5_cellscale1"
CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_NAME="ant_u_maze_cratecnn_crateheads_${CONFIG_NAME}_${HEADS_OPTIMIZER}_s${SEED}_ar${ACTION_REPEAT}_test"
GIF_DIR="${REPO_ROOT}/outputs/goal_reward_sweep/${GROUP_NAME}"
mkdir -p "${GIF_DIR}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

EXP_NAME="ppo_maze_${CONFIG_NAME}_cratecnn_crateheads_${HEADS_OPTIMIZER}_ar${ACTION_REPEAT}_ant_u_maze_b${BACKEND}_s${SEED}"
TAGS="maze,pixel_maze,ant_u_maze,env_ant_u_maze,backend_${BACKEND},seed_${SEED},encoder_crate_cnn,headarch_crate,opt_${HEADS_OPTIMIZER},healthy_reward_1,ctrl_0p5,cell_scale_1,no_terminate_unhealthy,action_repeat_${ACTION_REPEAT},test"

echo "Config=${CONFIG_NAME} optimizer=${HEADS_OPTIMIZER} seed=${SEED}"
echo "Project=${WANDB_PROJECT} entity=${WANDB_ENTITY} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "GIF dir=${GIF_DIR}"
echo "Exp: ${EXP_NAME}"

cd "${REPO_ROOT}"

WANDB_TAGS="${TAGS}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  --env-name ant_u_maze \
  --backend "${BACKEND}" \
  --n-envs 128 \
  --track \
  --wandb-project-name "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
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
