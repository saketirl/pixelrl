#!/bin/bash
#SBATCH --job-name=hum-goal-rs64-term
#SBATCH --output=slurm_logs/humanoid_goal_cratecnn_reward_sweep_64_seed0_stiefel_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

set -euo pipefail

# Humanoid reward-shaping sweep for the fixed 2.5m, 45-degree goal.
# This sweep terminates unhealthy humanoids and pushes harder on goal progress.
#
# 64 configs total:
#   progress_scale     in {50, 100, 200, 400}
#   success_easy_reward in {10, 25}
#   distance_scale     in {0.25, 0.5}
#   heading_scale      in {0.5, 1}
#   standing_scale     in {0, 2}
#
# Each Slurm array task launches 8 configs on one GPU.

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-3000000}"
BACKEND="${4:-spring}"
ACTION_REPEAT="${5:-4}"
JAX_MEM_FRACTION="${6:-0.11}"
START_STAGGER_SECONDS="${7:-10}"
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
HEADS_OPTIMIZER="stiefel"

PROGRESS_SCALE="20"
HEALTHY_REWARD="2"
SUCCESS_REWARD="50"

PROGRESS_SCALES=(50 100 200 400)
SUCCESS_EASY_REWARDS=(10 25)
DISTANCE_SCALES=(0.25 0.5)
HEADING_SCALES=(0.5 1)
STANDING_SCALES=(0 2)

NUM_PROGRESS=${#PROGRESS_SCALES[@]}
NUM_SUCCESS_EASY=${#SUCCESS_EASY_REWARDS[@]}
NUM_DISTANCE=${#DISTANCE_SCALES[@]}
NUM_HEADING=${#HEADING_SCALES[@]}
NUM_STANDING=${#STANDING_SCALES[@]}
NUM_CONFIGS=$((NUM_PROGRESS * NUM_SUCCESS_EASY * NUM_DISTANCE * NUM_HEADING * NUM_STANDING))
CONFIGS_PER_TASK=8

if (( TASK_ID < 0 || TASK_ID >= 8 )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..7." >&2
  exit 1
fi

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="humanoid_reward_sweep64_term_progress_cratecnn_stiefel_seed0_goal2p5_${GROUP_ID}"
SWEEP_DIR="${REPO_ROOT}/outputs/goal_reward_sweep/${GROUP_NAME}"
mkdir -p "${SWEEP_DIR}"
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
  --goal-healthy-reward "${HEALTHY_REWARD}"
  --goal-success-reward "${SUCCESS_REWARD}"
  --goal-gate-progress-by-standing
  --goal-terminate-when-unhealthy
  --save-rollout-gif
  --rollout-gif-step 1000000
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
  --heads-optimizer "${HEADS_OPTIMIZER}"
)

fmt_tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "${value}"
}

run_config() {
  local config_id="$1"
  local standing_idx=$((config_id % NUM_STANDING))
  local heading_idx=$(((config_id / NUM_STANDING) % NUM_HEADING))
  local distance_idx=$(((config_id / (NUM_STANDING * NUM_HEADING)) % NUM_DISTANCE))
  local success_easy_idx=$(((config_id / (NUM_STANDING * NUM_HEADING * NUM_DISTANCE)) % NUM_SUCCESS_EASY))
  local progress_idx=$((config_id / (NUM_STANDING * NUM_HEADING * NUM_DISTANCE * NUM_SUCCESS_EASY)))

  local progress_scale="${PROGRESS_SCALES[$progress_idx]}"
  local success_easy_reward="${SUCCESS_EASY_REWARDS[$success_easy_idx]}"
  local distance_scale="${DISTANCE_SCALES[$distance_idx]}"
  local standing_scale="${STANDING_SCALES[$standing_idx]}"
  local heading_scale="${HEADING_SCALES[$heading_idx]}"

  local progress_tag
  local success_easy_tag
  local distance_tag
  local standing_tag
  local heading_tag
  progress_tag="$(fmt_tag_value "${progress_scale}")"
  success_easy_tag="$(fmt_tag_value "${success_easy_reward}")"
  distance_tag="$(fmt_tag_value "${distance_scale}")"
  standing_tag="$(fmt_tag_value "${standing_scale}")"
  heading_tag="$(fmt_tag_value "${heading_scale}")"

  local config_name="progress${progress_tag}_easy${success_easy_tag}_dist${distance_tag}_heading${heading_tag}_stand${standing_tag}"
  local gif_dir="${SWEEP_DIR}/${config_name}"
  local exp_name="ppo_humanoid_goal_rs64term_${config_name}_cratecnn_stiefel_s${SEED}_goal2p5"
  local tags="goal,pixel_goal,humanoid_goal,reward_sweep64_term,goal_dist_2p5,terminate_unhealthy,encoder_crate_cnn,headarch_crate,opt_stiefel,seed_${SEED},action_repeat_${ACTION_REPEAT},gif_every_1m,healthy_${HEALTHY_REWARD},success_${SUCCESS_REWARD},gated_progress,progress_${progress_tag},success_easy_${success_easy_tag},distance_${distance_tag},heading_${heading_tag},stand_${standing_tag}"

  mkdir -p "${gif_dir}"

  echo "Running config_id=${config_id}/${NUM_CONFIGS} ${config_name}"
  echo "progress=${progress_scale} success_easy=${success_easy_reward} distance=${distance_scale} heading=${heading_scale} standing=${standing_scale}"
  echo "Exp: ${exp_name}"

  WANDB_TAGS="${tags}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --seed "${SEED}" \
    --goal-progress-reward-scale "${progress_scale}" \
    --goal-success-easy-reward "${success_easy_reward}" \
    --goal-standing-reward-scale "${standing_scale}" \
    --goal-heading-reward-scale "${heading_scale}" \
    --goal-distance-reward-scale "${distance_scale}" \
    --rollout-gif-dir "${gif_dir}" \
    --exp-name "${exp_name}"
}

echo "Humanoid reward sweep task=${TASK_ID} group=${GROUP_NAME}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "Output dir=${SWEEP_DIR}"

cd "${REPO_ROOT}"

pids=()
labels=()
start_config=$((TASK_ID * CONFIGS_PER_TASK))
end_config=$((start_config + CONFIGS_PER_TASK - 1))
if (( end_config >= NUM_CONFIGS )); then
  end_config=$((NUM_CONFIGS - 1))
fi

for config_id in $(seq "${start_config}" "${end_config}"); do
  run_config "${config_id}" &
  pids+=("$!")
  labels+=("config${config_id}")
  sleep "${START_STAGGER_SECONDS}"
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Run failed: ${labels[$i]}" >&2
    status=1
  fi
done

exit "${status}"
