#!/bin/bash
#SBATCH --job-name=hum-goal-exp-ref
#SBATCH --output=slurm_logs/humanoid_goal_exp_distance_refine_16_seed0_stiefel_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

set -euo pipefail

# Focused humanoid goal exponential distance-potential sweep around the current
# best reward: exp_distance_reward_scale=100, temperature=0.5.
#
# Each Slurm array task launches 2 configs on one GPU.

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-3100000}"
BACKEND="${4:-spring}"
ACTION_REPEAT="${5:-4}"
JAX_MEM_FRACTION="${6:-0.45}"
START_STAGGER_SECONDS="${7:-20}"
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
CONFIGS_PER_TASK=2
NUM_CONFIGS=16

PROGRESS_SCALE="400"
SUCCESS_REWARD="250"
SUCCESS_EASY_REWARD="25"
DISTANCE_SCALE="0.5"
HEADING_SCALE="0.5"
HEALTHY_REWARD="2"
STANDING_SCALE="0"

EXP_SCALES=(75 100 125 150)
EXP_TEMPERATURES=(0.35 0.5 0.65 0.8)

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="humanoid_exp_distance_refine16_cratecnn_stiefel_seed0_goal2p5_${GROUP_ID}"
SWEEP_DIR="${REPO_ROOT}/outputs/goal_exp_distance_refine/${GROUP_NAME}"
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
  --goal-progress-reward-scale "${PROGRESS_SCALE}"
  --goal-success-reward "${SUCCESS_REWARD}"
  --goal-success-easy-reward "${SUCCESS_EASY_REWARD}"
  --goal-distance-reward-scale "${DISTANCE_SCALE}"
  --goal-heading-reward-scale "${HEADING_SCALE}"
  --goal-healthy-reward "${HEALTHY_REWARD}"
  --goal-standing-reward-scale "${STANDING_SCALE}"
  --goal-close-reward-scale 0
  --goal-gate-progress-by-standing
  --goal-terminate-when-unhealthy
  --save-rollout-gif
  --save-final-rollout-gif
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
  local temp_idx=$((config_id % ${#EXP_TEMPERATURES[@]}))
  local scale_idx=$((config_id / ${#EXP_TEMPERATURES[@]}))
  local exp_scale="${EXP_SCALES[$scale_idx]}"
  local exp_temperature="${EXP_TEMPERATURES[$temp_idx]}"

  local exp_scale_tag
  local exp_temperature_tag
  exp_scale_tag="$(fmt_tag_value "${exp_scale}")"
  exp_temperature_tag="$(fmt_tag_value "${exp_temperature}")"

  local config_name="exp_s${exp_scale_tag}_t${exp_temperature_tag}"
  local gif_dir="${SWEEP_DIR}/${config_name}"
  local exp_name="ppo_humanoid_goal_exp_refine16_${config_name}_cratecnn_stiefel_s${SEED}_goal2p5"
  local tags="goal,pixel_goal,humanoid_goal,humanoid_exp_distance_refine16,goal_dist_2p5,terminate_unhealthy,encoder_crate_cnn,headarch_crate,opt_stiefel,seed_${SEED},action_repeat_${ACTION_REPEAT},gif_every_1m,final_gif,healthy_${HEALTHY_REWARD},progress_${PROGRESS_SCALE},success_${SUCCESS_REWARD},success_easy_${SUCCESS_EASY_REWARD},distance_${DISTANCE_SCALE},heading_${HEADING_SCALE},stand_${STANDING_SCALE},close_scale_0,exp_distance_scale_${exp_scale_tag},exp_distance_temp_${exp_temperature_tag}"
  if [[ "${exp_scale}" == "100" && "${exp_temperature}" == "0.5" ]]; then
    tags="${tags},current_best_control"
  fi

  mkdir -p "${gif_dir}"

  echo "Running config_id=${config_id}/${NUM_CONFIGS} ${config_name}"
  echo "exp_scale=${exp_scale} exp_temperature=${exp_temperature}"
  echo "Exp: ${exp_name}"

  WANDB_TAGS="${tags}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --seed "${SEED}" \
    --goal-exp-distance-reward-scale "${exp_scale}" \
    --goal-exp-distance-reward-temperature "${exp_temperature}" \
    --rollout-gif-dir "${gif_dir}" \
    --exp-name "${exp_name}"
}

echo "Humanoid exp distance refine sweep task=${TASK_ID} group=${GROUP_NAME}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "Output dir=${SWEEP_DIR}"

cd "${REPO_ROOT}"

pids=()
labels=()
start_config=$((TASK_ID * CONFIGS_PER_TASK))
end_config=$((start_config + CONFIGS_PER_TASK - 1))
if (( TASK_ID < 0 || start_config >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..7." >&2
  exit 1
fi
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
