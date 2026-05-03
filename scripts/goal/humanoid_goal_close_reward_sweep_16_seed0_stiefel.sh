#!/bin/bash
#SBATCH --job-name=hum-goal-close16
#SBATCH --output=slurm_logs/humanoid_goal_close_reward_sweep_16_seed0_stiefel_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

set -euo pipefail

# Humanoid goal close-range reward sweep for the fixed 2.5m, 45-degree goal.
# Config 0 is the current-best baseline control. Configs 1-15 add a
# close-range reward around the strict success radius.
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
SUCCESS_EASY_REWARD="25"
DISTANCE_SCALE="0.5"
HEADING_SCALE="0.5"
HEALTHY_REWARD="2"

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="humanoid_close_reward_sweep16_cratecnn_stiefel_seed0_goal2p5_${GROUP_ID}"
SWEEP_DIR="${REPO_ROOT}/outputs/goal_close_reward_sweep/${GROUP_NAME}"
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
  --goal-success-easy-reward "${SUCCESS_EASY_REWARD}"
  --goal-distance-reward-scale "${DISTANCE_SCALE}"
  --goal-heading-reward-scale "${HEADING_SCALE}"
  --goal-healthy-reward "${HEALTHY_REWARD}"
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

config_values() {
  local config_id="$1"
  case "${config_id}" in
    0) echo "baseline_current_best 50 2 0 1.0" ;;
    1) echo "close_r0p75_c25_s100_stand0 100 0 25 0.75" ;;
    2) echo "close_r0p75_c25_s100_stand2 100 2 25 0.75" ;;
    3) echo "close_r0p75_c25_s250_stand0 250 0 25 0.75" ;;
    4) echo "close_r0p75_c25_s250_stand2 250 2 25 0.75" ;;
    5) echo "close_r0p75_c50_s100_stand0 100 0 50 0.75" ;;
    6) echo "close_r0p75_c50_s100_stand2 100 2 50 0.75" ;;
    7) echo "close_r0p75_c50_s250_stand0 250 0 50 0.75" ;;
    8) echo "close_r0p75_c50_s250_stand2 250 2 50 0.75" ;;
    9) echo "close_r1p0_c25_s100_stand2 100 2 25 1.0" ;;
    10) echo "close_r1p0_c25_s250_stand0 250 0 25 1.0" ;;
    11) echo "close_r1p0_c25_s250_stand2 250 2 25 1.0" ;;
    12) echo "close_r1p0_c50_s100_stand0 100 0 50 1.0" ;;
    13) echo "close_r1p0_c50_s100_stand2 100 2 50 1.0" ;;
    14) echo "close_r1p0_c50_s250_stand0 250 0 50 1.0" ;;
    15) echo "close_r1p0_c50_s250_stand2 250 2 50 1.0" ;;
    *) echo "Invalid config_id=${config_id}" >&2; return 1 ;;
  esac
}

run_config() {
  local config_id="$1"
  local config_name
  local success_reward
  local standing_scale
  local close_scale
  local close_radius
  read -r config_name success_reward standing_scale close_scale close_radius < <(config_values "${config_id}")

  local success_tag
  local standing_tag
  local close_scale_tag
  local close_radius_tag
  success_tag="$(fmt_tag_value "${success_reward}")"
  standing_tag="$(fmt_tag_value "${standing_scale}")"
  close_scale_tag="$(fmt_tag_value "${close_scale}")"
  close_radius_tag="$(fmt_tag_value "${close_radius}")"

  local gif_dir="${SWEEP_DIR}/${config_name}"
  local exp_name="ppo_humanoid_goal_close16_${config_name}_cratecnn_stiefel_s${SEED}_goal2p5"
  local tags="goal,pixel_goal,humanoid_goal,humanoid_close_sweep16,goal_dist_2p5,terminate_unhealthy,encoder_crate_cnn,headarch_crate,opt_stiefel,seed_${SEED},action_repeat_${ACTION_REPEAT},gif_every_1m,final_gif,healthy_${HEALTHY_REWARD},progress_${PROGRESS_SCALE},success_easy_${SUCCESS_EASY_REWARD},distance_${DISTANCE_SCALE},heading_${HEADING_SCALE},success_${success_tag},stand_${standing_tag},close_scale_${close_scale_tag},close_radius_${close_radius_tag}"
  if [[ "${config_id}" == "0" ]]; then
    tags="${tags},baseline_control,current_best"
  fi

  mkdir -p "${gif_dir}"

  echo "Running config_id=${config_id}/${NUM_CONFIGS} ${config_name}"
  echo "success=${success_reward} standing=${standing_scale} close_scale=${close_scale} close_radius=${close_radius}"
  echo "Exp: ${exp_name}"

  WANDB_TAGS="${tags}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --seed "${SEED}" \
    --goal-success-reward "${success_reward}" \
    --goal-standing-reward-scale "${standing_scale}" \
    --goal-close-reward-scale "${close_scale}" \
    --goal-close-reward-radius "${close_radius}" \
    --rollout-gif-dir "${gif_dir}" \
    --exp-name "${exp_name}"
}

echo "Humanoid close reward sweep task=${TASK_ID} group=${GROUP_NAME}"
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
