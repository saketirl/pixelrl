#!/bin/bash
#SBATCH --job-name=ant-goal-reward-sweep
#SBATCH --output=slurm_logs/ant_goal_reward_sweep_8configs_4seeds_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

set -euo pipefail

# Ant goal reward-shaping sweep:
#   8 reward configs, one config per GPU/array task
#   each GPU runs 8 concurrent runs: {adam, stiefel} x seeds {0,1,2,3}
#
# The rollout GIF hook saves one deterministic policy rollout after the run
# first crosses ~1M steps.

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-3000000}"
BACKEND="${4:-spring}"
ACTION_REPEAT="${5:-4}"
JAX_MEM_FRACTION="${6:-0.11}"
START_STAGGER_SECONDS="${7:-12}"
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

SEEDS=(0 1 2 3)
OPTIMIZERS=(adam stiefel)

CONFIG_NAMES=(
  original_dist1
  progress10
  progress20
  progress50
  progress10_dist0p1
  progress20_dist0p1
  progress20_success50
  progress20_dist0p1_success50
)
PROGRESS_SCALES=(0 10 20 50 10 20 20 20)
DISTANCE_SCALES=(1.0 0.0 0.0 0.0 0.1 0.1 0.0 0.1)
SUCCESS_REWARDS=(0 0 0 0 0 0 50 50)

if (( TASK_ID < 0 || TASK_ID >= ${#CONFIG_NAMES[@]} )); then
  echo "Invalid TASK_ID=${TASK_ID}" >&2
  exit 1
fi

CONFIG_NAME="${CONFIG_NAMES[$TASK_ID]}"
PROGRESS_SCALE="${PROGRESS_SCALES[$TASK_ID]}"
DISTANCE_SCALE="${DISTANCE_SCALES[$TASK_ID]}"
SUCCESS_REWARD="${SUCCESS_REWARDS[$TASK_ID]}"

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="ant_goal_reward_${CONFIG_NAME}_ar${ACTION_REPEAT}_4seeds_${GROUP_ID}"
GIF_DIR="${REPO_ROOT}/outputs/goal_reward_sweep/${GROUP_NAME}"
mkdir -p "${GIF_DIR}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

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

run_config() {
  local optimizer="$1"
  local seed="$2"
  local exp_name="ppo_goal_reward_${CONFIG_NAME}_${optimizer}_ar${ACTION_REPEAT}_ant_goal_b${BACKEND}_s${seed}"
  local tags="goal,pixel_goal,ant_goal,reward_sweep,${CONFIG_NAME},progress_${PROGRESS_SCALE},distance_${DISTANCE_SCALE},success_${SUCCESS_REWARD},action_repeat_${ACTION_REPEAT},env_ant_goal,backend_${BACKEND},seed_${seed},encoder_crate_cnn,headarch_crate,opt_${optimizer},single_gpu_8runs,mem_fraction_${JAX_MEM_FRACTION},gif_at_1m"

  echo "Running config=${CONFIG_NAME} opt=${optimizer} seed=${seed} progress=${PROGRESS_SCALE} distance=${DISTANCE_SCALE} success=${SUCCESS_REWARD}"
  echo "Exp: ${exp_name}"

  WANDB_TAGS="${tags}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --seed "${seed}" \
    --heads-optimizer "${optimizer}" \
    --exp-name "${exp_name}"
}

echo "Reward config ${TASK_ID}: ${CONFIG_NAME}"
echo "progress=${PROGRESS_SCALE} distance=${DISTANCE_SCALE} success=${SUCCESS_REWARD}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "GIF dir=${GIF_DIR}"

cd "${REPO_ROOT}"

pids=()
labels=()
for optimizer in "${OPTIMIZERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_config "${optimizer}" "${seed}" &
    pids+=("$!")
    labels+=("${CONFIG_NAME}/${optimizer}/seed${seed}")
    sleep "${START_STAGGER_SECONDS}"
  done
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Run failed: ${labels[$i]}" >&2
    status=1
  fi
done

exit "${status}"
