#!/bin/bash
#SBATCH --job-name=hum-goal-final
#SBATCH --output=slurm_logs/humanoid_goal_final_archopt_6seeds_10m_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

set -euo pipefail

# Final humanoid_goal architecture/optimizer validation.
#
# 4 variants x 6 seeds = 24 runs:
#   0: crate_cnn encoder + crate heads + adam
#   1: crate_cnn encoder + crate heads + stiefel
#   2: cnn encoder       + mlp heads   + adam
#   3: cnn encoder       + mlp heads   + stiefel
#
# Each array task runs one variant and two seeds on one GPU:
#   variant_idx = SLURM_ARRAY_TASK_ID / 3
#   seed_pair   = SLURM_ARRAY_TASK_ID % 3
#   seeds       = {2 * seed_pair, 2 * seed_pair + 1}

WANDB_PROJECT="${1:-pixel-goal}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
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

if (( TASK_ID < 0 || TASK_ID > 11 )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..11." >&2
  exit 1
fi

VARIANT_IDX=$((TASK_ID / 3))
SEED_PAIR_IDX=$((TASK_ID % 3))
SEEDS=($((2 * SEED_PAIR_IDX)) $((2 * SEED_PAIR_IDX + 1)))

VARIANT_NAMES=(
  cratecnn_crateheads_adam
  cratecnn_crateheads_stiefel
  cnn_heads_adam
  cnn_heads_stiefel
)
ENCODER_TYPES=(crate_cnn crate_cnn cnn cnn)
HEAD_ARCHES=(crate crate mlp mlp)
HEADS_OPTIMIZERS=(adam stiefel adam stiefel)

VARIANT_NAME="${VARIANT_NAMES[$VARIANT_IDX]}"
ENCODER_TYPE="${ENCODER_TYPES[$VARIANT_IDX]}"
HEAD_ARCH="${HEAD_ARCHES[$VARIANT_IDX]}"
HEADS_OPTIMIZER="${HEADS_OPTIMIZERS[$VARIANT_IDX]}"

CONFIG_NAME="exp_s125_t0p35"
PROGRESS_SCALE="400"
SUCCESS_REWARD="250"
SUCCESS_EASY_REWARD="25"
DISTANCE_SCALE="0.5"
HEADING_SCALE="0.5"
HEALTHY_REWARD="2"
STANDING_SCALE="0"
CLOSE_REWARD_SCALE="0"
EXP_DISTANCE_REWARD_SCALE="125"
EXP_DISTANCE_REWARD_TEMPERATURE="0.35"

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ENCODER_SCALE="0.5"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="humanoid_goal_final_archopt_6seeds_10m_${CONFIG_NAME}_${GROUP_ID}"
SWEEP_DIR="${REPO_ROOT}/outputs/goal_final_archopt/${GROUP_NAME}"
GIF_DIR="${SWEEP_DIR}/${VARIANT_NAME}_seedpair${SEED_PAIR_IDX}"
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
  --goal-success-reward "${SUCCESS_REWARD}"
  --goal-success-easy-reward "${SUCCESS_EASY_REWARD}"
  --goal-distance-reward-scale "${DISTANCE_SCALE}"
  --goal-heading-reward-scale "${HEADING_SCALE}"
  --goal-healthy-reward "${HEALTHY_REWARD}"
  --goal-standing-reward-scale "${STANDING_SCALE}"
  --goal-close-reward-scale "${CLOSE_REWARD_SCALE}"
  --goal-exp-distance-reward-scale "${EXP_DISTANCE_REWARD_SCALE}"
  --goal-exp-distance-reward-temperature "${EXP_DISTANCE_REWARD_TEMPERATURE}"
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
  --encoder-type "${ENCODER_TYPE}"
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-stiefel-lr 0.001
  --max-grad-norm 0.05
)

if [[ "${HEAD_ARCH}" == "crate" ]]; then
  ARCH_ARGS+=(
    --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
    --use-crate-head
    --crate-step-size "${CRATE_STEP_SIZE}"
    --sigreg-mode off
  )
else
  ARCH_ARGS+=(
    --encoder-tanh-scale "${ENCODER_SCALE}"
  )
fi

run_seed() {
  local seed="$1"
  local seed_gif_dir="${GIF_DIR}/seed${seed}"
  local exp_name="ppo_humanoid_goal_final_${CONFIG_NAME}_${VARIANT_NAME}_ar${ACTION_REPEAT}_b${BACKEND}_s${seed}_10m"
  local tags="neurips,goal,pixel_goal,humanoid_goal,humanoid_goal_final_6seed_10m,${CONFIG_NAME},goal_dist_2p5,backend_${BACKEND},action_repeat_${ACTION_REPEAT},seed_${seed},encoder_${ENCODER_TYPE},headarch_${HEAD_ARCH},opt_${HEADS_OPTIMIZER},progress_${PROGRESS_SCALE},success_${SUCCESS_REWARD},success_easy_${SUCCESS_EASY_REWARD},distance_${DISTANCE_SCALE},heading_${HEADING_SCALE},stand_${STANDING_SCALE},healthy_${HEALTHY_REWARD},exp_distance_scale_${EXP_DISTANCE_REWARD_SCALE},exp_distance_temp_0p35,two_runs_per_gpu,mem_fraction_${JAX_MEM_FRACTION},gif_every_1m,final_gif"

  mkdir -p "${seed_gif_dir}"

  echo "Running variant=${VARIANT_NAME} seed=${seed}"
  echo "Reward=${CONFIG_NAME} exp_scale=${EXP_DISTANCE_REWARD_SCALE} exp_temperature=${EXP_DISTANCE_REWARD_TEMPERATURE}"
  echo "Exp: ${exp_name}"

  WANDB_TAGS="${tags}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    "${ARCH_ARGS[@]}" \
    --seed "${seed}" \
    --heads-optimizer "${HEADS_OPTIMIZER}" \
    --rollout-gif-dir "${seed_gif_dir}" \
    --exp-name "${exp_name}"
}

echo "Final humanoid validation task=${TASK_ID} group=${GROUP_NAME}"
echo "Variant=${VARIANT_NAME} optimizer=${HEADS_OPTIMIZER} seeds=${SEEDS[*]}"
echo "Project=${WANDB_PROJECT} total_timesteps=${TOTAL_TIMESTEPS} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
echo "Output dir=${SWEEP_DIR}"

cd "${REPO_ROOT}"

pids=()
labels=()
for seed in "${SEEDS[@]}"; do
  run_seed "${seed}" &
  pids+=("$!")
  labels+=("${VARIANT_NAME}/seed${seed}")
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
