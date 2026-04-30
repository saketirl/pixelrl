#!/bin/bash
#SBATCH --job-name=maze-ant-4algos-ar4
#SBATCH --output=slurm_logs/maze_ant_umaze_4algos_6seeds_ar4_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

# Ant U-maze fixed-goal sweep:
#   algos in {cnn_adam, cnn_stiefel, crate_cnn_crate_head_adam, crate_cnn_crate_head_stiefel}
#   seeds in {0, 1, 2, 3, 4, 5}
#
# 4 x 6 = 24 runs.  Each SLURM array task launches 8 runs concurrently on
# one GPU, so the full sweep uses 3 GPU allocations.

WANDB_PROJECT="${1:-pixel-maze}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
ENV_NAME="${4:-ant_u_maze}"
BACKEND="${5:-spring}"
ACTION_REPEAT="${6:-4}"
RUNS_PER_GPU="${7:-8}"
JAX_MEM_FRACTION="${8:-0.11}"
START_STAGGER_SECONDS="${9:-15}"
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

SEEDS=(0 1 2 3 4 5)
ALGOS=(
  cnn_adam
  cnn_stiefel
  crate_cnn_crate_head_adam
  crate_cnn_crate_head_stiefel
)

NUM_SEEDS=${#SEEDS[@]}
NUM_ALGOS=${#ALGOS[@]}
NUM_CONFIGS=$((NUM_SEEDS * NUM_ALGOS))
START_CONFIG=$((TASK_ID * RUNS_PER_GPU))
END_CONFIG=$((START_CONFIG + RUNS_PER_GPU - 1))

if (( START_CONFIG >= NUM_CONFIGS )); then
  echo "Array task ${TASK_ID} has no work: start=${START_CONFIG}, configs=${NUM_CONFIGS}"
  exit 0
fi
if (( END_CONFIG >= NUM_CONFIGS )); then
  END_CONFIG=$((NUM_CONFIGS - 1))
fi

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="maze_${ENV_NAME}_4algos_6seeds_ar${ACTION_REPEAT}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
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
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

run_config() {
  local config_id="$1"
  local seed_idx=$((config_id % NUM_SEEDS))
  local algo_idx=$((config_id / NUM_SEEDS))
  local seed="${SEEDS[$seed_idx]}"
  local algo="${ALGOS[$algo_idx]}"
  local encoder_type=""
  local heads_optimizer=""
  local exp_name=""
  local tags=""
  local arch_args=()

  case "${algo}" in
    cnn_adam)
      encoder_type="cnn"
      heads_optimizer="adam"
      arch_args=(
        --encoder-type cnn
        --encoder-lr 3e-4
        --heads-adam-lr 3e-4
        --heads-stiefel-lr 0.001
        --max-grad-norm 0.05
        --encoder-tanh-scale 0.5
      )
      ;;
    cnn_stiefel)
      encoder_type="cnn"
      heads_optimizer="stiefel"
      arch_args=(
        --encoder-type cnn
        --encoder-lr 3e-4
        --heads-adam-lr 3e-4
        --heads-stiefel-lr 0.001
        --max-grad-norm 0.05
        --encoder-tanh-scale 0.5
      )
      ;;
    crate_cnn_crate_head_adam)
      encoder_type="crate_cnn"
      heads_optimizer="adam"
      arch_args=(
        --encoder-type crate_cnn
        --encoder-lr 3e-4
        --heads-adam-lr 3e-4
        --heads-stiefel-lr 0.001
        --max-grad-norm 0.05
        --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
        --use-crate-head
        --crate-step-size "${CRATE_STEP_SIZE}"
        --sigreg-mode off
        --actor-mean-tanh
        --actor-mean-scale "${ACTOR_MEAN_SCALE}"
        --bounded-global-logstd
        --actor-logstd-init="${ACTOR_LOGSTD_INIT}"
        --actor-logstd-min="${ACTOR_LOGSTD_MIN}"
        --actor-logstd-max="${ACTOR_LOGSTD_MAX}"
      )
      ;;
    crate_cnn_crate_head_stiefel)
      encoder_type="crate_cnn"
      heads_optimizer="stiefel"
      arch_args=(
        --encoder-type crate_cnn
        --encoder-lr 3e-4
        --heads-adam-lr 3e-4
        --heads-stiefel-lr 0.001
        --max-grad-norm 0.05
        --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
        --use-crate-head
        --crate-step-size "${CRATE_STEP_SIZE}"
        --sigreg-mode off
        --actor-mean-tanh
        --actor-mean-scale "${ACTOR_MEAN_SCALE}"
        --bounded-global-logstd
        --actor-logstd-init="${ACTOR_LOGSTD_INIT}"
        --actor-logstd-min="${ACTOR_LOGSTD_MIN}"
        --actor-logstd-max="${ACTOR_LOGSTD_MAX}"
      )
      ;;
    *)
      echo "Unknown algo=${algo}" >&2
      return 1
      ;;
  esac

  exp_name="ppo_maze_${algo}_ar${ACTION_REPEAT}_${ENV_NAME}_b${BACKEND}_s${seed}_c${config_id}"
  tags="first_big_run, maze,pixel_maze,fixed_goal,dense_distance_reward,action_repeat_${ACTION_REPEAT},env_${ENV_NAME},backend_${BACKEND},seed_${seed},algo_${algo},encoder_${encoder_type},opt_${heads_optimizer},sweep4algos6seeds,concurrent_gpu,mem_fraction_${JAX_MEM_FRACTION}"
  if [[ "${encoder_type}" == "crate_cnn" ]]; then
    tags="${tags},encoder_crate_step_${ENCODER_CRATE_STEP_SIZE},head_crate_step_${CRATE_STEP_SIZE},headarch_crate,bounded_global_logstd,actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},sigreg_off"
  fi

  echo "Running config=${config_id}/${NUM_CONFIGS} algo=${algo} seed=${seed} env=${ENV_NAME} backend=${BACKEND} action_repeat=${ACTION_REPEAT}"
  echo "Exp: ${exp_name}"

  WANDB_TAGS="${tags}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}" \
  uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --heads-optimizer "${heads_optimizer}" \
    --exp-name "${exp_name}" \
    "${arch_args[@]}"
}

echo "Running array task ${TASK_ID}: configs ${START_CONFIG}..${END_CONFIG} of ${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Project=${WANDB_PROJECT} env=${ENV_NAME} backend=${BACKEND} action_repeat=${ACTION_REPEAT} runs_per_gpu=${RUNS_PER_GPU} mem_fraction=${JAX_MEM_FRACTION}"

cd "${REPO_ROOT}"

pids=()
config_ids=()
for config_id in $(seq "${START_CONFIG}" "${END_CONFIG}"); do
  run_config "${config_id}" &
  pids+=("$!")
  config_ids+=("${config_id}")
  sleep "${START_STAGGER_SECONDS}"
done

status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  config_id="${config_ids[$i]}"
  if ! wait "${pid}"; then
    echo "Config ${config_id} failed" >&2
    status=1
  fi
done

exit "${status}"
