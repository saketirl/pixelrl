#!/bin/bash
#SBATCH --job-name=cratecnn-boundstd-crate-muon-allenv6
#SBATCH --output=slurm_logs/baseline_crate_cnn_meanbound_boundedstd_crate_muon_allenv_6seeds_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-13%8

set -euo pipefail

# Optax Muon baseline for the CRATECNN + CRATE-head staging architecture.
# Mirrors:
#   scripts/staging/crate_cnn_meanbound_boundedstd_crate_stiefel_allenv_4seeds.sh
# but uses real Optax Muon for actor/critic head matrices and runs 6 seeds.
#
# Swept:
#   env  in {halfcheetah, walker2d, ant, humanoid, reacher, swimmer, pusher, hopper, inverted_pendulum}
#   seed in {0, 1, 2, 3, 4, 5}
#
# 9 x 6 = 54 logical configs, packed RUNS_PER_GPU at a time.

WANDB_PROJECT="${1:-encoder}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
RUNS_PER_GPU="${4:-${RUNS_PER_GPU:-4}}"
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
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

ENVS=(halfcheetah walker2d ant humanoid reacher swimmer pusher hopper inverted_pendulum)
BACKENDS=(spring spring spring spring generalized generalized generalized positional generalized)
SEEDS=(0 1 2 3 4 5)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_SEEDS))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$(((NUM_CONFIGS + RUNS_PER_GPU - 1) / RUNS_PER_GPU - 1))." >&2
  exit 1
fi

if (( RUNS_PER_GPU < 1 )); then
  echo "RUNS_PER_GPU must be >= 1; got ${RUNS_PER_GPU}." >&2
  exit 1
fi
if (( RUNS_PER_GPU != 4 )); then
  echo "This launcher's SBATCH array is configured for RUNS_PER_GPU=4; got ${RUNS_PER_GPU}." >&2
  echo "Edit #SBATCH --array if you want a different packing factor." >&2
  exit 1
fi

START_IDX=$((TASK_ID * RUNS_PER_GPU))
if (( START_IDX >= NUM_CONFIGS )); then
  echo "TASK_ID=${TASK_ID} start=${START_IDX} >= NUM_CONFIGS=${NUM_CONFIGS}; nothing to do."
  exit 0
fi
END_IDX=$((START_IDX + RUNS_PER_GPU - 1))
if (( END_IDX >= NUM_CONFIGS )); then
  END_IDX=$((NUM_CONFIGS - 1))
fi

CRATE_STEP_SIZE="0.1"
ENCODER_CRATE_STEP_SIZE="0.1"
ACTOR_MEAN_SCALE="1.0"
ACTOR_LOGSTD_INIT="-2.0"
ACTOR_LOGSTD_MIN="-5.0"
ACTOR_LOGSTD_MAX="-1.4"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="baseline_crate_cnn_meanbound_boundedstd_crate_optax_muon_allenv6_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

case "${RUNS_PER_GPU}" in
  1) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.60}" ;;
  2) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.35}" ;;
  3|4) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.20}" ;;
  *) JAX_MEM_FRACTION="${JAX_MEM_FRACTION:-0.10}" ;;
esac

echo "Running logical configs ${START_IDX}..${END_IDX}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Packing: runs_per_gpu=${RUNS_PER_GPU} jax_mem_fraction=${JAX_MEM_FRACTION}"
echo "Config: encoder=crate_cnn encoder_crate_step_size=${ENCODER_CRATE_STEP_SIZE} actor_mean_tanh=true actor_mean_scale=${ACTOR_MEAN_SCALE} bounded_global_logstd=true actor_logstd_init=${ACTOR_LOGSTD_INIT} actor_logstd_min=${ACTOR_LOGSTD_MIN} actor_logstd_max=${ACTOR_LOGSTD_MAX} crate_head=true heads_optimizer=optax_muon"

WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"

pids=()
for RUN_IDX in $(seq "${START_IDX}" "${END_IDX}"); do
  SEED_IDX=$((RUN_IDX % NUM_SEEDS))
  ENV_IDX=$((RUN_IDX / NUM_SEEDS))

  SEED="${SEEDS[$SEED_IDX]}"
  ENV_NAME="${ENVS[$ENV_IDX]}"
  BACKEND="${BACKENDS[$ENV_IDX]}"
  EXP_NAME="ppo_baseline_cratecnn_meanbound_boundedstd_crate_optaxmuon_${ENV_NAME}_b${BACKEND}_s${SEED}_t${RUN_IDX}"
  CHILD_LOG="${REPO_ROOT}/slurm_logs/baseline_crate_cnn_meanbound_boundedstd_crate_muon_allenv_6seeds_${GROUP_ID}_run${RUN_IDX}_${ENV_NAME}_s${SEED}.out"

  (
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
    export PYTHONUNBUFFERED=1
    export WANDB_RUN_GROUP="${GROUP_NAME}"
    export WANDB_TAGS="baseline,optax_muon,crate_cnn_meanbound_boundedstd,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},encoder_crate_cnn,actor_mean_tanh,actor_mean_scale_${ACTOR_MEAN_SCALE},bounded_global_logstd,actor_logstd_init_${ACTOR_LOGSTD_INIT},actor_logstd_min_${ACTOR_LOGSTD_MIN},actor_logstd_max_${ACTOR_LOGSTD_MAX},opt_muon,headarch_crate,sigreg_off,encoder_crate_step_${ENCODER_CRATE_STEP_SIZE},head_crate_step_${CRATE_STEP_SIZE},allenv9,seeds012345,runs_per_gpu_${RUNS_PER_GPU},jax_mem_fraction_${JAX_MEM_FRACTION}"
    uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
      --env-name "${ENV_NAME}" \
      --backend "${BACKEND}" \
      --n-envs 128 \
      --track \
      --debug-repr \
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
      --action-repeat 4 \
      --anneal-lr \
      --muon-ns-steps 5 \
      --actor-muon-max-grad-norm 100 \
      --critic-muon-max-grad-norm 1 \
      --actor-mean-tanh \
      --actor-mean-scale "${ACTOR_MEAN_SCALE}" \
      --bounded-global-logstd \
      --actor-logstd-init="${ACTOR_LOGSTD_INIT}" \
      --actor-logstd-min="${ACTOR_LOGSTD_MIN}" \
      --actor-logstd-max="${ACTOR_LOGSTD_MAX}" \
      --encoder-type crate_cnn \
      --encoder-lr 3e-4 \
      --heads-adam-lr 3e-4 \
      --heads-muon-lr 0.001 \
      --max-grad-norm 0.05 \
      --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}" \
      --use-crate-head \
      --crate-step-size "${CRATE_STEP_SIZE}" \
      --sigreg-mode off \
      --heads-optimizer muon \
      --exp-name "${EXP_NAME}" \
      "${WANDB_ENTITY_ARGS[@]}"
  ) > "${CHILD_LOG}" 2>&1 &
  pids+=("$!")
  echo "Launched run_idx=${RUN_IDX} env=${ENV_NAME} backend=${BACKEND} seed=${SEED} pid=${pids[-1]} log=${CHILD_LOG}"
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
