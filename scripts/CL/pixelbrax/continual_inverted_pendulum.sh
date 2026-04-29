#!/bin/bash
#SBATCH --job-name=cl-invpend-10tasks
#SBATCH --output=slurm_logs/cl_inverted_pendulum_10tasks_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2

set -euo pipefail

# Continual InvertedPendulum launcher using the same PPO/Stiefel hyperparameters as
# scripts/full_runs/full_run.sh, restricted to InvertedPendulum generalized dynamics. This
# runs one seed for each head optimizer condition: Stiefel heads and Adam heads.
# Each run sees ten tasks total by default: task 0 uses default Brax
# dynamics, then nine sampled tasks.

WANDB_PROJECT="${1:-continual_pixelbrax}"
WANDB_ENTITY="${2:-rl-power}"
SWITCH_EVERY_ENV_STEPS="${3:-999680}"
NUM_TASKS="${4:-10}"
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

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

SEED=0
OPT_CONDITIONS=(stiefel adam)
NUM_RUNS=${#OPT_CONDITIONS[@]}

if (( TASK_ID < 0 || TASK_ID >= NUM_RUNS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_RUNS - 1))." >&2
  exit 1
fi

if (( NUM_TASKS < 1 )); then
  echo "NUM_TASKS must be >= 1; got ${NUM_TASKS}." >&2
  exit 1
fi

TOTAL_TIMESTEPS=$((SWITCH_EVERY_ENV_STEPS * NUM_TASKS))
ROLLOUT_ENV_STEPS=$((128 * 10))
if (( SWITCH_EVERY_ENV_STEPS % ROLLOUT_ENV_STEPS != 0 )); then
  echo "SWITCH_EVERY_ENV_STEPS=${SWITCH_EVERY_ENV_STEPS} must be divisible by ${ROLLOUT_ENV_STEPS}." >&2
  exit 1
fi

OPT_CONDITION="${OPT_CONDITIONS[$TASK_ID]}"
OPT_ARGS=()
if [[ "${OPT_CONDITION}" == "stiefel" ]]; then
  OPT_ARGS+=(--heads-optimizer stiefel)
else
  OPT_ARGS+=(--heads-optimizer adam)
fi

ENV_NAME="inverted_pendulum"
BACKEND="generalized"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cl_inverted_pendulum_${NUM_TASKS}tasks_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,continual_dynamics,inverted_pendulum,backend_${BACKEND},seed_${SEED},tasks_${NUM_TASKS},switch_${SWITCH_EVERY_ENV_STEPS},opt_${OPT_CONDITION},cnn_heads_stiefel_me_hparams"

EXP_NAME="ppo_cl_inverted_pendulum_${NUM_TASKS}tasks_${OPT_CONDITION}_s${SEED}_t${TASK_ID}"

BASE_CONFIG="${REPO_ROOT}/configs/continual/inverted_pendulum_dynamics.yaml"
RUN_CONFIG="${REPO_ROOT}/slurm_logs/continual_inverted_pendulum_${NUM_TASKS}tasks_${GROUP_ID}_${OPT_CONDITION}_${TASK_ID}.yaml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing continual dynamics config: ${BASE_CONFIG}" >&2
  exit 1
fi
sed "s/^switch_every_env_steps:.*/switch_every_env_steps: ${SWITCH_EVERY_ENV_STEPS}/" "${BASE_CONFIG}" > "${RUN_CONFIG}"

echo "Running TASK_ID=${TASK_ID}/${NUM_RUNS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} opt=${OPT_CONDITION} tasks=${NUM_TASKS} switch_every_env_steps=${SWITCH_EVERY_ENV_STEPS} total_timesteps=${TOTAL_TIMESTEPS}"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"
echo "Exp: ${EXP_NAME}"
echo "Continual config: ${RUN_CONFIG}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --n-envs 128
  --track
  --debug-repr
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
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --anneal-lr
  --stiefel-dual-lr 0.01
  --stiefel-dual-steps 5
  --actor-stiefel-max-grad-norm 100
  --critic-stiefel-max-grad-norm 1
  --continual-dynamics-config "${RUN_CONFIG}"
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

ARCH_ARGS=(
  --encoder-type cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-stiefel-lr 0.001
  --max-grad-norm 0.05
  --encoder-tanh-scale 0.5
)

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  "${OPT_ARGS[@]}"
