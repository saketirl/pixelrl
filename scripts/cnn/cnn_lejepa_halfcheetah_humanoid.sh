#!/bin/bash
#SBATCH --job-name=cnn-lejepa
#SBATCH --output=slurm_logs/cnn_lejepa_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

set -euo pipefail

# Run four CNN + LeJEPA PPO jobs:
#   TASK_ID=0 -> halfcheetah + Adam heads
#   TASK_ID=1 -> halfcheetah + MUON heads
#   TASK_ID=2 -> humanoid + Adam heads
#   TASK_ID=3 -> humanoid + MUON heads

WANDB_PROJECT="scott"
WANDB_ENTITY="${1:-}"
TOTAL_TIMESTEPS="${2:-10000000}"
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
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

ENVS=(halfcheetah humanoid)
OPT_CONDITIONS=(adam muon)

NUM_ENVS=${#ENVS[@]}
NUM_OPTS=${#OPT_CONDITIONS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_OPTS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

ENV_IDX=$((TASK_ID / NUM_OPTS))
OPT_IDX=$((TASK_ID % NUM_OPTS))

ENV_NAME="${ENVS[$ENV_IDX]}"
OPT_CONDITION="${OPT_CONDITIONS[$OPT_IDX]}"
BACKEND="spring"
SEED=0

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_lejepa_hc_hum_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn,lejepa,env_${ENV_NAME},opt_${OPT_CONDITION},backend_${BACKEND},seed_${SEED}"

EXP_NAME="ppo_cnn_lejepa_${OPT_CONDITION}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} encoder=cnn lejepa=true opt=${OPT_CONDITION} seed=${SEED}"
echo "Exp: ${EXP_NAME}"

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
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

CNN_ARGS=(
  --encoder-type cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --weight-decay 0.0
  --max-grad-norm 0.5
  --encoder-tanh-scale 0.5
  --lejepa
  --lejepa-lambda 0.1
  --lejepa-proj-dim 128
  --lejepa-num-views 2
  --lejepa-num-slices 64
  --lejepa-t-points 17
  --lejepa-t-min -5.0
  --lejepa-t-max 5.0
)

OPT_ARGS=()
if [[ "${OPT_CONDITION}" == "muon" ]]; then
  OPT_ARGS+=(--use-heads-muon)
else
  OPT_ARGS+=(--no-use-heads-muon)
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${CNN_ARGS[@]}" \
  "${OPT_ARGS[@]}"
