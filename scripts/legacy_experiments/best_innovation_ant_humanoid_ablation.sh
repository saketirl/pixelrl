#!/bin/bash
#SBATCH --job-name=best-innovation-ant-humanoid
#SBATCH --output=slurm_logs/best_innovation_ant_humanoid_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# 8-run grid:
#   env in {humanoid, ant}
#   head optimizer in {muon, adam}
#   innovation in {off, on}
# CRATE heads are fixed on.
# Best innovation hyperparameters are fixed to the current best humanoid setting.

WANDB_PROJECT="${1:-encoder}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
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

ENVS=(humanoid ant)
HEAD_OPTS=(muon adam)
INNOVATION_MODES=(off on)
SEED="0"
BACKEND="spring"
NUM_ENVS=${#ENVS[@]}
NUM_HEAD_OPTS=${#HEAD_OPTS[@]}
NUM_INNOVATION=${#INNOVATION_MODES[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_HEAD_OPTS * NUM_INNOVATION))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

ENV_IDX=$((TASK_ID % NUM_ENVS))
HEAD_OPT_IDX=$(((TASK_ID / NUM_ENVS) % NUM_HEAD_OPTS))
INNOVATION_IDX=$((TASK_ID / (NUM_ENVS * NUM_HEAD_OPTS)))

ENV_NAME="${ENVS[$ENV_IDX]}"
HEAD_OPT="${HEAD_OPTS[$HEAD_OPT_IDX]}"
INNOVATION_MODE="${INNOVATION_MODES[$INNOVATION_IDX]}"

INNOVATION_COEF="0.0"
if [[ "${INNOVATION_MODE}" == "on" ]]; then
  INNOVATION_COEF="1e-3"
fi

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="best_innovation_ant_humanoid_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="best_innovation,innovation_aux,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},headopt_${HEAD_OPT},headarch_crate,innovation_${INNOVATION_MODE},proj_64"

EXP_NAME="ppo_${HEAD_OPT}_${ENV_NAME}_bestinnovation_crate_innovation${INNOVATION_MODE}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} head_opt=${HEAD_OPT} innovation=${INNOVATION_MODE} seed=${SEED}"
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
  --encoder-type innovation_cnn
  --sigreg-mode off
  --innovation-coef "${INNOVATION_COEF}"
  --innovation-proj-dim 64
  --innovation-num-slices 16
  --innovation-num-t 8
  --innovation-t-max 5.0
  --innovation-warmup-updates 500
  --innovation-ramp-updates 500
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --max-grad-norm 0.05
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --use-crate-head
  --exp-name "${EXP_NAME}"
)

if [[ "${HEAD_OPT}" == "muon" ]]; then
  COMMON_ARGS+=(--use-heads-muon)
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
