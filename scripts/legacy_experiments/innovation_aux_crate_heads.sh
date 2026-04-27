#!/bin/bash
#SBATCH --job-name=innovation-aux-crate-heads
#SBATCH --output=slurm_logs/innovation_aux_crate_heads_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Humanoid-only innovation-aware auxiliary experiment.
# Keeps the known-good policy path: innovation_cnn encoder feeding Muon+CRATE heads.
# The innovation projector is auxiliary only; actor/critic still consume the full 512-d latent.
# Sweep:
#   innovation_proj_dim in {64, 128}
#   innovation_coef in {1e-4, 1e-3}

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

ENV_NAME="humanoid"
BACKEND="spring"
SEED="0"
PROJ_DIMS=(64 128)
INNOVATION_COEFS=(1e-4 1e-3)
NUM_DIMS=${#PROJ_DIMS[@]}
NUM_COEFS=${#INNOVATION_COEFS[@]}
NUM_CONFIGS=$((NUM_DIMS * NUM_COEFS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

DIM_IDX=$((TASK_ID % NUM_DIMS))
COEF_IDX=$((TASK_ID / NUM_DIMS))

PROJ_DIM="${PROJ_DIMS[$DIM_IDX]}"
INNOVATION_COEF="${INNOVATION_COEFS[$COEF_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="innovation_aux_crate_heads_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="innovation_aux,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},proj_${PROJ_DIM},innovation_coef_${INNOVATION_COEF},headopt_muon,headarch_crate"

EXP_NAME="ppo_muon_${ENV_NAME}_innovationaux_proj${PROJ_DIM}_coef${COEF_IDX}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} innovation_proj_dim=${PROJ_DIM} innovation_coef=${INNOVATION_COEF} seed=${SEED}"
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
  --innovation-proj-dim "${PROJ_DIM}"
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
  --use-heads-muon
  --use-crate-head
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py"   "${COMMON_ARGS[@]}"
