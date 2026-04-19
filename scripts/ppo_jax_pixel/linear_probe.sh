#!/bin/bash
#SBATCH --job-name=linear-probe
#SBATCH --output=slurm_logs/linear_probe_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%6

set -euo pipefail

# CNN training with online linear probe analysis.
# Array job sweeps:
#   - head optimizer: {MUON, Adam}
#   - seed:           {0,1,2,3,4,5} by default
#
# Usage:
#   # Default seed sweep (12 tasks total: 2 opts x 6 seeds)
#   sbatch scripts/ppo_jax_pixel/linear_probe.sh [env_name] [backend]
#
#   # Single-seed run (2 tasks total: MUON + Adam)
#   sbatch --array=0-1 scripts/ppo_jax_pixel/linear_probe.sh [env_name] [backend] [seed]
#
# Defaults:
#   env_name = inverted_pendulum
#   backend  = generalized
#   seeds    = 0 1 2 3 4 5
#
# IMPORTANT: always submit from inside the worktree directory:
#   cd /home/guests/arjun/pixelrl/.worktrees/vit-muon
#   sbatch scripts/ppo_jax_pixel/linear_probe.sh

ENV_NAME="${1:-inverted_pendulum}"
BACKEND="${2:-generalized}"
SEED_OVERRIDE="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"
mkdir -p "${REPO_ROOT}/probe_results"
mkdir -p "${REPO_ROOT}/checkpoints"

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

WANDB_PROJECT="${WANDB_PROJECT:-linear_probe}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

OPT_NAMES=(muon adam)
if [[ -n "${SEED_OVERRIDE}" ]]; then
  SEEDS=("${SEED_OVERRIDE}")
else
  SEEDS=(0 1 2 3 4 5)
fi

NUM_OPTS=${#OPT_NAMES[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_OPTS * NUM_SEEDS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1)) for ${NUM_OPTS} opts x ${NUM_SEEDS} seeds." >&2
  echo "If using a fixed seed override, submit with --array=0-1." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
OPT_IDX=$((TASK_ID / NUM_SEEDS))

SEED="${SEEDS[$SEED_IDX]}"
OPT_NAME="${OPT_NAMES[$OPT_IDX]}"
if [[ "${OPT_NAME}" == "muon" ]]; then
  OPT_FLAG="--use-heads-muon"
else
  OPT_FLAG="--no-use-heads-muon"
fi

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="linear_probe_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="linear_probe,cnn,${OPT_NAME}_heads,env_${ENV_NAME},backend_${BACKEND},seed_${SEED}"

EXP_NAME="probe_cnn_${OPT_NAME}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running linear probe experiment (task ${TASK_ID}/${NUM_CONFIGS})"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED}"
echo "Head optimizer: ${OPT_NAME}"
echo "Exp: ${EXP_NAME}"
echo "REPO_ROOT: ${REPO_ROOT}"

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
  --env-name "${ENV_NAME}" \
  --backend "${BACKEND}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --track \
  --wandb-project-name "${WANDB_PROJECT}" \
  --n-envs 128 \
  --hw 84 \
  --num-steps 10 \
  --num-minibatches 32 \
  --update-epochs 4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --frame-stack 4 \
  --action-repeat 4 \
  --anneal-lr \
  --muon-dual-lr 0.01 \
  --muon-dual-steps 5 \
  --actor-muon-max-grad-norm 100 \
  --critic-muon-max-grad-norm 1 \
  ${OPT_FLAG} \
  --debug-repr \
  --log-interval 1 \
  --encoder-type cnn \
  --encoder-lr 3e-4 \
  --heads-adam-lr 3e-4 \
  --heads-muon-lr 0.001 \
  --max-grad-norm 0.05 \
  --encoder-tanh-scale 0.5 \
  --probe-interval 100000 \
  --probe-n-eval-steps 400 \
  --probe-output-dir "probe_results/${EXP_NAME}" \
  --save-checkpoint \
  --checkpoint-dir "checkpoints" \
  ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"}
