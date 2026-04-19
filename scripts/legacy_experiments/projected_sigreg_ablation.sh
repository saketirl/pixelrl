#!/bin/bash
#SBATCH --job-name=proj-sigreg-humanoid
#SBATCH --output=slurm_logs/proj_sigreg_humanoid_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

# Humanoid-only projected SIGReg ablation with one seed.
# Sweeps:
#   optimizer in {adam, muon}
#   encoder_crate_block in {off, on}
#   sigreg_proj_dim in {32, 48}
# Warmup/ramp are fixed on for all runs.

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
ENCODER_SCALE="0.5"
OPT_CONDITIONS=(adam muon)
CRATE_CONDITIONS=(off on)
PROJ_DIMS=(32 48)
SIGREG_COEF="1e-3"
SIGREG_NUM_SLICES="16"
SIGREG_NUM_T="8"
SIGREG_T_MAX="5.0"
SIGREG_WARMUP_UPDATES="500"
SIGREG_RAMP_UPDATES="500"
ENCODER_CRATE_STEP_SIZE="0.1"

NUM_OPTS=${#OPT_CONDITIONS[@]}
NUM_CRATE=${#CRATE_CONDITIONS[@]}
NUM_DIMS=${#PROJ_DIMS[@]}
NUM_CONFIGS=$((NUM_OPTS * NUM_CRATE * NUM_DIMS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

OPT_IDX=$((TASK_ID % NUM_OPTS))
CRATE_IDX=$(((TASK_ID / NUM_OPTS) % NUM_CRATE))
DIM_IDX=$((TASK_ID / (NUM_OPTS * NUM_CRATE)))

OPT_CONDITION="${OPT_CONDITIONS[$OPT_IDX]}"
CRATE_CONDITION="${CRATE_CONDITIONS[$CRATE_IDX]}"
PROJ_DIM="${PROJ_DIMS[$DIM_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="proj_sigreg_humanoid_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="proj_sigreg_humanoid,encoder_sigreg_cnn,opt_${OPT_CONDITION},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},proj_${PROJ_DIM},encoder_crate_${CRATE_CONDITION},sigreg_warmup_on"

EXP_NAME="ppo_projsigreg_${OPT_CONDITION}_${ENV_NAME}_proj${PROJ_DIM}_crate${CRATE_CONDITION}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} optimizer=${OPT_CONDITION} proj_dim=${PROJ_DIM} encoder_crate=${CRATE_CONDITION} seed=${SEED}"
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

ARCH_ARGS=(
  --encoder-type sigreg_cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --max-grad-norm 0.05
  --encoder-tanh-scale "${ENCODER_SCALE}"
  --encoder-crate-step-size "${ENCODER_CRATE_STEP_SIZE}"
  --sigreg-mode projected
  --sigreg-coef "${SIGREG_COEF}"
  --sigreg-proj-dim "${PROJ_DIM}"
  --sigreg-num-slices "${SIGREG_NUM_SLICES}"
  --sigreg-num-t "${SIGREG_NUM_T}"
  --sigreg-t-max "${SIGREG_T_MAX}"
  --sigreg-warmup-updates "${SIGREG_WARMUP_UPDATES}"
  --sigreg-ramp-updates "${SIGREG_RAMP_UPDATES}"
)

if [[ "${CRATE_CONDITION}" == "on" ]]; then
  ARCH_ARGS+=(--encoder-use-crate-block)
else
  ARCH_ARGS+=(--no-encoder-use-crate-block)
fi

OPT_ARGS=()
if [[ "${OPT_CONDITION}" == "muon" ]]; then
  OPT_ARGS+=(--use-heads-muon)
else
  OPT_ARGS+=(--no-use-heads-muon)
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py"   "${COMMON_ARGS[@]}"   "${ARCH_ARGS[@]}"   "${OPT_ARGS[@]}"
