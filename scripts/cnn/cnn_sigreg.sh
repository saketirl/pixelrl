#!/bin/bash
#SBATCH --job-name=cnn-sigreg
#SBATCH --output=slurm_logs/cnn_sigreg_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

set -euo pipefail

# Vanilla CNN + SIGReg baseline on humanoid.
# - encoder: cnn
# - SIGReg on pre-tanh CNN latents
# - TASK_ID=0 => Adam actor/critic, TASK_ID=1 => MUON actor/critic

WANDB_PROJECT="${1:-scott}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WANDB_TRACKING="${WANDB_TRACKING:-1}"
TOTAL_MODES=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"
mkdir -p "${REPO_ROOT}/probe_results"

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

if (( TASK_ID < 0 || TASK_ID >= TOTAL_MODES )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((TOTAL_MODES - 1))." >&2
  exit 1
fi

env_name="humanoid"
backend="spring"
seed=0

if (( TASK_ID == 0 )); then
  OPT_CONDITION="adam"
else
  OPT_CONDITION="muon"
fi

CNN_SIGREG_COEF="${CNN_SIGREG_COEF:-1e-3}"
CNN_SIGREG_WARMUP_UPDATES="${CNN_SIGREG_WARMUP_UPDATES:-0}"
CNN_SIGREG_RAMPUP_UPDATES="${CNN_SIGREG_RAMPUP_UPDATES:-50}"
CNN_SIGREG_NUM_SLICES="${CNN_SIGREG_NUM_SLICES:-64}"
CNN_SIGREG_T_POINTS="${CNN_SIGREG_T_POINTS:-17}"
CNN_SIGREG_T_MIN="${CNN_SIGREG_T_MIN:--5.0}"
CNN_SIGREG_T_MAX="${CNN_SIGREG_T_MAX:-5.0}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_sigreg_humanoid_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn,humanoid,sigreg,opt_${OPT_CONDITION},env_${env_name},backend_${backend},seed_${seed}"

EXP_NAME="ppo_cnn_sigreg_${OPT_CONDITION}_${env_name}_b${backend}_s${seed}_t${TASK_ID}"

COMMON_ARGS=(
  --env-name "${env_name}"
  --backend "${backend}"
  --n-envs 128
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
  --seed "${seed}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --anneal-lr
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --weight-decay 0.0
  --max-grad-norm 0.5
  --encoder-tanh-scale 0.5
  --exp-name "${EXP_NAME}"
  --probe-interval 0
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${WANDB_TRACKING}" == "0" || "${WANDB_TRACKING}" == "false" || "${WANDB_TRACKING}" == "off" ]]; then
  COMMON_ARGS+=(--no-track)
else
  COMMON_ARGS+=(--track)
fi

ENCODER_ARGS=(
  --encoder-type cnn
  --cnn-sigreg-coef "${CNN_SIGREG_COEF}"
  --cnn-sigreg-warmup-updates "${CNN_SIGREG_WARMUP_UPDATES}"
  --cnn-sigreg-rampup-updates "${CNN_SIGREG_RAMPUP_UPDATES}"
  --cnn-sigreg-num-slices "${CNN_SIGREG_NUM_SLICES}"
  --cnn-sigreg-t-points "${CNN_SIGREG_T_POINTS}"
  --cnn-sigreg-t-min "${CNN_SIGREG_T_MIN}"
  --cnn-sigreg-t-max "${CNN_SIGREG_T_MAX}"
)

if [[ "${OPT_CONDITION}" == "muon" ]]; then
  ENCODER_ARGS+=(--use-heads-muon)
else
  ENCODER_ARGS+=(--no-use-heads-muon)
fi

echo "Running: env=${env_name} project=${WANDB_PROJECT} opt=${OPT_CONDITION} exp=${EXP_NAME}"

echo "Config:"
echo "  TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "  CNN_SIGREG_COEF=${CNN_SIGREG_COEF}"
echo "  WANDB_TRACKING=${WANDB_TRACKING}"

echo "  OPT_CONDITION=${OPT_CONDITION}"

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ENCODER_ARGS[@]}"
