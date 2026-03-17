#!/bin/bash
#SBATCH --job-name=cnn-steifel-hum
#SBATCH --output=slurm_logs/cnn_steifel_hum_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

set -euo pipefail

# Run four Humanoid PPO jobs:
#   TASK_ID=0 -> Stiefel encoder + actor/critic MUON (use-heads-muon)
#   TASK_ID=1 -> Stiefel encoder + actor/critic Adam (no-use-heads-muon)
#   TASK_ID=2 -> Vanilla CNN encoder + actor/critic MUON + CRATE head
#   TASK_ID=3 -> Vanilla CNN encoder + actor/critic MUON + CRATE head + CNN SIGReg

WANDB_PROJECT="${1:-scott}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
STIEFEL_LATENT_DIM="${4:-64}"
WANDB_TRACKING="${WANDB_TRACKING:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PROBE_INTERVAL="${PROBE_INTERVAL:-100000}"
CRATE_STEP_SIZE="${CRATE_STEP_SIZE:-0.1}"

CNN_SIGREG_COEF="${CNN_SIGREG_COEF:-1e-3}"
CNN_SIGREG_WARMUP_UPDATES="${CNN_SIGREG_WARMUP_UPDATES:-0}"
CNN_SIGREG_RAMPUP_UPDATES="${CNN_SIGREG_RAMPUP_UPDATES:-50}"
CNN_SIGREG_NUM_SLICES="${CNN_SIGREG_NUM_SLICES:-64}"
CNN_SIGREG_T_POINTS="${CNN_SIGREG_T_POINTS:-17}"
CNN_SIGREG_T_MIN="${CNN_SIGREG_T_MIN:--5.0}"
CNN_SIGREG_T_MAX="${CNN_SIGREG_T_MAX:-5.0}"

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

MODES=(stiefel_muon stiefel_adam cnn_crate_muon cnn_crate_muon_sigreg)
NUM_MODES=${#MODES[@]}

if (( TASK_ID < 0 || TASK_ID >= NUM_MODES )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_MODES - 1))." >&2
  exit 1
fi

MODE="${MODES[$TASK_ID]}"
ENV_NAME="humanoid"
BACKEND="spring"
SEED=0

LATENT_DYN_COEF="${LATENT_DYN_COEF:-1e-3}"
LATENT_COV_COEF="${LATENT_COV_COEF:-0.0}"
LATENT_VAR_COEF="${LATENT_VAR_COEF:-0.0}"
LATENT_WHITEN_COEF="${LATENT_WHITEN_COEF:-0.0}"
LATENT_SUBSPACE_COEF="${LATENT_SUBSPACE_COEF:-0.0}"

case "${MODE}" in
  stiefel_muon)
    OPT_CONDITION="muon"
    MODE_TAG="stiefel_muon"
    WANDB_TAGS="cnn,steifel,env_${ENV_NAME},mode_${MODE_TAG},opt_${OPT_CONDITION},backend_${BACKEND},seed_${SEED},latent_dim_${STIEFEL_LATENT_DIM}"
    EXP_NAME="ppo_cnn_steifel_${OPT_CONDITION}_${ENV_NAME}_z${STIEFEL_LATENT_DIM}_b${BACKEND}_s${SEED}_t${TASK_ID}"
    ENCODER_ARGS=(
      --encoder-type stiefel_cnn
      --stiefel-latent-dim "${STIEFEL_LATENT_DIM}"
      --use-latent-dynamics
      --latent-transition-type linear
      --latent-dyn-coef "${LATENT_DYN_COEF}"
      --latent-dyn-warmup-updates 10
      --latent-dyn-rampup-updates 50
      --latent-cov-coef "${LATENT_COV_COEF}"
      --latent-var-coef "${LATENT_VAR_COEF}"
      --latent-whiten-coef "${LATENT_WHITEN_COEF}"
      --latent-subspace-coef "${LATENT_SUBSPACE_COEF}"
    )
    ;;
  stiefel_adam)
    OPT_CONDITION="adam"
    MODE_TAG="stiefel_adam"
    WANDB_TAGS="cnn,steifel,env_${ENV_NAME},mode_${MODE_TAG},opt_${OPT_CONDITION},backend_${BACKEND},seed_${SEED},latent_dim_${STIEFEL_LATENT_DIM}"
    EXP_NAME="ppo_cnn_steifel_${OPT_CONDITION}_${ENV_NAME}_z${STIEFEL_LATENT_DIM}_b${BACKEND}_s${SEED}_t${TASK_ID}"
    ENCODER_ARGS=(
      --encoder-type stiefel_cnn
      --stiefel-latent-dim "${STIEFEL_LATENT_DIM}"
      --use-latent-dynamics
      --latent-transition-type linear
      --latent-dyn-coef "${LATENT_DYN_COEF}"
      --latent-dyn-warmup-updates 10
      --latent-dyn-rampup-updates 50
      --latent-cov-coef "${LATENT_COV_COEF}"
      --latent-var-coef "${LATENT_VAR_COEF}"
      --latent-whiten-coef "${LATENT_WHITEN_COEF}"
      --latent-subspace-coef "${LATENT_SUBSPACE_COEF}"
    )
    ;;
  cnn_crate_muon)
    OPT_CONDITION="muon"
    MODE_TAG="cnn_crate_muon"
    WANDB_TAGS="cnn,vanilla,crate,env_${ENV_NAME},mode_${MODE_TAG},opt_${OPT_CONDITION},backend_${BACKEND},seed_${SEED}"
    EXP_NAME="ppo_cnn_crate_${OPT_CONDITION}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"
    ENCODER_ARGS=(
      --encoder-type cnn
      --use-crate-head
      --crate-step-size "${CRATE_STEP_SIZE}"
    )
    ;;
  cnn_crate_muon_sigreg)
    OPT_CONDITION="muon"
    MODE_TAG="cnn_crate_muon_sigreg"
    WANDB_TAGS="cnn,vanilla,crate,sigreg,env_${ENV_NAME},mode_${MODE_TAG},opt_${OPT_CONDITION},backend_${BACKEND},seed_${SEED}"
    EXP_NAME="ppo_cnn_crate_sigreg_${OPT_CONDITION}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"
    ENCODER_ARGS=(
      --encoder-type cnn
      --use-crate-head
      --crate-step-size "${CRATE_STEP_SIZE}"
      --cnn-sigreg-coef "${CNN_SIGREG_COEF}"
      --cnn-sigreg-warmup-updates "${CNN_SIGREG_WARMUP_UPDATES}"
      --cnn-sigreg-rampup-updates "${CNN_SIGREG_RAMPUP_UPDATES}"
      --cnn-sigreg-num-slices "${CNN_SIGREG_NUM_SLICES}"
      --cnn-sigreg-t-points "${CNN_SIGREG_T_POINTS}"
      --cnn-sigreg-t-min "${CNN_SIGREG_T_MIN}"
      --cnn-sigreg-t-max "${CNN_SIGREG_T_MAX}"
    )
    ;;
  *)
    echo "Invalid mode: ${MODE}" >&2
    exit 1
    ;;
esac

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_steifel_hum_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS

echo "Running TASK_ID=${TASK_ID}/${NUM_MODES} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} mode=${MODE_TAG} opt=${OPT_CONDITION} seed=${SEED}"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
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
  --seed "${SEED}"
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
  --probe-interval "${PROBE_INTERVAL}"
  --probe-n-eval-steps 400
  --probe-output-dir "probe_results/${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${WANDB_TRACKING}" == "0" || "${WANDB_TRACKING}" == "false" || "${WANDB_TRACKING}" == "off" ]]; then
  COMMON_ARGS+=(--no-track)
else
  COMMON_ARGS+=(--track)
fi

if [[ "${OPT_CONDITION}" == "muon" ]]; then
  COMMON_ARGS+=(--use-heads-muon)
else
  COMMON_ARGS+=(--no-use-heads-muon)
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${ENCODER_ARGS[@]}"
