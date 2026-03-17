#!/bin/bash
#SBATCH --job-name=cnn-lejepa-ablate
#SBATCH --output=slurm_logs/cnn_lejepa_ablate_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-95%12

set -euo pipefail

WANDB_PROJECT="scott"
WANDB_ENTITY="${1:-}"
TOTAL_TIMESTEPS="${2:-10000000}"
OPT_NAME="${3:-muon}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
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
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

SEEDS=(0 1 2 3 4 5)
CONFIG_NAMES=(
  baseline
  legacy
  auxcoef
  ramp
  sigreg
  view_shift
  view_shift_noise
  layernorm_off
  sweep_aux_1e4
  sweep_aux_1e3
  sweep_aux_1e2
  sweep_lambda_01
  sweep_lambda_03
  sweep_lambda_05
  sweep_slices_256
  sweep_slices_512
)

NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=${#CONFIG_NAMES[@]}
NUM_TASKS=$((NUM_SEEDS * NUM_CONFIGS))

if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_TASKS - 1))." >&2
  exit 1
fi

CONFIG_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))
SEED="${SEEDS[$SEED_IDX]}"
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"

if [[ "${OPT_NAME}" == "muon" ]]; then
  OPT_FLAG="--use-heads-muon"
elif [[ "${OPT_NAME}" == "adam" ]]; then
  OPT_FLAG="--no-use-heads-muon"
else
  echo "Invalid OPT_NAME=${OPT_NAME}. Expected muon or adam." >&2
  exit 1
fi

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_lejepa_humanoid_ablation_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn,lejepa_ablation,env_humanoid,opt_${OPT_NAME},seed_${SEED},config_${CONFIG_NAME}"

EXP_NAME="ppo_cnn_lejepa_${CONFIG_NAME}_humanoid_${OPT_NAME}_s${SEED}_t${TASK_ID}"

COMMON_ARGS=(
  --env-name humanoid
  --backend spring
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
  --encoder-type cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --weight-decay 0.0
  --max-grad-norm 0.5
  --encoder-tanh-scale 0.5
  --probe-interval 100000
  --probe-n-eval-steps 400
  --probe-output-dir "probe_results/${EXP_NAME}"
  --save-checkpoint
  --checkpoint-dir checkpoints
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ "${OPT_NAME}" == "muon" ]]; then
  COMMON_ARGS+=(--use-heads-muon)
else
  COMMON_ARGS+=(--no-use-heads-muon)
fi

append_safe_lejepa_defaults() {
  CONFIG_ARGS+=(
    --lejepa
    --lejepa-lambda 0.1
    --lejepa-aux-coef 1e-3
    --lejepa-warmup-updates 10
    --lejepa-rampup-updates 50
    --lejepa-proj-dim 128
    --lejepa-num-views 2
    --lejepa-num-slices 256
    --lejepa-t-points 17
    --lejepa-t-min -5.0
    --lejepa-t-max 5.0
    --lejepa-view-mode shift
    --lejepa-sigreg-mode epps_pulley
  )
}

CONFIG_ARGS=()
case "${CONFIG_NAME}" in
  baseline)
    ;;
  legacy)
    CONFIG_ARGS+=(
      --lejepa
      --lejepa-lambda 0.1
      --lejepa-aux-coef 1.0
      --lejepa-warmup-updates 0
      --lejepa-rampup-updates 0
      --lejepa-proj-dim 128
      --lejepa-num-views 2
      --lejepa-num-slices 64
      --lejepa-t-points 17
      --lejepa-t-min -5.0
      --lejepa-t-max 5.0
      --lejepa-view-mode shift
      --lejepa-sigreg-mode legacy
    )
    ;;
  auxcoef)
    CONFIG_ARGS+=(
      --lejepa
      --lejepa-lambda 0.1
      --lejepa-aux-coef 1e-3
      --lejepa-warmup-updates 0
      --lejepa-rampup-updates 0
      --lejepa-proj-dim 128
      --lejepa-num-views 2
      --lejepa-num-slices 64
      --lejepa-t-points 17
      --lejepa-t-min -5.0
      --lejepa-t-max 5.0
      --lejepa-view-mode shift
      --lejepa-sigreg-mode legacy
    )
    ;;
  ramp)
    CONFIG_ARGS+=(
      --lejepa
      --lejepa-lambda 0.1
      --lejepa-aux-coef 1e-3
      --lejepa-warmup-updates 10
      --lejepa-rampup-updates 50
      --lejepa-proj-dim 128
      --lejepa-num-views 2
      --lejepa-num-slices 64
      --lejepa-t-points 17
      --lejepa-t-min -5.0
      --lejepa-t-max 5.0
      --lejepa-view-mode shift
      --lejepa-sigreg-mode legacy
    )
    ;;
  sigreg)
    append_safe_lejepa_defaults
    ;;
  view_shift)
    append_safe_lejepa_defaults
    ;;
  view_shift_noise)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-view-mode shift_noise)
    ;;
  layernorm_off)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--no-lejepa-use-layernorm)
    ;;
  sweep_aux_1e4)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-aux-coef 1e-4)
    ;;
  sweep_aux_1e3)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-aux-coef 1e-3)
    ;;
  sweep_aux_1e2)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-aux-coef 1e-2)
    ;;
  sweep_lambda_01)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-lambda 0.1)
    ;;
  sweep_lambda_03)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-lambda 0.3)
    ;;
  sweep_lambda_05)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-lambda 0.5)
    ;;
  sweep_slices_256)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-num-slices 256)
    ;;
  sweep_slices_512)
    append_safe_lejepa_defaults
    CONFIG_ARGS+=(--lejepa-num-slices 512)
    ;;
  *)
    echo "Unhandled CONFIG_NAME=${CONFIG_NAME}" >&2
    exit 1
    ;;
esac

echo "Running TASK_ID=${TASK_ID}/${NUM_TASKS} group=${GROUP_NAME}"
echo "Config: env=humanoid backend=spring opt=${OPT_NAME} seed=${SEED} variant=${CONFIG_NAME}"
echo "Exp: ${EXP_NAME}"

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax.py" \
  "${COMMON_ARGS[@]}" \
  "${CONFIG_ARGS[@]}"
