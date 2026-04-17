#!/bin/bash
#SBATCH --job-name=diagnose-ant-phase-change-splitcnn-crate-seedfactor-encoderfinalmuon-4seeds-1m
#SBATCH --output=slurm_logs/diagnose_ant_phase_change_splitcnn_crate_seedfactor_encoderfinalmuon_4seeds_1m_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

WANDB_PROJECT="${1:-encoder}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-1000000}"
FACTOR_MODE="${4:-fixed_init}"
FIXED_SEED="${5:-0}"
ENCODER_MUON_LR="${6:-0.001}"
ENCODER_MUON_MAX_GRAD_NORM="${7:-1.0}"
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

SEEDS=(0 1 2 3)
BACKEND="spring"

if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$(( ${#SEEDS[@]} - 1 ))." >&2
  exit 1
fi

VARIED_SEED="${SEEDS[$TASK_ID]}"
case "${FACTOR_MODE}" in
  fixed_init)
    INIT_SEED="${FIXED_SEED}"
    DATA_SEED="${VARIED_SEED}"
    FACTOR_TAG="fixed_init"
    ;;
  fixed_data)
    INIT_SEED="${VARIED_SEED}"
    DATA_SEED="${FIXED_SEED}"
    FACTOR_TAG="fixed_data"
    ;;
  *)
    echo "Unsupported FACTOR_MODE=${FACTOR_MODE}. Expected fixed_init or fixed_data." >&2
    exit 1
    ;;
esac

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="diagnose_ant_phase_change_splitcnn_crate_seedfactor_encoderfinalmuon_${FACTOR_TAG}_fix${FIXED_SEED}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="ant_phase_change,diagnosis,env_ant,backend_${BACKEND},seedfactor,${FACTOR_TAG},fixed_seed_${FIXED_SEED},varied_seed_${VARIED_SEED},init_seed_${INIT_SEED},data_seed_${DATA_SEED},headopt_muon,headarch_crate,encoder_split_cnn,encoder_final_muon,innovation_off,lr_anneal_off,probe_off,1m_steps"

EXP_NAME="ppo_muon_ant_phasechange_diag_cnncrate_seedfactor_encoderfinalmuon_${FACTOR_TAG}_fix${FIXED_SEED}_i${INIT_SEED}_d${DATA_SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${#SEEDS[@]} group=${GROUP_NAME}"
echo "Config: env=ant backend=${BACKEND} factor_mode=${FACTOR_MODE} fixed_seed=${FIXED_SEED} varied_seed=${VARIED_SEED} init_seed=${INIT_SEED} data_seed=${DATA_SEED} encoder=split_cnn crate_head=true anneal_lr=false encoder_final_muon=true encoder_muon_lr=${ENCODER_MUON_LR} encoder_muon_max_grad_norm=${ENCODER_MUON_MAX_GRAD_NORM} probe_interval=0"
echo "Exp: ${EXP_NAME}"

COMMON_ARGS=(
  --env-name "ant"
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
  --seed "${VARIED_SEED}"
  --init-seed "${INIT_SEED}"
  --data-seed "${DATA_SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --encoder-type split_cnn
  --sigreg-mode off
  --vicreg-var-coef 0.0
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --encoder-muon-lr "${ENCODER_MUON_LR}"
  --max-grad-norm 0.05
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --encoder-muon-max-grad-norm "${ENCODER_MUON_MAX_GRAD_NORM}"
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --use-heads-muon
  --use-encoder-final-muon
  --use-crate-head
  --probe-interval 0
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
