#!/bin/bash
#SBATCH --job-name=diagnose-ant-phase-change-cnn-crate-4seeds-1m
#SBATCH --output=slurm_logs/diagnose_ant_phase_change_cnn_crate_4seeds_1m_%A_%a.out
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
VICREG_VAR_COEF="${4:-0.0}"
VICREG_VAR_TARGET="${5:-0.1}"
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

SEED="${SEEDS[$TASK_ID]}"
if [[ "${VICREG_VAR_COEF}" == "0" || "${VICREG_VAR_COEF}" == "0.0" || "${VICREG_VAR_COEF}" == "0e0" ]]; then
  VICREG_LABEL="vicregoff"
  VICREG_TAG="vicreg_off"
else
  VICREG_LABEL="vicregon"
  VICREG_TAG="vicreg_on"
fi
GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="diagnose_ant_phase_change_cnn_crate_4seeds_1m_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="ant_phase_change,diagnosis,env_ant,backend_${BACKEND},seed_${SEED},headopt_muon,headarch_crate,encoder_cnn,innovation_off,lr_anneal_off,${VICREG_TAG},1m_steps,probe_100k"

EXP_NAME="ppo_muon_ant_phasechange_diag_cnncrate_${VICREG_LABEL}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${#SEEDS[@]} group=${GROUP_NAME}"
echo "Config: env=ant backend=${BACKEND} seed=${SEED} encoder=cnn crate_head=true anneal_lr=false vicreg_var_coef=${VICREG_VAR_COEF} vicreg_var_target=${VICREG_VAR_TARGET}"
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
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --encoder-type cnn
  --sigreg-mode off
  --vicreg-var-coef "${VICREG_VAR_COEF}"
  --vicreg-var-target "${VICREG_VAR_TARGET}"
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
  --probe-interval 100000
  --probe-n-eval-steps 400
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" "${COMMON_ARGS[@]}"
