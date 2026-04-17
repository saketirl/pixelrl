#!/bin/bash
#SBATCH --job-name=diagnose-ant-phase-change-cnn-crate-bottleneckreg-4seeds-1m
#SBATCH --output=slurm_logs/diagnose_ant_phase_change_cnn_crate_bottleneckreg_4seeds_1m_%A_%a.out
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
BOTTLENECK_VAR_COEF="${4:-0.001}"
BOTTLENECK_VAR_TARGET="${5:-0.05}"
BOTTLENECK_VAR_BOTTOM_FRAC="${6:-0.25}"
BOTTLENECK_PRE_LN_COEF="${7:-0.001}"
BOTTLENECK_PRE_LN_MAX_STD="${8:-12.0}"
BOTTLENECK_REG_STOP_UPDATES="${9:-40}"
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
GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="diagnose_ant_phase_change_cnn_crate_bottleneckreg_4seeds_1m_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="ant_phase_change,diagnosis,env_ant,backend_${BACKEND},seed_${SEED},headopt_muon,headarch_crate,encoder_cnn,innovation_off,lr_anneal_off,bottleneck_reg,1m_steps,probe_100k"

EXP_NAME="ppo_muon_ant_phasechange_diag_cnncrate_bneckreg_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${#SEEDS[@]} group=${GROUP_NAME}"
echo "Config: env=ant backend=${BACKEND} seed=${SEED} encoder=cnn crate_head=true anneal_lr=false bottleneck_var_coef=${BOTTLENECK_VAR_COEF} bottleneck_var_target=${BOTTLENECK_VAR_TARGET} bottleneck_var_bottom_frac=${BOTTLENECK_VAR_BOTTOM_FRAC} bottleneck_pre_ln_coef=${BOTTLENECK_PRE_LN_COEF} bottleneck_pre_ln_max_std=${BOTTLENECK_PRE_LN_MAX_STD} bottleneck_reg_stop_updates=${BOTTLENECK_REG_STOP_UPDATES}"
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
  --vicreg-var-coef 0.0
  --bottleneck-var-coef "${BOTTLENECK_VAR_COEF}"
  --bottleneck-var-target "${BOTTLENECK_VAR_TARGET}"
  --bottleneck-var-bottom-frac "${BOTTLENECK_VAR_BOTTOM_FRAC}"
  --bottleneck-pre-ln-coef "${BOTTLENECK_PRE_LN_COEF}"
  --bottleneck-pre-ln-max-std "${BOTTLENECK_PRE_LN_MAX_STD}"
  --bottleneck-reg-stop-updates "${BOTTLENECK_REG_STOP_UPDATES}"
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
