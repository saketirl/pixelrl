#!/bin/bash
#SBATCH --job-name=cnn-enc-final-muon
#SBATCH --output=slurm_logs/cnn_enc_final_muon_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Humanoid-only default-CNN experiment.
# Encoder uses Adam everywhere except the final CNN Dense kernel, which uses manifold MUON.
# No SIGReg, no projector bottleneck, no encoder-side regularizers.
# Sweep:
#   actor/critic optimizer in {muon, adam}
#   actor/critic head architecture in {plain, crate}

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
HEAD_OPT_CONDITIONS=(muon adam)
HEAD_ARCH_CONDITIONS=(plain crate)
NUM_HEAD_OPTS=${#HEAD_OPT_CONDITIONS[@]}
NUM_HEAD_ARCH=${#HEAD_ARCH_CONDITIONS[@]}
NUM_CONFIGS=$((NUM_HEAD_OPTS * NUM_HEAD_ARCH))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

HEAD_OPT_IDX=$((TASK_ID % NUM_HEAD_OPTS))
HEAD_ARCH_IDX=$((TASK_ID / NUM_HEAD_OPTS))

HEAD_OPT_CONDITION="${HEAD_OPT_CONDITIONS[$HEAD_OPT_IDX]}"
HEAD_ARCH_CONDITION="${HEAD_ARCH_CONDITIONS[$HEAD_ARCH_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_enc_final_muon_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn_enc_final_muon,env_${ENV_NAME},backend_${BACKEND},seed_${SEED},headopt_${HEAD_OPT_CONDITION},headarch_${HEAD_ARCH_CONDITION},encoder_final_muon_on,encoder_cnn,sigreg_off"

EXP_NAME="ppo_${HEAD_OPT_CONDITION}_${ENV_NAME}_cnnencfinalmuon_head${HEAD_ARCH_CONDITION}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} head_opt=${HEAD_OPT_CONDITION} head_arch=${HEAD_ARCH_CONDITION} seed=${SEED}"
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
  --encoder-type cnn
  --sigreg-mode off
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --encoder-muon-lr 0.001
  --max-grad-norm 0.05
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --encoder-muon-max-grad-norm 1
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --use-encoder-final-muon
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

HEAD_ARGS=()
if [[ "${HEAD_OPT_CONDITION}" == "muon" ]]; then
  HEAD_ARGS+=(--use-heads-muon)
else
  HEAD_ARGS+=(--no-use-heads-muon)
fi

if [[ "${HEAD_ARCH_CONDITION}" == "crate" ]]; then
  HEAD_ARGS+=(--use-crate-head)
else
  HEAD_ARGS+=(--no-use-crate-head)
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py"   "${COMMON_ARGS[@]}"   "${HEAD_ARGS[@]}"
