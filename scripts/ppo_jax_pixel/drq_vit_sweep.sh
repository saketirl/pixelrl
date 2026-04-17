#!/bin/bash
#SBATCH --job-name=drq-vit
#SBATCH --output=slurm_logs/drq_vit_sweep_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2

set -euo pipefail

# Two parallel runs of a token-compressed DrQViT encoder on halfcheetah:
#   GPU 0: actor/critic optimised with Muon
#   GPU 1: actor/critic optimised with Adam
#
# Usage:
#   sbatch scripts/ppo_jax_pixel/drq_vit_sweep.sh [wandb_project] [wandb_entity] [total_timesteps] [seed]
# Example:
#   sbatch scripts/ppo_jax_pixel/drq_vit_sweep.sh benchmark my_entity 10000000 0

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"
ENV_NAME="halfcheetah"
DRQ_TOKEN_DOWNSAMPLE=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

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

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="drq_vit_sweep_${ENV_NAME}_${GROUP_ID}"
export WANDB_RUN_GROUP="${GROUP_NAME}"

echo "Running env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "Using worktree repo root: ${REPO_ROOT}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend spring
  --n-envs 128
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --hw 84
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --encoder-lr 5e-5
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
  --weight-decay 1e-4
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --max-grad-norm 0.5
  --seed "${SEED}"
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --anneal-lr
  --encoder-warmup-updates 500
  --encoder-type drq_vit
  --encoder-tanh-scale 0.25
  --vit-hidden-size 192
  --vit-mlp-dim 768
  --vit-num-heads 6
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --drq-vit-stem-channels 32
  --drq-vit-token-downsample "${DRQ_TOKEN_DOWNSAMPLE}"
  --no-drq-vit-apply-output-tanh
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"

export WANDB_TAGS="drq_vit_sweep,encoder_drq_vit,drq_token_downsample_${DRQ_TOKEN_DOWNSAMPLE},drq_out_tanh_false,heads_muon_true,env_${ENV_NAME},seed_${SEED}"
CUDA_VISIBLE_DEVICES=0 uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
  "${COMMON_ARGS[@]}" \
  --use-heads-muon \
  --exp-name "ppo_drq_vit_tok${DRQ_TOKEN_DOWNSAMPLE}_notanh_muon_${ENV_NAME}_s${SEED}" \
  > "${REPO_ROOT}/slurm_logs/drq_vit_sweep_${GROUP_ID}_muon.out" 2>&1 &
PID_MUON=$!

export WANDB_TAGS="drq_vit_sweep,encoder_drq_vit,drq_token_downsample_${DRQ_TOKEN_DOWNSAMPLE},drq_out_tanh_false,heads_muon_false,env_${ENV_NAME},seed_${SEED}"
CUDA_VISIBLE_DEVICES=1 uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
  "${COMMON_ARGS[@]}" \
  --no-use-heads-muon \
  --exp-name "ppo_drq_vit_tok${DRQ_TOKEN_DOWNSAMPLE}_notanh_adam_${ENV_NAME}_s${SEED}" \
  > "${REPO_ROOT}/slurm_logs/drq_vit_sweep_${GROUP_ID}_adam.out" 2>&1 &
PID_ADAM=$!

echo "Launched Muon run (PID ${PID_MUON}) on GPU 0"
echo "Launched Adam  run (PID ${PID_ADAM}) on GPU 1"

wait ${PID_MUON}
MUON_EXIT=$?
wait ${PID_ADAM}
ADAM_EXIT=$?

echo "Muon run exit code: ${MUON_EXIT}"
echo "Adam  run exit code: ${ADAM_EXIT}"

[[ ${MUON_EXIT} -eq 0 && ${ADAM_EXIT} -eq 0 ]]
