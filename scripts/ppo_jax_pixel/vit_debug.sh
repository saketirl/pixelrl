#!/bin/bash
#SBATCH --job-name=vit-debug
#SBATCH --output=slurm_logs/vit_debug_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4

set -euo pipefail

# Four parallel runs of the ViTEncoder gold-standard setup (conv stem enabled)
# on halfcheetah:
#   width profile {all-128, all-512} x actor/critic heads Muon {on, off}
#
# Usage:
#   sbatch scripts/ppo_jax_pixel/vit_debug.sh [wandb_project] [wandb_entity] [total_timesteps] [seed]
# Example:
#   sbatch scripts/ppo_jax_pixel/vit_debug.sh benchmark my_entity 10000000 0

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
SEED="${4:-0}"
ENV_NAME="halfcheetah"

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
GROUP_NAME="vit_debug_${ENV_NAME}_${GROUP_ID}"
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
  --encoder-type vit
  --encoder-tanh-scale 0.25
  --vit-patch-size 21
  --vit-num-heads 8
  --vit-num-layers 4
  --vit-dropout-rate 0.0
  --vit-attention-dropout-rate 0.0
  --vit-conv-stem-channels 64
  --vit-conv-stem-kernel 3
  --vit-use-conv-stem
  --no-vit-use-cls-token
  --no-vit-apply-output-tanh
  --no-vit-qk-stiefel
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"

launch_run() {
  local gpu="$1"
  local width_label="$2"
  local hidden_dim="$3"
  local proj_dim="$4"
  local mlp_dim="$5"
  local muon_flag="$6"   # true/false

  local muon_cli
  local muon_tag
  local run_suffix
  if [[ "${muon_flag}" == "true" ]]; then
    muon_cli="--use-heads-muon"
    muon_tag="heads_muon_true"
    run_suffix="muon"
  else
    muon_cli="--no-use-heads-muon"
    muon_tag="heads_muon_false"
    run_suffix="adam"
  fi

  export WANDB_TAGS="vit_debug,encoder_vit,vit_conv_stem_true,width_${width_label},hidden_${hidden_dim},proj_${proj_dim},mlp_${mlp_dim},${muon_tag},env_${ENV_NAME},seed_${SEED}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
    "${COMMON_ARGS[@]}" \
    --vit-hidden-size "${hidden_dim}" \
    --vit-proj-dim "${proj_dim}" \
    --vit-mlp-dim "${mlp_dim}" \
    ${muon_cli} \
    --exp-name "ppo_vit_debug_${width_label}_${run_suffix}_${ENV_NAME}_s${SEED}" \
    > "${REPO_ROOT}/slurm_logs/vit_debug_${GROUP_ID}_${width_label}_${run_suffix}.out" 2>&1 &
}

# "all-128" profile: hidden/proj both 128; keep ViT MLP at 4x hidden (512)
launch_run 0 all128 128 128 512 true
PID_128_MUON=$!
launch_run 1 all128 128 128 512 false
PID_128_ADAM=$!

# "all-512" profile: hidden/proj both 512; keep ViT MLP at 4x hidden (2048)
launch_run 2 all512 512 512 2048 true
PID_512_MUON=$!
launch_run 3 all512 512 512 2048 false
PID_512_ADAM=$!

echo "Launched all128 Muon run (PID ${PID_128_MUON}) on GPU 0"
echo "Launched all128 Adam run (PID ${PID_128_ADAM}) on GPU 1"
echo "Launched all512 Muon run (PID ${PID_512_MUON}) on GPU 2"
echo "Launched all512 Adam run (PID ${PID_512_ADAM}) on GPU 3"

wait "${PID_128_MUON}"
EXIT_128_MUON=$?
wait "${PID_128_ADAM}"
EXIT_128_ADAM=$?
wait "${PID_512_MUON}"
EXIT_512_MUON=$?
wait "${PID_512_ADAM}"
EXIT_512_ADAM=$?

echo "all128 Muon exit code: ${EXIT_128_MUON}"
echo "all128 Adam exit code: ${EXIT_128_ADAM}"
echo "all512 Muon exit code: ${EXIT_512_MUON}"
echo "all512 Adam exit code: ${EXIT_512_ADAM}"

[[ ${EXIT_128_MUON} -eq 0 && ${EXIT_128_ADAM} -eq 0 && ${EXIT_512_MUON} -eq 0 && ${EXIT_512_ADAM} -eq 0 ]]
