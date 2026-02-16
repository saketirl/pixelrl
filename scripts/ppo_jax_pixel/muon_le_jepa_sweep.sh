#!/bin/bash
#SBATCH --job-name=ppo-muon-le-jepa-sweep
#SBATCH --output=slurm_logs/ppo_muon_le_jepa_sweep_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%8

set -euo pipefail

export PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH:-}"
export WANDB_API_KEY=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

# Usage:
#   sbatch scripts/ppo_jax_pixel/muon_le_jepa_sweep.sh [env_name] [wandb_project] [wandb_entity] [seed]
# Example:
#   sbatch scripts/ppo_jax_pixel/muon_le_jepa_sweep.sh halfcheetah benchmark my_entity 0
ENV_NAME="${1:-halfcheetah}"
WANDB_PROJECT="${2:-benchmark}"
WANDB_ENTITY="${3:-}"
SEED="${4:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# 32 unique configs: 2 x 2 x 2 x 2 x 2
JEPA_LAMBDAS=(5e-5 2e-4)
JEPA_EMA_TAUS=(0.996 0.999)
JEPA_WARMUP_UPDATES=(0 50)
SIGREG_WEIGHTS=(0.5 1.0)
PROJ_DIMS=(128 256)

# Keep remaining LeJEPA/JEPA settings fixed.
JEPA_RAMPUP_UPDATES=200
SIGREG_NUM_SLICES=64
SIGREG_T_POINTS=17
SIGREG_T_MIN=-5.0
SIGREG_T_MAX=5.0

NUM_LAMBDAS=${#JEPA_LAMBDAS[@]}
NUM_TAUS=${#JEPA_EMA_TAUS[@]}
NUM_WARMUPS=${#JEPA_WARMUP_UPDATES[@]}
NUM_SIGREG=${#SIGREG_WEIGHTS[@]}
NUM_PROJS=${#PROJ_DIMS[@]}
NUM_CONFIGS=$((NUM_LAMBDAS * NUM_TAUS * NUM_WARMUPS * NUM_SIGREG * NUM_PROJS))

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))."
  exit 1
fi

LAMBDA_IDX=$((TASK_ID % NUM_LAMBDAS))
TAU_IDX=$(((TASK_ID / NUM_LAMBDAS) % NUM_TAUS))
WARMUP_IDX=$(((TASK_ID / (NUM_LAMBDAS * NUM_TAUS)) % NUM_WARMUPS))
SIGREG_IDX=$(((TASK_ID / (NUM_LAMBDAS * NUM_TAUS * NUM_WARMUPS)) % NUM_SIGREG))
PROJ_IDX=$((TASK_ID / (NUM_LAMBDAS * NUM_TAUS * NUM_WARMUPS * NUM_SIGREG)))

JEPA_LAMBDA="${JEPA_LAMBDAS[$LAMBDA_IDX]}"
JEPA_EMA_TAU="${JEPA_EMA_TAUS[$TAU_IDX]}"
JEPA_WARMUP_UPDATE="${JEPA_WARMUP_UPDATES[$WARMUP_IDX]}"
SIGREG_WEIGHT="${SIGREG_WEIGHTS[$SIGREG_IDX]}"
PROJ_DIM="${PROJ_DIMS[$PROJ_IDX]}"

# Compact tokens for readable W&B run names.
LAMBDA_TOKEN="${JEPA_LAMBDA//-/m}"
LAMBDA_TOKEN="${LAMBDA_TOKEN//./p}"
TAU_TOKEN="${JEPA_EMA_TAU//./p}"
SIGREG_TOKEN="${SIGREG_WEIGHT//./p}"
EXP_NAME="ppo_muon_lejepa_sweep32_t${TASK_ID}_jl${LAMBDA_TOKEN}_tau${TAU_TOKEN}_wu${JEPA_WARMUP_UPDATE}_sw${SIGREG_TOKEN}_pd${PROJ_DIM}"

GROUP_NAME="${ENV_NAME}_ppo_muon_le_jepa_sweep32_${SLURM_JOB_ID:-local}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="ppo_muon_le_jepa_sweep32,method_le_jepa,env_${ENV_NAME},seed_${SEED},task_${TASK_ID},jl_${JEPA_LAMBDA},tau_${JEPA_EMA_TAU},wu_${JEPA_WARMUP_UPDATE},sw_${SIGREG_WEIGHT},pd_${PROJ_DIM}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} env=${ENV_NAME} seed=${SEED} group=${GROUP_NAME}"
echo "jepa_lambda=${JEPA_LAMBDA} jepa_ema_tau=${JEPA_EMA_TAU} warmup=${JEPA_WARMUP_UPDATE} rampup=${JEPA_RAMPUP_UPDATES} sigreg_weight=${SIGREG_WEIGHT} sigreg_num_slices=${SIGREG_NUM_SLICES} proj_dim=${PROJ_DIM}"

COMMON_ARGS=(
  --env-name "${ENV_NAME}"
  --backend spring
  --n-envs 128
  --hw 84
  --total-timesteps 10000000
  --num-steps 10
  --num-minibatches 32
  --update-epochs 4
  --encoder-lr 3e-4
  --heads-muon-lr 0.001
  --heads-adam-lr 3e-4
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.1
  --ent-coef 0.0
  --vf-coef 0.5
  --max-grad-norm 0.05
  --seed 1
  --log-interval 1
  --frame-stack 4
  --action-repeat 4
  --encoder-type cnn
  --anneal-lr
  --track
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --encoder-tanh-scale 0.5
  --wandb-project-name "${WANDB_PROJECT}"
  --jepa-mode le_jepa
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

uv run ppo_pixelbrax_jax2_muon.py \
  "${COMMON_ARGS[@]}" \
  --jepa-lambda "${JEPA_LAMBDA}" \
  --jepa-warmup-updates "${JEPA_WARMUP_UPDATE}" \
  --jepa-rampup-updates "${JEPA_RAMPUP_UPDATES}" \
  --jepa-ema-tau "${JEPA_EMA_TAU}" \
  --sigreg-weight "${SIGREG_WEIGHT}" \
  --sigreg-num-slices "${SIGREG_NUM_SLICES}" \
  --sigreg-t-points "${SIGREG_T_POINTS}" \
  --sigreg-t-min "${SIGREG_T_MIN}" \
  --sigreg-t-max "${SIGREG_T_MAX}" \
  --proj-dim "${PROJ_DIM}"
