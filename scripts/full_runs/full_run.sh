#!/bin/bash
#SBATCH --job-name=cnn-muon-me
#SBATCH --output=slurm_logs/cnn_heads_muon_me_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-107%8

set -euo pipefail

# Sweep CNN Muon PPO runs across all PixelBrax environments and 6 seeds,
# comparing head optimizer choice (MUON vs Adam).
#
# Swept:
#   env         in {halfcheetah, walker2d, ant, humanoid, reacher, swimmer, pusher, hopper, inverted_pendulum}
#   head_opt    in {muon, adam}  (use_heads_muon true/false)
#   seed        in {0, 1, 2, 3, 4, 5}
#
# 9 x 1 x 2 x 6 = 108 configs

WANDB_PROJECT="${1:-benchmark}"
WANDB_ENTITY="${2:-}"
TOTAL_TIMESTEPS="${3:-10000000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/ppo_pixelbrax.py" || ! -f "${REPO_ROOT}/encoders.py" ]]; then
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

ENVS=(halfcheetah walker2d ant humanoid reacher swimmer pusher hopper inverted_pendulum)
BACKENDS=(spring spring spring spring generalized generalized generalized positional generalized)
ENCODER_TYPES=(cnn)
OPT_CONDITIONS=(muon adam)
SEEDS=(0 1 2 3 4 5)

NUM_ENVS=${#ENVS[@]}
NUM_BACKENDS=${#BACKENDS[@]}
NUM_ENCODERS=${#ENCODER_TYPES[@]}
NUM_OPTS=${#OPT_CONDITIONS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_CONFIGS=$((NUM_ENVS * NUM_ENCODERS * NUM_OPTS * NUM_SEEDS))

if (( NUM_ENVS != NUM_BACKENDS )); then
  echo "ENVS/BACKENDS length mismatch: ${NUM_ENVS} vs ${NUM_BACKENDS}" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= NUM_CONFIGS )); then
  echo "Invalid TASK_ID=${TASK_ID}. Expected 0..$((NUM_CONFIGS - 1))." >&2
  exit 1
fi

SEED_IDX=$((TASK_ID % NUM_SEEDS))
OPT_IDX=$(((TASK_ID / NUM_SEEDS) % NUM_OPTS))
ENC_IDX=$(((TASK_ID / (NUM_SEEDS * NUM_OPTS)) % NUM_ENCODERS))
ENV_IDX=$((TASK_ID / (NUM_SEEDS * NUM_OPTS * NUM_ENCODERS)))

SEED="${SEEDS[$SEED_IDX]}"
OPT_CONDITION="${OPT_CONDITIONS[$OPT_IDX]}"
ENCODER_TYPE="${ENCODER_TYPES[$ENC_IDX]}"
ENV_NAME="${ENVS[$ENV_IDX]}"
BACKEND="${BACKENDS[$ENV_IDX]}"

GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GROUP_NAME="cnn_heads_muon_me_"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="cnn_heads_muon_me,encoder_${ENCODER_TYPE},opt_${OPT_CONDITION},env_${ENV_NAME},backend_${BACKEND},seed_${SEED},allenv9,sweep6seeds"

EXP_NAME="ppo_cmp_${ENCODER_TYPE}_${OPT_CONDITION}_${ENV_NAME}_b${BACKEND}_s${SEED}_t${TASK_ID}"

echo "Running TASK_ID=${TASK_ID}/${NUM_CONFIGS} group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} encoder=${ENCODER_TYPE} opt=${OPT_CONDITION} seed=${SEED}"
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
  --muon-dual-lr 0.01
  --muon-dual-steps 5
  --actor-muon-max-grad-norm 100
  --critic-muon-max-grad-norm 1
  --exp-name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

ARCH_ARGS=(
  --encoder-type cnn
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
  --heads-muon-lr 0.001
  --max-grad-norm 0.05
  --encoder-tanh-scale 0.5
)

OPT_ARGS=()
if [[ "${OPT_CONDITION}" == "muon" ]]; then
  OPT_ARGS+=(--use-heads-muon)
else
  OPT_ARGS+=(--no-use-heads-muon)
fi

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_pixelbrax_jax2_muon.py" \
  "${COMMON_ARGS[@]}" \
  "${ARCH_ARGS[@]}" \
  "${OPT_ARGS[@]}"