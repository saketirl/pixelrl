#!/bin/bash
#SBATCH --job-name=slip-hum-stiefel
#SBATCH --output=slurm_logs/slippery_humanoid_stiefel_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# Stiefel-head variant of the successful Slippery Humanoid state-observation
# config. This keeps the Adam trunk/Adam scalar params from the working config,
# but routes actor/critic matrix params through optimizers/manifold_stiefel_optax.py.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"
PHASE_EVERY_ENV_STEPS="${3:-999936}"
NUM_PHASES="${4:-20}"
SEED="${5:-0}"
SLIPPERY_SCHEDULE_SEED="${6:-14}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
if [[ ! -f "${REPO_ROOT}/ppo_brax.py" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"

if [[ -d "${REPO_ROOT}/pixelbrax/brax/brax" ]]; then
  BRAX_PYTHONPATH="${REPO_ROOT}/pixelbrax/brax"
else
  BRAX_PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"
fi
export PYTHONPATH="${BRAX_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

ENV_NAME="humanoid"
BACKEND="${BACKEND:-spring}"
N_ENVS="${N_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
ADAM_EPS="${ADAM_EPS:-1e-5}"
HEADS_STIEFEL_LR="${HEADS_STIEFEL_LR:-0.001}"
STIEFEL_DUAL_LR="${STIEFEL_DUAL_LR:-0.01}"
STIEFEL_DUAL_STEPS="${STIEFEL_DUAL_STEPS:-5}"
STIEFEL_MSIGN_STEPS="${STIEFEL_MSIGN_STEPS:-5}"
ACTOR_STIEFEL_MAX_GRAD_NORM="${ACTOR_STIEFEL_MAX_GRAD_NORM:-100}"
CRITIC_STIEFEL_MAX_GRAD_NORM="${CRITIC_STIEFEL_MAX_GRAD_NORM:-1}"
ACTOR_MEAN_SCALE="${ACTOR_MEAN_SCALE:-1.0}"
ACTOR_LOGSTD_INIT="${ACTOR_LOGSTD_INIT:--2.0}"
ACTOR_LOGSTD_MIN="${ACTOR_LOGSTD_MIN:--5.0}"
ACTOR_LOGSTD_MAX="${ACTOR_LOGSTD_MAX:--1.4}"
OBS_NORM_CLIP="${OBS_NORM_CLIP:-10.0}"
NETWORK_CRATE_STEP_SIZE="${NETWORK_CRATE_STEP_SIZE:-0.1}"
CRATE_STEP_SIZE="${CRATE_STEP_SIZE:-0.1}"

ROLLOUT_ENV_STEPS=$((N_ENVS * NUM_STEPS))
if (( PHASE_EVERY_ENV_STEPS % N_ENVS != 0 )); then
  echo "PHASE_EVERY_ENV_STEPS=${PHASE_EVERY_ENV_STEPS} must be divisible by N_ENVS=${N_ENVS}." >&2
  exit 1
fi
if (( ROLLOUT_ENV_STEPS % NUM_MINIBATCHES != 0 )); then
  echo "batch=${ROLLOUT_ENV_STEPS} must be divisible by NUM_MINIBATCHES=${NUM_MINIBATCHES}." >&2
  exit 1
fi

CHANGE_EVERY_POLICY_STEPS=$((PHASE_EVERY_ENV_STEPS / N_ENVS))
SLIPPERY_CHANGE_EVERY=$((CHANGE_EVERY_POLICY_STEPS * ACTION_REPEAT))
TOTAL_TIMESTEPS=$((PHASE_EVERY_ENV_STEPS * NUM_PHASES))
if (( TOTAL_TIMESTEPS % ROLLOUT_ENV_STEPS != 0 )); then
  TOTAL_TIMESTEPS=$(((TOTAL_TIMESTEPS / ROLLOUT_ENV_STEPS) * ROLLOUT_ENV_STEPS))
fi
MINIBATCH_SIZE=$((ROLLOUT_ENV_STEPS / NUM_MINIBATCHES))

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="slippery_humanoid_stiefel_crate_obsnorm_${GROUP_ID}"
EXP_NAME="ppo_brax_sliphum_stiefel_crate_obsnorm_ph${NUM_PHASES}_phase${PHASE_EVERY_ENV_STEPS}_sched${SLIPPERY_SCHEDULE_SEED}_s${SEED}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_humanoid_stiefel,brax_state,humanoid,backend_${BACKEND},seed_${SEED},schedule_seed_${SLIPPERY_SCHEDULE_SEED},phase_every_${PHASE_EVERY_ENV_STEPS},phases_${NUM_PHASES},action_repeat_${ACTION_REPEAT},headopt_stiefel,lr_${LEARNING_RATE},stiefel_lr_${HEADS_STIEFEL_LR},obs_norm,crate_network,crate_heads,bounded_std,actor_mean_tanh"

echo "Running Stiefel Slippery Humanoid config group=${GROUP_NAME}"
echo "Config: env=${ENV_NAME} backend=${BACKEND} seed=${SEED} schedule_seed=${SLIPPERY_SCHEDULE_SEED} total_timesteps=${TOTAL_TIMESTEPS}"
echo "Schedule: phase_every_env_steps=${PHASE_EVERY_ENV_STEPS} phases=${NUM_PHASES} slippery_change_every=${SLIPPERY_CHANGE_EVERY}"
echo "PPO: trunk_lr=${LEARNING_RATE} stiefel_lr=${HEADS_STIEFEL_LR} anneal_lr=true batch=${ROLLOUT_ENV_STEPS} minibatch_size=${MINIBATCH_SIZE}"
echo "Architecture: obs_norm=true crate_network=true crate_heads=true bounded_std=true actor_mean_tanh=true"

cd "${REPO_ROOT}"
uv run python "${REPO_ROOT}/ppo_brax.py" \
  --env-name "${ENV_NAME}" \
  --backend "${BACKEND}" \
  --n-envs "${N_ENVS}" \
  --track \
  --wandb-project-name "${WANDB_PROJECT}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --adam-eps "${ADAM_EPS}" \
  --anneal-lr \
  --num-steps "${NUM_STEPS}" \
  --num-minibatches "${NUM_MINIBATCHES}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.05 \
  --actor-critic-activation relu \
  --heads-optimizer stiefel \
  --heads-stiefel-lr "${HEADS_STIEFEL_LR}" \
  --stiefel-dual-lr "${STIEFEL_DUAL_LR}" \
  --stiefel-dual-steps "${STIEFEL_DUAL_STEPS}" \
  --stiefel-msign-steps "${STIEFEL_MSIGN_STEPS}" \
  --actor-stiefel-max-grad-norm "${ACTOR_STIEFEL_MAX_GRAD_NORM}" \
  --critic-stiefel-max-grad-norm "${CRITIC_STIEFEL_MAX_GRAD_NORM}" \
  --actor-mean-tanh \
  --actor-mean-scale "${ACTOR_MEAN_SCALE}" \
  --bounded-global-logstd \
  --actor-logstd-init "${ACTOR_LOGSTD_INIT}" \
  --actor-logstd-min "${ACTOR_LOGSTD_MIN}" \
  --actor-logstd-max "${ACTOR_LOGSTD_MAX}" \
  --obs-normalize \
  --obs-norm-clip "${OBS_NORM_CLIP}" \
  --use-crate-network \
  --network-crate-step-size "${NETWORK_CRATE_STEP_SIZE}" \
  --use-crate-head \
  --crate-step-size "${CRATE_STEP_SIZE}" \
  --reward-normalize \
  --seed "${SEED}" \
  --log-interval 1 \
  --action-repeat "${ACTION_REPEAT}" \
  --slippery \
  --slippery-change-every "${SLIPPERY_CHANGE_EVERY}" \
  --slippery-schedule-seed "${SLIPPERY_SCHEDULE_SEED}" \
  --exp-name "${EXP_NAME}" \
  ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"}
