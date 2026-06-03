#!/bin/bash
#SBATCH --job-name=kevin-config
#SBATCH --output=slurm_logs/kevin_config_%j.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

set -euo pipefail

# Kevin Guo lop-jax nonstationary Slippery Ant BP setup, packed as vmapped
# independent agents. Defaults match rlopt/scripts/hyperparams/nonstationary/bp.py.

WANDB_PROJECT="${1:-continual_brax}"
WANDB_ENTITY="${2:-rl-power}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." 2>/dev/null && pwd || true)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/parallel_ppo_brax.py" && -n "${SCRIPT_REPO_ROOT}" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi
if [[ ! -f "${REPO_ROOT}/parallel_ppo_brax.py" ]]; then
  REPO_ROOT="/home/guests/arjun/pixelrl"
fi

mkdir -p "${REPO_ROOT}/slurm_logs"

if [[ -d "${REPO_ROOT}/pixelbrax/brax/brax" ]]; then
  BRAX_PYTHONPATH="${REPO_ROOT}/pixelbrax/brax"
else
  BRAX_PYTHONPATH="/home/guests/arjun/pixelrl/pixelbrax/brax"
fi
export PYTHONPATH="${BRAX_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"

WANDB_KEY_FILE="${REPO_ROOT}/secrets/wandb_api_key.txt"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
fi

ENV_NAME="${ENV_NAME:-ant}"
BACKEND="${BACKEND:-positional}"
SEED="${SEED:-2025}"
N_SEEDS="${N_SEEDS:-256}"
NUM_AGENTS="${NUM_AGENTS:-${N_SEEDS}}"
AGENT_SEED_STRIDE="${AGENT_SEED_STRIDE:-1}"
FRICTION_SEED="${FRICTION_SEED:-0}"

N_ENVS="${N_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-2048}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
CHANGE_EVERY="${CHANGE_EVERY:-2000000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
# Kevin's adam_with_param_counts uses Adam eps=1e-8.
ADAM_EPS="${ADAM_EPS:-1e-8}"
ENT_COEF="${ENT_COEF:-0.01}"
# parallel_ppo_brax.py uses 0.5 * value MSE; Kevin's PPO uses unhalved value
# MSE. Use vf_coef=2.0 here so the effective value-loss weight is 1.0.
VF_COEF="${VF_COEF:-2.0}"
LOG_INTERVAL="${LOG_INTERVAL:-8}"
MAX_LOGGED_AGENTS="${MAX_LOGGED_AGENTS:-0}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

GROUP_ID="${SLURM_JOB_ID:-local}"
GROUP_NAME="kevin_config_slippery_ant_bp_${BACKEND}${EXP_SUFFIX:+_${EXP_SUFFIX}}_${GROUP_ID}"
EXP_NAME="kevin_config_slippery_ant_bp_n${NUM_AGENTS}_${BACKEND}${EXP_SUFFIX:+_${EXP_SUFFIX}}"
export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_TAGS="continual_rl,slippery_ant,kevin_config,lop_jax_bp,parallel_ppo,vmapped_agents,brax_state,${ENV_NAME},backend_${BACKEND},num_agents_${NUM_AGENTS},seed_${SEED},friction_seed_${FRICTION_SEED},change_every_${CHANGE_EVERY},total_${TOTAL_TIMESTEPS},n_envs_${N_ENVS},num_steps_${NUM_STEPS},minibatches_${NUM_MINIBATCHES},epochs_${UPDATE_EPOCHS},lr_${LEARNING_RATE},ent_${ENT_COEF},vf_${VF_COEF},no_reward_norm,no_obs_norm,no_action_clip,unbounded_global_logstd,actor_mean_unbounded${EXP_SUFFIX:+,${EXP_SUFFIX}}"

cd "${REPO_ROOT}"

echo "Running Kevin config group=${GROUP_NAME}"
echo "Packed runs on this GPU: NUM_AGENTS=${NUM_AGENTS} training seeds ${SEED}..$((SEED + (NUM_AGENTS - 1) * AGENT_SEED_STRIDE)) stride=${AGENT_SEED_STRIDE}"
echo "Fixed friction schedule seed: ${FRICTION_SEED}"
echo "PPO: n_envs=${N_ENVS} num_steps=${NUM_STEPS} minibatches=${NUM_MINIBATCHES} epochs=${UPDATE_EPOCHS} total=${TOTAL_TIMESTEPS}"

CMD=(
  uv run python "${REPO_ROOT}/parallel_ppo_brax.py"
  --env-name "${ENV_NAME}"
  --backend "${BACKEND}"
  --num-agents "${NUM_AGENTS}"
  --agent-seed-stride "${AGENT_SEED_STRIDE}"
  --n-envs "${N_ENVS}"
  --track
  --wandb-project-name "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --learning-rate "${LEARNING_RATE}"
  --adam-eps "${ADAM_EPS}"
  --base-optimizer adam
  --heads-optimizer adam
  --weight-decay 0.0
  --num-steps "${NUM_STEPS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --update-epochs "${UPDATE_EPOCHS}"
  --gamma 0.99
  --gae-lambda 0.95
  --clip-eps 0.2
  --ent-coef "${ENT_COEF}"
  --vf-coef "${VF_COEF}"
  --max-grad-norm 1000000000
  --actor-critic-activation relu
  --network-arch lop
  --seed "${SEED}"
  --log-interval "${LOG_INTERVAL}"
  --action-repeat 1
  --slippery
  --slippery-change-every "${CHANGE_EVERY}"
  --slippery-start-at-schedule
  --slippery-schedule-seed-mode fixed
  --slippery-schedule-seed "${FRICTION_SEED}"
  --actor-logstd-init 0.0
  --actor-logstd-min -5.0
  --actor-logstd-max 2.0
  --max-logged-agents "${MAX_LOGGED_AGENTS}"
  --max-printed-agents 16
  --exp-name "${EXP_NAME}"
  --clip-vloss
  --no-reward-normalize
  --no-obs-normalize
  --no-actor-mean-tanh
  --no-bounded-global-logstd
  --no-clip-global-logstd
  --no-clip-actions
)

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
