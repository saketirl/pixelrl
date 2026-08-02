#!/bin/bash
#SBATCH --job-name=lqr-gentle-final
#SBATCH --output=slurm_logs/lqr_linear_rep_overnight/gentle_final_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --array=0-39%40

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/guests/arjun/pixelrl}"

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Prelocked final stability confirmation on a genuinely untouched paired seed
# bank. Hyperparameters, the common final checkpoint, and all acceptance rules
# were fixed before these runs; this bank permits no seed-wise or aggregate
# hyperparameter, checkpoint, or stopping-time selection. Analytic LQR
# quantities remain evaluation-only and are not used by training or selection.
#
# Both stages use independent upstairs/downstairs actions, epsilon=0, common
# exogenous noise, and no parameter synchronization. With gamma=exp(-beta*dt),
# the centered rate and L-step martingale residual are
#
#   psi_t = Psi(z_t,a_t) - Psi(z_t,pi(z_t)),
#   e_t   = V(z_t) - dt*sum_{l=0}^{L-1} gamma^l (r_{t+l}-psi_{t+l})
#           - gamma^L V(z_{t+L}),                 L=30,
#
# and the displayed value loss is mean_t(e_t^2)/2.
#
# Curvature disclosure:
#   Stage 1 (tasks 0--19):
#       Psi(z,a) = MLP_A([z,a]) - a^T R a             (c_R=1, c_a=1).
#   Stage 2 (tasks 20--39):
#       Psi(z,a) = MLP_A([z,a])                       (c_R=0, c_a=1),
#       with no explicit -a^T R a term or other analytic LQR training input.
#
# The action-curvature coefficient c_R is the only cross-stage configuration
# difference. In particular, both stages use warmup 300, exclusive joint
# actor/critic stop 600, actor cadence 8, and MLP actor-rate scale 0.01.
# Equal actor/critic stops freeze every learned parameter from iteration 600
# onward while rollout and loss diagnostics continue through iteration 999.
INIT_SEEDS=(75101 75102 75103 75104 75105 75106 75107 75108 75109 75110 75111 75112 75113 75114 75115 75116 75117 75118 75119 75120)
NOISE_SEEDS=(76101 76102 76103 76104 76105 76106 76107 76108 76109 76110 76111 76112 76113 76114 76115 76116 76117 76118 76119 76120)

CURVATURE_SCALES=(1 0)
STAGE_NAMES=(stage1 stage2)

STAGE_INDEX=$((SLURM_ARRAY_TASK_ID / 20))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 20))
INIT_SEED=${INIT_SEEDS[$SEED_INDEX]}
NOISE_SEED=${NOISE_SEEDS[$SEED_INDEX]}
CURVATURE_SCALE=${CURVATURE_SCALES[$STAGE_INDEX]}
STAGE_NAME=${STAGE_NAMES[$STAGE_INDEX]}
RUN_DIR="lqr_ac_compare/linear_rep_head_results/overnight_two_stage/${STAGE_NAME}/final_gentle/seed_${SEED_INDEX}"

uv run python lqr_ac_compare/compare_linear_rep_nonlinear_heads.py \
    --head-types mlp \
    --training-coupling independent_actions \
    --observation-noises 0 \
    --action-curvature-scale "$CURVATURE_SCALE" \
    --advantage-gradient full_window \
    --hidden-dim 64 \
    --mlp-policy-output-scale 0 \
    --mlp-value-output-scale 0.1 \
    --mlp-advantage-output-scale 0 \
    --mlp-advantage-action-input-scale 1 \
    --mlp-policy-action-limit 1 \
    --start-time-weighting uniform \
    --encoder-lr-scale 1 \
    --iters 1000 \
    --eta-actor 0.01 \
    --mlp-actor-lr-scale 0.01 \
    --mlp-actor-lr-schedule constant \
    --mlp-actor-update-every 8 \
    --actor-warmup-iters 300 \
    --actor-stop-iters 600 \
    --critic-stop-iters 600 \
    --eta-critic 0.01 \
    --eta-advantage 0.01 \
    --exploration-std 0.1 \
    --n-steps 30 \
    --alpha 0.5 \
    --eval-episodes 64 \
    --eval-every 20 \
    --log-every 250 \
    --init-seed "$INIT_SEED" \
    --noise-seed "$NOISE_SEED" \
    --eval-seed 77101 \
    --output-dir "$RUN_DIR" \
    --no-plots
