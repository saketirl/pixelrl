#!/bin/bash
#SBATCH --job-name=lqr-s2-eps001
#SBATCH --output=slurm_logs/lqr_stage2_reviewer_eps001/run_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --array=0-19%20

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/guests/arjun/pixelrl}"

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Controlled observation-noise replication of the accepted Stage-2 reviewer
# experiment. Every optimization and seed setting is retained; only
#
#     epsilon: 0 -> 0.01,
#     o_t = M s_t + epsilon xi_t,  xi_t ~ N(0,I_d),
#
# changes. Downstairs observes s_t without observation noise. The branches
# take their own actions and receive no action, trajectory, parameter,
# gradient, optimizer-state, or batch synchronization. Common process and
# exploration innovations are variance-reducing common random numbers only.
# Stage 2 remains pure MLP:
#
#     Psi(z,a) = MLP_A([z,a])                    (c_R=0),
#
# while the physical environment reward retains -a^T R a.
INIT_SEEDS=(75101 75102 75103 75104 75105 75106 75107 75108 75109 75110 75111 75112 75113 75114 75115 75116 75117 75118 75119 75120)
NOISE_SEEDS=(76101 76102 76103 76104 76105 76106 76107 76108 76109 76110 76111 76112 76113 76114 76115 76116 76117 76118 76119 76120)

SEED_INDEX=$SLURM_ARRAY_TASK_ID
INIT_SEED=${INIT_SEEDS[$SEED_INDEX]}
NOISE_SEED=${NOISE_SEEDS[$SEED_INDEX]}
RUN_DIR="lqr_ac_compare/linear_rep_head_results/overnight_two_stage/stage2/reviewer_epsilon001/seed_${SEED_INDEX}"

uv run python lqr_ac_compare/compare_linear_rep_nonlinear_heads.py \
    --head-types mlp \
    --training-coupling independent_actions \
    --observation-noises 0.01 \
    --action-curvature-scale 0 \
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
    --train-initial-state-std 0 \
    --exploration-std 0.1 \
    --exploration-schedule constant \
    --exploration-final-std 0.1 \
    --exploration-decay-end-iters 600 \
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
