#!/bin/bash
#SBATCH --job-name=stg3-plain-stiefel-gaps
#SBATCH --output=slurm_logs/staging3_plainheads_stiefel_gaps_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%8

set -euo pipefail

# Fill missing canonical plainheads_stiefel seeds from plots/staging_data/seed_coverage.csv.
# Missing:
#   ant: seed 0, 1, 2, 3
#   humanoid: seed 3
#   swimmer: seed 3
#   pusher: seed 0, 1, 2
#   hopper: seed 0

CONFIG_NAME="plainheads_stiefel"
HEADARCH="plain"
HEADS_OPTIMIZER="stiefel"

ENVS=(ant ant ant ant humanoid swimmer pusher pusher pusher hopper)
BACKENDS=(spring spring spring spring spring generalized generalized generalized generalized positional)
SEEDS=(0 1 2 3 3 3 0 1 2 0)

GAP_SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/staging/gaps"
if [[ ! -f "${GAP_SCRIPT_DIR}/common_gap_launcher.sh" ]]; then
  GAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${GAP_SCRIPT_DIR}/common_gap_launcher.sh"
run_staging_gap_task "$@"
