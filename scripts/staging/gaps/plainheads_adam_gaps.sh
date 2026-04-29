#!/bin/bash
#SBATCH --job-name=stg3-plain-adam-gaps
#SBATCH --output=slurm_logs/staging3_plainheads_adam_gaps_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

# Fill missing canonical plainheads_adam seeds from plots/staging_data/seed_coverage.csv.
# Missing:
#   pusher: seed 0, 2, 3
#   inverted_pendulum: seed 0

CONFIG_NAME="plainheads_adam"
HEADARCH="plain"
HEADS_OPTIMIZER="adam"

ENVS=(pusher pusher pusher inverted_pendulum)
BACKENDS=(generalized generalized generalized generalized)
SEEDS=(0 2 3 0)

GAP_SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/staging/gaps"
if [[ ! -f "${GAP_SCRIPT_DIR}/common_gap_launcher.sh" ]]; then
  GAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${GAP_SCRIPT_DIR}/common_gap_launcher.sh"
run_staging_gap_task "$@"
