#!/bin/bash
#SBATCH --job-name=stg3-crate-stiefel-gaps
#SBATCH --output=slurm_logs/staging3_crate_stiefel_gaps_%A.out
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --mem=1GB

set -euo pipefail

# No missing canonical crate_stiefel seeds were found in plots/staging_data/seed_coverage.csv.

CONFIG_NAME="crate_stiefel"
HEADARCH="crate"
HEADS_OPTIMIZER="stiefel"

ENVS=()
BACKENDS=()
SEEDS=()

GAP_SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/staging/gaps"
if [[ ! -f "${GAP_SCRIPT_DIR}/common_gap_launcher.sh" ]]; then
  GAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${GAP_SCRIPT_DIR}/common_gap_launcher.sh"
run_staging_gap_task "$@"
