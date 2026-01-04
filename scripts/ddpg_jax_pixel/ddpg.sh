#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1

# Load Python module

module load python/3.9
export PYTHONPATH="/users/apraka15/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH}"

uv run ddpg_pixelbrax_jax.py \
  --env-name walker2d \
  --n-envs 16 \
  --track True \
  --total-timesteps 1000000

