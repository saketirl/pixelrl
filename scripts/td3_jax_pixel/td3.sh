#!/bin/bash

#SBATCH --output=slurm_logs/td3_%j.out
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

# Environment (change as needed: halfcheetah, walker2d, hopper, ant, humanoid)
ENV_NAME=${1:-"halfcheetah"}
SEED=${2:-0}

uv run td3_pixelbrax_jax.py \
  --env-name ${ENV_NAME} \
  --backend spring \
  --n-envs 128 \
  --hw 84 \
  --total-timesteps 10000000 \
  --gamma 0.99 \
  --tau 0.005 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --start-timesteps 25000 \
  --exploration-noise 0.1 \
  --max-action 1.0 \
  --actor-lr 1e-4 \
  --critic-lr 1e-3 \
  --policy-noise 0.2 \
  --noise-clip 0.5 \
  --policy-frequency 2 \
  --seed ${SEED} \
  --log-interval 10 \
  --track
