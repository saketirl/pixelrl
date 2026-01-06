#!/bin/bash

#SBATCH --output=slurm_logs/reppo_%j.out
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

uv run reppo_pixelbrax.py \
  --env-name ${ENV_NAME} \
  --backend spring \
  --n-envs 128 \
  --hw 84 \
  --total-timesteps 10000000 \
  --num-steps 128 \
  --num-minibatches 16 \
  --update-epochs 10 \
  --learning-rate 2e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --ent-coef 0.01 \
  --kl-coef 0.1 \
  --kl-target 0.01 \
  --polyak 0.005 \
  --exploration-noise-min 1.0 \
  --exploration-noise-max 2.0 \
  --use-kl-constraint \
  --use-entropy-constraint \
  --actor-min-std 0.1 \
  --seed ${SEED} \
  --log-interval 10 \
  --anneal-lr \
  --track
