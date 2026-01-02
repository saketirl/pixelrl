#!/bin/bash

#SBATCH --output=slurm_logs/ppo_%j.out
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

# uv run ppo_pixelbrax_jax.py \
#   --env-name ${ENV_NAME} \
#   --backend spring \
#   --n-envs 512 \
#   --hw 84 \
#   --total-timesteps 10000000 \
#   --num-steps 256 \
#   --num-minibatches 32 \
#   --update-epochs 4 \
#   --learning-rate 3e-5 \
#   --gamma 0.99 \
#   --gae-lambda 0.95 \
#   --clip-eps 0.2 \
#   --ent-coef 0.01 \
#   --vf-coef 0.5 \
#   --max-grad-norm 0.5 \
#   --seed ${SEED} \
#   --log-interval 1 \
#   --anneal-lr \
#   --track

# uv run ppo_pixelbrax_jax2.py \
#   --env-name ${ENV_NAME} \
#   --backend spring \
#   --n-envs 128 \
#   --hw 84 \
#   --total-timesteps 10000000 \
#   --num-steps 128 \
#   --num-minibatches 16 \
#   --update-epochs 10 \
#   --learning-rate 2e-4 \
#   --gamma 0.99 \
#   --gae-lambda 0.95 \
#   --clip-eps 0.1 \
#   --ent-coef 0.0 \
#   --vf-coef 0.5 \
#   --max-grad-norm 0.5 \
#   --seed ${SEED} \
#   --log-interval 1 \
#   --anneal-lr \
#   --track

uv run ppo_pixelbrax_jax2.py \
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
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --seed ${SEED} \
  --log-interval 1 \
  --anneal-lr \
  --track \
