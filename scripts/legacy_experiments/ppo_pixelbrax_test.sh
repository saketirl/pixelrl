#!/bin/bash

cd /home/guests/saket/pixelenvs/pixelrl

export PYTHONPATH="/home/guests/saket/pixelenvs/pixelrl/pixelbrax/brax:${PYTHONPATH}"
uv run python -m wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

uv run ppo_pixelbrax_jax2.py \
  --env-name halfcheetah \
  --backend spring \
  --n-envs 128 \
  --hw 84 \
  --total-timesteps 10000000 \
  --num-steps 10 \
  --num-minibatches 32 \
  --update-epochs 4 \
  --learning-rate 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --seed 1 \
  --log-interval 1 \
  --frame-stack 4 \
  --action-repeat 4 \
  --anneal-lr \
  --track \
  --debug-repr &> ppo_base_tmp.out
