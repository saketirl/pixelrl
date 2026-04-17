#!/bin/bash

cd /home/guests/saket/pixelenvs/pixelrl

export PYTHONPATH="/home/guests/saket/pixelenvs/pixelrl/pixelbrax/brax:${PYTHONPATH}"
uv run python -m wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

uv run crate_exps/ppo_temporal_spatial_crate.py --env-name halfcheetah \
  --n-envs 128 \
  --total-timesteps 10000000 \
  --num-steps 10 \
  --temporal-stack 4 \
  --channel-stack 4 \
  --patch-size 14 \
  --embed-dim 256 \
  --depth 2 \
  --num-heads 4 \
  --learning-rate 3e-4 \
  --warmup-steps 50000 \
  --ista-lambda 0.0 \
  --action-repeat 4 \
  --vf-coef 0.5 \
  --gae-lambda 0.95 \
  --gamma 0.99 \
  --update-epochs 4 \
  --num-minibatches 32 \
  --anneal-lr \
  --track \
  --debug-repr \
  --seed 0 \
  --gpu 0 \
  --frame-skip 1 \
  --emb-dropout 0.1 \
  --ent-coef 0.001 \
  --weight-decay 0.01 \
  --encoder-lr-scale 0.10 \
  --temporal-decay 0.5 \
  --parseval-coef 0.01 &> tmp_crate_6.out

uv run crate_exps/ppo_temporal_spatial_crate.py --env-name halfcheetah \
  --n-envs 128 \
  --total-timesteps 10000000 \
  --num-steps 10 \
  --temporal-stack 4 \
  --channel-stack 4 \
  --patch-size 21 \
  --embed-dim 256 \
  --depth 2 \
  --num-heads 4 \
  --learning-rate 3e-4 \
  --warmup-steps 50000 \
  --ista-lambda 0.0 \
  --action-repeat 4 \
  --vf-coef 0.5 \
  --gae-lambda 0.95 \
  --gamma 0.99 \
  --update-epochs 4 \
  --num-minibatches 32 \
  --anneal-lr \
  --track \
  --debug-repr \
  --seed 0 \
  --gpu 0 \
  --frame-skip 1 \
  --emb-dropout 0.1 \
  --ent-coef 0.001 \
  --weight-decay 0.01 \
  --encoder-lr-scale 0.10 \
  --temporal-decay 0.5 \
  --parseval-coef 0.01 &> tmp_crate_6.out

uv run crate_exps/ppo_temporal_spatial_crate.py --env-name halfcheetah \
  --n-envs 128 \
  --total-timesteps 10000000 \
  --num-steps 10 \
  --temporal-stack 4 \
  --channel-stack 4 \
  --patch-size 14 \
  --embed-dim 256 \
  --depth 2 \
  --num-heads 4 \
  --learning-rate 3e-4 \
  --warmup-steps 50000 \
  --ista-lambda 0.0 \
  --action-repeat 4 \
  --vf-coef 0.5 \
  --gae-lambda 0.95 \
  --gamma 0.99 \
  --update-epochs 4 \
  --num-minibatches 32 \
  --anneal-lr \
  --track \
  --debug-repr \
  --seed 0 \
  --gpu 0 \
  --frame-skip 1 \
  --emb-dropout 0.1 \
  --ent-coef 0.001 \
  --weight-decay 0.001 \
  --encoder-lr-scale 0.10 \
  --temporal-decay 0.5 \
  --parseval-coef 0.01 &> tmp_crate_6.out