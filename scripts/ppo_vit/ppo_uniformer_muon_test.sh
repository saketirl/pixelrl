#!/bin/bash

cd /home/guests/saket/pixelenvs/pixelrl

export PYTHONPATH="/home/guests/saket/pixelenvs/pixelrl/pixelbrax/brax:${PYTHONPATH}"
uv run python -m wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

uv run vit_exps/ppo_pixelbrax_uniformer_muon.py \
    --env-name halfcheetah \
    --n-envs 128 \
    --total-timesteps 10000000 \
    --num-steps 10 \
    --frame-stack 8 \
    --uniformer-local-depth 1 \
    --uniformer-global-depth 1 \
    --uniformer-local-dim 64 \
    --uniformer-global-dim 128 \
    --uniformer-head-dim 32 \
    --uniformer-local-conv-size 3 5 5 \
    --encoder-output-dim 256 \
    --encoder-output-scale 0.1 \
    --encoder-output-activation tanh \
    --encoder-lr 5e-4 \
    --heads-adam-lr 5e-4 \
    --heads-muon-lr 0.001 \
    --weight-decay 0.01 \
    --action-repeat 4 \
    --vf-coef 0.5 \
    --gae-lambda 0.95 \
    --gamma 0.99 \
    --update-epochs 4 \
    --num-minibatches 32 \
    --anneal-lr \
    --track \
    --debug-repr \
    --muon-dual-lr 0.01 \
    --muon-dual-steps 5 \
    --actor-muon-max-grad-norm 100 \
    --critic-muon-max-grad-norm 1 \
    --seed 0 &> tmp_3.out