#!/usr/bin/env bash

SCRIPT="crate_exps/ppo_temporal_spatial_crate.py"
LOGDIR="logs/temporal_spatial_sweep_seed0_no_targetkl"
mkdir -p "$LOGDIR"

BASE_ARGS=(
  --env-name halfcheetah
  --n-envs 128
  --total-timesteps 10000000
  --num-steps 10
  --temporal-stack 8
  --channel-stack 4
  --patch-size 14
  --embed-dim 256
  --depth 2
  --num-heads 4
  --learning-rate 3e-4
  --warmup-steps 50000
  --ista-lambda 0.0
  --action-repeat 4
  --vf-coef 0.5
  --gae-lambda 0.95
  --gamma 0.99
  --update-epochs 4
  --num-minibatches 32
  --anneal-lr
  --track
  --debug-repr
  --seed 0
)

# 12 runs: no --target-kl at all, sweep other hyperparams
# Baseline values: emb-dropout=0.1, ent-coef=0.001, weight-decay=0.01, encoder-lr-scale=0.05
NAMES=(
  "01_baseline"
  "02_embdrop_0.0"
  "03_embdrop_0.2"
  "04_embdrop_0.3"
  "05_ent_0.0"
  "06_ent_0.003"
  "07_ent_0.005"
  "08_wd_0.0"
  "09_wd_0.05"
  "10_encLR_0.02"
  "11_encLR_0.10"
  "12_depth_4"
)

EXTRA_FLAGS=(
  "--emb-dropout 0.1 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.0 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.2 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.3 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.1 --ent-coef 0.000 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.1 --ent-coef 0.003 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.1 --ent-coef 0.005 --weight-decay 0.01 --encoder-lr-scale 0.05"
  "--emb-dropout 0.1 --ent-coef 0.001 --weight-decay 0.00 --encoder-lr-scale 0.05"
  "--emb-dropout 0.1 --ent-coef 0.001 --weight-decay 0.05 --encoder-lr-scale 0.05"
  "--emb-dropout 0.1 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.02"
  "--emb-dropout 0.1 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.10"
  "--depth 4 --emb-dropout 0.1 --ent-coef 0.001 --weight-decay 0.01 --encoder-lr-scale 0.05"
)

# GPU pool
FREE_GPUS=(0 1 2 3)

# Track running jobs
RUN_PIDS=()
RUN_GPUS=()

launch_job () {
  local idx="$1"
  local gpu="$2"
  local name="${NAMES[$idx]}"
  local logfile="$LOGDIR/${name}.log"

  echo "[LAUNCH] ${name} on GPU ${gpu} -> ${logfile}"

  # shellcheck disable=SC2086
  uv run "$SCRIPT" "${BASE_ARGS[@]}" --gpu "$gpu" ${EXTRA_FLAGS[$idx]} >"$logfile" 2>&1 &

  local pid=$!
  RUN_PIDS+=("$pid")
  RUN_GPUS+=("$gpu")
}

reap_finished () {
  local new_pids=()
  local new_gpus=()

  for i in "${!RUN_PIDS[@]}"; do
    local pid="${RUN_PIDS[$i]}"
    local gpu="${RUN_GPUS[$i]}"

    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_gpus+=("$gpu")
    else
      wait "$pid" 2>/dev/null
      echo "[DONE] pid=${pid} freed GPU ${gpu}"
      FREE_GPUS+=("$gpu")
    fi
  done

  RUN_PIDS=("${new_pids[@]}")
  RUN_GPUS=("${new_gpus[@]}")
}

# Schedule: max 4 concurrent
for idx in "${!NAMES[@]}"; do
  while ((${#FREE_GPUS[@]} == 0)); do
    reap_finished
    sleep 5
  done

  gpu="${FREE_GPUS[-1]}"
  unset 'FREE_GPUS[-1]'

  launch_job "$idx" "$gpu"
done

# Wait remaining
while ((${#RUN_PIDS[@]} > 0)); do
  reap_finished
  sleep 5
done

echo "All 12 runs completed. Logs are in: $LOGDIR"

