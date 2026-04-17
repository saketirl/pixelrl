#!/usr/bin/env bash

SCRIPT="crate_exps/ppo_temporal_spatial_crate_muon.py"
LOGDIR="logs/temporal_spatial_muon_sweep"
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
  --encoder-lr 3e-4
  --heads-adam-lr 3e-4
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

# 12 runs: sweep heads_muon_lr (6 values) x emb_dropout (2 values)
# heads_muon_lr: 0.05, 0.02, 0.01, 0.005, 0.002, 0.0005
# emb_dropout: 0.1, 0.5
# Adam LR fixed at 3e-4

NAMES=(
  "01_muonLR_0.050_embdrop_0.1"
  "02_muonLR_0.020_embdrop_0.1"
  "03_muonLR_0.010_embdrop_0.1"
  "04_muonLR_0.005_embdrop_0.1"
  "05_muonLR_0.002_embdrop_0.1"
  "06_muonLR_0.0005_embdrop_0.1"
  "07_muonLR_0.050_embdrop_0.5"
  "08_muonLR_0.020_embdrop_0.5"
  "09_muonLR_0.010_embdrop_0.5"
  "10_muonLR_0.005_embdrop_0.5"
  "11_muonLR_0.002_embdrop_0.5"
  "12_muonLR_0.0005_embdrop_0.5"
)

EXTRA_FLAGS=(
  "--heads-muon-lr 0.05 --emb-dropout 0.1"
  "--heads-muon-lr 0.02 --emb-dropout 0.1"
  "--heads-muon-lr 0.01 --emb-dropout 0.1"
  "--heads-muon-lr 0.005 --emb-dropout 0.1"
  "--heads-muon-lr 0.002 --emb-dropout 0.1"
  "--heads-muon-lr 0.0005 --emb-dropout 0.1"
  "--heads-muon-lr 0.05 --emb-dropout 0.5"
  "--heads-muon-lr 0.02 --emb-dropout 0.5"
  "--heads-muon-lr 0.01 --emb-dropout 0.5"
  "--heads-muon-lr 0.005 --emb-dropout 0.5"
  "--heads-muon-lr 0.002 --emb-dropout 0.5"
  "--heads-muon-lr 0.0005 --emb-dropout 0.5"
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
