# CRATE CNN Head Scaling

This directory contains the actor/critic head scaling experiment based on:

- `scripts/staging/crate_cnn_meanbound_boundedstd_crate_stiefel_allenv_4seeds.sh`
- "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities" (`arXiv:2503.14858`)

The experiment is a repo-faithful PPO adaptation of the paper's width-vs-depth comparison. It keeps the pixel encoder fixed as `crate_cnn` and scales only the actor/critic CRATE heads on `ant` and `humanoid`.

## Setup

From the repo root:

```bash
git submodule update --init --recursive
uv sync
export PYTHONPATH="$(pwd)/pixelbrax/brax:${PYTHONPATH:-}"
```

The submodule command is required after a fresh clone because this repo uses the vendored Brax checkout at `pixelbrax/brax`.

For W&B, either export a key:

```bash
export WANDB_API_KEY="..."
```

or place it at:

```bash
secrets/wandb_api_key.txt
```

The launcher also checks the parent repo for `secrets/wandb_api_key.txt` when run from a worktree.

## Launcher

Script:

```bash
scripts/scaling/crate_cnn_head_scaling_ant_humanoid_4seeds.sh
```

Submit with:

```bash
sbatch scripts/scaling/crate_cnn_head_scaling_ant_humanoid_4seeds.sh [WANDB_PROJECT] [WANDB_ENTITY] [TOTAL_TIMESTEPS]
```

Defaults:

- `WANDB_PROJECT=encoder`
- `WANDB_ENTITY=""`
- `TOTAL_TIMESTEPS=10000000`

Example:

```bash
sbatch scripts/scaling/crate_cnn_head_scaling_ant_humanoid_4seeds.sh encoder "" 10000000
```

## Sweep

The SLURM array is `0-55%4`, for 56 total tasks and 4 concurrent tasks.

Task dimensions:

- Envs: `ant`, `humanoid`
- Backends: `spring`, `spring`
- Seeds: `0`, `1`, `2`, `3`
- Scaling configs:
  - baseline/depth: `head_width=256`, `head_crate_layers=1`
  - width: `head_width=512`, `head_crate_layers=1`
  - width: `head_width=1024`, `head_crate_layers=1`
  - width: `head_width=2048`, `head_crate_layers=1`
  - depth: `head_width=256`, `head_crate_layers=8`
  - depth: `head_width=256`, `head_crate_layers=16`
  - depth: `head_width=256`, `head_crate_layers=32`

The depth curve is locked to `head_crate_layers={1,8,16,32}`, with `256 x 1` shared as the width baseline.

Task mapping:

```text
seed_idx  = TASK_ID % 4
scale_idx = (TASK_ID / 4) % 7
env_idx   = TASK_ID / 28
```

## Learning Rate Scaling

Only actor/critic head learning rates are scaled:

```text
scale = sqrt(head_width / 256) * head_crate_layers
heads_stiefel_lr = 0.001 * scale
heads_adam_lr    = 0.0003 * scale
```

This is the baseline-preserving form of `sqrt(width) * layers`: `head_width=256` and `head_crate_layers=1` gives `scale=1`. The encoder LR stays fixed at `3e-4` because the encoder is not scaled.

## Memory Notes

The script sets:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
```

unless the caller already set it. The trainer respects this override. The most memory-risky condition is the width sweep at `head_width=2048`; if jobs OOM, reduce concurrency by editing the SLURM array suffix from `%4` to `%2`.

## Checks

Syntax-check the launcher:

```bash
bash -n scripts/scaling/crate_cnn_head_scaling_ant_humanoid_4seeds.sh
```

Verify the trainer exposes the scaling flags:

```bash
uv run python ppo_pixelbrax.py --help | rg "head-hidden-dim|head-crate-layers"
```

Small compile smoke:

```bash
PYTHONPATH="$(pwd)/pixelbrax/brax:${PYTHONPATH:-}" JAX_PLATFORMS=cpu \
uv run python ppo_pixelbrax.py \
  --env-name ant \
  --backend spring \
  --n-envs 2 \
  --total-timesteps 20 \
  --num-steps 1 \
  --num-minibatches 1 \
  --update-epochs 1 \
  --encoder-type crate_cnn \
  --use-crate-head \
  --head-hidden-dim 512 \
  --head-crate-layers 2
```
