# SlipperyAnt JAX Memory Fraction Probe - 2026-04-25

## Context

Goal: estimate how much GPU memory the default state-observation SlipperyAnt PPO run actually needs, independent of JAX's default preallocation behavior.

Default per-run PPO config preserved:

- `N_ENVS=128`
- `NUM_STEPS=10`
- `NUM_MINIBATCHES=32`
- `UPDATE_EPOCHS=4`
- `batch=1280`
- `minibatch_size=40`

Implemented and launched:

- `ppo_slippery_brax_memfrac.py`, which respects externally provided `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- `scripts/CL/brax/slippery_ant_wrapper_lr1e4_3seeds_per_gpu.sh`, launched as job `71479`.

## Full 3-Seed Run

Submitted:

```bash
sbatch scripts/CL/brax/slippery_ant_wrapper_lr1e4_3seeds_per_gpu.sh
```

Job `71479` started on `worker-11` with:

- `SEEDS="0 1 2"`
- `JAX_MEM_FRACTION=0.25`

Observed with `nvidia-smi`:

- Three Python processes on one H100.
- About `20.8 GiB` reserved per process.
- About `62.5-64.4 GiB` total reserved.
- GPU utilization samples reached `99-100%`.

Interpretation: 3 default runs per H100 at `0.25` are viable and keep the GPU busy.

## Probe Results

Short single-seed probes used:

```bash
SEEDS=0 JAX_MEM_FRACTION=<fraction> \
  scripts/CL/brax/slippery_ant_wrapper_lr1e4_3seeds_per_gpu.sh \
  continual_brax rl-power 4999936 20 2560
```

Results:

| Fraction | Job | Timesteps | Result | Observed allocation |
| --- | ---: | ---: | --- | ---: |
| `0.25` | `71480` | `2560` | completed | `~21.5 GiB` |
| `0.20` | `71481` | `2560` | completed | not sampled |
| `0.15` | `71482` | `2560` | completed | not sampled |
| `0.10` | `71483` | `2560` | completed | not sampled |
| `0.05` | `71484` | `2560` | completed | `~4.6 GiB` |
| `0.03` | `71486` | `2560` | completed | not sampled |
| `0.02` | `71487` | `2560` | completed | not sampled |
| `0.01` | `71488` | `2560` | completed | not sampled |

Longer confirmation probe:

```bash
SEEDS=0 JAX_MEM_FRACTION=0.01 \
  scripts/CL/brax/slippery_ant_wrapper_lr1e4_3seeds_per_gpu.sh \
  continual_brax rl-power 4999936 20 12800
```

Job `71489` completed 10 PPO updates successfully.

Observed during active compute:

- GPU memory: `~2.0 GiB`
- GPU utilization: `91%`
- Final log: `Average SPS: 498` for the short 12.8k-step probe, dominated by startup/W&B overhead.

## Findings

- The original `~50 GiB` seen for a single run was allocator reservation from `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`, not real live tensor need.
- A default single SlipperyAnt state PPO run can execute with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.01` on an 80 GB H100 for at least 10 PPO updates.
- Active allocation at `0.01` was about `2.0 GiB`, so the live requirement for this shape is far below `20 GiB`.
- Since shapes are fixed by `N_ENVS`, `NUM_STEPS`, network size, and minibatch geometry, longer total timesteps should not substantially increase required device memory after compilation.

## Recommendation

- Keep the already-launched full job `71479` at `0.25`; it is healthy and already running.
- For the next full multi-seed launcher, lower the default from `0.25` to `0.05` or `0.10`.
- Practical starting points:
  - `0.10`: conservative, likely enough for 6-8 default runs by memory, though compute contention may dominate.
  - `0.05`: more aggressive, still completed probes and should fit many processes by memory.
  - `0.01`: proven for short probes, but too tight to make the default without a longer full-run validation.
- The next bottleneck is likely compute/process contention, not memory capacity. Increase processes per GPU gradually and compare aggregate SPS, not just whether memory fits.
