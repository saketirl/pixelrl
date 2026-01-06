# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PixelRL is a reinforcement learning framework for pixel-based continuous control tasks using JAX and Flax. It implements DDPG and PPO algorithms for training agents on physics-based environments with pixel observations rendered from 3D simulation via PixelBrax.

**Tech Stack**: JAX, Flax, MuJoCo 3.2.6 (MJX backend), Optax, Wandb, Python 3.9, UV package manager

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run DDPG training
uv run ddpg_pixelbrax_jax.py --env-name walker2d --n-envs 16 --total-timesteps 1000000

# Run PPO training
uv run ppo_pixelbrax_jax.py --env-name walker2d --n-envs 64 --total-timesteps 1000000

# Test JAX/MuJoCo setup
python test.py

# HPC submission (SLURM)
sbatch scripts/ddpg_jax_pixel/ddpg.sh
sbatch scripts/ppo_jax_pixel/ppo.sh
```

**Required environment variable** (for Brax source):
```bash
export PYTHONPATH="/users/apraka15/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH}"
```

## Architecture

### Core Data Flow
```
PixelBrax Environment (make_pixel_brax)
    ↓ RGB observations (84x84x3)
CNN Encoder (Conv 32→64→64, strides 4,2,1)
    ↓ 256-dim latent
Actor/Critic Networks
    ↓ Actions / Q-values
```

### Key Components

| File | Purpose |
|------|---------|
| `pixelbrax/env_utils.py` | Environment factory (`make_pixel_brax()`), PixelEnv wrapper, camera configs |
| `ddpg_pixelbrax_jax.py` | DDPG: Actor-Critic with replay buffer, soft target updates |
| `ppo_pixelbrax_jax.py` | PPO: On-policy with GAE, clipped loss, JIT-compiled training |
| `pure_jax_ppo.py` | Reference PPO without pixel observations (MLP baseline) |
| `pixelbrax/brax/` | Bundled Brax physics engine source |
| `pixelbrax/renderer/` | 3D rendering pipeline for pixel observations |
| `clean_rl_reference/` | reference code of gold-standard implementations of RL algorithms |


### Environment Creation

```python
from pixelbrax import make_pixel_brax

envs, timestep_fn, reset_fn = make_pixel_brax(
    backend="spring",           # "spring", "generalized", "positional"
    env_name="halfcheetah",     # "walker2d", "swimmer", etc.
    n_envs=16,                  # Parallel environments
    seed=0,
    hw=84,                      # Pixel observation size
    distractor=None,            # None or "videos" for DAVIS backgrounds
    video_path="datasets/DAVIS-2017-trainval-480p/",
)
```

### Training Patterns

Both algorithms follow:
1. Config via dataclass/dict with all hyperparameters
2. Single call to `make_pixel_brax()` for environment
3. JAX key splitting + Flax `init()` for networks
4. `TrainState` objects with optimizer state
5. Core loops use `jax.jit()` and `jax.vmap()` for batching
6. Wandb integration for logging (`--track` flag)

**Pixel handling**: Stored as `uint8` in replay buffer, normalized to [0,1] in network forward passes.

## Git Workflow

- **Main branch**: `continual`
- **Remote**: https://github.com/saketirl/pixelrl.git
