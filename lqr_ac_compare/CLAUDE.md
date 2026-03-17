# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Project Overview

This sub-folder verifies the similarity between the **upstairs** (high-dimensional observation) and **downstairs** (low-dimensional latent state) formulations of actor–critic learning described in `Stiefel_Ascent_RLC.pdf`.

The code runs two learning setups on the **same underlying LQR dynamics**:
- **Upstairs**: deep linear networks with **L = 3**, using **Cayley retraction** to stay on the Stiefel/orthogonal constraints.
- **Downstairs**: shallow low-dimensional analogue using the **Theorem-2-style Stiefel-direction** update.

Primary goal: confirm that the effective learning dynamics match qualitatively (and, after tuning, quantitatively) between upstairs and downstairs.

## Key Algorithms

### Batch Multi-step Updates (CT-DDPG style)

The `*_batch.py` files implement batch multi-step returns to avoid O(1/dt) variance blow-up:

```
# Instead of single-step TD (variance ~ 1/dt):
delta_t = r_t - psi_t + (gamma * V_{t+1} - V_t) / dt

# Use multi-step returns (variance ~ O(1)):
G_t = sum_{l=0}^{L-1} gamma^l r_{t+l} dt + gamma^L V_{t+L}
```

### Critical Initialization for DPG

For deterministic policy gradient (DPG) to work correctly, the advantage gradient `Zeff @ o` must equal the true gradient `dA/da` at the current policy:

```python
# For LQR: dA/da|_{a=π(s)} = -2(Weff + K@M.T) @ o
# NOT just -K @ M.T (the optimal action direction)
target_Zeff = -2 * (Weff_current + K_opt @ M.T)
```

### Stiefel-Compatible LQR

Standard LQR has `||K|| ≠ 1` and `P ≠ I`, violating Stiefel constraints. Use compatible construction:
```python
G = -alpha * I  # stable diagonal
H = unit_norm_random  # ||H|| = 1
Q = H @ H.T + 2*alpha*I
R = I
# Result: P_opt = I (orthogonal), K_opt = H.T (unit norm)
```

## Repo Layout

### Core Files
- `lqr_env.py` - LQR environment with latent state `s_t ∈ R^{d_s}` and upstairs observation `o_t = M s_t + noise`

### Downstairs (low-dimensional)
- `downstairs_models.py` - Shallow actor/critic with Stiefel constraints
- `downstairs_learning_batch.py` - Batch multi-step CT-DDPG update
- `run_downstairs_batch.py` - Entry point for downstairs training

### Upstairs (high-dimensional, L=3 DLN)
- `upstairs_models.py` - Deep linear network actor/critic (L=3)
- `upstairs_learning_batch.py` - Batch multi-step update for L=3 networks
- `run_upstairs_batch.py` - Entry point for upstairs training

### Legacy/Reference
- `run_experiment.py` - Original combined upstairs+downstairs runner
- `run_simple_ac.py` - Simple actor-critic without Stiefel constraints

## How to Run

```bash
# Downstairs learning (recommended: --compatible for Stiefel-valid LQR)
python run_downstairs_batch.py --compatible --iters 3000 --L 10 --eta_actor 0.01 --eta_critic 0.05

# Upstairs learning
python run_upstairs_batch.py --compatible --iters 3000 --L 10 --eta_actor 0.01 --eta_critic 0.05

# With optimal initialization (actor starts at optimal)
python run_upstairs_batch.py --compatible --optimal_init --iters 1000

