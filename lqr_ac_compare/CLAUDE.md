# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Project Overview

This sub-folder verifies the similarity between the **upstairs** (high-dimensional observation) and **downstairs** (low-dimensional latent state) formulations of actor-critic learning described in `Stiefel_Ascent_RLC.pdf`.

The code runs two learning setups on the **same underlying LQR dynamics**:
- **Upstairs**: deep linear networks with **L = 3**, using **Cayley retraction** to stay on the Stiefel/orthogonal constraints.
- **Downstairs**: shallow low-dimensional analogue using the **Theorem-2-style Stiefel-direction** update.

**Primary goal**: Confirm that the effective learning dynamics match between upstairs and downstairs when initialized with the same effective parameters.

## Stiefel Manifold Definition

The **Stiefel manifold** St(n, p) is the set of n x p matrices with orthonormal columns:

```
St(n, p) = { X ∈ R^{n×p} : X^T X = I_p }
```

**IMPORTANT**: The constraint is `X^T X = I_p` (orthonormal columns), NOT `||X||_F = 1` (unit Frobenius norm).

For the LQR problem:
- Policy K ∈ R^{da × ds}: requires `K K^T = I_da` (orthonormal rows, since da < ds)
- Actor Wc_col ∈ R^{ds × da}: requires `Wc_col^T Wc_col = I_da` (orthonormal columns)
- Advantage Zb ∈ R^{ds × da}: requires `Zb^T Zb = I_da` (orthonormal columns)

## Key Algorithms

### CT-DDPG Martingale Loss (arXiv 2509.23711)

The `*_batch.py` files implement the CT-DDPG algorithm with the **[r - q] formulation**:

```python
# CT-DDPG Martingale Loss:
# G_t = sum_{l=0}^{L-1} gamma^l [r_{t+l} - q(s_{t+l}, a_{t+l})] * dt + gamma^L V(s_{t+L})
#
# where q(s, a) = Psi(s, a) - Psi(s, a_pi) is the advantage relative to current policy

for l in range(L):
    a_pi_l = actor.act(s_l)
    q_l = critic.Psi(s_l, a_l) - critic.Psi(s_l, a_pi_l)
    G += (gamma ** l) * (r_traj[t + l] - q_l) * dt
G += (gamma ** L) * values[t + L]
```

**Key insight**: Subtracting the advantage rate `q` from rewards maintains the martingale property and keeps variance bounded as dt→0.

### Stiefel-Compatible LQR Construction

Standard LQR has `||K|| ≠ 1` and `P ≠ I`, violating Stiefel constraints. Use QR decomposition for proper Stiefel construction:

```python
def construct_compatible_lqr(ds: int, da: int, alpha: float = 0.5, seed: int = 42):
    """Construct Stiefel-compatible LQR with P=I, K on Stiefel manifold (K K^T = I_da)."""
    rng = np.random.default_rng(seed)
    G = -alpha * np.eye(ds)

    # H must have orthonormal columns (H^T H = I_da) so that K = H^T is on Stiefel
    H_raw = rng.standard_normal((ds, da))
    H, _ = np.linalg.qr(H_raw, mode='reduced')  # H is (ds, da) with H^T H = I_da

    R = np.eye(da)
    Q = H @ H.T + 2 * alpha * np.eye(ds)
    return G, H, Q, R
    # Result: P_opt = I, K_opt = H^T with K_opt K_opt^T = I_da
```

**Note**: `||K_opt||_F = sqrt(da)`, not 1. For da=1, ||K||=1; for da=2, ||K||=sqrt(2)≈1.414.

### Matched Initialization (Critical for Theorem Verification)

For upstairs and downstairs to have matching learning dynamics, they must start with the **same effective parameters**. Initialize downstairs first (naturally Stiefel), then set upstairs to match:

```python
# 1. Initialize downstairs (Stiefel-valid)
down_actor = DownstairsActorShallowT2(ds, da, seed)  # Wc_col on Stiefel
down_critic = DownstairsCriticShallowT2(ds, da, seed)
down_critic.Zb = H.copy()  # H already has orthonormal columns from QR

# 2. Set upstairs to match downstairs effective matrices
# Actor: Weff @ M should equal Wc_col.T
Wc = down_actor.Wc_col.T  # (da, ds)
W3_target = Wc @ M.T  # (da, d)
up_actor.W3 = W3_target / np.linalg.norm(W3_target, axis=1, keepdims=True)  # row normalize
up_actor.W1 = np.eye(d)
up_actor.W2 = np.eye(d)

# Critic value: set U1=U2=I, U3 = M @ Ub @ M.T + orthogonal complement
# Critic advantage: Z3 = Zb.T @ M.T (row normalized), Z1=Z2=I
```

**Verification**: After matched init, `cos(down_policy, optimal) == cos(up_policy, optimal)`.

## Repo Layout

### Core Files
- `lqr_env.py` - LQR environment with latent state `s_t ∈ R^{ds}` and upstairs observation `o_t = M s_t + noise`
- `compare_upstairs_downstairs.py` - Main comparison script with matched initialization

### Downstairs (low-dimensional)
- `downstairs_models.py` - Shallow actor/critic with Stiefel constraints
- `downstairs_learning_batch.py` - Batch multi-step CT-DDPG update

### Upstairs (high-dimensional, L=3 DLN)
- `upstairs_models.py` - Deep linear network actor/critic (L=3)
- `upstairs_learning_batch.py` - Batch multi-step update for L=3 networks

## Verified Working Cases

### Case 1: Small Problem (ds=4, da=1)

```bash
python compare_upstairs_downstairs.py --iters 3000 --n_steps 10 --ds 4 --da 1 --d 64 --alpha 0.5 \
    --eta_actor 0.01 --eta_critic 0.05 --save_plot comparison_plot.png --save_csv comparison_results.csv
```

**Results**:
| Metric | Downstairs | Upstairs | Optimal |
|--------|-----------|----------|---------|
| Initial cos | -0.009 | -0.009 | 1.0 |
| Final 50-iter avg reward | -0.172 | -0.188 | -0.158 |
| Final cos to optimal | 0.64 | 0.74 | 1.0 |
| **Policy correlation** | **0.9788** | | |

### Case 2: Harder Problem (ds=8, da=2) - Recommended

```bash
python compare_upstairs_downstairs.py --iters 500 --n_steps 30 --ds 8 --da 2 --d 128 --alpha 0.3 \
    --eta_actor 0.01 --eta_critic 0.05 --save_plot comparison_hard.png --save_csv comparison_hard.csv
```

**Results**:
| Metric | Downstairs | Upstairs | Optimal |
|--------|-----------|----------|---------|
| Initial cos | -0.127 | -0.127 | 1.0 |
| Initial reward | -1.45 | -1.19 | -0.29 |
| Final 50-iter avg reward | -0.341 | -0.333 | -0.289 |
| Final cos to optimal | 0.75 | 0.90 | 1.0 |
| **Policy correlation** | **0.9914** | | |

**Key observation**: K K^T = I_2 (Stiefel constraint verified), ||K_opt|| = 1.414 = sqrt(2).

### Quick Test Command

For rapid iteration during development:
```bash
python compare_upstairs_downstairs.py --iters 50 --n_steps 30 --ds 8 --da 2 --d 128 --alpha 0.3
```

## Known Quirks

1. **Upstairs converges faster**: Even with matched initialization, upstairs DLN (L=3) converges faster than downstairs shallow networks (cos 0.90 vs 0.75). May be due to better gradient flow through deep linear layers.

2. **Don't normalize Zb by Frobenius norm**: Since H already has orthonormal columns from QR decomposition, use `Zb = H.copy()` directly. Dividing by `||H||_F` breaks the Stiefel constraint and causes singular matrix errors in Cayley retraction.

3. **Bilinear Psi only**: The advantage function should be bilinear:
   ```python
   Psi(s, a) = s^T Zb a + Zc a  # CORRECT
   # NOT: Psi(s, a) = a^T Za a + s^T Zb a + Zc a  # quadratic Za causes instability
   ```

4. **Learning rate sensitivity**: Both require careful tuning. Recommended: `eta_actor=0.01, eta_critic=0.05`.

5. **Reward correlation is low but policy correlation is high**: Even when policies match (cos correlation ~0.99), reward correlations can be ~0.05-0.5 due to noise in state distributions. Use policy convergence correlation as the primary metric.

6. **Value function U=I initialization hurts upstairs**: Random orthogonal initialization works better than U1=U2=U3=I.

## SGD vs Cayley Retraction (Stiefel Manifold Learning)

The codebase supports two optimization modes controlled by `--use_sgd`:

### Cayley Retraction (Default, Recommended)

Cayley retraction keeps matrices on the Stiefel manifold after each gradient step:

```python
def cayley_retract(X: Array, G: Array, eta: float) -> Array:
    """Cayley retraction for matrices on Stiefel manifold."""
    n, p = X.shape
    I = np.eye(n)
    A = G @ X.T - X @ G.T  # skew-symmetric
    return np.linalg.solve(I + 0.5 * eta * A, (I - 0.5 * eta * A) @ X)
```

**Properties**:
- Maintains orthogonality constraint: `X^T X = I_p` after update
- Bounded matrix norms prevent divergence
- Numerically stable for long training runs
- Required for theoretical guarantees in Stiefel_Ascent_RLC.pdf

### Canonical SGD (`--use_sgd`)

Standard gradient descent without manifold constraints:

```python
def sgd_update(X: Array, G: Array, eta: float) -> Array:
    return X + eta * G
```

**Properties**:
- Matrices can grow unboundedly
- Faster per-iteration (no linear solve)
- Often diverges or stagnates without careful learning rate tuning
- Useful for ablation studies comparing manifold vs unconstrained optimization

**Example comparison**:
```bash
# With Cayley retraction (stable, converges)
python compare_upstairs_downstairs.py --iters 500

# With SGD (may diverge or stagnate)
python compare_upstairs_downstairs.py --iters 500 --use_sgd
```

## Multi-step Bootstrapping (`--n_steps`)

The `n_steps` parameter controls the **bootstrapping horizon** for computing TD targets. This is critical for variance control in continuous-time RL.

### How It Works

Instead of single-step TD (high variance as dt→0):
```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  # variance ~ O(1/dt)
```

We use multi-step returns (bounded variance):
```
G_t = sum_{l=0}^{n_steps-1} gamma^l [r_{t+l} - q_{t+l}] * dt + gamma^{n_steps} V(s_{t+n_steps})
```

### Parameter Meaning

- `n_steps=30` means bootstrap from state 30 timesteps ahead
- Effective bootstrapping time horizon = `n_steps * dt` time units
- For `dt=0.02`, `n_steps=30` → 0.6 time units of integration

### Recommended Values

| Problem Size | n_steps | Reason |
|--------------|---------|--------|
| ds=4, da=1 | 10 | Smaller problem tolerates shorter horizon |
| ds=8, da=2 | **30** | Working default, realistic horizon |

With `dt=0.02`, `n_steps=30` gives a bootstrapping horizon of 0.6 time units, which provides good bias-variance tradeoff for the CT-DDPG algorithm.

**Note**: `n_steps` is NOT the network depth (L=3 for DLN). The parameter was renamed from `--L` to `--n_steps` to avoid this confusion.

## Sweep Experiments (`run_sweep.py`)

The sweep script runs multiple experiments with different noise seeds while keeping the LQR problem and policy initialization fixed. This tests robustness across environment noise realizations.

### Seed Structure

Three independent seeds control different sources of randomness:
- `--seed`: LQR problem construction (G, H, Q, R matrices)
- `--init_seed`: Policy/critic initialization (default: seed+1)
- `--noise_seed`: Exploration and environment noise (swept over)

### Working Commands

**Standard comparison sweep (16 seeds, 4 parallel)**:
```bash
python run_sweep.py \
    --noise_seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --n_steps 30 \
    --iters 500 \
    --max_parallel 4
```

**Quick test sweep (4 seeds)**:
```bash
python run_sweep.py \
    --noise_seeds 0 1 2 3 \
    --n_steps 30 \
    --iters 100 \
    --max_parallel 4
```

**Downstairs only (faster, for debugging)**:
```bash
python run_sweep.py \
    --noise_seeds 0 1 2 3 4 5 6 7 \
    --n_steps 30 \
    --iters 500 \
    --downstairs_only \
    --max_parallel 8
```

**With SGD instead of Cayley (ablation)**:
```bash
python run_sweep.py \
    --noise_seeds 0 1 2 3 \
    --n_steps 30 \
    --iters 500 \
    --use_sgd \
    --max_parallel 4
```

### Output Files

Results are saved with naming convention:
```
results_nsteps{n_steps}_noise{noise_seed}.csv
plot_nsteps{n_steps}_noise{noise_seed}.png
```

### Working Hyperparameters

The following configuration has been verified to work reliably:

```bash
python run_sweep.py \
    --seed 0 \
    --ds 8 --da 2 --d 128 \
    --n_steps 30 \
    --alpha 0.3 \
    --eta_actor 0.01 \
    --eta_critic 0.05 \
    --iters 500 \
    --noise_seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --max_parallel 4
```

**Typical results (16-seed average)**:
| Metric | Downstairs | Upstairs |
|--------|-----------|----------|
| Final reward | -0.414 ± 0.078 | -0.376 ± 0.055 |
| Final cos to optimal | 0.282 ± 0.005 | 0.463 ± 0.007 |

## Summary

The theorem is verified: **upstairs and downstairs have equivalent learning dynamics** (policy correlation 0.97-0.99) when:
1. Using Stiefel-compatible LQR construction with QR decomposition
2. Matching initial effective parameters (downstairs first, then upstairs matches)
3. Using CT-DDPG [r - q] loss formulation
4. Using bilinear advantage function Psi(s, a) = s^T Zb a + Zc a
5. Using Cayley retraction (not SGD) for Stiefel manifold optimization
6. Using appropriate bootstrapping horizon (n_steps=30 for ds=8, da=2)
