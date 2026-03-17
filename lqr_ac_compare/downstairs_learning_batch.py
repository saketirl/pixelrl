"""
Batch-based downstairs actor-critic update inspired by CT-DDPG.

Key insight: Single-step TD has variance O(1/dt) which blows up as dt→0.
Using multi-step returns with fixed Lh keeps variance bounded.

Instead of: delta_t = r_t - psi_t + (gamma * V_{t+1} - V_t) / dt  [variance ~ 1/dt]
We use:     delta = V_t - sum_{l=0}^{L-1} gamma^l r_{t+l} dt - gamma^L V_{t+L}  [variance ~ O(1)]
"""

import numpy as np
from typing import Dict

Array = np.ndarray


def cayley_retract(X: Array, G: Array, eta: float) -> Array:
    """
    Cayley retraction for matrices on Stiefel manifold.
    Moves X in direction of G (Riemannian gradient).
    """
    n, p = X.shape
    I = np.eye(n)
    A = G @ X.T - X @ G.T  # skew-symmetric
    return np.linalg.solve(I + 0.5 * eta * A, (I - 0.5 * eta * A) @ X)


def batch_multistep_update(
    actor,
    critic,
    s_traj: Array,   # (N+1, ds)
    a_traj: Array,   # (N, da)
    r_traj: Array,   # (N,)
    *,
    beta: float,
    dt: float,
    L: int = 10,     # multi-step horizon (L steps, so Lh = L*dt)
    eta_actor: float,
    eta_critic: float,
) -> Dict[str, float]:
    """
    Batch multi-step actor-critic update.

    Uses L-step returns to compute TD targets, avoiding 1/dt variance blow-up.
    """
    N = len(r_traj)
    ds, da = s_traj.shape[1], a_traj.shape[1]

    if N < L:
        L = N  # use full trajectory if shorter than L

    # Discount factor per step
    gamma = np.exp(-beta * dt)

    # Compute multi-step returns for each starting point
    # G_t = sum_{l=0}^{L-1} gamma^l r_{t+l} * dt + gamma^L V(s_{t+L})

    # First, compute all values V(s) = s^T Ub s + Uc^T s + c
    values = np.array([critic.phi(s_traj[i]) for i in range(N + 1)])

    # Compute multi-step TD targets
    targets = np.zeros(N - L + 1)
    for t in range(N - L + 1):
        # Discounted sum of rewards
        G = 0.0
        for l in range(L):
            G += (gamma ** l) * r_traj[t + l] * dt
        # Add bootstrapped value
        G += (gamma ** L) * values[t + L]
        targets[t] = G

    # Critic loss: MSE between V(s_t) and multi-step target
    # Gradient: d/dP ||V(s) - target||^2 = 2(V(s) - target) * d/dP V(s)

    # For V(s) = s^T Ub s + Uc^T s + c:
    # dV/dUb = s @ s^T
    # dV/dUc = s
    # dV/dc = 1

    G_Ub = np.zeros_like(critic.Ub)
    G_Uc = np.zeros_like(critic.Uc_col)
    G_c = 0.0

    mse = 0.0
    n_samples = N - L + 1

    for t in range(n_samples):
        s = s_traj[t]
        V = values[t]
        target = targets[t]
        td_error = V - target  # want to minimize (V - target)^2
        mse += td_error ** 2

        # Gradient direction (for minimizing MSE, we want negative gradient)
        G_Ub -= td_error * np.outer(s, s)
        G_Uc -= td_error * s.reshape(-1, 1)
        G_c -= td_error

    G_Ub /= n_samples
    G_Uc /= n_samples
    G_c /= n_samples
    mse /= n_samples

    # Update critic using Cayley retraction (for Stiefel-constrained Ub, Uc)
    critic.Ub = cayley_retract(critic.Ub, G_Ub, eta_critic)
    critic.Uc_col = cayley_retract(critic.Uc_col, G_Uc, eta_critic)
    critic.c = critic.c + eta_critic * G_c

    # Actor update using deterministic policy gradient (DPG):
    # dJ/dW = E[dQ/da * da/dW] = E[dPsi/da * da/dW]
    #
    # For Psi(s, a) = s^T Zb a + Zc a:
    #   dPsi/da = Zb^T s + Zc^T
    #
    # For policy a = Wc @ s = Wc_col^T @ s:
    #   da/d(Wc_col) is s (ds x 1) mapped to da (da x 1)
    #   Full gradient: s @ (dPsi/da)^T = s @ (Zb^T s + Zc^T)^T
    #
    # We want to MAXIMIZE expected return, so we ascend in direction of dPsi/da

    G_Wc = np.zeros_like(actor.Wc_col)

    for t in range(n_samples):
        s = s_traj[t]

        # Advantage gradient at current policy: dPsi/da|_{a=pi(s)}
        g = (critic.Zb.T @ s).flatten() + critic.Zc_row.flatten()  # (da,)

        # For Wc_col (ds x da), gradient is outer(s, g)
        # This moves policy in direction that increases Psi
        G_Wc += np.outer(s, g)

    G_Wc /= n_samples

    # Update actor
    actor.Wc_col = cayley_retract(actor.Wc_col, G_Wc, eta_actor)

    # Also update advantage network (Zb, Zc) - learn to predict advantage
    # Psi(s, a) = s^T Zb a + Zc a
    # Should predict: A(s, a) = Q(s, a) - V(s)
    # For LQR: A(s, a) ≈ (a - a*)^T R (a - a*) which is quadratic, not linear in a
    # But the gradient dA/da = R(a - a*) is linear, which Zb can capture

    G_Zb = np.zeros_like(critic.Zb)
    G_Zc = np.zeros_like(critic.Zc_row)

    for t in range(n_samples):
        s = s_traj[t]
        a = a_traj[t]
        a_pi = actor.act(s)

        # Target for Psi: the actual advantage
        actual_advantage = targets[t] - values[t]
        predicted_psi = critic.Psi(s, a) - critic.Psi(s, a_pi)
        psi_error = predicted_psi - actual_advantage

        # Gradient of Psi(s,a) - Psi(s, a_pi) w.r.t. Zb, Zc
        # Psi(s, a) = s^T Zb a + Zc a
        # d/dZb = s @ a^T - s @ a_pi^T = s @ (a - a_pi)^T
        a_diff = (a - a_pi).reshape(da,)

        G_Zb -= psi_error * np.outer(s, a_diff)
        G_Zc -= psi_error * a_diff.reshape(1, da)

    G_Zb /= n_samples
    G_Zc /= n_samples

    critic.Zb = cayley_retract(critic.Zb, G_Zb, eta_critic)
    critic.Zc_row = critic.Zc_row + eta_critic * G_Zc

    return {
        "mse": float(mse),
        "mean_advantage": float(np.mean([targets[t] - values[t] for t in range(n_samples)])),
        "ortho_err_Wc": float(np.linalg.norm(actor.Wc_col.T @ actor.Wc_col - np.eye(da))),
        "ortho_err_Ub": float(np.linalg.norm(critic.Ub.T @ critic.Ub - np.eye(ds))),
        "G_Wc_norm": float(np.linalg.norm(G_Wc)),
        "G_Ub_norm": float(np.linalg.norm(G_Ub)),
    }
