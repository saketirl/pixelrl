"""
Batch-based downstairs actor-critic update implementing CT-DDPG.

Key insight: Single-step TD has variance O(1/dt) which blows up as dt→0.
Using multi-step returns with fixed Lh keeps variance bounded.

CT-DDPG Martingale Loss (from arXiv 2509.23711):
  ℒᴹ = (V(s_t) - ∑_{l=0}^{L-1} γ^l [r_{t+l} - q(s_{t+l}, a_{t+l})] dt - γ^L V(s_{t+L}))²

The key is subtracting the advantage rate q from rewards in the integral.
This maintains the martingale property and keeps variance bounded as dt→0.

Alternative: Generator-based update (use_generator=True):
  δ_t = r_t - ψ(s_t, a_t) + L^π V(s_t) - β V(s_t)
  where L^π V(s_t) ≈ (V(s_{t+1}) - V(s_t)) / dt (sample-based estimate)
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
    use_generator: bool = False,  # Use generator-based TD error instead of multi-step
) -> Dict[str, float]:
    """
    Batch multi-step actor-critic update.

    Uses L-step returns to compute TD targets, avoiding 1/dt variance blow-up.

    If use_generator=True, uses generator-based TD error instead:
        δ_t = r_t - ψ(s_t, a_t) + (V(s_{t+1}) - V(s_t))/dt - β V(s_t)
    """
    N = len(r_traj)
    ds, da = s_traj.shape[1], a_traj.shape[1]

    # Discount factor per step
    gamma = np.exp(-beta * dt)

    # First, compute all values V(s) = s^T Ub s + Uc^T s + c
    values = np.array([critic.phi(s_traj[i]) for i in range(N + 1)])

    # ========== GENERATOR-BASED UPDATE ==========
    if use_generator:
        # Generator-based TD error:
        # δ_t = r_t - ψ(s_t, a_t) + L^π V(s_t) - β V(s_t)
        # where L^π V(s_t) ≈ (V(s_{t+1}) - V(s_t)) / dt

        weights = np.array([gamma ** t for t in range(N)])

        G_Ub = np.zeros_like(critic.Ub)
        G_Uc = np.zeros_like(critic.Uc_col)
        G_c = 0.0
        G_Zb = np.zeros_like(critic.Zb)
        G_Zc = np.zeros_like(critic.Zc_row)
        G_Wc = np.zeros_like(actor.Wc_col)

        mse = 0.0
        for t in range(N):
            s = s_traj[t]
            a = a_traj[t]
            r = r_traj[t]
            a_pi = actor.act(s)

            V_t = values[t]
            V_next = values[t + 1]

            # Sample-based generator estimate
            L_V_est = (V_next - V_t) / dt

            # Advantage rate
            psi = critic.Psi(s, a) - critic.Psi(s, a_pi)

            # TD error: δ = r - ψ + L^π V - β V
            delta = r - psi + L_V_est - beta * V_t
            mse += delta ** 2

            w = weights[t] * dt

            # Critic value gradients
            G_Ub -= w * delta * np.outer(s, s)
            G_Uc -= w * delta * s.reshape(-1, 1)
            G_c -= w * delta

            # Critic advantage gradients
            a_diff = a - a_pi
            G_Zb += w * delta * np.outer(s, a_diff)
            G_Zc += w * delta * a_diff.reshape(1, da)

            # Actor gradient (DPG)
            g = (critic.Zb.T @ s).flatten() + critic.Zc_row.flatten()
            G_Wc += w * np.outer(s, g)

        # Normalize
        total_weight = float(np.sum(weights * dt))
        G_Ub /= total_weight
        G_Uc /= total_weight
        G_c /= total_weight
        G_Zb /= total_weight
        G_Zc /= total_weight
        G_Wc /= total_weight
        mse /= N

        # Apply updates
        critic.Ub = cayley_retract(critic.Ub, G_Ub, eta_critic)
        critic.Uc_col = cayley_retract(critic.Uc_col, G_Uc, eta_critic)
        critic.c = critic.c + eta_critic * G_c
        critic.Zb = cayley_retract(critic.Zb, G_Zb, eta_critic)
        critic.Zc_row = critic.Zc_row + eta_critic * G_Zc
        actor.Wc_col = cayley_retract(actor.Wc_col, G_Wc, eta_actor)

        return {
            "mse": float(mse),
            "mean_advantage": 0.0,
            "ortho_err_Wc": float(np.linalg.norm(actor.Wc_col.T @ actor.Wc_col - np.eye(da))),
            "ortho_err_Ub": float(np.linalg.norm(critic.Ub.T @ critic.Ub - np.eye(ds))),
            "G_Wc_norm": float(np.linalg.norm(G_Wc)),
            "G_Ub_norm": float(np.linalg.norm(G_Ub)),
        }

    # ========== MULTI-STEP UPDATE (DEFAULT) ==========
    if N < L:
        L = N  # use full trajectory if shorter than L

    # CT-DDPG Martingale Loss (arXiv 2509.23711):
    # G_t = sum_{l=0}^{L-1} gamma^l [r_{t+l} - q(s_{t+l}, a_{t+l})] * dt + gamma^L V(s_{t+L})
    #
    # The advantage rate q(s, a) is subtracted from rewards to maintain
    # the martingale property and keep variance bounded as dt→0.

    # Compute multi-step TD targets with CT-DDPG [r - q] formulation
    # q(s, a) = Psi(s, a) - Psi(s, a_pi) is the advantage relative to current policy
    targets = np.zeros(N - L + 1)
    for t in range(N - L + 1):
        G = 0.0
        for l in range(L):
            s_l = s_traj[t + l]
            a_l = a_traj[t + l]
            a_pi_l = actor.act(s_l)
            # q(s, a) is the advantage at trajectory action relative to policy
            q_l = critic.Psi(s_l, a_l) - critic.Psi(s_l, a_pi_l)
            G += (gamma ** l) * (r_traj[t + l] - q_l) * dt
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
    #   Full gradient: s @ (dPsi/da)^T
    #
    # We want to MAXIMIZE expected return, so we ascend in direction of dPsi/da

    G_Wc = np.zeros_like(actor.Wc_col)

    for t in range(n_samples):
        s = s_traj[t]

        # Advantage gradient at current policy: dPsi/da|_{a=pi(s)}
        # dPsi/da = Zb^T*s + Zc^T
        g = (critic.Zb.T @ s).flatten() + critic.Zc_row.flatten()

        # For Wc_col (ds x da), gradient is outer(s, g)
        # This moves policy in direction that increases Psi
        G_Wc += np.outer(s, g)

    G_Wc /= n_samples

    # Update actor
    actor.Wc_col = cayley_retract(actor.Wc_col, G_Wc, eta_actor)

    # Also update advantage network (Zb, Zc) - learn to predict advantage
    # Psi(s, a) = s^T Zb a + Zc a
    # Should predict: A(s, a) = Q(s, a) - V(s)

    G_Zb = np.zeros_like(critic.Zb)
    G_Zc = np.zeros_like(critic.Zc_row)

    for t in range(n_samples):
        s = s_traj[t]
        a = a_traj[t].reshape(da,)
        a_pi = actor.act(s).reshape(da,)

        # Target for Psi: the actual advantage
        actual_advantage = targets[t] - values[t]
        predicted_psi = critic.Psi(s, a) - critic.Psi(s, a_pi)
        psi_error = predicted_psi - actual_advantage

        # Gradient of Psi(s,a) - Psi(s, a_pi) w.r.t. Zb, Zc
        # Psi(s, a) = s^T Zb a + Zc a
        # d/dZb = s @ (a - a_pi)^T
        # d/dZc = (a - a_pi)^T
        a_diff = a - a_pi

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
