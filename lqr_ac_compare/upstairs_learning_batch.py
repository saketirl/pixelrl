"""
Batch multi-step upstairs actor-critic update with deep linear networks (L=3).

Implements CT-DDPG Martingale Loss (from arXiv 2509.23711):
  ℒᴹ = (V(o_t) - ∑_{l=0}^{L-1} γ^l [r_{t+l} - q(o_{t+l}, a_{t+l})] dt - γ^L V(o_{t+L}))²

The key is subtracting the advantage rate q from rewards in the integral.
This maintains the martingale property and keeps variance bounded as dt→0.

Alternative: Generator-based update (use_generator=True):
  δ_t = r_t - ψ(o_t, a_t) + L^π V(o_t) - β V(o_t)
  where L^π V(o_t) ≈ (V(o_{t+1}) - V(o_t)) / dt (sample-based estimate)

Architecture (L=3 deep linear networks):
-----------------------------------------
Actor:  a = W3 @ W2 @ W1 @ o
  - Trainable: W1, W2 (d×d orthogonal)
  - Frozen: W3 (da×d)

Critic Value: φ(o) = o^T (U3 U2 U1) o + (Up3 Up2 Up1) o + c
  - Trainable: U1, U2, U3, Up1, Up2 (all d×d orthogonal), c (scalar)
  - Frozen: Up3 (1×d)

Critic Advantage: Ψ(o, a) = a^T (Z3 Z2 Z1) o + (Zp3 Zp2 Zp1) a
  - Trainable: Z1, Z2, Zp2 (d×d orthogonal)
  - Frozen: Z3 (da×d), Zp1 (d×da), Zp3 (1×d)
"""

import numpy as np
from typing import Dict

Array = np.ndarray


def cayley_retract(X: Array, G: Array, eta: float) -> Array:
    """Cayley retraction for orthogonal matrices: X^T X = I."""
    n = X.shape[0]
    I = np.eye(n, dtype=X.dtype)
    A = G @ X.T - X @ G.T  # skew-symmetric
    return np.linalg.solve(I + 0.5 * eta * A, (I - 0.5 * eta * A) @ X)


def update_matrix(X: Array, G: Array, eta: float, use_cayley: bool) -> Array:
    """Update matrix using either Cayley retraction or standard gradient descent."""
    if use_cayley:
        return cayley_retract(X, G, eta)
    else:
        return X + eta * G


# ============================================================
# Gradient computations for L=3 deep linear networks
# ============================================================

def grad_phi_U(o: Array, U1: Array, U2: Array, U3: Array) -> Dict[str, Array]:
    """
    Gradient of φ_quad(o) = o^T (U3 U2 U1) o w.r.t. U1, U2, U3.

    Chain rule:
      dφ/dU1 = (U2^T U3^T o) @ o^T
      dφ/dU2 = (U3^T o) @ (U1 o)^T
      dφ/dU3 = o @ (U2 U1 o)^T
    """
    o = o.reshape(-1)
    # Forward pass
    v1 = U1 @ o           # (d,)
    v2 = U2 @ v1          # (d,)
    # Backward pass
    u3_o = U3.T @ o       # (d,)
    u2_u3_o = U2.T @ u3_o # (d,)

    return {
        "U1": np.outer(u2_u3_o, o),
        "U2": np.outer(u3_o, v1),
        "U3": np.outer(o, v2),
    }


def grad_phi_Up(o: Array, Up1: Array, Up2: Array, Up3: Array) -> Dict[str, Array]:
    """
    Gradient of φ_lin(o) = (Up3 Up2 Up1) o w.r.t. Up1, Up2.
    Up3 is frozen (1×d).
    """
    o = o.reshape(-1)
    # Backward: Up3^T is (d, 1)
    u = (Up2.T @ Up3.T).reshape(-1)  # (d,)

    # Forward
    x1 = Up1 @ o  # (d,)

    return {
        "Up1": np.outer(u, o),
        "Up2": Up3.T @ x1.reshape(1, -1),  # (d, d)
    }


def grad_Psi_Z(o: Array, a: Array, Z1: Array, Z2: Array, Z3: Array) -> Dict[str, Array]:
    """
    Gradient of Ψ_bilin(o, a) = a^T (Z3 Z2 Z1) o w.r.t. Z1, Z2.
    Z3 is frozen.
    """
    o = o.reshape(-1)
    a = a.reshape(-1)

    # Forward
    z1_o = Z1 @ o  # (d,)

    # Backward
    z3_a = Z3.T @ a       # (d,)
    z2_z3_a = Z2.T @ z3_a # (d,)

    return {
        "Z1": np.outer(z2_z3_a, o),
        "Z2": np.outer(z3_a, z1_o),
    }


def grad_Psi_Zp2(a: Array, Zp1: Array, Zp2: Array, Zp3: Array) -> Array:
    """
    Gradient of Ψ_lin(a) = (Zp3 Zp2 Zp1) a w.r.t. Zp2.
    Zp1, Zp3 are frozen.
    """
    a = a.reshape(-1)
    x1 = Zp1 @ a  # (d,)
    # dΨ/dZp2 = Zp3^T @ x1^T = outer(Zp3^T, x1)
    return Zp3.T @ x1.reshape(1, -1)


def grad_a_Psi(o: Array, Z1: Array, Z2: Array, Z3: Array,
               Zp1: Array, Zp2: Array, Zp3: Array) -> Array:
    """
    Gradient of Ψ(o, a) w.r.t. a:
      ∂Ψ/∂a = (Z3 Z2 Z1) o + (Zp3 Zp2 Zp1)^T
            = Zeff @ o + Zp1^T @ Zp2^T @ Zp3^T
    """
    o = o.reshape(-1)
    Zeff = Z3 @ Z2 @ Z1  # (da, d)
    term1 = (Zeff @ o).reshape(-1)
    term2 = (Zp1.T @ Zp2.T @ Zp3.T).reshape(-1)
    return term1 + term2


def grad_actor_W(o: Array, g_a: Array, W1: Array, W2: Array, W3: Array) -> Dict[str, Array]:
    """
    Gradient of J = g_a^T @ a = g_a^T @ (W3 W2 W1 o) w.r.t. W1, W2.
    W3 is frozen.

    Chain rule:
      dJ/dW1 = (W2^T W3^T g_a) @ o^T
      dJ/dW2 = (W3^T g_a) @ (W1 o)^T
    """
    o = o.reshape(-1)
    g_a = g_a.reshape(-1)

    # Forward
    x1 = W1 @ o  # (d,)

    # Backward
    w3_g = W3.T @ g_a       # (d,)
    w2_w3_g = W2.T @ w3_g   # (d,)

    return {
        "W1": np.outer(w2_w3_g, o),
        "W2": np.outer(w3_g, x1),
    }


# ============================================================
# Batch multi-step update
# ============================================================

def upstairs_batch_multistep_update(
    actor,
    critic,
    o_traj: Array,   # (N+1, d) observations
    a_traj: Array,   # (N, da) actions
    r_traj: Array,   # (N,) rewards
    *,
    beta: float,
    dt: float,
    L: int = 10,     # multi-step horizon
    eta_actor: float,
    eta_critic: float,
    use_generator: bool = False,  # Use generator-based TD error instead of multi-step
    use_cayley: bool = True,  # Use Cayley retraction (False = standard gradient descent)
) -> Dict[str, float]:
    """
    Batch multi-step actor-critic update for upstairs L=3 deep linear networks.

    Uses L-step returns to avoid 1/dt variance blow-up.

    If use_generator=True, uses generator-based TD error instead:
        δ_t = r_t - ψ(o_t, a_t) + (V(o_{t+1}) - V(o_t))/dt - β V(o_t)
    """
    N = len(r_traj)
    d = o_traj.shape[1]
    da = a_traj.shape[1]

    gamma = np.exp(-beta * dt)

    # Compute all values
    values = np.array([critic.phi_upstairs(o_traj[i]) for i in range(N + 1)])

    # ========== GENERATOR-BASED UPDATE ==========
    if use_generator:
        weights = np.array([gamma ** t for t in range(N)])

        gU1 = np.zeros_like(critic.U1)
        gU2 = np.zeros_like(critic.U2)
        gU3 = np.zeros_like(critic.U3)
        gUp1 = np.zeros_like(critic.Up1)
        gUp2 = np.zeros_like(critic.Up2)
        gc = 0.0
        gZ1 = np.zeros_like(critic.Z1)
        gZ2 = np.zeros_like(critic.Z2)
        gZp2 = np.zeros_like(critic.Zp2)
        gW1 = np.zeros_like(actor.W1)
        gW2 = np.zeros_like(actor.W2)

        mse = 0.0
        for t in range(N):
            o = o_traj[t]
            a = a_traj[t]
            r = r_traj[t]
            a_pi = actor.act_upstairs(o)

            V_t = values[t]
            V_next = values[t + 1]

            # Sample-based generator estimate
            L_V_est = (V_next - V_t) / dt

            # Advantage rate
            psi = critic.psi_upstairs(o, a, actor)

            # TD error: δ = r - ψ + L^π V - β V
            delta = r - psi + L_V_est - beta * V_t
            mse += delta ** 2

            w = weights[t] * dt

            # Critic value gradients
            dU = grad_phi_U(o, critic.U1, critic.U2, critic.U3)
            dUp = grad_phi_Up(o, critic.Up1, critic.Up2, critic.Up3)
            gU1 -= w * delta * dU["U1"]
            gU2 -= w * delta * dU["U2"]
            gU3 -= w * delta * dU["U3"]
            gUp1 -= w * delta * dUp["Up1"]
            gUp2 -= w * delta * dUp["Up2"]
            gc -= w * delta

            # Critic advantage gradients
            dZ_a = grad_Psi_Z(o, a, critic.Z1, critic.Z2, critic.Z3)
            dZ_pi = grad_Psi_Z(o, a_pi, critic.Z1, critic.Z2, critic.Z3)
            dZp2_a = grad_Psi_Zp2(a, critic.Zp1, critic.Zp2, critic.Zp3)
            dZp2_pi = grad_Psi_Zp2(a_pi, critic.Zp1, critic.Zp2, critic.Zp3)
            gZ1 += w * delta * (dZ_a["Z1"] - dZ_pi["Z1"])
            gZ2 += w * delta * (dZ_a["Z2"] - dZ_pi["Z2"])
            gZp2 += w * delta * (dZp2_a - dZp2_pi)

            # Actor gradient (DPG)
            g_a = grad_a_Psi(o, critic.Z1, critic.Z2, critic.Z3,
                             critic.Zp1, critic.Zp2, critic.Zp3)
            dW = grad_actor_W(o, g_a, actor.W1, actor.W2, actor.W3)
            gW1 += w * dW["W1"]
            gW2 += w * dW["W2"]

        # Normalize
        total_weight = float(np.sum(weights * dt))
        gU1 /= total_weight
        gU2 /= total_weight
        gU3 /= total_weight
        gUp1 /= total_weight
        gUp2 /= total_weight
        gc /= total_weight
        gZ1 /= total_weight
        gZ2 /= total_weight
        gZp2 /= total_weight
        gW1 /= total_weight
        gW2 /= total_weight
        mse /= N

        # Apply updates
        critic.U1 = update_matrix(critic.U1, gU1, eta_critic, use_cayley)
        critic.U2 = update_matrix(critic.U2, gU2, eta_critic, use_cayley)
        critic.U3 = update_matrix(critic.U3, gU3, eta_critic, use_cayley)
        critic.Up1 = update_matrix(critic.Up1, gUp1, eta_critic, use_cayley)
        critic.Up2 = update_matrix(critic.Up2, gUp2, eta_critic, use_cayley)
        critic.c = critic.c + eta_critic * gc
        critic.Z1 = update_matrix(critic.Z1, gZ1, eta_critic, use_cayley)
        critic.Z2 = update_matrix(critic.Z2, gZ2, eta_critic, use_cayley)
        critic.Zp2 = update_matrix(critic.Zp2, gZp2, eta_critic, use_cayley)
        actor.W1 = update_matrix(actor.W1, gW1, eta_actor, use_cayley)
        actor.W2 = update_matrix(actor.W2, gW2, eta_actor, use_cayley)

        return {
            "mse": float(mse),
            "ortho_err_W1": float(np.linalg.norm(actor.W1.T @ actor.W1 - np.eye(d))),
            "ortho_err_U1": float(np.linalg.norm(critic.U1.T @ critic.U1 - np.eye(d))),
            "G_W1_norm": float(np.linalg.norm(gW1)),
            "G_U1_norm": float(np.linalg.norm(gU1)),
        }

    # ========== MULTI-STEP UPDATE (DEFAULT) ==========
    if N < L:
        L = N

    # Compute multi-step TD targets with CT-DDPG [r - q] formulation
    # G_t = sum_{l=0}^{L-1} gamma^l [r_{t+l} - q(o_{t+l}, a_{t+l})] * dt + gamma^L V(o_{t+L})
    n_samples = N - L + 1
    targets = np.zeros(n_samples)
    for t in range(n_samples):
        G = 0.0
        for l in range(L):
            o_l = o_traj[t + l]
            a_l = a_traj[t + l]
            # q(o, a) is the advantage rate at the trajectory action
            q_l = critic.psi_upstairs(o_l, a_l, actor)
            G += (gamma ** l) * (r_traj[t + l] - q_l) * dt
        G += (gamma ** L) * values[t + L]
        targets[t] = G

    # ==================== Critic Update ====================
    # Gradient accumulators for value function
    gU1 = np.zeros_like(critic.U1)
    gU2 = np.zeros_like(critic.U2)
    gU3 = np.zeros_like(critic.U3)
    gUp1 = np.zeros_like(critic.Up1)
    gUp2 = np.zeros_like(critic.Up2)
    gc = 0.0

    # Gradient accumulators for advantage function
    gZ1 = np.zeros_like(critic.Z1)
    gZ2 = np.zeros_like(critic.Z2)
    gZp2 = np.zeros_like(critic.Zp2)

    mse = 0.0

    for t in range(n_samples):
        o = o_traj[t]
        V = values[t]
        target = targets[t]
        td_error = V - target
        mse += td_error ** 2

        # Value function gradients (minimize MSE)
        dU = grad_phi_U(o, critic.U1, critic.U2, critic.U3)
        dUp = grad_phi_Up(o, critic.Up1, critic.Up2, critic.Up3)

        gU1 -= td_error * dU["U1"]
        gU2 -= td_error * dU["U2"]
        gU3 -= td_error * dU["U3"]
        gUp1 -= td_error * dUp["Up1"]
        gUp2 -= td_error * dUp["Up2"]
        gc -= td_error

        # Advantage function gradients
        a = a_traj[t]
        a_pi = actor.act_upstairs(o)
        actual_advantage = target - V

        # Predicted advantage
        pred_psi = critic.psi_upstairs(o, a, actor)
        psi_error = pred_psi - actual_advantage

        # Gradients of Psi(o, a) - Psi(o, a_pi)
        dZ_a = grad_Psi_Z(o, a, critic.Z1, critic.Z2, critic.Z3)
        dZ_pi = grad_Psi_Z(o, a_pi, critic.Z1, critic.Z2, critic.Z3)
        dZp2_a = grad_Psi_Zp2(a, critic.Zp1, critic.Zp2, critic.Zp3)
        dZp2_pi = grad_Psi_Zp2(a_pi, critic.Zp1, critic.Zp2, critic.Zp3)

        gZ1 -= psi_error * (dZ_a["Z1"] - dZ_pi["Z1"])
        gZ2 -= psi_error * (dZ_a["Z2"] - dZ_pi["Z2"])
        gZp2 -= psi_error * (dZp2_a - dZp2_pi)

    # Normalize
    gU1 /= n_samples
    gU2 /= n_samples
    gU3 /= n_samples
    gUp1 /= n_samples
    gUp2 /= n_samples
    gc /= n_samples
    gZ1 /= n_samples
    gZ2 /= n_samples
    gZp2 /= n_samples
    mse /= n_samples

    # Apply critic updates
    critic.U1 = update_matrix(critic.U1, gU1, eta_critic, use_cayley)
    critic.U2 = update_matrix(critic.U2, gU2, eta_critic, use_cayley)
    critic.U3 = update_matrix(critic.U3, gU3, eta_critic, use_cayley)
    critic.Up1 = update_matrix(critic.Up1, gUp1, eta_critic, use_cayley)
    critic.Up2 = update_matrix(critic.Up2, gUp2, eta_critic, use_cayley)
    critic.c = critic.c + eta_critic * gc

    critic.Z1 = update_matrix(critic.Z1, gZ1, eta_critic, use_cayley)
    critic.Z2 = update_matrix(critic.Z2, gZ2, eta_critic, use_cayley)
    critic.Zp2 = update_matrix(critic.Zp2, gZp2, eta_critic, use_cayley)

    # ==================== Actor Update ====================
    # DPG: maximize E[Ψ(o, π(o))] by following ∂Ψ/∂a
    gW1 = np.zeros_like(actor.W1)
    gW2 = np.zeros_like(actor.W2)

    for t in range(n_samples):
        o = o_traj[t]

        # Advantage gradient at current policy
        g_a = grad_a_Psi(o, critic.Z1, critic.Z2, critic.Z3,
                         critic.Zp1, critic.Zp2, critic.Zp3)

        # Actor gradient via chain rule
        dW = grad_actor_W(o, g_a, actor.W1, actor.W2, actor.W3)
        gW1 += dW["W1"]
        gW2 += dW["W2"]

    gW1 /= n_samples
    gW2 /= n_samples

    # Apply actor updates (gradient ASCENT for maximizing return)
    actor.W1 = update_matrix(actor.W1, gW1, eta_actor, use_cayley)
    actor.W2 = update_matrix(actor.W2, gW2, eta_actor, use_cayley)

    # Diagnostics
    ortho_W1 = float(np.linalg.norm(actor.W1.T @ actor.W1 - np.eye(d)))
    ortho_U1 = float(np.linalg.norm(critic.U1.T @ critic.U1 - np.eye(d)))

    return {
        "mse": float(mse),
        "ortho_err_W1": ortho_W1,
        "ortho_err_U1": ortho_U1,
        "G_W1_norm": float(np.linalg.norm(gW1)),
        "G_U1_norm": float(np.linalg.norm(gU1)),
    }
