
import numpy as np
from typing import Dict, Optional, Sequence, Tuple

Array = np.ndarray


# -------------------------
# Cayley retraction (square orthogonal)
# -------------------------

def cayley_update_square(X: Array, direction: Array, eta: float) -> Array:
    """
    Square Cayley retraction that preserves X^T X = I exactly (numerically stable for small eta).
      A = D X^T - X D^T   (skew-symmetric)
      X+ = (I + (eta/2)A)^{-1} (I - (eta/2)A) X
    """
    d = X.shape[0]
    I = np.eye(d, dtype=X.dtype)
    A = direction @ X.T - X @ direction.T
    B = (I - 0.5 * eta * A) @ X
    return np.linalg.solve(I + 0.5 * eta * A, B)


# -------------------------
# TD error proxy (continuous-time -> discrete)
# -------------------------

def td_error(phi_t: float, phi_tp1: float, r_t: float, psi_t: float, beta: float, dt: float) -> float:
    return float(r_t - psi_t + (np.exp(-beta * dt) * phi_tp1 - phi_t) / dt)


# -------------------------
# Closed-form partials for L=3 models
# -------------------------

def dphi_dU_L3(o: Array, U1: Array, U2: Array, U3: Array) -> Dict[str, Array]:
    o = o.reshape(-1)
    u3T_o = U3.T @ o
    u2T_u3T_o = U2.T @ u3T_o

    v1 = U1 @ o
    v2 = U2 @ v1

    return {
        "U1": np.outer(u2T_u3T_o, o),
        "U2": np.outer(u3T_o, v1),
        "U3": np.outer(o, v2),
    }

def dphi_dUp_L3(o: Array, Up1: Array, Up2: Array, Up3: Array) -> Dict[str, Array]:
    o = o.reshape(-1)

    u = Up2.T @ Up3.T  # (d,1)
    gUp1 = u @ o.reshape(1, -1)

    x2 = Up1 @ o
    gUp2 = Up3.T @ x2.reshape(1, -1)

    return {"Up1": gUp1, "Up2": gUp2}

def dPsi_dZ_L3(o: Array, a: Array, Z1: Array, Z2: Array, Z3: Array) -> Dict[str, Array]:
    o = o.reshape(-1)
    a = a.reshape(-1)

    z3T_a = Z3.T @ a
    z2T_z3T_a = Z2.T @ z3T_a
    z1o = Z1 @ o

    return {
        "Z1": np.outer(z2T_z3T_a, o),
        "Z2": np.outer(z3T_a, z1o),
    }

def dPsi_dZp2_L3(a: Array, Zp1: Array, Zp3: Array) -> Array:
    a = a.reshape(-1)
    u = Zp3.T  # (d,1)
    x = (Zp1 @ a).reshape(-1)  # (d,)
    return u @ x.reshape(1, -1)

def grad_a_Psi(o: Array, Zeff: Array, Zp1: Array, Zp2: Array, Zp3: Array) -> Array:
    """
    ∂_a Ψ(o,a) = Zeff o + (Zp3 Zp2 Zp1)^T
    """
    o = o.reshape(-1)
    term1 = (Zeff @ o).reshape(-1)  # (da,)
    term2 = (Zp1.T @ (Zp2.T @ Zp3.T)).reshape(-1)
    return term1 + term2


# -------------------------
# Upstairs update with subsampled timesteps (minibatch)
# -------------------------

def upstairs_actor_critic_update_L3(
    actor,
    critic,
    o_traj: Array,  # (N+1, d)
    a_traj: Array,  # (N, da)
    r_traj: Array,  # (N,)
    *,
    beta: float,
    dt: float,
    eta_actor: float,
    eta_value: float,
    eta_adv: float,
    batch_size: int = 32,
    batch_indices: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Implements upstairs actor-critic update for L=3 using Cayley retraction.
    To keep runtime reasonable for d=512, gradients are estimated using a minibatch
    of timesteps from the trajectory.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = a_traj.shape[0]
    d = o_traj.shape[1]

    if batch_indices is None:
        bs = min(batch_size, N)
        idx = rng.choice(N, size=bs, replace=False)
    else:
        idx = np.array(list(batch_indices), dtype=int)

    # scale factor to approximate full integral sum
    scale_full = float(N / max(len(idx), 1))

    # accumulators
    gW1 = np.zeros_like(actor.W1)
    gW2 = np.zeros_like(actor.W2)

    gU1 = np.zeros_like(critic.U1)
    gU2 = np.zeros_like(critic.U2)
    gU3 = np.zeros_like(critic.U3)
    gUp1 = np.zeros_like(critic.Up1)
    gUp2 = np.zeros_like(critic.Up2)
    gc = 0.0

    gZ1 = np.zeros_like(critic.Z1)
    gZ2 = np.zeros_like(critic.Z2)
    gZp2 = np.zeros_like(critic.Zp2)

    td2 = 0.0

    for k in idx:
        t = k * dt
        w = float(np.exp(-beta * t))

        o_t = o_traj[k].reshape(d,)
        o_tp1 = o_traj[k + 1].reshape(d,)
        a_t = a_traj[k].reshape(-1)
        r_t = float(r_traj[k])

        phi_t = float(critic.phi_upstairs(o_t))
        phi_tp1 = float(critic.phi_upstairs(o_tp1))
        psi_t = float(critic.psi_upstairs(o_t, a_t, actor))
        delta = td_error(phi_t, phi_tp1, r_t, psi_t, beta=beta, dt=dt)
        td2 += delta * delta

        # value semigrads
        dphiU = dphi_dU_L3(o_t, critic.U1, critic.U2, critic.U3)
        dphiUp = dphi_dUp_L3(o_t, critic.Up1, critic.Up2, critic.Up3)

        sU = scale_full * (w * delta * dt)
        gU1 += sU * dphiU["U1"]
        gU2 += sU * dphiU["U2"]
        gU3 += sU * dphiU["U3"]
        gUp1 += sU * dphiUp["Up1"]
        gUp2 += sU * dphiUp["Up2"]
        gc += sU

        # advantage semigrads (note e^{-2βt} weight)
        a_pi = actor.act_upstairs(o_t)

        dPsi_a = dPsi_dZ_L3(o_t, a_t, critic.Z1, critic.Z2, critic.Z3)
        dPsi_pi = dPsi_dZ_L3(o_t, a_pi, critic.Z1, critic.Z2, critic.Z3)
        dpsiZ1 = dPsi_a["Z1"] - dPsi_pi["Z1"]
        dpsiZ2 = dPsi_a["Z2"] - dPsi_pi["Z2"]

        dZp2_a = dPsi_dZp2_L3(a_t, critic.Zp1, critic.Zp3)
        dZp2_pi = dPsi_dZp2_L3(a_pi, critic.Zp1, critic.Zp3)
        dpsiZp2 = dZp2_a - dZp2_pi

        sZ = scale_full * ((w * w) * delta * dt)
        gZ1 += sZ * dpsiZ1
        gZ2 += sZ * dpsiZ2
        gZp2 += sZ * dpsiZp2

        # actor policy gradient
        g_a = grad_a_Psi(o_t, critic.Zeff(), critic.Zp1, critic.Zp2, critic.Zp3)  # (da,)

        gpre = (actor.W3.T @ g_a).reshape(-1)      # (d,)
        x1 = (actor.W1 @ o_t).reshape(-1)          # (d,)
        gW2 += scale_full * (w * dt) * np.outer(gpre, x1)
        gW1 += scale_full * (w * dt) * np.outer(actor.W2.T @ gpre, o_t)

    # Apply updates
    # Actor ascent
    actor.W1 = cayley_update_square(actor.W1, +gW1, eta_actor)
    actor.W2 = cayley_update_square(actor.W2, +gW2, eta_actor)

    # Critic descent on squared-TD objective (semi-gradient)
    critic.U1 = cayley_update_square(critic.U1, -gU1, eta_value)
    critic.U2 = cayley_update_square(critic.U2, -gU2, eta_value)
    critic.U3 = cayley_update_square(critic.U3, -gU3, eta_value)
    critic.Up1 = cayley_update_square(critic.Up1, -gUp1, eta_value)
    critic.Up2 = cayley_update_square(critic.Up2, -gUp2, eta_value)
    critic.c = float(critic.c - eta_value * gc)

    critic.Z1 = cayley_update_square(critic.Z1, -gZ1, eta_adv)
    critic.Z2 = cayley_update_square(critic.Z2, -gZ2, eta_adv)
    critic.Zp2 = cayley_update_square(critic.Zp2, -gZp2, eta_adv)

    # Diagnostics
    ortho_W1 = float(np.linalg.norm(actor.W1.T @ actor.W1 - np.eye(d)))
    ortho_U1 = float(np.linalg.norm(critic.U1.T @ critic.U1 - np.eye(d)))
    return {
        "td_mse_batch": float(td2 / max(len(idx), 1)),
        "ortho_err_W1": ortho_W1,
        "ortho_err_U1": ortho_U1,
        "batch_size": float(len(idx)),
    }
