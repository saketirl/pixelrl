
import numpy as np
from typing import Dict

Array = np.ndarray


def cayley_retract_stiefel(X: Array, G: Array, eta: float) -> Array:
    """
    Cayley retraction for Stiefel X ∈ R^{n×p} with X^T X = I_p.
      A = G X^T - X G^T ∈ R^{n×n} skew-symmetric
      X+ = (I + (eta/2)A)^{-1} (I - (eta/2)A) X
    """
    n, _ = X.shape
    I = np.eye(n, dtype=X.dtype)
    A = G @ X.T - X @ G.T
    B = (I - 0.5 * eta * A) @ X
    return np.linalg.solve(I + 0.5 * eta * A, B)


def td_error(phi_t: float, phi_tp1: float, r_t: float, psi_t: float, beta: float, dt: float) -> float:
    return float(r_t - psi_t + (np.exp(-beta * dt) * phi_tp1 - phi_t) / dt)


def downstairs_actor_critic_update_T2(
    actor,
    critic,
    s_traj: Array,   # (N+1, ds)
    a_traj: Array,   # (N, da)
    r_traj: Array,   # (N,)
    *,
    beta: float,
    dt: float,
    L_eff: int,
    eta_actor: float,
    eta_value: float,
    eta_adv: float,
) -> Dict[str, float]:
    """
    Downstairs shallow update using the Theorem-2-style Stiefel-direction coefficients:
      Ub   update direction scaled by L_eff
      Uc   update direction scaled by (L_eff-1)
      Zb   update direction scaled by (L_eff-1)
      Wc   update direction scaled by (L_eff-1)

    Zc_row is unconstrained (plain step).
    """
    N = a_traj.shape[0]
    ds, da = s_traj.shape[1], a_traj.shape[1]

    G_Ub = np.zeros_like(critic.Ub)
    G_Uc = np.zeros_like(critic.Uc_col)
    G_c = 0.0
    G_Zb = np.zeros_like(critic.Zb)
    G_Zc = np.zeros_like(critic.Zc_row)
    G_Wc = np.zeros_like(actor.Wc_col)

    td2 = 0.0

    for k in range(N):
        t = k * dt
        w = float(np.exp(-beta * t))

        s_t = s_traj[k].reshape(ds,)
        s_tp1 = s_traj[k + 1].reshape(ds,)
        a_t = a_traj[k].reshape(da,)
        r_t = float(r_traj[k])

        phi_t = float(critic.phi(s_t))
        phi_tp1 = float(critic.phi(s_tp1))
        psi_t = float(critic.psi(s_t, a_t, actor))
        delta = td_error(phi_t, phi_tp1, r_t, psi_t, beta=beta, dt=dt)
        td2 += delta * delta

        a_pi = actor.act(s_t).reshape(da,)
        a_bar = (a_t - a_pi).reshape(da,)

        # critic directions
        G_Ub += (w * delta * dt) * (s_t.reshape(ds, 1) @ s_t.reshape(1, ds))
        G_Uc += (w * delta * dt) * s_t.reshape(ds, 1)
        G_c += (w * delta * dt)

        G_Zb += (w * delta * dt) * (s_t.reshape(ds, 1) @ a_bar.reshape(1, da))
        G_Zc += (w * delta * dt) * a_bar.reshape(1, da)

        # actor direction: g_t = ∂_a Ψ|_{a=π(s)} = Zb^T s + Zc^T
        g_t = (critic.Zb.T @ s_t.reshape(ds, 1)).reshape(da,) + critic.Zc_row.reshape(da,)
        G_Wc += (w * dt) * (s_t.reshape(ds, 1) @ g_t.reshape(1, da))

    # Apply updates with depth coefficients (L_eff=3 to match upstairs depth)
    critic.Ub = cayley_retract_stiefel(critic.Ub, (L_eff * G_Ub), eta_value)
    critic.Uc_col = cayley_retract_stiefel(critic.Uc_col, ((L_eff - 1) * G_Uc), eta_value)
    critic.c = float(critic.c + eta_value * G_c)

    critic.Zb = cayley_retract_stiefel(critic.Zb, ((L_eff - 1) * G_Zb), eta_adv)
    critic.Zc_row = critic.Zc_row + eta_adv * (max(L_eff - 2, 0) * G_Zc)

    actor.Wc_col = cayley_retract_stiefel(actor.Wc_col, ((L_eff - 1) * G_Wc), eta_actor)

    return {
        "td_mse": float(td2 / max(N, 1)),
        "ortho_err_Wc": float(np.linalg.norm(actor.Wc_col.T @ actor.Wc_col - np.eye(da))),
        "ortho_err_Zb": float(np.linalg.norm(critic.Zb.T @ critic.Zb - np.eye(da))),
    }
