
"""JAX learning functions for upstairs and downstairs actor-critic."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Tuple
from functools import partial

from models_jax import (
    ActorUpParams, CriticUpParams, ActorDnParams, CriticDnParams,
    actor_up_act, actor_up_effective_matrix,
    critic_up_phi, critic_up_psi, critic_up_Zeff,
    actor_dn_act, critic_dn_phi, critic_dn_psi,
)

Array = jnp.ndarray


# -------------------------
# Cayley retraction
# -------------------------

@jax.jit
def cayley_update_square(X: Array, direction: Array, eta: float) -> Array:
    """
    Square Cayley retraction that preserves X^T X = I.
      A = D X^T - X D^T   (skew-symmetric)
      X+ = (I + (eta/2)A)^{-1} (I - (eta/2)A) X
    """
    d = X.shape[0]
    I = jnp.eye(d)
    A = direction @ X.T - X @ direction.T
    B = (I - 0.5 * eta * A) @ X
    return jnp.linalg.solve(I + 0.5 * eta * A, B)


@jax.jit
def cayley_retract_stiefel(X: Array, G: Array, eta: float) -> Array:
    """Cayley retraction for Stiefel X in R^{n x p}."""
    n, _ = X.shape
    I = jnp.eye(n)
    A = G @ X.T - X @ G.T
    B = (I - 0.5 * eta * A) @ X
    return jnp.linalg.solve(I + 0.5 * eta * A, B)


# -------------------------
# TD error
# -------------------------

@jax.jit
def td_error(phi_t: float, phi_tp1: float, r_t: float, psi_t: float,
             beta: float, dt: float) -> float:
    return r_t - psi_t + (jnp.exp(-beta * dt) * phi_tp1 - phi_t) / dt


# -------------------------
# Upstairs gradient computation (vectorized)
# -------------------------

@partial(jax.jit, static_argnums=(7, 8))
def upstairs_compute_grads(
    actor_params: ActorUpParams,
    critic_params: CriticUpParams,
    o_traj: Array,  # (N+1, d)
    a_traj: Array,  # (N, da)
    r_traj: Array,  # (N,)
    beta: float,
    dt: float,
    N: int,
    d: int,
) -> Tuple[Dict[str, Array], float]:
    """Compute gradients for upstairs actor-critic update."""

    def step_grads(k):
        t = k * dt
        w = jnp.exp(-beta * t)

        o_t = o_traj[k]
        o_tp1 = o_traj[k + 1]
        a_t = a_traj[k]
        r_t = r_traj[k]

        phi_t = critic_up_phi(critic_params, o_t)
        phi_tp1 = critic_up_phi(critic_params, o_tp1)
        psi_t = critic_up_psi(critic_params, o_t, a_t, actor_params)
        delta = td_error(phi_t, phi_tp1, r_t, psi_t, beta, dt)

        # Value gradients (dphi/dU)
        u3T_o = critic_params.U3.T @ o_t
        u2T_u3T_o = critic_params.U2.T @ u3T_o
        v1 = critic_params.U1 @ o_t
        v2 = critic_params.U2 @ v1

        gU1 = jnp.outer(u2T_u3T_o, o_t)
        gU2 = jnp.outer(u3T_o, v1)
        gU3 = jnp.outer(o_t, v2)

        # dphi/dUp
        u = critic_params.Up2.T @ critic_params.Up3.T
        gUp1 = u @ o_t.reshape(1, -1)
        x2 = critic_params.Up1 @ o_t
        gUp2 = critic_params.Up3.T @ x2.reshape(1, -1)

        # Advantage gradients
        a_pi = actor_up_act(actor_params, o_t)
        a_bar = a_t - a_pi

        z3T_a = critic_params.Z3.T @ a_t
        z2T_z3T_a = critic_params.Z2.T @ z3T_a
        z1o = critic_params.Z1 @ o_t

        z3T_api = critic_params.Z3.T @ a_pi
        z2T_z3T_api = critic_params.Z2.T @ z3T_api

        gZ1 = jnp.outer(z2T_z3T_a, o_t) - jnp.outer(z2T_z3T_api, o_t)
        gZ2 = jnp.outer(z3T_a, z1o) - jnp.outer(z3T_api, z1o)

        # dPsi/dZp2
        u_zp = critic_params.Zp3.T
        x_a = critic_params.Zp1 @ a_t
        x_api = critic_params.Zp1 @ a_pi
        gZp2 = u_zp @ x_a.reshape(1, -1) - u_zp @ x_api.reshape(1, -1)

        # Actor gradient
        Zeff = critic_up_Zeff(critic_params)
        g_a = (Zeff @ o_t) + (critic_params.Zp1.T @ (critic_params.Zp2.T @ critic_params.Zp3.T)).squeeze()

        gpre = actor_params.W3.T @ g_a
        x1 = actor_params.W1 @ o_t
        gW2 = jnp.outer(gpre, x1)
        gW1 = jnp.outer(actor_params.W2.T @ gpre, o_t)

        sU = w * delta * dt
        sZ = (w * w) * delta * dt
        sW = w * dt

        return (
            sU * gU1, sU * gU2, sU * gU3,
            sU * gUp1, sU * gUp2, sU,
            sZ * gZ1, sZ * gZ2, sZ * gZp2,
            sW * gW1, sW * gW2,
            delta * delta
        )

    # Vectorize over all timesteps
    results = jax.vmap(step_grads)(jnp.arange(N))

    # Sum gradients
    gU1 = results[0].sum(axis=0)
    gU2 = results[1].sum(axis=0)
    gU3 = results[2].sum(axis=0)
    gUp1 = results[3].sum(axis=0)
    gUp2 = results[4].sum(axis=0)
    gc = results[5].sum()
    gZ1 = results[6].sum(axis=0)
    gZ2 = results[7].sum(axis=0)
    gZp2 = results[8].sum(axis=0)
    gW1 = results[9].sum(axis=0)
    gW2 = results[10].sum(axis=0)
    td2 = results[11].sum()

    grads = {
        "gU1": gU1, "gU2": gU2, "gU3": gU3,
        "gUp1": gUp1, "gUp2": gUp2, "gc": gc,
        "gZ1": gZ1, "gZ2": gZ2, "gZp2": gZp2,
        "gW1": gW1, "gW2": gW2,
    }

    return grads, td2


def upstairs_actor_critic_update(
    actor_params: ActorUpParams,
    critic_params: CriticUpParams,
    o_traj: Array,
    a_traj: Array,
    r_traj: Array,
    *,
    beta: float,
    dt: float,
    eta_actor: float,
    eta_value: float,
    eta_adv: float,
) -> Tuple[ActorUpParams, CriticUpParams, Dict[str, float]]:
    """Full upstairs actor-critic update."""
    N = a_traj.shape[0]
    d = o_traj.shape[1]

    grads, td2 = upstairs_compute_grads(
        actor_params, critic_params,
        o_traj, a_traj, r_traj,
        beta, dt, N, d
    )

    # Apply Cayley updates
    new_W1 = cayley_update_square(actor_params.W1, grads["gW1"], eta_actor)
    new_W2 = cayley_update_square(actor_params.W2, grads["gW2"], eta_actor)

    new_U1 = cayley_update_square(critic_params.U1, -grads["gU1"], eta_value)
    new_U2 = cayley_update_square(critic_params.U2, -grads["gU2"], eta_value)
    new_U3 = cayley_update_square(critic_params.U3, -grads["gU3"], eta_value)
    new_Up1 = cayley_update_square(critic_params.Up1, -grads["gUp1"], eta_value)
    new_Up2 = cayley_update_square(critic_params.Up2, -grads["gUp2"], eta_value)
    new_c = critic_params.c - eta_value * grads["gc"]

    new_Z1 = cayley_update_square(critic_params.Z1, -grads["gZ1"], eta_adv)
    new_Z2 = cayley_update_square(critic_params.Z2, -grads["gZ2"], eta_adv)
    new_Zp2 = cayley_update_square(critic_params.Zp2, -grads["gZp2"], eta_adv)

    new_actor = ActorUpParams(W1=new_W1, W2=new_W2, W3=actor_params.W3)
    new_critic = CriticUpParams(
        U1=new_U1, U2=new_U2, U3=new_U3,
        Up1=new_Up1, Up2=new_Up2, Up3=critic_params.Up3,
        c=new_c,
        Z1=new_Z1, Z2=new_Z2, Z3=critic_params.Z3,
        Zp1=critic_params.Zp1, Zp2=new_Zp2, Zp3=critic_params.Zp3,
    )

    ortho_W1 = float(jnp.linalg.norm(new_W1.T @ new_W1 - jnp.eye(d)))
    ortho_U1 = float(jnp.linalg.norm(new_U1.T @ new_U1 - jnp.eye(d)))

    stats = {
        "td_mse_batch": float(td2 / N),
        "ortho_err_W1": ortho_W1,
        "ortho_err_U1": ortho_U1,
        "batch_size": float(N),
    }

    return new_actor, new_critic, stats


# -------------------------
# Downstairs gradient computation (vectorized)
# -------------------------

@partial(jax.jit, static_argnums=(7, 8, 9))
def downstairs_compute_grads(
    actor_params: ActorDnParams,
    critic_params: CriticDnParams,
    s_traj: Array,  # (N+1, ds)
    a_traj: Array,  # (N, da)
    r_traj: Array,  # (N,)
    beta: float,
    dt: float,
    N: int,
    ds: int,
    da: int,
) -> Tuple[Dict[str, Array], float]:
    """Compute gradients for downstairs actor-critic update."""

    def step_grads(k):
        t = k * dt
        w = jnp.exp(-beta * t)

        s_t = s_traj[k]
        s_tp1 = s_traj[k + 1]
        a_t = a_traj[k]
        r_t = r_traj[k]

        phi_t = critic_dn_phi(critic_params, s_t)
        phi_tp1 = critic_dn_phi(critic_params, s_tp1)
        psi_t = critic_dn_psi(critic_params, s_t, a_t, actor_params)
        delta = td_error(phi_t, phi_tp1, r_t, psi_t, beta, dt)

        a_pi = actor_dn_act(actor_params, s_t)
        a_bar = a_t - a_pi

        # Critic gradients
        gUb = jnp.outer(s_t, s_t)
        gUc = s_t.reshape(-1, 1)
        gZb = jnp.outer(s_t, a_bar)
        gZc = a_bar.reshape(1, -1)

        # Actor gradient
        g_t = (critic_params.Zb.T @ s_t) + critic_params.Zc_row.squeeze()
        gWc = jnp.outer(s_t, g_t)

        sU = w * delta * dt
        sW = w * dt

        return sU * gUb, sU * gUc, sU, sU * gZb, sU * gZc, sW * gWc, delta * delta

    results = jax.vmap(step_grads)(jnp.arange(N))

    grads = {
        "gUb": results[0].sum(axis=0),
        "gUc": results[1].sum(axis=0),
        "gc": results[2].sum(),
        "gZb": results[3].sum(axis=0),
        "gZc": results[4].sum(axis=0),
        "gWc": results[5].sum(axis=0),
    }
    td2 = results[6].sum()

    return grads, td2


def downstairs_actor_critic_update(
    actor_params: ActorDnParams,
    critic_params: CriticDnParams,
    s_traj: Array,
    a_traj: Array,
    r_traj: Array,
    *,
    beta: float,
    dt: float,
    L_eff: int,
    eta_actor: float,
    eta_value: float,
    eta_adv: float,
) -> Tuple[ActorDnParams, CriticDnParams, Dict[str, float]]:
    """Full downstairs actor-critic update with depth coefficients."""
    N = a_traj.shape[0]
    ds = s_traj.shape[1]
    da = a_traj.shape[1]

    grads, td2 = downstairs_compute_grads(
        actor_params, critic_params,
        s_traj, a_traj, r_traj,
        beta, dt, N, ds, da
    )

    # Apply updates with depth coefficients
    new_Ub = cayley_retract_stiefel(critic_params.Ub, L_eff * grads["gUb"], eta_value)
    new_Uc_col = cayley_retract_stiefel(critic_params.Uc_col, (L_eff - 1) * grads["gUc"], eta_value)
    new_c = critic_params.c + eta_value * grads["gc"]

    new_Zb = cayley_retract_stiefel(critic_params.Zb, (L_eff - 1) * grads["gZb"], eta_adv)
    new_Zc_row = critic_params.Zc_row + eta_adv * max(L_eff - 2, 0) * grads["gZc"]

    new_Wc_col = cayley_retract_stiefel(actor_params.Wc_col, (L_eff - 1) * grads["gWc"], eta_actor)

    new_actor = ActorDnParams(Wc_col=new_Wc_col)
    new_critic = CriticDnParams(
        Ub=new_Ub, Uc_col=new_Uc_col, c=new_c,
        Zb=new_Zb, Zc_row=new_Zc_row,
    )

    ortho_Wc = float(jnp.linalg.norm(new_Wc_col.T @ new_Wc_col - jnp.eye(da)))
    ortho_Zb = float(jnp.linalg.norm(new_Zb.T @ new_Zb - jnp.eye(da)))

    stats = {
        "td_mse": float(td2 / N),
        "ortho_err_Wc": ortho_Wc,
        "ortho_err_Zb": ortho_Zb,
    }

    return new_actor, new_critic, stats
