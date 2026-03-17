
"""JAX models for upstairs and downstairs actor-critic."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, NamedTuple

Array = jnp.ndarray


# -------------------------
# Initialization functions
# -------------------------

def stiefel_init(key: random.PRNGKey, n: int, p: int) -> Array:
    """X in R^{n x p} with orthonormal columns."""
    A = random.normal(key, (n, p))
    Q, _ = jnp.linalg.qr(A, mode="reduced")
    return Q


def orthogonal_init(key: random.PRNGKey, n: int) -> Array:
    return stiefel_init(key, n, n)


def row_unit_init(key: random.PRNGKey, m: int, n: int, eps: float = 1e-12) -> Array:
    A = random.normal(key, (m, n))
    return A / (jnp.linalg.norm(A, axis=1, keepdims=True) + eps)


# -------------------------
# Upstairs Actor (L=3 DLN)
# -------------------------

class ActorUpParams(NamedTuple):
    """Parameters for upstairs actor."""
    W1: Array  # (d, d) orthogonal, trainable
    W2: Array  # (d, d) orthogonal, trainable
    W3: Array  # (da, d) row-unit, frozen


def init_actor_up(key: random.PRNGKey, d: int, da: int) -> ActorUpParams:
    k1, k2, k3 = random.split(key, 3)
    return ActorUpParams(
        W1=orthogonal_init(k1, d),
        W2=orthogonal_init(k2, d),
        W3=row_unit_init(k3, da, d),
    )


@jax.jit
def actor_up_effective_matrix(params: ActorUpParams) -> Array:
    return params.W3 @ (params.W2 @ params.W1)  # (da, d)


@jax.jit
def actor_up_act(params: ActorUpParams, o: Array) -> Array:
    """Compute action from observation."""
    return actor_up_effective_matrix(params) @ o


# -------------------------
# Upstairs Critic (L=3 DLN)
# -------------------------

class CriticUpParams(NamedTuple):
    """Parameters for upstairs critic."""
    # Value function: phi(o) = o^T (U3 U2 U1) o + (Up3 Up2 Up1) o + c
    U1: Array   # (d, d) orthogonal, trainable
    U2: Array   # (d, d) orthogonal, trainable
    U3: Array   # (d, d) orthogonal, trainable
    Up1: Array  # (d, d) orthogonal, trainable
    Up2: Array  # (d, d) orthogonal, trainable
    Up3: Array  # (1, d) row-unit, frozen
    c: float    # scalar, trainable

    # Advantage: Psi(o,a) = a^T (Z3 Z2 Z1) o + (Zp3 Zp2 Zp1) a
    Z1: Array   # (d, d) orthogonal, trainable
    Z2: Array   # (d, d) orthogonal, trainable
    Z3: Array   # (da, d) row-unit, frozen
    Zp1: Array  # (d, da) stiefel, frozen
    Zp2: Array  # (d, d) orthogonal, trainable
    Zp3: Array  # (1, d) row-unit, frozen


def init_critic_up(key: random.PRNGKey, d: int, da: int) -> CriticUpParams:
    keys = random.split(key, 13)
    return CriticUpParams(
        U1=orthogonal_init(keys[0], d),
        U2=orthogonal_init(keys[1], d),
        U3=orthogonal_init(keys[2], d),
        Up1=orthogonal_init(keys[3], d),
        Up2=orthogonal_init(keys[4], d),
        Up3=row_unit_init(keys[5], 1, d),
        c=0.0,
        Z1=orthogonal_init(keys[6], d),
        Z2=orthogonal_init(keys[7], d),
        Z3=row_unit_init(keys[8], da, d),
        Zp1=stiefel_init(keys[9], d, da),
        Zp2=orthogonal_init(keys[10], d),
        Zp3=row_unit_init(keys[11], 1, d),
    )


@jax.jit
def critic_up_Ueff(params: CriticUpParams) -> Array:
    return params.U3 @ (params.U2 @ params.U1)


@jax.jit
def critic_up_Up_eff(params: CriticUpParams) -> Array:
    return params.Up2 @ params.Up1


@jax.jit
def critic_up_Zeff(params: CriticUpParams) -> Array:
    return params.Z3 @ (params.Z2 @ params.Z1)


@jax.jit
def critic_up_phi(params: CriticUpParams, o: Array) -> float:
    """Value function phi(o)."""
    quad = o.T @ (critic_up_Ueff(params) @ o)
    lin = (params.Up3 @ (critic_up_Up_eff(params) @ o)).squeeze()
    return quad + lin + params.c


@jax.jit
def critic_up_Psi(params: CriticUpParams, o: Array, a: Array) -> float:
    """Raw advantage Psi(o, a)."""
    mixed = a.T @ (critic_up_Zeff(params) @ o)
    lin = (params.Zp3 @ (params.Zp2 @ (params.Zp1 @ a))).squeeze()
    return mixed + lin


@jax.jit
def critic_up_psi(params: CriticUpParams, o: Array, a: Array, actor_params: ActorUpParams) -> float:
    """Centered advantage psi(o, a) = Psi(o, a) - Psi(o, pi(o))."""
    a_pi = actor_up_act(actor_params, o)
    return critic_up_Psi(params, o, a) - critic_up_Psi(params, o, a_pi)


# -------------------------
# Downstairs Actor (Shallow)
# -------------------------

class ActorDnParams(NamedTuple):
    """Parameters for downstairs actor."""
    Wc_col: Array  # (ds, da) stiefel columns, trainable


def init_actor_dn(key: random.PRNGKey, ds: int, da: int) -> ActorDnParams:
    return ActorDnParams(Wc_col=stiefel_init(key, ds, da))


@jax.jit
def actor_dn_act(params: ActorDnParams, s: Array) -> Array:
    """Compute action from state: a = Wc^T s."""
    return params.Wc_col.T @ s


# -------------------------
# Downstairs Critic (Shallow)
# -------------------------

class CriticDnParams(NamedTuple):
    """Parameters for downstairs critic."""
    Ub: Array      # (ds, ds) orthogonal, trainable
    Uc_col: Array  # (ds, 1) stiefel, trainable
    c: float       # scalar, trainable
    Zb: Array      # (ds, da) stiefel, trainable
    Zc_row: Array  # (1, da) unconstrained, trainable


def init_critic_dn(key: random.PRNGKey, ds: int, da: int) -> CriticDnParams:
    k1, k2, k3, k4 = random.split(key, 4)
    return CriticDnParams(
        Ub=orthogonal_init(k1, ds),
        Uc_col=stiefel_init(k2, ds, 1),
        c=0.0,
        Zb=stiefel_init(k3, ds, da),
        Zc_row=0.01 * random.normal(k4, (1, da)),
    )


@jax.jit
def critic_dn_phi(params: CriticDnParams, s: Array) -> float:
    """Value function phi(s)."""
    quad = s.T @ (params.Ub @ s)
    lin = (params.Uc_col.T @ s).squeeze()
    return quad + lin + params.c


@jax.jit
def critic_dn_Psi(params: CriticDnParams, s: Array, a: Array) -> float:
    """Raw advantage Psi(s, a)."""
    mixed = s.T @ (params.Zb @ a)
    lin_a = (params.Zc_row @ a).squeeze()
    return mixed + lin_a


@jax.jit
def critic_dn_psi(params: CriticDnParams, s: Array, a: Array, actor_params: ActorDnParams) -> float:
    """Centered advantage psi(s, a) = Psi(s, a) - Psi(s, pi(s))."""
    a_pi = actor_dn_act(actor_params, s)
    return critic_dn_Psi(params, s, a) - critic_dn_Psi(params, s, a_pi)
