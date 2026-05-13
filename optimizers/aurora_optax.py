"""
Aurora optimizer for JAX/Optax.

Based on Tilde Research's Aurora optimizer:
https://github.com/tilde-research/aurora-release/blob/main/src/aurora.py

Aurora is a Muon-style optimizer for 2D matrices. It uses momentum, computes
an approximately leverage-uniform polar update, and applies decoupled weight
decay. Non-matrix parameters fall back to Adam updates.
"""
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from optax._src import base

from optimizers.manifold_stiefel_optax import msign


class AuroraState(NamedTuple):
    """State for Aurora with Adam fallback."""

    step: jnp.ndarray
    momentum: base.Updates
    adam_mu: base.Updates
    adam_nu: base.Updates


def _zeros_like_tree(params: base.Params) -> base.Updates:
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def _aurora_project(
    update: jnp.ndarray,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
    msign_steps: int = 5,
) -> jnp.ndarray:
    """
    Projects a 2D update with Aurora's leverage-uniform polar iteration.

    Wide matrices are transposed to tall matrices, projected, then transposed
    back, matching the reference implementation.
    """
    rows, cols = update.shape

    if rows == cols:
        return msign(update, steps=msign_steps)

    should_transpose = rows < cols
    if should_transpose:
        update = update.T
        rows, cols = cols, rows

    update32 = update.astype(jnp.float32)
    target_row_sq = cols / rows
    row_norm = jnp.maximum(
        jnp.linalg.norm(update32, axis=-1, keepdims=True),
        eps,
    )
    diagonal_scale = 1.0 / row_norm

    def projection_step(carry, step_idx):
        diagonal_scale = carry
        projected = msign(diagonal_scale * update32, steps=msign_steps)
        row_sq = jnp.maximum(
            jnp.sum(projected.astype(jnp.float32) ** 2, axis=-1, keepdims=True),
            eps * eps,
        )
        next_scale = diagonal_scale * (target_row_sq / row_sq) ** pp_beta
        diagonal_scale = jnp.where(
            step_idx < pp_iterations - 1,
            next_scale,
            diagonal_scale,
        )
        return diagonal_scale, projected

    _, projected_steps = jax.lax.scan(
        projection_step,
        diagonal_scale,
        jnp.arange(pp_iterations),
    )
    projected = projected_steps[-1].astype(update.dtype)

    if should_transpose:
        projected = projected.T

    return projected


def aurora_update(
    W: jnp.ndarray,
    G: jnp.ndarray,
    momentum: jnp.ndarray,
    learning_rate: float = 0.05,
    weight_decay: float = 0.025,
    mu: float = 0.95,
    nesterov: bool = True,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
    msign_steps: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes one Aurora update for a single 2D matrix parameter.

    Args:
        W: Current parameter matrix.
        G: Gradient for W.
        momentum: Previous Aurora momentum buffer.
        learning_rate: Step size.
        weight_decay: Decoupled weight decay.
        mu: Momentum coefficient.
        nesterov: Whether to use Nesterov-style momentum.
        pp_iterations: Aurora projection iterations.
        pp_beta: Projection damping exponent.
        eps: Numerical stability constant.
        msign_steps: Newton-Schulz steps for the polar factor.

    Returns:
        A tuple of (parameter_update, new_momentum), where parameter_update is
        suitable for optax.apply_updates.
    """
    if W.ndim != 2:
        raise ValueError(f"Aurora expects 2D weights, got shape {W.shape}")

    new_momentum = mu * momentum + (1.0 - mu) * G
    if nesterov:
        update = (1.0 - mu) * G + mu * new_momentum
    else:
        update = new_momentum

    update = _aurora_project(
        update,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
        eps=eps,
        msign_steps=msign_steps,
    )

    rows, cols = G.shape
    update = update * jnp.sqrt(jnp.maximum(1.0, rows / cols))
    parameter_update = -learning_rate * update - learning_rate * weight_decay * W
    return parameter_update, new_momentum


def _adam_update(
    G: jnp.ndarray,
    mu: jnp.ndarray,
    nu: jnp.ndarray,
    step: jnp.ndarray,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    new_mu = b1 * mu + (1.0 - b1) * G
    new_nu = b2 * nu + (1.0 - b2) * (G**2)
    count = step + 1
    mu_hat = new_mu / (1.0 - b1**count)
    nu_hat = new_nu / (1.0 - b2**count)
    update = -learning_rate * mu_hat / (jnp.sqrt(nu_hat) + eps)
    return update, new_mu, new_nu


def aurora(
    learning_rate: float = 0.05,
    weight_decay: float = 0.025,
    mu: float = 0.95,
    nesterov: bool = True,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
    msign_steps: int = 5,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
) -> base.GradientTransformation:
    """
    Aurora optimizer with Adam fallback for non-2D parameters.

    Aurora is applied only to rank-2 parameters whose minimum dimension is
    greater than 1. Other leaves use Adam with the same learning rate.
    """

    def init_fn(params: base.Params) -> AuroraState:
        zeros = _zeros_like_tree(params)
        return AuroraState(
            step=jnp.array(0),
            momentum=zeros,
            adam_mu=zeros,
            adam_nu=zeros,
        )

    def update_fn(
        updates: base.Updates,
        state: AuroraState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, AuroraState]:
        if params is None:
            raise ValueError("Aurora requires params to be passed to update()")

        def update_param(g, p, momentum, adam_mu, adam_nu):
            if p.ndim == 2 and min(p.shape) > 1:
                param_update, new_momentum = aurora_update(
                    W=p,
                    G=g,
                    momentum=momentum,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    mu=mu,
                    nesterov=nesterov,
                    pp_iterations=pp_iterations,
                    pp_beta=pp_beta,
                    eps=eps,
                    msign_steps=msign_steps,
                )
                return param_update, new_momentum, adam_mu, adam_nu

            param_update, new_adam_mu, new_adam_nu = _adam_update(
                G=g,
                mu=adam_mu,
                nu=adam_nu,
                step=state.step,
                learning_rate=learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
            )
            return param_update, momentum, new_adam_mu, new_adam_nu

        mapped = jax.tree_util.tree_map(
            update_param,
            updates,
            params,
            state.momentum,
            state.adam_mu,
            state.adam_nu,
        )
        is_update_record = lambda x: isinstance(x, tuple) and len(x) == 4
        new_updates = jax.tree_util.tree_map(
            lambda x: x[0],
            mapped,
            is_leaf=is_update_record,
        )
        new_momentum = jax.tree_util.tree_map(
            lambda x: x[1],
            mapped,
            is_leaf=is_update_record,
        )
        new_adam_mu = jax.tree_util.tree_map(
            lambda x: x[2],
            mapped,
            is_leaf=is_update_record,
        )
        new_adam_nu = jax.tree_util.tree_map(
            lambda x: x[3],
            mapped,
            is_leaf=is_update_record,
        )
        new_state = AuroraState(
            step=state.step + 1,
            momentum=new_momentum,
            adam_mu=new_adam_mu,
            adam_nu=new_adam_nu,
        )
        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
