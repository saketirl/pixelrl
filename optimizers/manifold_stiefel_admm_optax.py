"""
ADMM Manifold Stiefel optimizer for JAX/Optax.

This is a stateless ADMM variant of manifold Muon/Stiefel optimization.
It follows the ADMM solver described by Sam Buchanan:
https://sdbuchanan.com/blog/manifold-muon/
"""
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from optax._src import base

from optimizers.manifold_stiefel_optax import msign


class ManifoldStiefelADMMState(NamedTuple):
    """State for the ADMM Manifold Stiefel optimizer."""

    step: jnp.ndarray


def _sym(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (x + x.T)


def manifold_stiefel_admm_update(
    W: jnp.ndarray,
    G: jnp.ndarray,
    eta: float = 0.02,
    steps: int = 10,
    rho: float = 4.0,
    msign_steps: int = 5,
) -> jnp.ndarray:
    """
    Single stateless ADMM manifold Stiefel update.

    Args:
        W: Current weight matrix.
        G: Gradient for W.
        eta: Learning rate for the primal weight update.
        steps: Number of ADMM iterations for the inner solve.
        rho: ADMM penalty parameter.
        msign_steps: Number of Newton-Schulz iterations for matrix sign.

    Returns:
        Updated weight matrix with the same shape as W.
    """
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T

    Lambda = -0.25 * (W.T @ G + G.T @ W)
    X = G + 2.0 * W @ Lambda
    Omega = jnp.zeros_like(X)
    eye = jnp.eye(W.shape[1], dtype=W.dtype)
    inv_rho = 1.0 / rho

    def admm_step(carry, _):
        Lambda, X, Omega = carry

        P = W.T @ (inv_rho * Omega + X - G)
        Lambda = 0.5 * _sym(P)

        B = G + 2.0 * W @ Lambda - inv_rho * Omega
        thresholded_basis = 0.5 * (
            eye + msign(B.T @ B - (inv_rho**2) * eye, steps=msign_steps)
        )
        X = (B - inv_rho * msign(B, steps=msign_steps)) @ thresholded_basis

        Omega = Omega + rho * (X - 2.0 * W @ Lambda - G)
        return (Lambda, X, Omega), None

    (Lambda, _, _), _ = jax.lax.scan(
        admm_step, (Lambda, X, Omega), None, length=steps
    )

    A = msign(G + 2.0 * W @ Lambda, steps=msign_steps)
    new_W = W - eta * A
    new_W = msign(new_W, steps=msign_steps)

    if should_transpose:
        new_W = new_W.T

    return new_W


def manifold_stiefel_admm(
    learning_rate: float = 0.02,
    steps: int = 10,
    rho: float = 4.0,
    msign_steps: int = 5,
    min_ndim: int = 2,
) -> base.GradientTransformation:
    """
    ADMM Manifold Stiefel optimizer.

    Matrix parameters use the stateless ADMM manifold update. Other parameters
    use standard gradient descent updates.
    """

    def init_fn(params: base.Params) -> ManifoldStiefelADMMState:
        return ManifoldStiefelADMMState(step=jnp.array(0))

    def update_fn(
        updates: base.Updates,
        state: ManifoldStiefelADMMState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ManifoldStiefelADMMState]:
        if params is None:
            raise ValueError(
                "ADMM Manifold Stiefel requires params to be passed to update()"
            )

        def update_param(g, p):
            if p.ndim >= min_ndim and min(p.shape) > 1:
                new_W = manifold_stiefel_admm_update(
                    W=p,
                    G=g,
                    eta=learning_rate,
                    steps=steps,
                    rho=rho,
                    msign_steps=msign_steps,
                )
                return new_W - p
            return -learning_rate * g

        new_updates = jax.tree_util.tree_map(update_param, updates, params)
        new_state = ManifoldStiefelADMMState(step=state.step + 1)
        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
