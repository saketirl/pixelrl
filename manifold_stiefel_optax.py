"""
Manifold Stiefel optimizer for JAX/Optax.

Based on the manifold optimizer implementation from thinky-manifolds.

This optimizer performs gradient descent on the nuclear norm objective,
keeping weight matrices on the Stiefel manifold (orthonormal columns/rows).

Key insight: Lambda is recomputed each update, NOT stored across steps.
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Any, Tuple
import optax
from optax._src import base


def msign(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    """
    Matrix sign function using Newton-Schulz iteration.

    Computes the orthogonal polar factor of a matrix via iterative refinement.
    For a matrix G = U @ S @ V^T (SVD), returns U @ V^T.

    Uses: X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)

    Args:
        G: Input matrix of shape (m, n) where m >= n
        steps: Number of iteration steps

    Returns:
        Matrix sign of G with same shape
    """
    # Normalize for numerical stability
    norm = jnp.linalg.norm(G, ord='fro')
    x = G / (norm + 1e-7)

    eye = jnp.eye(x.shape[-1], dtype=x.dtype)

    def newton_schulz_step(x, _):
        xtx = x.T @ x
        x = 0.5 * x @ (3.0 * eye - xtx)
        return x, None

    x, _ = jax.lax.scan(newton_schulz_step, x, None, length=steps)

    return jnp.where(jnp.isnan(x), 0.0, x)


def manifold_stiefel_update(
    W: jnp.ndarray,
    G: jnp.ndarray,
    eta: float = 0.02,
    alpha: float = 0.01,
    steps: int = 5,
    msign_steps: int = 5,
) -> jnp.ndarray:
    """
    Single manifold Stiefel update step.

    Implements gradient descent on || G + W @ (L + L^T) ||_*

    Args:
        W: Current weight matrix
        G: Gradient
        eta: Learning rate for weight update
        alpha: Learning rate for dual variable
        steps: Number of dual optimization steps

    Returns:
        new_W: Updated weight matrix
    """
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T

    # Initialize Lambda (recomputed each update, not stored)
    Lambda = -0.25 * (W.T @ G + G.T @ W)

    # Dual optimization
    def dual_step(Lambda, step_idx):
        A = msign(G + 2 * W @ Lambda, steps=msign_steps)
        H = W.T @ A + A.T @ W
        # Decaying step size
        step_size = alpha * (1.0 - step_idx / steps)
        Lambda = Lambda - step_size * H
        return Lambda, None

    Lambda, _ = jax.lax.scan(dual_step, Lambda, jnp.arange(steps))

    # Final update
    A = msign(G + 2 * W @ Lambda, steps=msign_steps)
    new_W = W - eta * A
    new_W = msign(new_W, steps=msign_steps)

    if should_transpose:
        new_W = new_W.T

    return new_W


class ManifoldStiefelState(NamedTuple):
    """State for Manifold Stiefel optimizer (just step counter)."""
    step: jnp.ndarray


def manifold_stiefel_update_per_head(
    W: jnp.ndarray,
    G: jnp.ndarray,
    eta: float = 0.02,
    alpha: float = 0.01,
    dual_steps: int = 5,
    msign_steps: int = 5,
) -> jnp.ndarray:
    """
    Manifold Stiefel update applied independently per attention head.

    Expected shape for W/G is (in_dim, num_heads, head_dim). This enforces
    within-head orthogonality by updating each (in_dim, head_dim) slice.
    """
    if W.ndim != 3:
        raise ValueError(f"Expected rank-3 tensor for per-head update, got {W.shape}")

    w_heads = jnp.swapaxes(W, 0, 1)
    g_heads = jnp.swapaxes(G, 0, 1)

    def update_one_head(w_h, g_h):
        return manifold_stiefel_update(
            W=w_h,
            G=g_h,
            eta=eta,
            alpha=alpha,
            steps=dual_steps,
            msign_steps=msign_steps,
        )

    updated_heads = jax.vmap(update_one_head)(w_heads, g_heads)
    return jnp.swapaxes(updated_heads, 0, 1)


def manifold_stiefel(
    learning_rate: float = 0.02,
    dual_lr: float = 0.01,
    dual_steps: int = 5,
    msign_steps: int = 5,
    min_ndim: int = 2,
) -> base.GradientTransformation:
    """
    Manifold Stiefel optimizer.

    Keeps weight matrices on the Stiefel manifold (orthonormal structure)
    while optimizing.

    Only applies to parameters with ndim >= min_ndim. Other parameters
    use standard gradient descent.

    Args:
        learning_rate: Learning rate for weight updates (eta)
        dual_lr: Learning rate for dual variable optimization (alpha)
        dual_steps: Number of dual optimization steps per update
        msign_steps: Number of iterations for matrix sign computation
        min_ndim: Minimum number of dimensions for manifold optimization

    Returns:
        Optax GradientTransformation

    Example:
        >>> optimizer = manifold_stiefel(learning_rate=0.02)
        >>> opt_state = optimizer.init(params)
        >>> updates, opt_state = optimizer.update(grads, opt_state, params)
        >>> params = optax.apply_updates(params, updates)
    """

    def init_fn(params: base.Params) -> ManifoldStiefelState:
        return ManifoldStiefelState(step=jnp.array(0))

    def update_fn(
        updates: base.Updates,
        state: ManifoldStiefelState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ManifoldStiefelState]:
        if params is None:
            raise ValueError("Manifold Stiefel requires params to be passed to update()")

        def update_param(g, p):
            """Update a single parameter."""
            if p.ndim >= min_ndim and min(p.shape) > 1:
                # Apply manifold Stiefel for matrices
                new_W = manifold_stiefel_update(
                    W=p,
                    G=g,
                    eta=learning_rate,
                    alpha=dual_lr,
                    steps=dual_steps,
                    msign_steps=msign_steps,
                )
                # Return the update (difference from current)
                return new_W - p
            else:
                # Standard gradient descent for non-matrix params
                return -learning_rate * g

        new_updates = jax.tree_util.tree_map(update_param, updates, params)
        new_state = ManifoldStiefelState(step=state.step + 1)

        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def manifold_stiefel_per_head(
    learning_rate: float = 0.02,
    dual_lr: float = 0.01,
    dual_steps: int = 5,
    msign_steps: int = 5,
    min_ndim: int = 2,
) -> base.GradientTransformation:
    """
    Manifold Stiefel optimizer with special handling for rank-3 attention kernels.

    - Rank-3 tensors are updated head-wise (within-head orthogonality).
    - Rank-2+ matrices use standard manifold Stiefel fallback.
    - Lower-rank tensors use simple SGD updates.
    """

    def init_fn(params: base.Params) -> ManifoldStiefelState:
        return ManifoldStiefelState(step=jnp.array(0))

    def update_fn(
        updates: base.Updates,
        state: ManifoldStiefelState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ManifoldStiefelState]:
        if params is None:
            raise ValueError("Manifold Stiefel requires params to be passed to update()")

        def update_param(g, p):
            if p.ndim == 3 and p.shape[1] > 1 and min(p.shape[0], p.shape[-1]) > 1:
                new_W = manifold_stiefel_update_per_head(
                    W=p,
                    G=g,
                    eta=learning_rate,
                    alpha=dual_lr,
                    dual_steps=dual_steps,
                    msign_steps=msign_steps,
                )
                return new_W - p
            if p.ndim >= min_ndim and min(p.shape) > 1:
                new_W = manifold_stiefel_update(
                    W=p,
                    G=g,
                    eta=learning_rate,
                    alpha=dual_lr,
                    steps=dual_steps,
                    msign_steps=msign_steps,
                )
                return new_W - p
            return -learning_rate * g

        new_updates = jax.tree_util.tree_map(update_param, updates, params)
        new_state = ManifoldStiefelState(step=state.step + 1)
        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
