"""
Manifold MUON optimizer for JAX/Optax.

Based on: https://github.com/sdbuch/thinky-manifolds/blob/main/src/manifold_muon.py
Uses the Polar Express algorithm for matrix sign: https://arxiv.org/abs/2505.16932

This optimizer performs gradient descent on the nuclear norm objective,
keeping weight matrices on the Stiefel manifold (orthonormal columns/rows).
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Any, Tuple
import optax
from optax._src import base


# Polar Express coefficients for matrix sign function
ABC_LIST = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

ABC_LIST_STABLE = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


def msign(G: jnp.ndarray, steps: int = 10) -> jnp.ndarray:
    """
    Matrix sign function using the Polar Express algorithm.

    Computes the orthogonal polar factor of a matrix via iterative refinement.
    For a matrix G = U @ S @ V^T (SVD), returns U @ V^T.

    Args:
        G: Input matrix of shape (..., m, n)
        steps: Number of iteration steps (default 10)

    Returns:
        Matrix sign of G with same shape
    """
    assert G.ndim >= 2
    should_transpose = G.shape[-2] > G.shape[-1]

    x = G.astype(jnp.float32)
    if should_transpose:
        x = jnp.swapaxes(x, -2, -1)

    # Normalize
    norm = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
    x = x / (norm * 1.01 + 1e-8)

    def iteration_step(x, step_idx):
        # Get coefficients (use last one if beyond list length)
        step_idx_clamped = jnp.minimum(step_idx, len(ABC_LIST_STABLE) - 1)
        a, b, c = ABC_LIST_STABLE[step_idx_clamped]

        s = x @ jnp.swapaxes(x, -2, -1)  # x @ x^T

        # Compute x = (a I + (b I + c S) S) x
        # y = c * s
        y = c * s
        # y.diagonal += b  ->  y = y + b * I
        eye = jnp.eye(y.shape[-1], dtype=y.dtype)
        y = y + b * eye
        # y = y @ s
        y = y @ s
        # y.diagonal += a  ->  y = y + a * I
        y = y + a * eye
        # x = y @ x
        x = y @ x
        return x, None

    # Run iterations
    x, _ = jax.lax.scan(iteration_step, x, jnp.arange(steps))

    if should_transpose:
        x = jnp.swapaxes(x, -2, -1)

    # Replace NaNs with zeros
    x = jnp.nan_to_num(x)
    return x


def msign_simple(G: jnp.ndarray, steps: int = 10) -> jnp.ndarray:
    """
    Simpler matrix sign using Newton-Schulz iteration.
    More stable but slower convergence than Polar Express.

    X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)
    """
    should_transpose = G.shape[-2] > G.shape[-1]

    x = G.astype(jnp.float32)
    if should_transpose:
        x = jnp.swapaxes(x, -2, -1)

    # Normalize
    norm = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
    x = x / (norm + 1e-8)

    eye = jnp.eye(x.shape[-1], dtype=x.dtype)

    def newton_schulz_step(x, _):
        xtx = jnp.swapaxes(x, -2, -1) @ x
        x = 0.5 * x @ (3.0 * eye - xtx)
        return x, None

    x, _ = jax.lax.scan(newton_schulz_step, x, None, length=steps)

    if should_transpose:
        x = jnp.swapaxes(x, -2, -1)

    return jnp.nan_to_num(x)


class ManifoldMuonState(NamedTuple):
    """State for Manifold MUON optimizer."""
    lambda_dual: base.Params  # Dual variables for each parameter
    step: jnp.ndarray  # Global step counter


def manifold_muon_update(
    W: jnp.ndarray,
    G: jnp.ndarray,
    Lambda: jnp.ndarray,
    eta: float = 0.1,
    alpha: float = 0.01,
    steps: int = 100,
    tol: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single manifold MUON update step.

    Implements gradient descent on the nuclear norm objective while
    keeping weights on the Stiefel manifold.

    Args:
        W: Current weight matrix
        G: Gradient
        Lambda: Dual variable (symmetric matrix)
        eta: Learning rate for weight update
        alpha: Learning rate for dual variable
        steps: Number of inner optimization steps
        tol: Tolerance for early stopping

    Returns:
        Tuple of (new_W, new_Lambda)
    """
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T

    # Initialize Lambda if needed (first step)
    Lambda = jnp.where(
        jnp.sum(jnp.abs(Lambda)) < 1e-10,
        -0.25 * (W.T @ G + G.T @ W),
        Lambda
    )

    def dual_step(carry, step_idx):
        Lambda = carry
        A = msign_simple(G + 2 * W @ Lambda, steps=5)
        H = W.T @ A + A.T @ W

        # Compute step size with decay
        step_size = alpha * (1.0 - step_idx / steps)
        Lambda_new = Lambda - step_size * H

        return Lambda_new, None

    # Run dual optimization
    Lambda, _ = jax.lax.scan(dual_step, Lambda, jnp.arange(steps))

    # Compute final update
    A = msign_simple(G + 2 * W @ Lambda, steps=5)
    new_W = W - eta * A
    new_W = msign_simple(new_W, steps=5)

    if should_transpose:
        new_W = new_W.T
        Lambda = Lambda.T

    return new_W, Lambda


def manifold_muon(
    learning_rate: float = 0.02,
    dual_lr: float = 0.01,
    dual_steps: int = 5,
    msign_steps: int = 5,
    min_ndim: int = 2,
) -> base.GradientTransformation:
    """
    Manifold MUON optimizer.

    Keeps weight matrices on the Stiefel manifold (orthonormal structure)
    while optimizing. Uses the Polar Express algorithm for efficient
    matrix sign computation.

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
        >>> optimizer = manifold_muon(learning_rate=0.02)
        >>> opt_state = optimizer.init(params)
        >>> updates, opt_state = optimizer.update(grads, opt_state, params)
        >>> params = optax.apply_updates(params, updates)
    """

    def init_fn(params: base.Params) -> ManifoldMuonState:
        # Initialize dual variables as zeros with same structure as params
        def init_lambda(p):
            if p.ndim >= min_ndim:
                # Lambda should be square: min(m,n) x min(m,n)
                min_dim = min(p.shape[0], p.shape[1])
                return jnp.zeros((min_dim, min_dim), dtype=p.dtype)
            else:
                return jnp.zeros_like(p)

        lambda_dual = jax.tree_util.tree_map(init_lambda, params)
        return ManifoldMuonState(lambda_dual=lambda_dual, step=jnp.array(0))

    def update_fn(
        updates: base.Updates,
        state: ManifoldMuonState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ManifoldMuonState]:
        if params is None:
            raise ValueError("Manifold MUON requires params to be passed to update()")

        def update_param(g, p, lam):
            """Update a single parameter."""
            if p.ndim >= min_ndim and min(p.shape) > 1:
                # Apply manifold MUON for matrices
                should_transpose = p.shape[0] < p.shape[1]
                W = p.T if should_transpose else p
                G = g.T if should_transpose else g

                # Ensure Lambda has correct shape
                min_dim = min(W.shape[0], W.shape[1])
                if lam.shape[0] != min_dim:
                    lam = jnp.zeros((min_dim, min_dim), dtype=W.dtype)

                # Initialize Lambda if zero
                lam = jnp.where(
                    jnp.sum(jnp.abs(lam)) < 1e-10,
                    -0.25 * (W.T @ G + G.T @ W),
                    lam
                )

                # Dual optimization steps
                def dual_step(lam, _):
                    A = msign_simple(G + 2 * W @ lam, steps=msign_steps)
                    H = W.T @ A + A.T @ W
                    return lam - dual_lr * H, None

                lam, _ = jax.lax.scan(dual_step, lam, None, length=dual_steps)

                # Compute update direction
                A = msign_simple(G + 2 * W @ lam, steps=msign_steps)
                new_W = W - learning_rate * A
                new_W = msign_simple(new_W, steps=msign_steps)

                # Compute the actual update (difference)
                if should_transpose:
                    update = new_W.T - p
                    new_lam = lam.T if lam.shape[0] != lam.shape[1] else lam
                else:
                    update = new_W - p
                    new_lam = lam

                return update, new_lam
            else:
                # Standard gradient descent for non-matrix params
                return -learning_rate * g, lam

        # Apply updates to all parameters - returns tree of (update, new_lam) tuples
        results = jax.tree_util.tree_map(
            update_param,
            updates,
            params,
            state.lambda_dual,
            is_leaf=lambda x: isinstance(x, jnp.ndarray)
        )

        # Unzip the results into separate trees
        new_updates = jax.tree_util.tree_map(lambda x: x[0], results)
        new_lambda = jax.tree_util.tree_map(lambda x: x[1], results)

        new_state = ManifoldMuonState(
            lambda_dual=new_lambda,
            step=state.step + 1
        )

        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def manifold_muon_with_schedule(
    learning_rate: optax.Schedule,
    dual_lr: float = 0.01,
    dual_steps: int = 5,
    msign_steps: int = 5,
    min_ndim: int = 2,
) -> base.GradientTransformation:
    """
    Manifold MUON with learning rate schedule.

    Args:
        learning_rate: Optax schedule for learning rate
        dual_lr: Learning rate for dual variable
        dual_steps: Number of dual optimization steps
        msign_steps: Iterations for matrix sign
        min_ndim: Minimum dimensions for manifold optimization

    Returns:
        Optax GradientTransformation with injectable hyperparams
    """

    def init_fn(params: base.Params) -> ManifoldMuonState:
        def init_lambda(p):
            if p.ndim >= min_ndim:
                min_dim = min(p.shape[0], p.shape[1])
                return jnp.zeros((min_dim, min_dim), dtype=p.dtype)
            else:
                return jnp.zeros_like(p)

        lambda_dual = jax.tree_util.tree_map(init_lambda, params)
        return ManifoldMuonState(lambda_dual=lambda_dual, step=jnp.array(0))

    def update_fn(
        updates: base.Updates,
        state: ManifoldMuonState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, ManifoldMuonState]:
        if params is None:
            raise ValueError("Manifold MUON requires params")

        lr = learning_rate(state.step)

        def update_param(g, p, lam):
            if p.ndim >= min_ndim and min(p.shape) > 1:
                should_transpose = p.shape[0] < p.shape[1]
                W = p.T if should_transpose else p
                G = g.T if should_transpose else g

                min_dim = min(W.shape[0], W.shape[1])
                if lam.shape[0] != min_dim:
                    lam = jnp.zeros((min_dim, min_dim), dtype=W.dtype)

                lam = jnp.where(
                    jnp.sum(jnp.abs(lam)) < 1e-10,
                    -0.25 * (W.T @ G + G.T @ W),
                    lam
                )

                def dual_step(lam, _):
                    A = msign_simple(G + 2 * W @ lam, steps=msign_steps)
                    H = W.T @ A + A.T @ W
                    return lam - dual_lr * H, None

                lam, _ = jax.lax.scan(dual_step, lam, None, length=dual_steps)

                A = msign_simple(G + 2 * W @ lam, steps=msign_steps)
                new_W = W - lr * A
                new_W = msign_simple(new_W, steps=msign_steps)

                if should_transpose:
                    update = new_W.T - p
                    new_lam = lam
                else:
                    update = new_W - p
                    new_lam = lam

                return update, new_lam
            else:
                return -lr * g, lam

        # Apply updates - returns tree of (update, new_lam) tuples
        results = jax.tree_util.tree_map(
            update_param,
            updates,
            params,
            state.lambda_dual,
            is_leaf=lambda x: isinstance(x, jnp.ndarray)
        )

        # Unzip the results
        new_updates = jax.tree_util.tree_map(lambda x: x[0], results)
        new_lambda = jax.tree_util.tree_map(lambda x: x[1], results)

        new_state = ManifoldMuonState(
            lambda_dual=new_lambda,
            step=state.step + 1
        )

        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


# Convenience function to combine with gradient clipping
def manifold_muon_with_clipping(
    learning_rate: float = 0.02,
    max_grad_norm: float = 1.0,
    dual_lr: float = 0.01,
    dual_steps: int = 5,
    msign_steps: int = 5,
) -> base.GradientTransformation:
    """
    Manifold MUON with gradient clipping.

    Args:
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        dual_lr: Dual variable learning rate
        dual_steps: Dual optimization steps
        msign_steps: Matrix sign iterations

    Returns:
        Chained GradientTransformation
    """
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        manifold_muon(
            learning_rate=learning_rate,
            dual_lr=dual_lr,
            dual_steps=dual_steps,
            msign_steps=msign_steps,
        ),
    )
