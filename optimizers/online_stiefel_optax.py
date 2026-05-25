"""
Online Manifold Muon/Stiefel optimizer for JAX/Optax.

This implementation keeps matrix parameters on a scaled Stiefel manifold and
stores the online dual variables across training steps.
"""
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from optax._src import base


_ABC_LIST = [
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
    (float(a) / 1.01, float(b) / 1.01**3, float(c) / 1.01**5)
    if idx < len(_ABC_LIST) - 1
    else (float(a), float(b), float(c))
    for idx, (a, b, c) in enumerate(_ABC_LIST)
]


class OnlineStiefelState(NamedTuple):
    """State for online Stiefel optimization."""

    step: jnp.ndarray
    dual_lambda: base.Updates
    dual_velocity: base.Updates
    momentum: base.Updates
    last_tangent: base.Updates
    last_constraint_residual: base.Updates


def _sym(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _orient_tall(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
    transposed = matrix.shape[-2] < matrix.shape[-1]
    return (matrix.T if transposed else matrix), transposed


def _restore_orientation(matrix: jnp.ndarray, transposed: bool) -> jnp.ndarray:
    return matrix.T if transposed else matrix


def matrix_sign(matrix: jnp.ndarray, steps: int = 10) -> jnp.ndarray:
    """Matrix sign / polar factor using the Polar Express iteration."""
    transposed = matrix.shape[-2] > matrix.shape[-1]
    x = matrix.T if transposed else matrix
    compute_dtype = jnp.promote_types(x.dtype, jnp.float32)
    x = x.astype(compute_dtype)
    norm = jnp.linalg.norm(x, ord="fro")
    norm = jnp.where(norm == 0.0, 1.0, norm)
    x = x / (norm * 1.01)
    eye = jnp.eye(x.shape[-2], dtype=x.dtype)
    coeffs = jnp.asarray(ABC_LIST_STABLE, dtype=x.dtype)

    def sign_step(x, step):
        a, b, c = coeffs[jnp.minimum(step, coeffs.shape[0] - 1)]
        s = x @ x.T
        y = (c * s + b * eye) @ s + a * eye
        return y @ x, None

    x, _ = jax.lax.scan(sign_step, x, jnp.arange(steps))
    x = x.T if transposed else x
    return jnp.nan_to_num(x.astype(matrix.dtype))


def scale_radius(weight: jnp.ndarray, scale: str) -> jnp.ndarray:
    if scale == "none":
        return jnp.array(1.0, dtype=weight.dtype)
    if scale == "ratio":
        return jnp.sqrt(jnp.array(weight.shape[0] / weight.shape[1], dtype=weight.dtype))
    raise ValueError(f"unknown online Stiefel scale: {scale}")


def retract_to_scaled_stiefel(weight: jnp.ndarray, scale: str) -> jnp.ndarray:
    radius = scale_radius(weight, scale)
    return radius * matrix_sign(weight)


def online_dual_ascent_step_tall(
    weight: jnp.ndarray,
    gradient: jnp.ndarray,
    dual_lambda: jnp.ndarray,
    dual_velocity: jnp.ndarray,
    alpha: float,
    beta: float,
    msign_steps: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dual_lambda = _sym(dual_lambda)
    dual_velocity = _sym(dual_velocity)
    lambda_tilde = _sym(dual_lambda + beta * dual_velocity)
    tangent = matrix_sign(gradient + 2.0 * weight @ lambda_tilde, steps=msign_steps)

    residual = _sym(weight.T @ tangent + tangent.T @ weight)
    velocity_next = _sym(beta * dual_velocity - alpha * residual)
    lambda_next = _sym(dual_lambda + velocity_next)
    return tangent, lambda_next, velocity_next, residual


def online_dual_ascent_step(
    weight: jnp.ndarray,
    gradient: jnp.ndarray,
    dual_lambda: jnp.ndarray,
    dual_velocity: jnp.ndarray,
    alpha: float = 1e-2,
    beta: float = 0.9,
    msign_steps: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    weight_tall, transposed = _orient_tall(weight)
    gradient_tall = gradient.T if transposed else gradient
    tangent_tall, lambda_next, velocity_next, residual = online_dual_ascent_step_tall(
        weight_tall,
        gradient_tall,
        dual_lambda,
        dual_velocity,
        alpha,
        beta,
        msign_steps=msign_steps,
    )
    return _restore_orientation(tangent_tall, transposed), lambda_next, velocity_next, residual


def online_stiefel_update(
    W: jnp.ndarray,
    G: jnp.ndarray,
    dual_lambda: jnp.ndarray,
    dual_velocity: jnp.ndarray,
    momentum_buffer: jnp.ndarray,
    *,
    learning_rate: float,
    dual_lr: float = 1e-2,
    dual_beta: float = 0.9,
    momentum: float = 0.9,
    weight_decay: float = 0.01,
    scale: str = "none",
    msign_steps: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    radius = scale_radius(W, scale)
    next_momentum = momentum * momentum_buffer + G
    unit_W = W / radius
    tangent, lambda_next, velocity_next, residual = online_dual_ascent_step(
        unit_W,
        next_momentum,
        dual_lambda,
        dual_velocity,
        alpha=dual_lr,
        beta=dual_beta,
        msign_steps=msign_steps,
    )

    update = radius * tangent + weight_decay * W
    next_W = W - learning_rate * update
    next_W = retract_to_scaled_stiefel(next_W, scale)
    return next_W, lambda_next, velocity_next, next_momentum, tangent, residual


def _init_dual_leaf(param: jnp.ndarray) -> jnp.ndarray:
    if param.ndim >= 2 and min(param.shape) > 1:
        tall_param, _ = _orient_tall(param)
        k = tall_param.shape[-1]
        return jnp.zeros((k, k), dtype=param.dtype)
    return jnp.zeros((), dtype=param.dtype)


def online_stiefel(
    learning_rate: float = 0.02,
    dual_lr: float = 1e-2,
    dual_beta: float = 0.9,
    momentum: float = 0.9,
    weight_decay: float = 0.01,
    scale: str = "none",
    msign_steps: int = 10,
    min_ndim: int = 2,
) -> base.GradientTransformation:
    """
    Online Manifold Muon/Stiefel optimizer.

    Matrix leaves use online dual ascent and Stiefel retraction. Non-matrix
    leaves fall back to SGD, matching the local custom optimizer convention.
    """
    if scale not in {"none", "ratio"}:
        raise ValueError(f"Unsupported online Stiefel scale={scale!r}")

    def init_fn(params: base.Params) -> OnlineStiefelState:
        dual = jax.tree_util.tree_map(_init_dual_leaf, params)
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        scalar_zeros = jax.tree_util.tree_map(
            lambda p: jnp.zeros((), dtype=p.dtype),
            params,
        )
        return OnlineStiefelState(
            step=jnp.array(0),
            dual_lambda=dual,
            dual_velocity=dual,
            momentum=zeros,
            last_tangent=zeros,
            last_constraint_residual=scalar_zeros,
        )

    def update_fn(
        updates: base.Updates,
        state: OnlineStiefelState,
        params: Optional[base.Params] = None,
    ) -> Tuple[base.Updates, OnlineStiefelState]:
        if params is None:
            raise ValueError("Online Stiefel requires params to be passed to update()")

        def update_param(g, p, dual_lambda, dual_velocity, momentum_buffer):
            if p.ndim >= min_ndim and min(p.shape) > 1:
                next_W, lambda_next, velocity_next, next_momentum, tangent, residual = (
                    online_stiefel_update(
                        W=p,
                        G=g,
                        dual_lambda=dual_lambda,
                        dual_velocity=dual_velocity,
                        momentum_buffer=momentum_buffer,
                        learning_rate=learning_rate,
                        dual_lr=dual_lr,
                        dual_beta=dual_beta,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        scale=scale,
                        msign_steps=msign_steps,
                    )
                )
                return (
                    next_W - p,
                    lambda_next,
                    velocity_next,
                    next_momentum,
                    tangent,
                    jnp.linalg.norm(residual, ord="fro"),
                )

            return (
                -learning_rate * g,
                dual_lambda,
                dual_velocity,
                momentum_buffer,
                jnp.zeros_like(p),
                jnp.zeros((), dtype=p.dtype),
            )

        mapped = jax.tree_util.tree_map(
            update_param,
            updates,
            params,
            state.dual_lambda,
            state.dual_velocity,
            state.momentum,
        )
        is_update_record = lambda x: isinstance(x, tuple) and len(x) == 6
        new_updates = jax.tree_util.tree_map(lambda x: x[0], mapped, is_leaf=is_update_record)
        new_lambda = jax.tree_util.tree_map(lambda x: x[1], mapped, is_leaf=is_update_record)
        new_velocity = jax.tree_util.tree_map(lambda x: x[2], mapped, is_leaf=is_update_record)
        new_momentum = jax.tree_util.tree_map(lambda x: x[3], mapped, is_leaf=is_update_record)
        new_tangent = jax.tree_util.tree_map(lambda x: x[4], mapped, is_leaf=is_update_record)
        new_residual = jax.tree_util.tree_map(lambda x: x[5], mapped, is_leaf=is_update_record)
        new_state = OnlineStiefelState(
            step=state.step + 1,
            dual_lambda=new_lambda,
            dual_velocity=new_velocity,
            momentum=new_momentum,
            last_tangent=new_tangent,
            last_constraint_residual=new_residual,
        )
        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
