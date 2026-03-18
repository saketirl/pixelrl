"""JAX implementation of SIGReg for PPO/Flax training code."""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


def sigreg_loss(
    embeddings: jax.Array,
    rng: jax.Array,
    num_slices: int = 16,
    num_t: int = 8,
    t_max: float = 5.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the SIGReg loss on a batch of embeddings.

    Args:
        embeddings: Array of shape ``(batch, embedding_dim)``.
        rng: JAX PRNG key used to sample random projection directions.
        num_slices: Number of random 1D projections.
        num_t: Number of characteristic-function evaluation points.
        t_max: Maximum absolute value in the symmetric ``t`` grid.

    Returns:
        Tuple ``(loss, re_loss, im_loss)`` averaged over the ``t`` grid.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (batch, dim), got {embeddings.shape}")
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if num_t <= 0:
        raise ValueError(f"num_t must be positive, got {num_t}")

    embedding_dim = embeddings.shape[-1]
    t_grid = jnp.linspace(-t_max, t_max, num=num_t, dtype=embeddings.dtype)

    directions = jax.random.normal(rng, (num_slices, embedding_dim), dtype=embeddings.dtype)
    directions = directions / (jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
    projections = embeddings @ directions.T
    projections = projections.T  # (num_slices, batch)

    phases = t_grid[:, None, None] * projections[None, :, :]
    cos_terms = jnp.cos(phases)
    sin_terms = jnp.sin(phases)

    re = jnp.mean(cos_terms, axis=2)
    im = jnp.mean(sin_terms, axis=2)
    target = jnp.exp(-0.5 * jnp.square(t_grid))[:, None]

    re_loss = jnp.mean(jnp.square(re - target), axis=1)
    im_loss = jnp.mean(jnp.square(im), axis=1)
    total_loss = re_loss + im_loss

    return jnp.mean(total_loss), jnp.mean(re_loss), jnp.mean(im_loss)


def sigreg_loss_masked(
    embeddings: jax.Array,
    mask: jax.Array,
    rng: jax.Array,
    num_slices: int = 16,
    num_t: int = 8,
    t_max: float = 5.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Masked SIGReg loss used for innovation residuals."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (batch, dim), got {embeddings.shape}")
    if mask.ndim != 1 or mask.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Expected mask with shape ({embeddings.shape[0]},), got {mask.shape}"
        )
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if num_t <= 0:
        raise ValueError(f"num_t must be positive, got {num_t}")

    weights = mask.astype(embeddings.dtype)
    weight_sum = jnp.sum(weights)
    safe_weight_sum = weight_sum + 1e-12
    embedding_dim = embeddings.shape[-1]
    t_grid = jnp.linspace(-t_max, t_max, num=num_t, dtype=embeddings.dtype)

    directions = jax.random.normal(rng, (num_slices, embedding_dim), dtype=embeddings.dtype)
    directions = directions / (jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
    projections = embeddings @ directions.T
    projections = projections.T

    phases = t_grid[:, None, None] * projections[None, :, :]
    cos_terms = jnp.cos(phases)
    sin_terms = jnp.sin(phases)
    weights_broadcast = weights[None, None, :]

    re = jnp.sum(cos_terms * weights_broadcast, axis=2) / safe_weight_sum
    im = jnp.sum(sin_terms * weights_broadcast, axis=2) / safe_weight_sum
    target = jnp.exp(-0.5 * jnp.square(t_grid))[:, None]

    re_loss = jnp.mean(jnp.square(re - target), axis=1)
    im_loss = jnp.mean(jnp.square(im), axis=1)
    total_loss = re_loss + im_loss

    zero = jnp.asarray(0.0, dtype=embeddings.dtype)
    valid = weight_sum > 0.5
    return (
        jnp.where(valid, jnp.mean(total_loss), zero),
        jnp.where(valid, jnp.mean(re_loss), zero),
        jnp.where(valid, jnp.mean(im_loss), zero),
    )


@dataclass(frozen=True)
class SIGReg:
    """Stateless SIGReg wrapper for JAX/Flax training loops."""

    embedding_dim: int
    num_slices: int = 16
    num_t: int = 8
    t_max: float = 5.0

    def __call__(self, embeddings: jax.Array, rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {embeddings.shape[-1]}"
            )
        return sigreg_loss(
            embeddings=embeddings,
            rng=rng,
            num_slices=self.num_slices,
            num_t=self.num_t,
            t_max=self.t_max,
        )
