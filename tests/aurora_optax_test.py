import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pytest

jax.config.update("jax_platforms", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizers.aurora_optax import aurora, aurora_update


def _row_sq(W):
    return jnp.sum(W**2, axis=-1)


def test_aurora_update_returns_finite_matrix_update_and_momentum():
    W = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
    G = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    momentum = jnp.zeros_like(W)

    update, new_momentum = aurora_update(W, G, momentum)

    assert update.shape == W.shape
    assert new_momentum.shape == W.shape
    assert jnp.all(jnp.isfinite(update))
    assert jnp.allclose(new_momentum, 0.05 * G)


def test_aurora_update_balances_tall_matrix_row_energy():
    W = jax.random.normal(jax.random.PRNGKey(0), (12, 4))
    G = jax.random.normal(jax.random.PRNGKey(1), (12, 4))
    momentum = jnp.zeros_like(W)

    update, _ = aurora_update(
        W,
        G,
        momentum,
        learning_rate=1.0,
        weight_decay=0.0,
        pp_iterations=2,
    )

    direction = -update / jnp.sqrt(W.shape[0] / W.shape[1])
    target_row_sq = W.shape[1] / W.shape[0]
    assert jnp.std(_row_sq(direction)) < 0.15
    assert jnp.allclose(jnp.mean(_row_sq(direction)), target_row_sq, atol=0.1)


def test_aurora_handles_wide_matrices():
    W = jax.random.normal(jax.random.PRNGKey(0), (4, 12))
    G = jax.random.normal(jax.random.PRNGKey(1), (4, 12))
    momentum = jnp.zeros_like(W)

    update, new_momentum = aurora_update(W, G, momentum)

    assert update.shape == W.shape
    assert new_momentum.shape == W.shape
    assert jnp.all(jnp.isfinite(update))


def test_aurora_optax_transform_uses_adam_fallback_for_bias():
    params = {
        "kernel": jax.random.normal(jax.random.PRNGKey(0), (8, 3)),
        "bias": jnp.ones((3,)),
    }
    grads = {
        "kernel": jax.random.normal(jax.random.PRNGKey(1), (8, 3)),
        "bias": jnp.full((3,), 2.0),
    }
    tx = aurora(learning_rate=0.05)
    state = tx.init(params)

    updates, new_state = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert new_state.step == state.step + 1
    assert updates["kernel"].shape == params["kernel"].shape
    assert updates["bias"].shape == params["bias"].shape
    assert jnp.allclose(updates["bias"], jnp.full((3,), -0.05), atol=1e-6)
    assert jnp.all(jnp.isfinite(new_params["kernel"]))
    assert jnp.allclose(new_state.momentum["kernel"], 0.05 * grads["kernel"])
    assert jnp.allclose(new_state.adam_mu["bias"], 0.1 * grads["bias"])


def test_aurora_optax_transform_requires_params():
    params = {"kernel": jax.random.normal(jax.random.PRNGKey(0), (8, 3))}
    grads = {"kernel": jnp.ones((8, 3))}
    tx = aurora()
    state = tx.init(params)

    with pytest.raises(ValueError, match="requires params"):
        tx.update(grads, state)
