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

from optimizers.manifold_stiefel_admm_optax import (
    manifold_stiefel_admm,
    manifold_stiefel_admm_update,
)
from optimizers.manifold_stiefel_optax import manifold_stiefel


def _orthonormal_matrix(key, shape):
    rows, cols = shape
    if rows >= cols:
        q, _ = jnp.linalg.qr(jax.random.normal(key, shape))
        return q[:, :cols]

    q, _ = jnp.linalg.qr(jax.random.normal(key, (cols, rows)))
    return q[:, :rows].T


def _assert_stiefel_close(W, atol=5e-3):
    if W.shape[0] >= W.shape[1]:
        gram = W.T @ W
    else:
        gram = W @ W.T
    eye = jnp.eye(gram.shape[0], dtype=W.dtype)
    assert jnp.allclose(gram, eye, atol=atol)


@pytest.mark.parametrize("shape", [(8, 3), (3, 8)])
def test_admm_update_preserves_stiefel_constraint(shape):
    w_key, g_key = jax.random.split(jax.random.PRNGKey(sum(shape)))
    W = _orthonormal_matrix(w_key, shape)
    G = jax.random.normal(g_key, shape)

    new_W = manifold_stiefel_admm_update(W, G)

    assert new_W.shape == W.shape
    _assert_stiefel_close(new_W)


def test_admm_optax_transform_returns_updates_for_apply_updates():
    params = {
        "kernel": _orthonormal_matrix(jax.random.PRNGKey(0), (8, 3)),
        "bias": jnp.ones((3,)),
    }
    grads = {
        "kernel": jax.random.normal(jax.random.PRNGKey(1), (8, 3)),
        "bias": jnp.full((3,), 2.0),
    }
    tx = manifold_stiefel_admm(learning_rate=0.02)
    state = tx.init(params)

    updates, new_state = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert new_state.step == state.step + 1
    assert updates["kernel"].shape == params["kernel"].shape
    assert jnp.allclose(updates["bias"], -0.02 * grads["bias"])
    _assert_stiefel_close(new_params["kernel"])


def test_admm_optax_transform_requires_params():
    params = {"kernel": _orthonormal_matrix(jax.random.PRNGKey(0), (8, 3))}
    grads = {"kernel": jnp.ones((8, 3))}
    tx = manifold_stiefel_admm()
    state = tx.init(params)

    with pytest.raises(ValueError, match="requires params"):
        tx.update(grads, state)


def test_admm_matches_existing_transform_interface_and_invariants():
    params = {
        "kernel": _orthonormal_matrix(jax.random.PRNGKey(0), (8, 3)),
        "bias": jnp.ones((3,)),
    }
    grads = {
        "kernel": jax.random.normal(jax.random.PRNGKey(1), (8, 3)),
        "bias": jnp.full((3,), 2.0),
    }
    admm_tx = manifold_stiefel_admm(learning_rate=0.02, msign_steps=5)
    dual_tx = manifold_stiefel(
        learning_rate=0.02,
        dual_lr=0.01,
        dual_steps=5,
        msign_steps=5,
    )

    admm_updates, admm_state = admm_tx.update(grads, admm_tx.init(params), params)
    dual_updates, dual_state = dual_tx.update(grads, dual_tx.init(params), params)
    admm_params = optax.apply_updates(params, admm_updates)
    dual_params = optax.apply_updates(params, dual_updates)

    assert admm_state.step == dual_state.step
    assert jax.tree_util.tree_structure(admm_updates) == jax.tree_util.tree_structure(
        dual_updates
    )
    assert admm_updates["kernel"].shape == dual_updates["kernel"].shape
    assert jnp.allclose(admm_updates["bias"], dual_updates["bias"])
    _assert_stiefel_close(admm_params["kernel"])
    _assert_stiefel_close(dual_params["kernel"])
