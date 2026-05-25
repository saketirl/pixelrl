"""Custom Optax optimizers used by the PPO trainers."""

from optimizers.aurora_optax import aurora, aurora_update
from optimizers.manifold_stiefel_admm_optax import (
    manifold_stiefel_admm,
    manifold_stiefel_admm_update,
)
from optimizers.manifold_stiefel_optax import (
    manifold_stiefel,
    manifold_stiefel_per_head,
    manifold_stiefel_update,
    manifold_stiefel_update_per_head,
    msign,
)
from optimizers.online_stiefel_optax import (
    OnlineStiefelState,
    matrix_sign,
    online_stiefel,
    online_stiefel_update,
)

__all__ = [
    "OnlineStiefelState",
    "aurora",
    "aurora_update",
    "matrix_sign",
    "manifold_stiefel",
    "manifold_stiefel_admm",
    "manifold_stiefel_admm_update",
    "manifold_stiefel_per_head",
    "manifold_stiefel_update",
    "manifold_stiefel_update_per_head",
    "msign",
    "online_stiefel",
    "online_stiefel_update",
]
