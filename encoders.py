"""Shared model components for PPO pixel experiments."""

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class CNNEncoder(nn.Module):
    """CNN encoder for pixel observations with LayerNorm for stability."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0

        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(self.tanh_scale * x)
        return x


class MLPEncoder(nn.Module):
    """MLP encoder baseline for flattened pixel observations."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        x = x.reshape((x.shape[0], -1))

        for _ in range(3):
            x = nn.Dense(1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)

        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(self.tanh_scale * x)
        return x


def build_encoder(encoder_type: str, tanh_scale: float = 0.5) -> nn.Module:
    """Build an encoder module by name."""
    kind = encoder_type.lower()
    if kind == "cnn":
        return CNNEncoder(tanh_scale=tanh_scale)
    if kind == "mlp":
        return MLPEncoder(tanh_scale=tanh_scale)
    raise ValueError(f"Unknown encoder_type='{encoder_type}'. Expected one of: ['cnn', 'mlp']")
