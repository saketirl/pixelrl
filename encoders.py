"""Shared model components for PPO pixel experiments."""

from dataclasses import dataclass
from typing import Optional

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp
import numpy as np


@dataclass
class ViTConfig:
    """Configuration for the ViT encoder."""

    patch_size: int = 14
    hidden_size: int = 192
    mlp_dim: int = 768
    num_heads: int = 3
    num_layers: int = 4
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_cls_token: bool = False
    use_conv_stem: bool = True
    conv_stem_channels: int = 64
    conv_stem_kernel: int = 3
    apply_output_tanh: bool = False


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to token embeddings."""

    posemb_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, inputs):
        if inputs.ndim != 3:
            raise ValueError(f"Expected rank-3 token input, got shape={inputs.shape}")
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pos_embedding = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        return inputs + pos_embedding


class MlpBlock(nn.Module):
    """Transformer MLP block from ViT."""

    mlp_dim: int
    out_dim: Optional[int] = None
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs):
        out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            self.mlp_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)
        x = nn.Dense(
            out_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)
        return x


class Encoder1DBlock(nn.Module):
    """Transformer encoder block used by ViT."""

    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            dropout_rate=self.attention_dropout_rate,
        )(x, deterministic=True)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)
        x = x + inputs

        y = nn.LayerNorm()(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
        )(y)
        return x + y


class Encoder(nn.Module):
    """Stack of ViT transformer blocks."""

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = AddPositionEmbs(name="posembed_input")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)

        for layer_idx in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{layer_idx}",
            )(x)
        return nn.LayerNorm(name="encoder_norm")(x)


class ViTEncoder(nn.Module):
    """ViT encoder adapted from google-research/vision_transformer."""

    tanh_scale: float = 0.5
    patch_size: int = 14
    hidden_size: int = 192
    mlp_dim: int = 768
    num_heads: int = 3
    num_layers: int = 4
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_cls_token: bool = False
    use_conv_stem: bool = True
    conv_stem_channels: int = 64
    conv_stem_kernel: int = 3
    apply_output_tanh: bool = False

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected NHWC image input, got shape={x.shape}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        x = x.astype(jnp.float32) / 255.0
        if self.use_conv_stem:
            x = nn.Conv(
                features=self.conv_stem_channels,
                kernel_size=(self.conv_stem_kernel, self.conv_stem_kernel),
                strides=(2, 2),
                padding="SAME",
                kernel_init=nn.initializers.xavier_uniform(),
                name="stem_conv_0",
            )(x)
            x = nn.LayerNorm(name="stem_ln_0")(x)
            x = nn.relu(x)

            x = nn.Conv(
                features=self.conv_stem_channels,
                kernel_size=(self.conv_stem_kernel, self.conv_stem_kernel),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nn.initializers.xavier_uniform(),
                name="stem_conv_1",
            )(x)
            x = nn.LayerNorm(name="stem_ln_1")(x)
            x = nn.relu(x)

        height, width = x.shape[1], x.shape[2]
        if (height % self.patch_size != 0) or (width % self.patch_size != 0):
            raise ValueError(
                f"Input spatial size ({height}, {width}) is not divisible by "
                f"patch_size={self.patch_size}"
            )

        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
            name="embedding",
        )(x)
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1, self.hidden_size))

        if self.use_cls_token:
            cls = self.param("cls", nn.initializers.zeros, (1, 1, self.hidden_size))
            cls = jnp.tile(cls, (batch_size, 1, 1))
            x = jnp.concatenate([cls, x], axis=1)

        x = Encoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name="Transformer",
        )(x)
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = jnp.mean(x, axis=1)

        x = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="proj_512",
        )(x)
        x = nn.LayerNorm(name="proj_norm")(x)
        if self.apply_output_tanh:
            x = nn.tanh(self.tanh_scale * x)
        return x


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


def build_encoder(
    encoder_type: str,
    tanh_scale: float = 0.5,
    vit_config: Optional[ViTConfig] = None,
) -> nn.Module:
    """Build an encoder module by name."""
    kind = encoder_type.lower()
    if kind == "cnn":
        return CNNEncoder(tanh_scale=tanh_scale)
    if kind == "mlp":
        return MLPEncoder(tanh_scale=tanh_scale)
    if kind == "vit":
        cfg = vit_config or ViTConfig()
        return ViTEncoder(
            tanh_scale=tanh_scale,
            patch_size=cfg.patch_size,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout_rate=cfg.dropout_rate,
            attention_dropout_rate=cfg.attention_dropout_rate,
            use_cls_token=cfg.use_cls_token,
            use_conv_stem=cfg.use_conv_stem,
            conv_stem_channels=cfg.conv_stem_channels,
            conv_stem_kernel=cfg.conv_stem_kernel,
            apply_output_tanh=cfg.apply_output_tanh,
        )
    raise ValueError(
        f"Unknown encoder_type='{encoder_type}'. Expected one of: ['cnn', 'mlp', 'vit']"
    )
