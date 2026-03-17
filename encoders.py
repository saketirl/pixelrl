"""Shared model components for PPO pixel experiments."""

from dataclasses import dataclass
from typing import Optional

import jax
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
    stem_c1: int = 24
    stem_c2: int = 48
    stem_c3: int = 96
    stem_c4: int = 192
    drq_stem_channels: int = 32
    drq_token_downsample: int = 1
    drq_apply_output_tanh: bool = False
    proj_dim: int = 512
    scott_use_swiglu: bool = True
    scott_num_register_tokens: int = 0
    scott_dropout_rate: float = 0.0
    scott_attention_dropout_rate: float = 0.0
    scott_stochastic_depth_rate: float = 0.0


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
    proj_dim: int = 512

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

        # Standard ViT patchify + linear projection in one conv op.
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
            self.proj_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="proj_head",
        )(x)
        x = nn.LayerNorm(name="proj_norm")(x)
        if self.apply_output_tanh:
            x = nn.tanh(self.tanh_scale * x)
        return x


class HybridViTEncoder(nn.Module):
    """Hybrid ViT encoder: 4-stage stride-2 conv stem → 1×1 proj → Transformer.

    Spatial resolution for 84×84 input (SAME padding, stride-2 ×4):
    84 → 42 → 21 → 11 → 6  ⟹  6×6 = 36 tokens (+ 1 CLS = 37).

    https://proceedings.neurips.cc/paper/2021/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf
    Uses layernorm instead of batchnorm
    """

    tanh_scale: float = 0.5
    hidden_size: int = 192
    mlp_dim: int = 576
    num_heads: int = 3
    num_layers: int = 12
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    apply_output_tanh: bool = False
    stem_c1: int = 24
    stem_c2: int = 48
    stem_c3: int = 96
    stem_c4: int = 192
    proj_dim: int = 512

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected NHWC image input, got shape={x.shape}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        x = x.astype(jnp.float32) / 255.0

        # 4-stage conv stem: each stage halves spatial resolution
        for i, channels in enumerate(
            [self.stem_c1, self.stem_c2, self.stem_c3, self.stem_c4]
        ):
            x = nn.Conv(
                features=channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                kernel_init=nn.initializers.xavier_uniform(),
                name=f"stem_conv_{i}",
            )(x)
            x = nn.LayerNorm(name=f"stem_ln_{i}")(x)
            x = nn.relu(x)

        # 1×1 projection to transformer hidden dim (no norm)
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
            name="proj_conv",
        )(x)

        # Reshape (B, H', W', D) → (B, N, D) token sequence
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1, self.hidden_size))

        # Prepend learnable CLS token
        cls = self.param("cls", nn.initializers.zeros, (1, 1, self.hidden_size))
        cls = jnp.tile(cls, (batch_size, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)

        # Transformer encoder (includes AddPositionEmbs, Dropout, L-1 blocks, LayerNorm)
        x = Encoder(
            num_layers=self.num_layers - 1,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name="Transformer",
        )(x)

        # Extract CLS token
        x = x[:, 0]

        # RL projection head
        x = nn.Dense(
            self.proj_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="proj_head",
        )(x)
        x = nn.LayerNorm(name="proj_norm")(x)
        if self.apply_output_tanh:
            x = nn.tanh(self.tanh_scale * x)
        return x


class DrQViTEncoder(nn.Module):
    """DrQ-style hybrid encoder: shallow VALID conv stem → bridge conv → Transformer.

    Spatial math for 84×84 input (VALID padding):
      84 →(3×3,s=2)→ 41 →(3×3,s=1)→ 39 →(s=1)→ 37 →(s=1)→ 35
      →bridge(3×3,s=2,pad=1): floor((35+2-3)/2)+1 = 18 → 18×18 = 324 tokens

    No LayerNorm in conv stem (pure conv+ReLU like original DrQ/DreamerV3).
    No CLS token — uses Global Average Pool instead.
    """

    tanh_scale: float = 0.5
    hidden_size: int = 192
    mlp_dim: int = 768
    num_heads: int = 6
    num_layers: int = 4
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    drq_stem_channels: int = 32
    token_downsample: int = 1
    apply_output_tanh: bool = False

    @nn.compact
    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected NHWC image input, got shape={x.shape}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.token_downsample <= 0:
            raise ValueError(
                f"token_downsample must be positive, got {self.token_downsample}"
            )
        x = x.astype(jnp.float32) / 255.0

        # 4-layer conv stem: no LayerNorm, VALID padding, uniform channels
        strides = [(2, 2), (1, 1), (1, 1), (1, 1)]
        for i, s in enumerate(strides):
            x = nn.Conv(
                features=self.drq_stem_channels,
                kernel_size=(3, 3),
                strides=s,
                padding="VALID",
                kernel_init=nn.initializers.xavier_uniform(),
                name=f"stem_conv_{i}",
            )(x)
            x = nn.relu(x)

        # Bridge conv: stride-2, pad=1 each side → 18×18 tokens
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            kernel_init=nn.initializers.xavier_uniform(),
            name="bridge_conv",
        )(x)

        # Optional patch-like compression before attention to reduce token count.
        if self.token_downsample > 1:
            height, width = x.shape[1], x.shape[2]
            if (height % self.token_downsample != 0) or (
                width % self.token_downsample != 0
            ):
                raise ValueError(
                    f"DrQ token map ({height}, {width}) is not divisible by "
                    f"token_downsample={self.token_downsample}"
                )
            x = nn.Conv(
                features=self.hidden_size,
                kernel_size=(self.token_downsample, self.token_downsample),
                strides=(self.token_downsample, self.token_downsample),
                padding="VALID",
                kernel_init=nn.initializers.xavier_uniform(),
                name="token_downsample",
            )(x)

        # Reshape (B, H', W', D) → (B, N, D)
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1, self.hidden_size))

        # Transformer encoder (pos emb + dropout + blocks + LayerNorm)
        x = Encoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name="Transformer",
        )(x)

        # Global Average Pool
        x = jnp.mean(x, axis=1)

        # RL projection head
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="proj_head",
        )(x)
        x = nn.LayerNorm(name="proj_norm")(x)
        if self.apply_output_tanh:
            x = nn.tanh(self.tanh_scale * x)
        return x


class CNNEncoder(nn.Module):
    """CNN encoder for pixel observations with LayerNorm for stability."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x, return_pre_tanh: bool = False):
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
        pre_tanh = self.tanh_scale * x
        if return_pre_tanh:
            return pre_tanh
        return nn.tanh(pre_tanh)


class StiefelCNN(nn.Module):
    """CNN encoder with a learnable Stiefel projection head.

    The convolutional trunk is shared with CNNEncoder, then a linear projection
    maps the 512-D representation to a low-dimensional latent with orthogonal
    columns.
    """

    tanh_scale: float = 0.5
    latent_dim: int = 32

    @nn.compact
    def __call__(self, x, return_preproj: bool = False):
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

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

        h = x.reshape((x.shape[0], -1))
        h = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = nn.LayerNorm()(h)
        h = nn.tanh(self.tanh_scale * h)

        z = nn.Dense(
            self.latent_dim,
            use_bias=False,
            kernel_init=orthogonal(1.0),
            name="stiefel_projection",
        )(h)

        if return_preproj:
            return h, z
        return z


class E2CCNNEncoder(nn.Module):
    """CNN encoder with VAE bottleneck for E2C auxiliary loss.

    Call with sample=False (default): returns 512-dim feature (identical interface to CNNEncoder).
    Call with sample=True, key=key: returns (z_sampled, z_mean, z_logvar) with bottleneck_dim-dim z.
    """

    tanh_scale: float = 0.5
    bottleneck_dim: int = 256

    @nn.compact
    def __call__(self, x, sample: bool = False, key=None):
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
        feat = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        feat = nn.LayerNorm()(feat)
        feat = nn.tanh(self.tanh_scale * feat)

        # Always create VAE bottleneck params so they are initialized regardless of sample mode
        z_mean = nn.Dense(
            self.bottleneck_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="z_mean_head",
        )(feat)
        z_logvar = jnp.clip(
            nn.Dense(
                self.bottleneck_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="z_logvar_head",
            )(feat),
            -10.0,
            2.0,
        )

        if not sample:
            return z_mean

        eps = jax.random.normal(key, z_mean.shape)
        z_sampled = z_mean + jnp.exp(0.5 * z_logvar) * eps
        return z_sampled, z_mean, z_logvar


class E2CDecoder(nn.Module):
    """Convolutional decoder for E2C: maps latent z → reconstructed 84×84×3 RGB frame.

    Architecture is the mirror image of the CNN encoder trunk.
    """

    @nn.compact
    def __call__(self, z):
        # z: (B, bottleneck_dim)
        x = nn.Dense(
            64 * 7 * 7,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        x = x.reshape((x.shape[0], 7, 7, 64))

        # Inverse of Conv(64, 3x3, s1, VALID): 7→9
        x = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Inverse of Conv(64, 4x4, s2, VALID): 9→20
        x = nn.ConvTranspose(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Inverse of Conv(32, 8x8, s4, VALID): 20→84
        x = nn.ConvTranspose(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # 1×1 conv to 3 output channels + sigmoid
        x = nn.Conv(
            features=3,
            kernel_size=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return nn.sigmoid(x)


class E2CTransition(nn.Module):
    """Locally-linear transition model for E2C.

    Models z_{t+1} = A(z)*z + B(z)*u + o(z) where:
    - A uses a rank-1 perturbation of the identity: A*z = z + v1*(v2^T z)
    - B is an input-dependent matrix from a dense layer
    - o is an input-dependent affine offset
    Returns predicted (z_next_mean, z_next_logvar).
    """

    latent_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, z_t, action_u):
        # z_t: (B, latent_dim),  action_u: (B, action_dim)
        h = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z_t)
        h = nn.relu(h)

        # Low-rank A: A*z = z + v1*(v2^T z)
        v1 = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        v2 = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        v2_dot_z = jnp.sum(v2 * z_t, axis=-1, keepdims=True)  # (B, 1)
        Az = z_t + v1 * v2_dot_z  # (B, latent_dim)

        # Input-dependent B matrix
        B_flat = nn.Dense(
            self.latent_dim * self.action_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(h)
        B_mat = B_flat.reshape((-1, self.latent_dim, self.action_dim))
        Bu = jnp.einsum("bla,ba->bl", B_mat, action_u)

        # Affine offset
        o = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)

        z_next_mean = Az + Bu + o
        z_next_logvar = jnp.clip(
            nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h),
            -10.0,
            2.0,
        )
        return z_next_mean, z_next_logvar



class LinearLatentTransition(nn.Module):
    """Deterministic linear latent transition model.

    Predicts z_{t+1} = F z_t + G a_t + b with explicit F and G matrices
    for direct inspection and condition diagnostics.
    """

    latent_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, z_t, action_u):
        f_matrix = self.param(
            "F",
            orthogonal(1.0),
            (self.latent_dim, self.latent_dim),
        )
        g_matrix = self.param(
            "G",
            nn.initializers.orthogonal(),
            (self.latent_dim, self.action_dim),
        )
        bias = self.param(
            "b",
            constant(0.0),
            (self.latent_dim,),
        )
        return jnp.dot(z_t, f_matrix.T) + jnp.dot(action_u, g_matrix.T) + bias

class BottleneckProjection(nn.Module):
    """Deterministic latent bottleneck applied on top of the CNN feature."""

    latent_dim: int

    @nn.compact
    def __call__(self, h_t):
        z_t = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(h_t)
        z_t = nn.LayerNorm()(z_t)
        return z_t


class LatentTransition(nn.Module):
    """Deterministic one-step latent dynamics model."""

    latent_dim: int
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z_t, action_u):
        x = jnp.concatenate([z_t, action_u], axis=-1)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.swish(x)
        z_hat_tp1 = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return z_hat_tp1


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


# ---------------------------------------------------------------------------
# SCOTT encoder helpers: sincos position embeddings
# ---------------------------------------------------------------------------

def _sincos_1d_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Returns (N, embed_dim) sincos positional embedding."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("n,d->nd", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (N, embed_dim)


def sincos_2d_pos_embed(
    embed_dim: int, grid_size: int, num_register_tokens: int = 0
) -> np.ndarray:
    """Returns (num_register_tokens + grid_size*grid_size, embed_dim) sincos pos embeddings.

    Register token positions get all-zero embeddings.
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = _sincos_1d_pos_embed(embed_dim // 2, grid_h.reshape(-1))
    emb_w = _sincos_1d_pos_embed(embed_dim // 2, grid_w.reshape(-1))
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (N, embed_dim)
    if num_register_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros((num_register_tokens, embed_dim)), pos_embed], axis=0
        )
    return pos_embed


def _trunc_normal(stddev: float = 0.02):
    return nn.initializers.truncated_normal(stddev=stddev)


def _scaled_trunc_normal(layer_id: int, stddev: float = 0.02):
    scale = np.sqrt(2.0 * layer_id)
    return nn.initializers.truncated_normal(stddev=stddev / scale)


# ---------------------------------------------------------------------------
# SCOTT encoder modules
# ---------------------------------------------------------------------------

class MaxBlurPool2D(nn.Module):
    """MaxPool followed by anti-aliased Gaussian blur (Pascal triangle kernel).

    Implements stride-2 downsampling with anti-aliasing as in SCOTT.
    Input/output are NHWC.
    """

    kernel_size: int = 3
    stride: int = 2
    max_pool_size: int = 2

    @nn.compact
    def __call__(self, x):
        # Upstream SCOTT uses a valid 2x2 max-pool before the blur+stride op.
        x = nn.max_pool(
            x,
            window_shape=(self.max_pool_size, self.max_pool_size),
            strides=(1, 1),
            padding="VALID",
        )
        C = x.shape[-1]
        blur = jnp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=jnp.float32) / 16.0
        kernel = jnp.tile(blur[None, None, :, :], (C, 1, 1, 1))
        x_nchw = x.transpose(0, 3, 1, 2)
        x_nchw = jax.lax.conv_general_dilated(
            x_nchw,
            kernel,
            window_strides=(self.stride, self.stride),
            padding="SAME",
            feature_group_count=C,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return x_nchw.transpose(0, 2, 3, 1)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network from SCOTT."""

    d_model: int
    output_layer_id: int

    @nn.compact
    def __call__(self, x):
        hidden_dim = int(128 * round((8 / 3 * self.d_model) / 128))
        x12 = nn.Dense(
            2 * hidden_dim,
            use_bias=False,
            kernel_init=_trunc_normal(),
            name="w12",
        )(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        return nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=_scaled_trunc_normal(self.output_layer_id),
            name="last_linear",
        )(hidden)


class ScottDropPath(nn.Module):
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        if deterministic or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = jax.random.bernoulli(self.make_rng("dropout"), keep_prob, shape=shape)
        return jnp.asarray(x / keep_prob * keep, x.dtype)


class ScottAttention(nn.Module):
    dim: int
    num_heads: int
    attention_dropout_rate: float
    projection_dropout_rate: float
    output_layer_id: int

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        head_dim = self.dim // self.num_heads
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
            )

        qkv = nn.Dense(
            3 * self.dim,
            use_bias=False,
            kernel_init=_trunc_normal(),
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)

        q = q * (head_dim ** -0.5)
        attn = jnp.einsum("bhid,bhjd->bhij", q, k)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attention_dropout_rate)(
            attn, deterministic=deterministic
        )

        y = jnp.einsum("bhij,bhjd->bhid", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.dim)
        y = nn.Dense(
            self.dim,
            use_bias=False,
            kernel_init=_scaled_trunc_normal(self.output_layer_id),
            name="proj",
        )(y)
        y = nn.Dropout(rate=self.projection_dropout_rate)(y, deterministic=deterministic)
        return y


class ScottMlpFFN(nn.Module):
    d_model: int
    mlp_dim: int
    dropout_rate: float
    output_layer_id: int

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = nn.Dense(
            self.mlp_dim,
            use_bias=False,
            kernel_init=_trunc_normal(),
            name="linear1",
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=_scaled_trunc_normal(self.output_layer_id),
            name="last_linear",
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class ScottBlock(nn.Module):
    """Upstream-style SCOTT transformer block."""

    d_model: int
    num_heads: int
    ffn_layer: str = "swiglu"
    mlp_dim: int = 0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    output_layer_id: int = 1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        y = nn.LayerNorm()(x)
        y = ScottAttention(
            dim=self.d_model,
            num_heads=self.num_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            projection_dropout_rate=self.dropout_rate,
            output_layer_id=self.output_layer_id,
            name="self_attn",
        )(y, deterministic=deterministic)
        x = x + ScottDropPath(self.drop_path_rate)(
            y, deterministic=deterministic
        )

        y = nn.LayerNorm()(x)
        if self.ffn_layer == "swiglu":
            y = SwiGLUFFN(
                d_model=self.d_model,
                output_layer_id=self.output_layer_id,
                name="ffn_layer",
            )(y)
        elif self.ffn_layer == "mlp":
            mlp_dim = self.mlp_dim if self.mlp_dim > 0 else 4 * self.d_model
            y = ScottMlpFFN(
                d_model=self.d_model,
                mlp_dim=mlp_dim,
                dropout_rate=self.dropout_rate,
                output_layer_id=self.output_layer_id,
                name="ffn_layer",
            )(y, deterministic=deterministic)
        else:
            raise ValueError(f"Unsupported SCOTT ffn_layer='{self.ffn_layer}'")

        x = x + ScottDropPath(self.drop_path_rate)(
            y, deterministic=deterministic
        )
        return x


class SparseCNNTokenizer(nn.Module):
    """Dense tokenizer with upstream-style active-mask propagation semantics."""

    embed_dim: int = 192

    @staticmethod
    def _grid_hw(height: int, width: int):
        def _dim(size):
            size = (size + 1) // 2
            size = size // 2
            size = (size + 1) // 2
            size = size // 2
            return size

        return _dim(height), _dim(width)

    def _token_mask(self, masks, batch_size: int, height: int, width: int):
        if masks is None:
            return None

        grid_h, grid_w = self._grid_hw(height, width)
        expected = grid_h * grid_w
        if masks.shape != (batch_size, expected):
            raise ValueError(
                f"Expected masks with shape ({batch_size}, {expected}), got {masks.shape}"
            )
        return masks.reshape(batch_size, grid_h, grid_w, 1)

    @staticmethod
    def _resize_active_mask(token_mask, target_h: int, target_w: int, dtype):
        if token_mask is None:
            return None

        grid_h, grid_w = token_mask.shape[1], token_mask.shape[2]
        row_idx = jnp.minimum(
            jnp.ceil((jnp.arange(target_h) + 1) * grid_h / target_h).astype(jnp.int32) - 1,
            grid_h - 1,
        )
        col_idx = jnp.minimum(
            jnp.ceil((jnp.arange(target_w) + 1) * grid_w / target_w).astype(jnp.int32) - 1,
            grid_w - 1,
        )
        active = jnp.take(token_mask, row_idx, axis=1)
        active = jnp.take(active, col_idx, axis=2)
        return active.astype(dtype)

    def _apply_stage_mask(self, x, token_mask):
        if token_mask is None:
            return x
        active_mask = self._resize_active_mask(
            token_mask, x.shape[1], x.shape[2], x.dtype
        )
        return x * active_mask

    def _apply_input_mask(self, x, token_mask):
        if token_mask is None:
            return x
        image_mask = self._resize_active_mask(
            token_mask, x.shape[1], x.shape[2], x.dtype
        )
        return x * image_mask

    def _apply_masks(self, x, masks):
        if masks is None:
            return x, None

        B, H, W, _ = x.shape
        token_mask = self._token_mask(masks, B, H, W)
        return self._apply_input_mask(x, token_mask), token_mask

    @nn.compact
    def __call__(self, x, masks=None):
        x, token_mask = self._apply_masks(x, masks)
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
            name="conv1",
        )(x)
        x = self._apply_stage_mask(x, token_mask)
        x = nn.relu(x)
        x = MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, name="pool1")(x)
        x = self._apply_stage_mask(x, token_mask)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
            name="conv2",
        )(x)
        x = self._apply_stage_mask(x, token_mask)
        x = nn.relu(x)
        x = MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, name="pool2")(x)
        x = self._apply_stage_mask(x, token_mask)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


class ScottTransformerEncoder(nn.Module):
    """SCOTT transformer with fixed sin-cos positions and upstream-style init."""

    num_layers: int
    num_heads: int
    d_model: int
    mlp_dim: int = 0
    ffn_layer: str = "swiglu"
    num_register_tokens: int = 0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.0
    start_layer_id: int = 0

    @nn.compact
    def __call__(self, x, masks=None, deterministic: bool = True):
        B, N, D = x.shape[0], x.shape[1], x.shape[2]
        grid_size = int(round(np.sqrt(N)))
        if grid_size * grid_size != N:
            raise ValueError(f"SCOTT expects a square token grid, got {N} tokens")

        msk_embed = self.param("msk_embed", _trunc_normal(1e-6), (1, 1, D))
        if masks is not None:
            keep = masks[:, :, None].astype(bool)
            x = jnp.where(keep, x, jnp.broadcast_to(msk_embed, x.shape))

        if self.num_register_tokens > 0:
            reg = self.param(
                "reg_embed",
                nn.initializers.truncated_normal(0.02),
                (1, self.num_register_tokens, D),
            )
            x = jnp.concatenate([jnp.tile(reg, (B, 1, 1)), x], axis=1)

        pos_emb = sincos_2d_pos_embed(D, grid_size, self.num_register_tokens)
        x = x + jnp.array(pos_emb, dtype=jnp.float32)[None, :, :]
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        if self.num_layers > 1:
            dpr = np.linspace(0.0, self.stochastic_depth_rate, self.num_layers)
        else:
            dpr = np.array([self.stochastic_depth_rate])
        for i in range(self.num_layers):
            x = ScottBlock(
                d_model=D,
                num_heads=self.num_heads,
                ffn_layer=self.ffn_layer,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                drop_path_rate=float(dpr[i]),
                output_layer_id=self.start_layer_id + i + 1,
                name=f"block_{i}",
            )(x, deterministic=deterministic)

        x = x[:, self.num_register_tokens:]
        return x


class ScottEncoder(nn.Module):
    """SCOTT encoder: SparseCNNTokenizer stem + ScottTransformerEncoder backbone.

    For RL: call with default args → returns (B, proj_dim) embedding.
    For MIM-JEPA: call with return_tokens=True, masks=keep_masks → returns (B, N, D) tokens.
    """

    tanh_scale: float = 0.5
    hidden_size: int = 192
    mlp_dim: int = 768
    num_heads: int = 3
    num_layers: int = 4
    apply_output_tanh: bool = False
    proj_dim: int = 512
    scott_use_swiglu: bool = True
    scott_num_register_tokens: int = 0
    scott_dropout_rate: float = 0.0
    scott_attention_dropout_rate: float = 0.0
    scott_stochastic_depth_rate: float = 0.0

    @nn.compact
    def __call__(self, x, return_tokens: bool = False, masks=None, deterministic: bool = True):
        if x.ndim != 4:
            raise ValueError(f"Expected NHWC image input, got shape={x.shape}")
        x = x.astype(jnp.float32) / 255.0

        x = SparseCNNTokenizer(embed_dim=self.hidden_size, name="tokenizer")(x, masks=masks)

        x = ScottTransformerEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.hidden_size,
            mlp_dim=self.mlp_dim,
            ffn_layer="swiglu" if self.scott_use_swiglu else "mlp",
            num_register_tokens=self.scott_num_register_tokens,
            dropout_rate=self.scott_dropout_rate,
            attention_dropout_rate=self.scott_attention_dropout_rate,
            stochastic_depth_rate=self.scott_stochastic_depth_rate,
            name="Transformer",
        )(x, masks=masks, deterministic=deterministic)

        if return_tokens:
            return x

        x = jnp.mean(x, axis=1)
        x = nn.Dense(
            self.proj_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="proj_head",
        )(x)
        x = nn.LayerNorm(name="proj_norm")(x)
        if self.apply_output_tanh:
            x = nn.tanh(self.tanh_scale * x)
        return x


class MIMJEPAPredictor(nn.Module):
    """Small transformer predictor for MIM-JEPA auxiliary loss.

    Takes context encoder output (with mask tokens at masked positions) and
    predicts the target encoder embeddings at all positions.
    """

    embed_dim: int = 192
    num_layers: int = 2
    num_heads: int = 3
    num_register_tokens: int = 0
    backbone_depth: int = 0
    use_swiglu: bool = True
    mlp_dim: int = 0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        D = x.shape[-1]
        if D != self.embed_dim:
            x = nn.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_init=_trunc_normal(),
                name="pred_proj",
            )(x)

        x = ScottTransformerEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.embed_dim,
            mlp_dim=self.mlp_dim,
            ffn_layer="swiglu" if self.use_swiglu else "mlp",
            num_register_tokens=self.num_register_tokens,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth_rate=self.stochastic_depth_rate,
            start_layer_id=self.backbone_depth,
            name="pred_transformer",
        )(x, masks=None, deterministic=deterministic)

        x = nn.silu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.embed_dim,
            use_bias=True,
            kernel_init=_trunc_normal(),
        )(x)
        return x


def build_encoder(
    encoder_type: str,
    tanh_scale: float = 0.5,
    vit_config: Optional[ViTConfig] = None,
    e2c_latent_dim: int = 256,
    stiefel_latent_dim: int = 32,
) -> nn.Module:
    """Build an encoder module by name."""
    kind = encoder_type.lower()
    if kind == "cnn":
        return CNNEncoder(tanh_scale=tanh_scale)
    if kind == "stiefel_cnn":
        return StiefelCNN(
            tanh_scale=tanh_scale,
            latent_dim=stiefel_latent_dim,
        )
    if kind == "e2c_cnn":
        return E2CCNNEncoder(tanh_scale=tanh_scale, bottleneck_dim=e2c_latent_dim)
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
            proj_dim=cfg.proj_dim,
        )
    if kind == "hybrid_vit":
        cfg = vit_config or ViTConfig()
        return HybridViTEncoder(
            tanh_scale=tanh_scale,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout_rate=cfg.dropout_rate,
            attention_dropout_rate=cfg.attention_dropout_rate,
            apply_output_tanh=cfg.apply_output_tanh,
            stem_c1=cfg.stem_c1,
            stem_c2=cfg.stem_c2,
            stem_c3=cfg.stem_c3,
            stem_c4=cfg.stem_c4,
            proj_dim=cfg.proj_dim,
        )
    if kind == "drq_vit":
        cfg = vit_config or ViTConfig()
        return DrQViTEncoder(
            tanh_scale=tanh_scale,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout_rate=cfg.dropout_rate,
            attention_dropout_rate=cfg.attention_dropout_rate,
            drq_stem_channels=cfg.drq_stem_channels,
            token_downsample=cfg.drq_token_downsample,
            apply_output_tanh=cfg.drq_apply_output_tanh,
        )
    if kind == "scott":
        cfg = vit_config or ViTConfig()
        return ScottEncoder(
            tanh_scale=tanh_scale,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            apply_output_tanh=cfg.apply_output_tanh,
            proj_dim=cfg.proj_dim,
            scott_use_swiglu=cfg.scott_use_swiglu,
            scott_num_register_tokens=cfg.scott_num_register_tokens,
            scott_dropout_rate=cfg.scott_dropout_rate,
            scott_attention_dropout_rate=cfg.scott_attention_dropout_rate,
            scott_stochastic_depth_rate=cfg.scott_stochastic_depth_rate,
        )
    raise ValueError(
        f"Unknown encoder_type='{encoder_type}'. "
        f"Expected one of: ['cnn', 'stiefel_cnn', 'e2c_cnn', 'mlp', 'vit', 'hybrid_vit', 'drq_vit', 'scott']"
    )
