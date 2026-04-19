"""Shared model components for PPO pixel experiments."""

from dataclasses import dataclass
from typing import Optional, Sequence

from flax import linen as nn
from flax.linen.initializers import constant, lecun_normal, orthogonal
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




class HiddenActivation(nn.Module):
    """Configurable encoder bottleneck activation."""

    hidden_activation: str = "tanh"
    tanh_scale: float = 0.5
    alpha: float = 0.1
    beta_init: float = 1.0

    @nn.compact
    def __call__(self, hidden: jnp.ndarray) -> jnp.ndarray:
        act = self.hidden_activation.lower()
        if act == "tanh":
            return nn.tanh(self.tanh_scale * hidden)
        if act in {"swish", "silu"}:
            return nn.swish(hidden)
        if act in {"swish_tanh", "swishtanh"}:
            hidden = nn.swish(hidden)
            hidden = nn.Dense(
                hidden.shape[-1],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="swish_tanh_dense",
            )(hidden)
            hidden = nn.LayerNorm(name="swish_tanh_ln")(hidden)
            return nn.tanh(self.tanh_scale * hidden)
        if act in {"swish_ta", "swishta"}:
            return nn.sigmoid(hidden) * (hidden + 2.0 * self.alpha) - self.alpha
        if act in {"swish_tb", "swishtb"}:
            beta = self.param("beta", lambda key, shape: jnp.full(shape, self.beta_init, dtype=hidden.dtype), (1,))
            return nn.sigmoid(beta * hidden) * (hidden + 2.0 * self.alpha) - self.alpha
        if act in {"swish_tc", "swishtc"}:
            beta = self.param("beta", lambda key, shape: jnp.full(shape, self.beta_init, dtype=hidden.dtype), (1,))
            return nn.sigmoid(beta * hidden) * (hidden + 2.0 * self.alpha / beta) - self.alpha / beta
        raise ValueError(
            f"Unsupported hidden_activation={self.hidden_activation!r}. Expected 'tanh', 'swish', 'silu', 'swish_tanh', 'swish_ta', 'swish_tb', or 'swish_tc'."
        )


class ResidualBlock(nn.Module):
    """Residual block used by the IMPALA-style ResNet encoder."""

    channels: int
    kernel_init_fn: nn.initializers.Initializer = lecun_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=self.kernel_init_fn,
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=self.kernel_init_fn,
            bias_init=constant(0.0),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    """Conv + pool + two residual blocks, matching the DEAC PPO encoder."""

    channels: int
    kernel_init_fn_conv: nn.initializers.Initializer = lecun_normal()
    kernel_init_fn_resblock: nn.initializers.Initializer = lecun_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=self.kernel_init_fn_conv,
            bias_init=constant(0.0),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(
            self.channels,
            kernel_init_fn=self.kernel_init_fn_resblock,
        )(x)
        x = ResidualBlock(
            self.channels,
            kernel_init_fn=self.kernel_init_fn_resblock,
        )(x)
        return x


class ResNetBase(nn.Module):
    """Shared IMPALA-style encoder adapted from DEAC's Brax PPO implementation."""

    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)
    conv_kernel_init: nn.initializers.Initializer = lecun_normal()
    resblock_kernel_init: nn.initializers.Initializer = lecun_normal()
    dense_kernel_init: nn.initializers.Initializer = orthogonal(np.sqrt(2))

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        x = x.astype(jnp.float32) / 255.0

        for channels in self.channels:
            x = ConvSequence(
                channels=channels,
                kernel_init_fn_conv=self.conv_kernel_init,
                kernel_init_fn_resblock=self.resblock_kernel_init,
            )(x)

        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        dense_pre_ln = None
        for hidden in self.hiddens:
            dense_pre_ln = nn.Dense(
                hidden,
                kernel_init=self.dense_kernel_init,
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(dense_pre_ln)

        if return_intermediates:
            return {
                "hidden": x,
                "dense_pre_ln": dense_pre_ln,
            }
        return x

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
    hidden_activation: str = "tanh"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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
        dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        hidden = nn.LayerNorm()(dense_pre_ln)
        hidden = HiddenActivation(
            hidden_activation=self.hidden_activation,
            tanh_scale=self.tanh_scale,
            name="hidden_activation",
        )(hidden)
        if return_intermediates:
            return {
                "hidden": hidden,
                "dense_pre_ln": dense_pre_ln,
            }
        return hidden


class CNNSwishTanhEncoder(nn.Module):
    """CNN encoder with a two-stage swish-then-tanh bottleneck."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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
        stage1_dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_stage1",
        )(x)
        stage1_hidden = nn.LayerNorm(name="ln_stage1")(stage1_dense_pre_ln)
        stage1_hidden = nn.swish(stage1_hidden)

        dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_stage2",
        )(stage1_hidden)
        hidden = nn.LayerNorm(name="ln_stage2")(dense_pre_ln)
        hidden = nn.tanh(self.tanh_scale * hidden)
        if return_intermediates:
            return {
                "hidden": hidden,
                "dense_pre_ln": dense_pre_ln,
                "stage1_hidden": stage1_hidden,
                "stage1_dense_pre_ln": stage1_dense_pre_ln,
            }
        return hidden



class CNNSwishTanhResidualEncoder(nn.Module):
    """CNN encoder with a stage-1 swish bottleneck and residual tanh correction."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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
        stage1_dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_stage1",
        )(x)
        stage1_hidden = nn.LayerNorm(name="ln_stage1")(stage1_dense_pre_ln)
        stage1_hidden = nn.swish(stage1_hidden)

        dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_stage2",
        )(stage1_hidden)
        stage2_tanh_hidden = nn.LayerNorm(name="ln_stage2")(dense_pre_ln)
        stage2_tanh_hidden = nn.tanh(self.tanh_scale * stage2_tanh_hidden)
        hidden = (stage1_hidden + stage2_tanh_hidden) / jnp.sqrt(2.0)
        if return_intermediates:
            return {
                "hidden": hidden,
                "dense_pre_ln": dense_pre_ln,
                "stage1_hidden": stage1_hidden,
                "stage1_dense_pre_ln": stage1_dense_pre_ln,
                "stage2_tanh_hidden": stage2_tanh_hidden,
            }
        return hidden


class CNNSwishTanhResidualLearnedEncoder(nn.Module):
    """CNN encoder with a stage-1 swish bottleneck and learned tanh residual gate."""

    tanh_scale: float = 0.5
    residual_gate_init: float = 0.25

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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
        stage1_dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_stage1",
        )(x)
        stage1_hidden = nn.LayerNorm(name="ln_stage1")(stage1_dense_pre_ln)
        stage1_hidden = nn.swish(stage1_hidden)

        dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_stage2",
        )(stage1_hidden)
        stage2_tanh_hidden = nn.LayerNorm(name="ln_stage2")(dense_pre_ln)
        stage2_tanh_hidden = nn.tanh(self.tanh_scale * stage2_tanh_hidden)
        gate_logit = self.param(
            "residual_gate_logit",
            constant(float(np.log(self.residual_gate_init / (1.0 - self.residual_gate_init)))),
            (),
        )
        residual_gate = nn.sigmoid(gate_logit)
        hidden = (stage1_hidden + residual_gate * stage2_tanh_hidden) / jnp.sqrt(
            1.0 + residual_gate**2
        )
        if return_intermediates:
            return {
                "hidden": hidden,
                "dense_pre_ln": dense_pre_ln,
                "stage1_hidden": stage1_hidden,
                "stage1_dense_pre_ln": stage1_dense_pre_ln,
                "stage2_tanh_hidden": stage2_tanh_hidden,
                "residual_gate": residual_gate,
            }
        return hidden


class SplitActorCriticCNNEncoder(nn.Module):
    """CNN encoder with a shared trunk and separate actor/critic bottlenecks."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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

        actor_dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense",
        )(x)
        actor_hidden = nn.LayerNorm(name="actor_ln")(actor_dense_pre_ln)
        actor_hidden = nn.tanh(self.tanh_scale * actor_hidden)

        critic_dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense",
        )(x)
        critic_hidden = nn.LayerNorm(name="critic_ln")(critic_dense_pre_ln)
        critic_hidden = nn.tanh(self.tanh_scale * critic_hidden)

        if return_intermediates:
            return {
                "hidden": actor_hidden,
                "dense_pre_ln": actor_dense_pre_ln,
                "actor_hidden": actor_hidden,
                "critic_hidden": critic_hidden,
                "actor_dense_pre_ln": actor_dense_pre_ln,
                "critic_dense_pre_ln": critic_dense_pre_ln,
            }
        return actor_hidden, critic_hidden


class SeparateActorCriticCNNEncoder(nn.Module):
    """Fully separate CNN encoders for actor and critic."""

    tanh_scale: float = 0.5

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        actor_encoder = CNNEncoder(
            tanh_scale=self.tanh_scale,
            hidden_activation="tanh",
            name="actor_encoder",
        )
        critic_encoder = CNNEncoder(
            tanh_scale=self.tanh_scale,
            hidden_activation="tanh",
            name="critic_encoder",
        )

        if return_intermediates:
            actor_debug = actor_encoder(x, return_intermediates=True)
            critic_debug = critic_encoder(x, return_intermediates=True)
            return {
                "hidden": actor_debug["hidden"],
                "dense_pre_ln": actor_debug["dense_pre_ln"],
                "actor_hidden": actor_debug["hidden"],
                "critic_hidden": critic_debug["hidden"],
                "actor_dense_pre_ln": actor_debug["dense_pre_ln"],
                "critic_dense_pre_ln": critic_debug["dense_pre_ln"],
            }

        actor_hidden = actor_encoder(x)
        critic_hidden = critic_encoder(x)
        return actor_hidden, critic_hidden


class CRATEFeedForward(nn.Module):
    """CRATE-style FeedForward layer implementing an ISTA step."""

    dim: int
    step_size: float = 0.1

    @nn.compact
    def __call__(self, x):
        weight = self.param(
            "weight",
            nn.initializers.kaiming_uniform(),
            (self.dim, self.dim),
        )
        x1 = x @ weight.T
        grad_1 = x1 @ weight
        grad_2 = x @ weight
        grad_update = self.step_size * (grad_2 - grad_1)
        return nn.swish(x + grad_update)


class CRATECNNEncoder(nn.Module):
    """Compact CNN encoder with a CRATE policy-facing bottleneck."""

    crate_step_size: float = 0.1
    tanh_scale: float = 0.5
    residual_scale: float = 0.0
    apply_output_tanh: bool = False

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        x = x.astype(jnp.float32) / 255.0

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
        dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        pre_crate_hidden = nn.LayerNorm()(dense_pre_ln)
        hidden = CRATEFeedForward(
            dim=512,
            step_size=self.crate_step_size,
            name="crate_block",
        )(pre_crate_hidden)
        crate_hidden = hidden
        tanh_hidden = None
        residual_hidden = None
        if self.apply_output_tanh:
            tanh_hidden = nn.LayerNorm(name="post_crate_ln")(hidden)
            tanh_hidden = nn.tanh(self.tanh_scale * tanh_hidden)
            hidden = tanh_hidden
            if self.residual_scale != 0.0:
                residual_hidden = nn.LayerNorm(name="residual_ln")(crate_hidden)
                hidden = tanh_hidden + self.residual_scale * residual_hidden

        if return_intermediates:
            intermediates = {
                "hidden": hidden,
                "crate_hidden": crate_hidden,
                "pre_crate_hidden": pre_crate_hidden,
                "dense_pre_ln": dense_pre_ln,
            }
            if tanh_hidden is not None:
                intermediates["tanh_hidden"] = tanh_hidden
            if residual_hidden is not None:
                intermediates["residual_hidden"] = residual_hidden
                intermediates["residual_scale"] = jnp.asarray(self.residual_scale)
            return intermediates
        return hidden


class ProjectedSIGRegCNNEncoder(nn.Module):
    """CNN encoder with optional CRATE block before a learned projector."""

    tanh_scale: float = 0.5
    proj_dim: int = 64
    use_crate_block: bool = False
    crate_step_size: float = 0.1

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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
        dense_pre_ln = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        z_raw = nn.LayerNorm()(dense_pre_ln)
        z_raw = nn.tanh(self.tanh_scale * z_raw)

        z_structured = z_raw
        if self.use_crate_block:
            z_structured = CRATEFeedForward(
                dim=512,
                step_size=self.crate_step_size,
                name="crate_block",
            )(z_structured)

        projector_pre_ln = nn.Dense(
            self.proj_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="projector",
        )(z_structured)
        u = nn.LayerNorm(name="projector_ln")(projector_pre_ln)

        if return_intermediates:
            return {
                "hidden": u,
                "dense_pre_ln": dense_pre_ln,
                "z_raw": z_raw,
                "z_structured": z_structured,
                "projector_pre_ln": projector_pre_ln,
                "u": u,
            }
        return u


class InnovationCNNEncoder(nn.Module):
    """CNN encoder with an auxiliary innovation projector branch."""

    tanh_scale: float = 0.5
    innovation_proj_dim: int = 64
    use_crate_block: bool = False
    crate_step_size: float = 0.1
    policy_on_projected: bool = False
    hidden_activation: str = "tanh"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
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
        dense_pre_ln = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        hidden = nn.LayerNorm()(dense_pre_ln)
        hidden = HiddenActivation(hidden_activation=self.hidden_activation, tanh_scale=self.tanh_scale, name="hidden_activation")(hidden)

        structured_hidden = hidden
        if self.use_crate_block:
            structured_hidden = CRATEFeedForward(
                dim=512,
                step_size=self.crate_step_size,
                name="crate_block",
            )(structured_hidden)

        projector_pre_ln = nn.Dense(
            self.innovation_proj_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="innovation_projector",
        )(structured_hidden)
        u = projector_pre_ln
        policy_hidden = u if self.policy_on_projected else hidden

        if return_intermediates:
            return {
                "hidden": hidden,
                "structured_hidden": structured_hidden,
                "policy_hidden": policy_hidden,
                "dense_pre_ln": dense_pre_ln,
                "u": u,
                "projector_pre_ln": projector_pre_ln,
            }
        return policy_hidden


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
    sigreg_proj_dim: int = 64,
    innovation_proj_dim: int = 64,
    use_crate_block: bool = False,
    crate_step_size: float = 0.1,
    innovation_hidden_activation: str = "tanh",
    residual_scale: float = 0.1,
) -> nn.Module:
    """Build an encoder module by name."""
    kind = encoder_type.lower()
    if kind == "resnet":
        return ResNetBase()
    if kind == "cnn":
        return CNNEncoder(tanh_scale=tanh_scale, hidden_activation="tanh")
    if kind == "cnn_swish":
        return CNNEncoder(tanh_scale=tanh_scale, hidden_activation="swish")
    if kind == "cnn_swish_ta":
        return CNNEncoder(tanh_scale=tanh_scale, hidden_activation="swish_ta")
    if kind == "cnn_swish_tb":
        return CNNEncoder(tanh_scale=tanh_scale, hidden_activation="swish_tb")
    if kind == "cnn_swish_tc":
        return CNNEncoder(tanh_scale=tanh_scale, hidden_activation="swish_tc")
    if kind in {"crate_cnn", "cratecnn"}:
        return CRATECNNEncoder(crate_step_size=crate_step_size)
    if kind in {"crate_cnn_tanh", "cratecnn_tanh", "cratecnntanh"}:
        return CRATECNNEncoder(
            crate_step_size=crate_step_size,
            tanh_scale=tanh_scale,
            apply_output_tanh=True,
        )
    if kind in {"crate_cnn_tanh_resid", "cratecnn_tanh_resid", "cratecnntanhresid"}:
        return CRATECNNEncoder(
            crate_step_size=crate_step_size,
            tanh_scale=tanh_scale,
            residual_scale=residual_scale,
            apply_output_tanh=True,
        )
    if kind == "cnn_swish_tanh":
        return CNNSwishTanhEncoder(tanh_scale=tanh_scale)
    if kind == "cnn_swish_tanh_resid":
        return CNNSwishTanhResidualEncoder(tanh_scale=tanh_scale)
    if kind == "cnn_swish_tanh_resid_learned":
        return CNNSwishTanhResidualLearnedEncoder(tanh_scale=tanh_scale)
    if kind == "split_cnn":
        return SplitActorCriticCNNEncoder(tanh_scale=tanh_scale)
    if kind == "separate_cnn":
        return SeparateActorCriticCNNEncoder(tanh_scale=tanh_scale)
    if kind == "sigreg_cnn":
        return ProjectedSIGRegCNNEncoder(
            tanh_scale=tanh_scale,
            proj_dim=sigreg_proj_dim,
            use_crate_block=use_crate_block,
            crate_step_size=crate_step_size,
        )
    if kind == "innovation_cnn":
        return InnovationCNNEncoder(
            tanh_scale=tanh_scale,
            innovation_proj_dim=innovation_proj_dim,
            use_crate_block=use_crate_block,
            crate_step_size=crate_step_size,
            policy_on_projected=False,
            hidden_activation=innovation_hidden_activation,
        )
    if kind == "innovation_direct_cnn":
        return InnovationCNNEncoder(
            tanh_scale=tanh_scale,
            innovation_proj_dim=innovation_proj_dim,
            use_crate_block=use_crate_block,
            crate_step_size=crate_step_size,
            policy_on_projected=True,
            hidden_activation=innovation_hidden_activation,
        )
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
    raise ValueError(
        f"Unknown encoder_type='{encoder_type}'. Expected one of: ['resnet', 'cnn', 'cnn_swish', 'cnn_swish_ta', 'cnn_swish_tb', 'cnn_swish_tc', 'crate_cnn', 'crate_cnn_tanh', 'crate_cnn_tanh_resid', 'cnn_swish_tanh', 'cnn_swish_tanh_resid', 'cnn_swish_tanh_resid_learned', 'split_cnn', 'separate_cnn', 'sigreg_cnn', 'innovation_cnn', 'innovation_direct_cnn', 'mlp', 'vit', 'hybrid_vit', 'drq_vit']"
    )
