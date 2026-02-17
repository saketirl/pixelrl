#!/usr/bin/env python
"""
PPO for PixelBrax environments with ViT-5 encoder.

Based on "ViT-5: Vision Transformer for Mid-2020s" architecture:
- RMSNorm instead of LayerNorm
- SwiGLU FFN instead of MLP
- Rotary Position Embeddings (RoPE)
- QK normalization for stable attention
- Register tokens for improved performance

Optimizer setup:
- Adam for ViT-5 encoder (trunk)
- Manifold MUON for actor/critic head matrices (2D+ params)
- Adam for actor/critic head vectors/scalars (biases, log_std)
"""
import os
import argparse
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Optional, Tuple

# Parse --gpu argument before importing JAX
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use")
_args, _ = _parser.parse_known_args()
if _args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import distrax
from flax.linen.initializers import constant, orthogonal, zeros, ones
from flax.training.train_state import TrainState

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax
from pixelbrax.env_utils import make_pixel_brax

# Import manifold MUON optimizer
from manifold_muon_optax import manifold_muon

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinism
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    gpu: int = None
    """GPU device ID to use (parsed early, before JAX import)"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "benchmark"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Environment arguments
    env_name: str = "halfcheetah"
    """the name of the environment"""
    backend: str = "spring"
    """the physics backend (spring, generalized, positional)"""
    n_envs: int = 512
    """the number of parallel game environments"""
    hw: int = 84
    """height/width of the observation images"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""

    # Learning rates
    encoder_lr: float = 3e-4
    """learning rate for ViT-5 encoder (AdamW)"""
    encoder_weight_decay: float = 0.01
    """weight decay for encoder AdamW (L2 regularization)"""
    heads_muon_lr: float = 0.02
    """learning rate for actor/critic head matrices (MUON)"""
    heads_adam_lr: float = 3e-4
    """learning rate for actor/critic head vectors/scalars (Adam)"""
    heads_weight_decay: float = 0.0
    """weight decay for heads AdamW (0.0 uses Adam instead)"""

    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    warmup_steps: int = 0
    """Number of steps for learning rate warmup (0 to disable)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_eps: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping (Adam)"""
    actor_muon_max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping (actor MUON)"""
    critic_muon_max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping (critic MUON)"""
    max_action: float = 1.0
    """maximum action value for clipping"""
    log_interval: int = 10
    """logging interval (in updates)"""

    # Manifold MUON optimizer arguments
    muon_dual_lr: float = 0.01
    """dual learning rate for MUON"""
    muon_dual_steps: int = 5
    """number of dual optimization steps for MUON"""
    muon_msign_steps: int = 5
    """number of matrix sign iterations for MUON"""

    # Data augmentation (DrQ-style)
    use_augmentation: bool = False
    """Toggle random shift data augmentation (DrQ-style)"""
    augment_pad: int = 4
    """Padding size for random shift augmentation"""

    # Hierarchical frame stacking (matching CRATE)
    temporal_stack: int = 8
    """Number of temporal tokens in the sequence"""
    channel_stack: int = 4
    """Number of frames stacked per token (channel-wise)"""

    # Action repeat
    action_repeat: int = 4
    """Number of times to repeat each action (frame skip)"""

    # Environment frame stacking
    frame_stack: int = 3
    """Number of frames stacked by the environment (default 3 for backward compat, use 1 for raw RGB)"""

    # ViT-5 Architecture arguments
    patch_size: int = 14
    """Patch size for ViT-5"""
    embed_dim: int = 256
    """Embedding dimension for ViT-5"""
    depth: int = 4
    """Number of transformer blocks"""
    num_heads: int = 4
    """Number of attention heads"""
    mlp_ratio: float = 2.67
    """MLP hidden dim ratio (SwiGLU uses 2/3 of standard MLP ratio)"""
    num_registers: int = 4
    """Number of register tokens"""
    use_rope: bool = True
    """Use rotary position embeddings"""
    qk_norm: bool = True
    """Use QK normalization in attention"""
    emb_dropout: float = 0.0
    """Dropout after embedding"""
    attn_dropout: float = 0.0
    """Dropout in attention"""
    drop_path: float = 0.0
    """Drop path rate for stochastic depth"""

    # Output normalization (to prevent representation explosion)
    output_tanh: bool = True
    """Apply tanh to bound encoder output"""
    output_tanh_scale: float = 0.5
    """Scale factor before tanh (lower = less saturation)"""
    output_norm: bool = False
    """Apply RMSNorm to encoder output before tanh"""

    # Debug/analysis flags
    debug_repr: bool = False
    """Toggle debug logging for encoder representations"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


# --------------------------------------------------------
#  ViT-5 Components (JAX/Flax implementation)
# --------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't center the activations.
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        # Compute in float32 for numerical stability
        dtype = x.dtype
        x = x.astype(jnp.float32)

        # RMS normalization
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x_normed = x * jax.lax.rsqrt(variance + self.epsilon)

        # Learnable scale parameter
        scale = self.param('scale', ones, (x.shape[-1],))

        return (scale * x_normed).astype(dtype)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Uses gated linear unit with SiLU (Swish) activation:
    SwiGLU(x) = (x @ W1 * SiLU(x @ W2)) @ W3
    """
    hidden_dim: int
    out_dim: int
    use_sub_ln: bool = False

    @nn.compact
    def __call__(self, x):
        # Two parallel projections
        w1 = nn.Dense(self.hidden_dim, use_bias=False, name='w1')(x)
        w2 = nn.Dense(self.hidden_dim, use_bias=False, name='w2')(x)

        # Gated activation: SiLU(w2) * w1
        hidden = nn.silu(w2) * w1

        # Optional sub-layer normalization
        if self.use_sub_ln:
            hidden = RMSNorm()(hidden)

        # Output projection
        return nn.Dense(self.out_dim, use_bias=False, name='w3')(hidden)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> jnp.ndarray:
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing='xy')
    grid = jnp.stack(grid, axis=0).reshape(2, -1).T  # (grid_size*grid_size, 2)

    # Half dimensions for each spatial direction
    half_dim = embed_dim // 4
    omega = 1.0 / (10000 ** (jnp.arange(half_dim) / half_dim))

    # Compute sin/cos embeddings
    pos_h = grid[:, 0:1] * omega  # (N, half_dim)
    pos_w = grid[:, 1:2] * omega  # (N, half_dim)

    pos_embed = jnp.concatenate([
        jnp.sin(pos_h), jnp.cos(pos_h),
        jnp.sin(pos_w), jnp.cos(pos_w)
    ], axis=-1)

    return pos_embed


def apply_rotary_emb(q: jnp.ndarray, k: jnp.ndarray, freqs_cos: jnp.ndarray, freqs_sin: jnp.ndarray):
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (B, H, N, D)
        k: Key tensor of shape (B, H, N, D)
        freqs_cos: Cosine frequencies of shape (N, D//2)
        freqs_sin: Sine frequencies of shape (N, D//2)

    Returns:
        Rotated q and k tensors
    """
    # Split into even and odd dimensions
    d = q.shape[-1]
    q1, q2 = q[..., :d//2], q[..., d//2:]
    k1, k2 = k[..., :d//2], k[..., d//2:]

    # Reshape freqs for broadcasting: (N, D//2) -> (1, 1, N, D//2)
    freqs_cos = freqs_cos[None, None, :, :]
    freqs_sin = freqs_sin[None, None, :, :]

    # Apply rotation
    q_rot = jnp.concatenate([
        q1 * freqs_cos - q2 * freqs_sin,
        q2 * freqs_cos + q1 * freqs_sin
    ], axis=-1)

    k_rot = jnp.concatenate([
        k1 * freqs_cos - k2 * freqs_sin,
        k2 * freqs_cos + k1 * freqs_sin
    ], axis=-1)

    return q_rot, k_rot


def compute_rope_freqs(seq_len: int, head_dim: int, theta: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute rotary position embedding frequencies.

    Args:
        seq_len: Sequence length
        head_dim: Dimension per head
        theta: Base for frequency computation

    Returns:
        Tuple of (cos_freqs, sin_freqs), each of shape (seq_len, head_dim//2)
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    # Position indices
    t = jnp.arange(seq_len, dtype=jnp.float32)

    # Outer product: (seq_len, head_dim//2)
    freqs = jnp.outer(t, freqs)

    return jnp.cos(freqs), jnp.sin(freqs)


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE and QK normalization.

    Features from ViT-5:
    - Rotary Position Embeddings (RoPE)
    - QK normalization for stable training
    - Support for register tokens (no RoPE applied to registers)
    """
    dim: int
    num_heads: int
    qk_norm: bool = True
    use_rope: bool = True
    num_prefix_tokens: int = 0  # CLS + registers (0 for interleaved spatial/temporal blocks)
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, freqs_cos=None, freqs_sin=None, deterministic=True):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads

        # QKV projection
        qkv = nn.Dense(self.dim * 3, use_bias=False, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK normalization for stable attention
        if self.qk_norm:
            q = RMSNorm(name='q_norm')(q)
            k = RMSNorm(name='k_norm')(k)

        # Apply RoPE
        if self.use_rope and freqs_cos is not None and freqs_sin is not None:
            if self.num_prefix_tokens > 0:
                # Split prefix (CLS/registers) and patches - RoPE only on patches
                q_prefix, q_patches = q[:, :, :self.num_prefix_tokens], q[:, :, self.num_prefix_tokens:]
                k_prefix, k_patches = k[:, :, :self.num_prefix_tokens], k[:, :, self.num_prefix_tokens:]

                # Apply RoPE only to patches
                q_patches, k_patches = apply_rotary_emb(q_patches, k_patches, freqs_cos, freqs_sin)

                # Concatenate back
                q = jnp.concatenate([q_prefix, q_patches], axis=2)
                k = jnp.concatenate([k_prefix, k_patches], axis=2)
            else:
                # No prefix tokens - apply RoPE to all tokens
                q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout, deterministic=deterministic)(attn)

        # Combine heads
        x = (attn @ v)
        x = jnp.swapaxes(x, 1, 2).reshape(B, N, C)

        # Output projection
        x = nn.Dense(self.dim, use_bias=False, name='proj')(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        return x


class ViT5Block(nn.Module):
    """Transformer block for ViT-5.

    Structure: Pre-norm -> Attention -> Residual -> Pre-norm -> SwiGLU -> Residual

    Features:
    - RMSNorm instead of LayerNorm
    - SwiGLU FFN instead of MLP
    - Optional layer scaling (gamma)
    - Drop path for stochastic depth
    """
    dim: int
    num_heads: int
    mlp_ratio: float = 2.67
    qk_norm: bool = True
    use_rope: bool = True
    num_prefix_tokens: int = 0  # CLS + registers (0 for interleaved blocks)
    dropout: float = 0.0
    drop_path: float = 0.0
    init_values: Optional[float] = None

    @nn.compact
    def __call__(self, x, freqs_cos=None, freqs_sin=None, deterministic=True):
        # Calculate SwiGLU hidden dim (uses 2/3 of standard MLP hidden)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        # Layer scale parameters
        if self.init_values is not None:
            gamma_1 = self.param('gamma_1', constant(self.init_values), (self.dim,))
            gamma_2 = self.param('gamma_2', constant(self.init_values), (self.dim,))
        else:
            gamma_1 = gamma_2 = None

        # Self-attention block
        attn_out = RMSNorm(name='norm1')(x)
        attn_out = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qk_norm=self.qk_norm,
            use_rope=self.use_rope,
            num_prefix_tokens=self.num_prefix_tokens,
            dropout=self.dropout,
            name='attn'
        )(attn_out, freqs_cos, freqs_sin, deterministic)

        if gamma_1 is not None:
            attn_out = gamma_1 * attn_out

        # Drop path (stochastic depth) - simplified for now
        x = x + attn_out

        # FFN block
        ffn_out = RMSNorm(name='norm2')(x)
        ffn_out = SwiGLU(
            hidden_dim=mlp_hidden_dim,
            out_dim=self.dim,
            name='mlp'
        )(ffn_out)

        if gamma_2 is not None:
            ffn_out = gamma_2 * ffn_out

        x = x + ffn_out

        return x


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings.

    Uses a single convolution to extract non-overlapping patches
    and project them to embedding dimension.
    """
    patch_size: int = 7
    embed_dim: int = 192

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        # Patch embedding via convolution
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            use_bias=True,
            name='proj'
        )(x)

        # Reshape to (B, num_patches, embed_dim)
        grid_size = H // self.patch_size
        x = x.reshape(B, grid_size * grid_size, self.embed_dim)

        return x, grid_size


class ViT5Encoder(nn.Module):
    """Vision Transformer 5 Encoder for RL with Temporal-Spatial Processing.

    Features from ViT-5:
    - RMSNorm instead of LayerNorm
    - SwiGLU FFN instead of standard MLP
    - Rotary Position Embeddings (RoPE)
    - QK normalization
    - Register tokens
    - Interleaved spatial-temporal attention (like CRATE)

    Input shape: (B, T, H, W, C_stacked) where:
    - T = temporal_stack (number of temporal tokens)
    - C_stacked = C * channel_stack (channel-stacked frames per token)
    """
    temporal_stack: int = 8
    channel_stack: int = 4
    patch_size: int = 14
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.67
    num_registers: int = 4
    use_rope: bool = True
    qk_norm: bool = True
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.0
    output_tanh: bool = True
    output_tanh_scale: float = 0.5
    output_norm: bool = False

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, H, W, C_stacked = x.shape
        head_dim = self.embed_dim // self.num_heads
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Normalize pixel values
        x = x.astype(jnp.float32) / 255.0

        # Patchify all frames: (B, T, H, W, C) → (B, T, num_patches, embed_dim)
        x = x.reshape(B * T, H, W, C_stacked)
        x, grid_size = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )(x)
        # x: (B*T, num_patches, embed_dim)
        x = x.reshape(B, T, num_patches, self.embed_dim)

        # Add spatial positional embeddings (shared across time)
        spatial_pos_embed = self.param(
            "spatial_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1, num_patches, self.embed_dim)
        )
        x = x + spatial_pos_embed

        # Add temporal positional embeddings (shared across patches)
        temporal_pos_embed = self.param(
            "temporal_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, T, 1, self.embed_dim)
        )
        x = x + temporal_pos_embed

        # Embedding dropout
        x = nn.Dropout(self.emb_dropout, deterministic=deterministic)(x)

        # Compute RoPE frequencies for patches
        if self.use_rope:
            spatial_freqs_cos, spatial_freqs_sin = compute_rope_freqs(num_patches, head_dim)
            temporal_freqs_cos, temporal_freqs_sin = compute_rope_freqs(T, head_dim)
        else:
            spatial_freqs_cos = spatial_freqs_sin = None
            temporal_freqs_cos = temporal_freqs_sin = None

        # Interleaved spatial and temporal blocks (like CRATE)
        for i in range(self.depth):
            # Spatial attention: attend over patches within each frame
            # Reshape: (B, T, P, D) → (B*T, P, D)
            x = x.reshape(B * T, num_patches, self.embed_dim)
            x = ViT5Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=self.qk_norm,
                use_rope=self.use_rope,
                num_prefix_tokens=0,  # No prefix tokens for spatial (applied per-frame)
                dropout=self.attn_dropout,
                drop_path=self.drop_path,
                name=f'spatial_block_{i}',
            )(x, spatial_freqs_cos, spatial_freqs_sin, deterministic)
            # Reshape back: (B*T, P, D) → (B, T, P, D)
            x = x.reshape(B, T, num_patches, self.embed_dim)

            # Temporal attention: attend over time for each patch position
            # Reshape: (B, T, P, D) → (B*P, T, D)
            x = jnp.transpose(x, (0, 2, 1, 3))  # (B, P, T, D)
            x = x.reshape(B * num_patches, T, self.embed_dim)
            x = ViT5Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=self.qk_norm,
                use_rope=self.use_rope,
                num_prefix_tokens=0,  # No prefix tokens for temporal
                dropout=self.attn_dropout,
                drop_path=self.drop_path,
                name=f'temporal_block_{i}',
            )(x, temporal_freqs_cos, temporal_freqs_sin, deterministic)
            # Reshape back: (B*P, T, D) → (B, T, P, D)
            x = x.reshape(B, num_patches, T, self.embed_dim)
            x = jnp.transpose(x, (0, 2, 1, 3))  # (B, T, P, D)

        # Final normalization
        x = RMSNorm(name='norm')(x)

        # Global average pooling over both time and patches
        # (B, T, P, D) → (B, D)
        x = jnp.mean(x, axis=(1, 2))

        # Optional output normalization to prevent representation explosion
        if self.output_norm:
            x = RMSNorm(name='output_norm')(x)

        # Optional tanh to bound output (like CNN encoder)
        if self.output_tanh:
            x = nn.tanh(self.output_tanh_scale * x)

        return x


class Critic(nn.Module):
    """Value network with 2 hidden layers using ReLU activation."""
    @nn.compact
    def __call__(self, x, return_activations: bool = False):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        h1 = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h1)
        h2 = nn.relu(x)
        out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h2)
        if return_activations:
            return out, (h1, h2)
        return out


class Actor(nn.Module):
    """Continuous action actor with 2 hidden layers using ReLU activation."""
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x, return_activations: bool = False):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        h1 = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h1)
        h2 = nn.relu(x)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(h2)
        actor_logstd = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,)
        )
        actor_logstd = jnp.clip(actor_logstd, self.log_std_min, self.log_std_max)
        if return_activations:
            return actor_mean, actor_logstd, (h1, h2)
        return actor_mean, actor_logstd


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@flax.struct.dataclass
class HierarchicalFrameStack:
    """
    Hierarchical frame stacking buffer (matching CRATE).
    Stores: (n_envs, total_frames, H, W, C)
    Returns: (n_envs, temporal_stack, H, W, C * channel_stack)
    """
    frames: jnp.ndarray
    temporal_stack: int = flax.struct.field(pytree_node=False)
    channel_stack: int = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, n_envs: int, temporal_stack: int, channel_stack: int, obs_shape: tuple):
        h, w, c = obs_shape
        total_frames = temporal_stack * channel_stack
        frames = jnp.zeros((n_envs, total_frames, h, w, c), dtype=jnp.uint8)
        return cls(frames=frames, temporal_stack=temporal_stack, channel_stack=channel_stack)

    def reset(self, obs: jnp.ndarray, done: jnp.ndarray = None):
        n_envs = obs.shape[0]
        total_frames = self.frames.shape[1]
        new_frames = jnp.broadcast_to(
            obs[:, None, :, :, :],
            (n_envs, total_frames, obs.shape[1], obs.shape[2], obs.shape[3])
        )
        if done is None:
            return self.replace(frames=new_frames)
        else:
            frames = jnp.where(done[:, None, None, None, None], new_frames, self.frames)
            return self.replace(frames=frames)

    def push(self, obs: jnp.ndarray):
        new_frames = jnp.concatenate([
            self.frames[:, 1:, :, :, :],
            obs[:, None, :, :, :]
        ], axis=1)
        return self.replace(frames=new_frames)

    def get_stacked(self) -> jnp.ndarray:
        """Get stacked observation: (n_envs, temporal_stack, H, W, C * channel_stack)"""
        n_envs, total_frames, H, W, C = self.frames.shape
        x = self.frames.reshape(n_envs, self.temporal_stack, self.channel_stack, H, W, C)
        x = jnp.transpose(x, (0, 1, 3, 4, 2, 5))
        x = x.reshape(n_envs, self.temporal_stack, H, W, self.channel_stack * C)
        return x


@flax.struct.dataclass
class RewardNormalizer:
    """Reward normalization using discounted returns (CleanRL style)."""
    return_rms_mean: jnp.array
    return_rms_var: jnp.array
    return_rms_count: jnp.array
    discounted_return: jnp.array
    gamma: float = 0.99

    @classmethod
    def create(cls, n_envs, gamma=0.99):
        return cls(
            return_rms_mean=jnp.array(0.0),
            return_rms_var=jnp.array(1.0),
            return_rms_count=jnp.array(1e-4),
            discounted_return=jnp.zeros(n_envs),
            gamma=gamma,
        )

    def update(self, rewards, dones):
        """Update running statistics with a batch of rewards."""
        new_discounted_return = rewards + self.gamma * self.discounted_return * (1.0 - dones)

        batch_mean = jnp.mean(new_discounted_return)
        batch_var = jnp.var(new_discounted_return)
        batch_count = new_discounted_return.size

        delta = batch_mean - self.return_rms_mean
        tot_count = self.return_rms_count + batch_count

        new_mean = self.return_rms_mean + delta * batch_count / tot_count
        m_a = self.return_rms_var * self.return_rms_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.return_rms_count * batch_count / tot_count
        new_var = M2 / tot_count

        new_discounted_return = jnp.where(dones, 0.0, new_discounted_return)

        return self.replace(
            return_rms_mean=new_mean,
            return_rms_var=new_var,
            return_rms_count=tot_count,
            discounted_return=new_discounted_return,
        )

    def normalize(self, rewards, clip=10.0, epsilon=1e-8):
        """Normalize rewards by std of discounted returns and clip."""
        normalized = rewards / jnp.sqrt(self.return_rms_var + epsilon)
        return jnp.clip(normalized, -clip, clip)


# --------------------------------------------------------
#  Debug Metrics for Representation Learning
# --------------------------------------------------------

def repr_health(hidden: jnp.ndarray) -> dict:
    """Compute representation health metrics for encoder output."""
    dim_std = jnp.std(hidden, axis=0)
    norms = jnp.linalg.norm(hidden, axis=-1)
    return {
        "repr/mean": jnp.mean(hidden),
        "repr/std": jnp.std(hidden),
        "repr/min": jnp.min(hidden),
        "repr/max": jnp.max(hidden),
        "repr/norm_mean": jnp.mean(norms),
        "repr/norm_std": jnp.std(norms),
        "repr/norm_min": jnp.min(norms),
        "repr/norm_max": jnp.max(norms),
        "repr/dead_dims": jnp.mean(dim_std < 0.01),
        "repr/dim_std_min": jnp.min(dim_std),
        "repr/dim_std_max": jnp.max(dim_std),
        "repr/dim_std_mean": jnp.mean(dim_std),
        "repr/sparsity": jnp.mean(jnp.abs(hidden) < 0.01),
        "repr/near_zero_frac": jnp.mean(jnp.abs(hidden) < 0.1),
    }


def compute_grad_norms(grads: dict) -> dict:
    """Compute gradient norms for encoder, actor, and critic."""
    def tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    return {
        "grads/encoder_norm": tree_norm(grads['network']),
        "grads/actor_norm": tree_norm(grads['actor']),
        "grads/critic_norm": tree_norm(grads['critic']),
    }


def patch_embed_metrics(patches: jnp.ndarray) -> dict:
    """Compute metrics for patch embeddings before transformer blocks."""
    # patches: (B, T, num_patches, embed_dim) or (B, num_patches, embed_dim)
    norms = jnp.linalg.norm(patches, axis=-1)
    return {
        "patches/mean": jnp.mean(patches),
        "patches/std": jnp.std(patches),
        "patches/norm_mean": jnp.mean(norms),
        "patches/norm_std": jnp.std(norms),
        "patches/sparsity": jnp.mean(jnp.abs(patches) < 0.01),
    }


def attention_metrics(attn_weights: jnp.ndarray, name: str = "attn") -> dict:
    """Compute attention pattern metrics.

    Args:
        attn_weights: (B, num_heads, N, N) attention weights
        name: prefix for metric names
    """
    # Entropy of attention distribution (higher = more uniform)
    entropy = -jnp.sum(attn_weights * jnp.log(attn_weights + 1e-10), axis=-1)

    # Max attention (how peaked the attention is)
    max_attn = jnp.max(attn_weights, axis=-1)

    # Diagonal attention (self-attention strength)
    B, H, N, _ = attn_weights.shape
    diag_mask = jnp.eye(N)[None, None, :, :]
    diag_attn = jnp.sum(attn_weights * diag_mask, axis=-1)

    return {
        f"{name}/entropy_mean": jnp.mean(entropy),
        f"{name}/entropy_std": jnp.std(entropy),
        f"{name}/max_attn_mean": jnp.mean(max_attn),
        f"{name}/diag_attn_mean": jnp.mean(diag_attn),
    }


def positional_embed_metrics(spatial_pos: jnp.ndarray, temporal_pos: jnp.ndarray) -> dict:
    """Compute metrics for positional embeddings."""
    metrics = {}

    # Spatial position embeddings
    spatial_norms = jnp.linalg.norm(spatial_pos, axis=-1)
    metrics["pos/spatial_norm_mean"] = jnp.mean(spatial_norms)
    metrics["pos/spatial_norm_std"] = jnp.std(spatial_norms)
    metrics["pos/spatial_mean"] = jnp.mean(spatial_pos)
    metrics["pos/spatial_std"] = jnp.std(spatial_pos)

    # Temporal position embeddings
    temporal_norms = jnp.linalg.norm(temporal_pos, axis=-1)
    metrics["pos/temporal_norm_mean"] = jnp.mean(temporal_norms)
    metrics["pos/temporal_norm_std"] = jnp.std(temporal_norms)
    metrics["pos/temporal_mean"] = jnp.mean(temporal_pos)
    metrics["pos/temporal_std"] = jnp.std(temporal_pos)

    return metrics


def swiglu_activation_metrics(hidden: jnp.ndarray, name: str) -> dict:
    """Compute activation metrics for SwiGLU FFN layers."""
    norms = jnp.linalg.norm(hidden, axis=-1)
    return {
        f"{name}/mean": jnp.mean(hidden),
        f"{name}/std": jnp.std(hidden),
        f"{name}/norm_mean": jnp.mean(norms),
        f"{name}/dead_frac": jnp.mean(jnp.abs(hidden) < 0.01),
        f"{name}/negative_frac": jnp.mean(hidden < 0),
    }


def per_layer_norms(params: dict) -> dict:
    """Compute weight norms for each transformer block layer."""
    metrics = {}
    flat = flax.traverse_util.flatten_dict(params['network']['params'])

    for key, value in flat.items():
        # Convert key tuple to string
        key_str = '/'.join(key)
        if 'spatial_block' in key_str or 'temporal_block' in key_str:
            # Extract block type and number
            if value.ndim >= 2:
                # Weight matrix - compute Frobenius norm
                metrics[f"weights/{key_str}_norm"] = jnp.linalg.norm(value)

    return metrics


def rmsnorm_scale_metrics(params: dict) -> dict:
    """Track RMSNorm scale parameters across layers."""
    metrics = {}
    flat = flax.traverse_util.flatten_dict(params['network']['params'])

    scales = []
    for key, value in flat.items():
        key_str = '/'.join(key)
        if 'scale' in key_str.lower() and value.ndim == 1:
            scales.append(value)
            metrics[f"rmsnorm/{key_str}_mean"] = jnp.mean(value)
            metrics[f"rmsnorm/{key_str}_std"] = jnp.std(value)

    if scales:
        all_scales = jnp.concatenate(scales)
        metrics["rmsnorm/all_scales_mean"] = jnp.mean(all_scales)
        metrics["rmsnorm/all_scales_std"] = jnp.std(all_scales)
        metrics["rmsnorm/all_scales_min"] = jnp.min(all_scales)
        metrics["rmsnorm/all_scales_max"] = jnp.max(all_scales)

    return metrics


def head_activation_metrics(activations: tuple, name: str) -> dict:
    """Compute activation metrics for ReLU layers in actor/critic heads."""
    metrics = {}
    for i, h in enumerate(activations):
        layer_name = f"{name}/layer{i+1}"
        norms = jnp.linalg.norm(h, axis=-1)
        metrics[f"{layer_name}/mean"] = jnp.mean(h)
        metrics[f"{layer_name}/std"] = jnp.std(h)
        metrics[f"{layer_name}/norm_mean"] = jnp.mean(norms)
        metrics[f"{layer_name}/dead_frac"] = jnp.mean(h == 0)  # ReLU dead units (exactly 0)
        metrics[f"{layer_name}/active_frac"] = jnp.mean(h > 0)  # ReLU active units
    return metrics


# --------------------------------------------------------
#  Optimizer: Adam for ViT-5 encoder, MUON for head matrices, Adam for head vectors
# --------------------------------------------------------

def create_vit5_adamw_heads_muon_optimizer(
    encoder_lr,  # Can be float or schedule
    encoder_weight_decay: float,
    heads_muon_lr: float,
    heads_adam_lr,  # Can be float or schedule
    heads_weight_decay: float = 0.0,
    muon_dual_lr: float = 0.01,
    muon_dual_steps: int = 5,
    muon_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    max_grad_norm: float = 0.5,
    actor_muon_max_grad_norm: float = 1.0,
    critic_muon_max_grad_norm: float = 1.0,
):
    """
    Create optimizer that uses:
    - AdamW for ViT-5 encoder (all params) with weight decay for regularization
    - Manifold MUON for actor head matrices (2D+ with min dim > 1) - with separate grad clipping
    - Manifold MUON for critic head matrices (2D+ with min dim > 1) - with separate grad clipping
    - Adam/AdamW for actor/critic head vectors/scalars (biases, log_std, etc.)
    """

    # AdamW for encoder (always use AdamW with weight decay for regularization)
    encoder_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=encoder_lr, eps=adam_eps, weight_decay=encoder_weight_decay
        ),
    )

    # Adam/AdamW for head vectors/scalars (with schedule support and grad clipping)
    if heads_weight_decay > 0:
        heads_adam_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=heads_adam_lr, eps=adam_eps, weight_decay=heads_weight_decay
            ),
        )
    else:
        heads_adam_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=heads_adam_lr, eps=adam_eps
            ),
        )

    # MUON for actor head matrices (with separate grad clipping)
    actor_muon_tx = optax.chain(
        optax.clip_by_global_norm(actor_muon_max_grad_norm),
        manifold_muon(
            learning_rate=heads_muon_lr,
            dual_lr=muon_dual_lr,
            dual_steps=muon_dual_steps,
            msign_steps=muon_msign_steps,
            min_ndim=2,
        ),
    )

    # MUON for critic head matrices (with separate grad clipping)
    critic_muon_tx = optax.chain(
        optax.clip_by_global_norm(critic_muon_max_grad_norm),
        manifold_muon(
            learning_rate=heads_muon_lr,
            dual_lr=muon_dual_lr,
            dual_steps=muon_dual_steps,
            msign_steps=muon_msign_steps,
            min_ndim=2,
        ),
    )

    # 4 transforms: encoder, actor_muon (matrices), critic_muon (matrices), heads_adam (vectors/scalars)
    transforms = {
        'encoder': encoder_tx,
        'actor_muon': actor_muon_tx,
        'critic_muon': critic_muon_tx,
        'heads_adam': heads_adam_tx,
    }

    # Label function
    def label_fn(params):
        def _label(path, param):
            # path[0] is 'network', 'actor', or 'critic'
            if path[0] == 'network':
                return 'encoder'
            else:
                # For actor/critic heads, check if matrix or vector/scalar
                is_matrix = param.ndim >= 2 and min(param.shape) > 1
                if is_matrix:
                    return 'actor_muon' if path[0] == 'actor' else 'critic_muon'
                else:
                    return 'heads_adam'

        flat = flax.traverse_util.flatten_dict(params)
        labeled = {k: _label(k, v) for k, v in flat.items()}
        return flax.core.freeze(flax.traverse_util.unflatten_dict(labeled))

    return optax.multi_transform(
        transforms=transforms,
        param_labels=label_fn,
    )


def make_pixelbrax_envs(args):
    """Create PixelBrax environments."""
    envs, _, _ = make_pixel_brax(
        backend=args.backend,
        env_name=args.env_name,
        n_envs=args.n_envs,
        seed=args.seed,
        hw=args.hw,
        distractor=None,
        video_path="datasets/DAVIS",
        video_set="train",
        return_float32=False,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
    )

    try:
        action_dim = envs.action_size
    except AttributeError:
        action_dim = envs.env.action_size

    return envs, action_dim


def random_shift(key: jax.random.PRNGKey, x: jnp.ndarray, pad: int = 4) -> jnp.ndarray:
    """DrQ-style random shift augmentation."""
    b, h, w, c = x.shape
    x_padded = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')

    key1, key2 = jax.random.split(key)
    crop_h = jax.random.randint(key1, (b,), 0, 2 * pad + 1)
    crop_w = jax.random.randint(key2, (b,), 0, 2 * pad + 1)

    def crop_single(x_pad, ch, cw):
        return jax.lax.dynamic_slice(x_pad, (ch, cw, 0), (h, w, c))

    return jax.vmap(crop_single)(x_padded, crop_h, crop_w)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.env_name}__vit5_muon__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # Environment setup
    print("=" * 60)
    print("PPO with ViT-5 Encoder + Manifold MUON for Actor/Critic Heads")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")
    print(f"frame_stack (env): {args.frame_stack}")

    total_frames = args.temporal_stack * args.channel_stack
    num_patches = (args.hw // args.patch_size) ** 2
    print(f"\nViT-5 Architecture:")
    print(f"  temporal_stack: {args.temporal_stack}")
    print(f"  channel_stack: {args.channel_stack}")
    print(f"  total_frames: {total_frames}")
    print(f"  patch_size: {args.patch_size}")
    print(f"  num_patches per frame: {num_patches}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"  depth (interleaved pairs): {args.depth}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  mlp_ratio: {args.mlp_ratio}")
    print(f"  num_registers: {args.num_registers}")
    print(f"  use_rope: {args.use_rope}")
    print(f"  qk_norm: {args.qk_norm}")
    print(f"  emb_dropout: {args.emb_dropout}")
    print(f"  attn_dropout: {args.attn_dropout}")
    print(f"  output_tanh: {args.output_tanh} (scale={args.output_tanh_scale})")
    print(f"  output_norm: {args.output_norm}")
    print(f"\nAttention pattern: S1 → T1 → S2 → T2 → ... → S{args.depth} → T{args.depth}")
    print(f"Total blocks: {args.depth * 2} ({args.depth} spatial + {args.depth} temporal)")

    heads_adam_type = "AdamW" if args.heads_weight_decay > 0 else "Adam"
    print(f"\nOptimizer config:")
    print(f"  encoder_lr (AdamW): {args.encoder_lr}")
    print(f"  encoder_weight_decay: {args.encoder_weight_decay}")
    print(f"  heads_muon_lr (MUON for matrices): {args.heads_muon_lr}")
    print(f"  heads_adam_lr ({heads_adam_type} for vectors): {args.heads_adam_lr}")
    print(f"  heads_weight_decay: {args.heads_weight_decay}")
    print(f"  muon_dual_lr: {args.muon_dual_lr}")
    print(f"  muon_dual_steps: {args.muon_dual_steps}")
    print(f"  max_grad_norm: {args.max_grad_norm}")
    print(f"  actor_muon_max_grad_norm: {args.actor_muon_max_grad_norm}")
    print(f"  critic_muon_max_grad_norm: {args.critic_muon_max_grad_norm}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  anneal_lr: {args.anneal_lr}")
    print("=" * 60)

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset(reset_rng)
    raw_obs_shape = init_env_state.pixels.shape[1:]  # (H, W, C)
    H, W, C = raw_obs_shape

    C_stacked = C * args.channel_stack
    obs_shape = (args.temporal_stack, H, W, C_stacked)

    print(f"raw_obs_shape: {raw_obs_shape}")
    print(f"obs_shape (T, H, W, C_stacked): {obs_shape}")

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

    # Learning rate schedule with warmup and optional annealing
    def lr_schedule(count, base_lr):
        """Learning rate schedule with warmup and optional annealing."""
        update_step = count // (args.num_minibatches * args.update_epochs)
        if args.warmup_steps > 0:
            warmup_ratio = jnp.minimum(count / args.warmup_steps, 1.0)
        else:
            warmup_ratio = 1.0
        if args.anneal_lr:
            anneal_ratio = 1.0 - update_step / args.num_updates
        else:
            anneal_ratio = 1.0
        return base_lr * warmup_ratio * anneal_ratio

    # Initialize ViT-5 network
    network = ViT5Encoder(
        temporal_stack=args.temporal_stack,
        channel_stack=args.channel_stack,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_registers=args.num_registers,
        use_rope=args.use_rope,
        qk_norm=args.qk_norm,
        emb_dropout=args.emb_dropout,
        attn_dropout=args.attn_dropout,
        drop_path=args.drop_path,
        output_tanh=args.output_tanh,
        output_tanh_scale=args.output_tanh_scale,
        output_norm=args.output_norm,
    )
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    # Dummy input
    dummy_obs = jnp.zeros((1,) + obs_shape)
    print(f"dummy_obs shape: {dummy_obs.shape}")

    network_params = network.init(network_key, dummy_obs, deterministic=True)
    dummy_hidden = network.apply(network_params, dummy_obs, deterministic=True)
    print(f"encoder output shape: {dummy_hidden.shape}")

    all_params = flax.core.freeze({
        'network': network_params,
        'actor': actor.init(actor_key, dummy_hidden),
        'critic': critic.init(critic_key, dummy_hidden),
    })

    # Count params by component and type
    def count_params(params, key):
        if key in params:
            return sum(p.size for p in jax.tree_util.tree_leaves(params[key]))
        return 0

    def count_by_type(params, key):
        if key not in params:
            return 0, 0
        flat = flax.traverse_util.flatten_dict(params[key])
        muon_count = sum(p.size for p in flat.values() if p.ndim >= 2 and min(p.shape) > 1)
        adam_count = sum(p.size for p in flat.values() if p.ndim < 2 or min(p.shape) <= 1)
        return muon_count, adam_count

    encoder_params_count = count_params(all_params, 'network')
    actor_params_count = count_params(all_params, 'actor')
    critic_params_count = count_params(all_params, 'critic')
    total_params = encoder_params_count + actor_params_count + critic_params_count

    actor_muon, actor_adam = count_by_type(all_params, 'actor')
    critic_muon, critic_adam = count_by_type(all_params, 'critic')

    print(f"\nParameter breakdown:")
    print(f"  ViT-5 Encoder (Adam): {encoder_params_count:,}")
    print(f"  Actor total: {actor_params_count:,} (MUON: {actor_muon:,}, Adam: {actor_adam:,})")
    print(f"  Critic total: {critic_params_count:,} (MUON: {critic_muon:,}, Adam: {critic_adam:,})")
    print(f"  Total: {total_params:,}")

    # Create learning rate schedules
    use_schedule = args.anneal_lr or args.warmup_steps > 0
    if use_schedule:
        encoder_lr = partial(lr_schedule, base_lr=args.encoder_lr)
        heads_adam_lr = partial(lr_schedule, base_lr=args.heads_adam_lr)
        print(f"\nLearning rate schedule: ENABLED (warmup={args.warmup_steps}, anneal={args.anneal_lr})")
    else:
        encoder_lr = args.encoder_lr
        heads_adam_lr = args.heads_adam_lr
        print(f"\nLearning rate schedule: DISABLED")

    # Create optimizer
    tx = create_vit5_adamw_heads_muon_optimizer(
        encoder_lr=encoder_lr,
        encoder_weight_decay=args.encoder_weight_decay,
        heads_muon_lr=args.heads_muon_lr,
        heads_adam_lr=heads_adam_lr,
        heads_weight_decay=args.heads_weight_decay,
        muon_dual_lr=args.muon_dual_lr,
        muon_dual_steps=args.muon_dual_steps,
        muon_msign_steps=args.muon_msign_steps,
        adam_eps=1e-5,
        max_grad_norm=args.max_grad_norm,
        actor_muon_max_grad_norm=args.actor_muon_max_grad_norm,
        critic_muon_max_grad_norm=args.critic_muon_max_grad_norm,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    network.apply = jax.jit(partial(network.apply, deterministic=True))
    actor.apply = jax.jit(actor.apply, static_argnames=['return_activations'])
    critic.apply = jax.jit(critic.apply, static_argnames=['return_activations'])

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and return updated key."""
        hidden = network.apply(agent_state.params['network'], next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params['actor'], hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic.apply(agent_state.params['critic'], hidden)

        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        hidden = network.apply(params['network'], x)
        actor_mean, actor_logstd = actor.apply(params['actor'], hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic.apply(params['critic'], hidden).squeeze(-1)

        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params['critic'],
            network.apply(agent_state.params['network'], next_obs)
        ).squeeze(-1)

        advantages = jnp.zeros((args.n_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values, aug_key):
        # Apply random shift augmentation if enabled
        if args.use_augmentation:
            x = random_shift(aug_key, x, pad=args.augment_pad)

        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss with clipping
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + jnp.clip(
                newvalue - mb_values, -args.clip_eps, args.clip_eps
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey, aug_key = jax.random.split(key, 3)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            aug_keys = jax.random.split(aug_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state = carry
                minibatch, mb_aug_key = inputs
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, (shuffled_storage, aug_keys)
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        # Get final gradients from last minibatch of last epoch
        final_grads = jax.tree_util.tree_map(lambda x: x[-1, -1], grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, final_grads, key

    # Start the game
    global_step = 0
    start_time = time.time()

    # Reset environment
    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    env_state = envs.reset(reset_rngs)

    # Initialize hierarchical frame stack with initial observation
    frame_stack = HierarchicalFrameStack.create(
        args.n_envs, args.temporal_stack, args.channel_stack, raw_obs_shape
    )
    frame_stack = frame_stack.reset(env_state.pixels)
    next_obs = frame_stack.get_stacked()
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    # Initialize reward normalizer
    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    def step_once(carry, step):
        agent_state, episode_stats, reward_norm, fs, env_state, obs, done, key = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        env_state = envs.step(env_state, action)
        raw_obs = env_state.pixels
        raw_reward = env_state.reward
        next_done = env_state.done.astype(jnp.bool_)

        fs = fs.push(raw_obs)
        fs = fs.reset(raw_obs, next_done)
        next_obs = fs.get_stacked()

        reward_norm = reward_norm.update(raw_reward, next_done.astype(jnp.float32))
        reward = reward_norm.normalize(raw_reward)

        new_episode_return = episode_stats.episode_returns + raw_reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=jnp.where(next_done, 0.0, new_episode_return),
            episode_lengths=jnp.where(next_done, 0, new_episode_length).astype(jnp.int32),
            returned_episode_returns=jnp.where(
                next_done, new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                next_done, new_episode_length, episode_stats.returned_episode_lengths
            ).astype(jnp.int32),
        )

        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key), storage

    def rollout(agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key, max_steps):
        (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key), storage = jax.lax.scan(
            step_once, (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key), jnp.arange(max_steps)
        )
        return agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, storage, key

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("\nStarting training...")
    cumulative_episodic_return = 0.0
    for iteration in range(1, args.num_updates + 1):
        iteration_time_start = time.time()
        agent_state, episode_stats, reward_normalizer, frame_stack, env_state, next_obs, next_done, storage, key = rollout(
            agent_state, episode_stats, reward_normalizer, frame_stack, env_state, next_obs, next_done, key
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, final_grads, key = update_ppo(
            agent_state,
            storage,
            key,
        )

        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            cumulative_episodic_return += avg_episodic_return
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))

            print(
                f"update={iteration} step={global_step} "
                f"ep_return={avg_episodic_return:.1f} "
                f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                f"loss={loss[-1, -1].item():.4f} "
                f"SPS={sps}"
            )

            if args.track:
                # Compute LR from schedule (optimizer step count)
                opt_step = iteration * args.update_epochs * args.num_minibatches
                if use_schedule:
                    encoder_lr_current = float(lr_schedule(opt_step, args.encoder_lr))
                    heads_adam_lr_current = float(lr_schedule(opt_step, args.heads_adam_lr))
                else:
                    encoder_lr_current = args.encoder_lr
                    heads_adam_lr_current = args.heads_adam_lr

                # Get log_std stats
                log_std = jax.device_get(agent_state.params['actor']['params']['log_std'])
                log_std_mean = float(np.mean(log_std))
                log_std_min = float(np.min(log_std))
                log_std_max = float(np.max(log_std))
                std_mean = float(np.mean(np.exp(log_std)))

                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/cumulative_episodic_return": cumulative_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/encoder_lr": encoder_lr_current,
                    "charts/heads_adam_lr": heads_adam_lr_current,
                    "charts/heads_muon_lr": args.heads_muon_lr,  # Fixed
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                    "policy/log_std_mean": log_std_mean,
                    "policy/log_std_min": log_std_min,
                    "policy/log_std_max": log_std_max,
                    "policy/std_mean": std_mean,
                }

                # Debug representation metrics
                if args.debug_repr:
                    # Sample observations from the rollout
                    batch_obs = storage.obs.reshape((-1,) + storage.obs.shape[2:])
                    # Use a subset to avoid OOM
                    sample_size = min(256, batch_obs.shape[0])
                    sample_obs = batch_obs[:sample_size]

                    # Get encoder output
                    hidden = network.apply(agent_state.params['network'], sample_obs)

                    # Representation health metrics
                    health_metrics = repr_health(hidden)
                    for k, v in health_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    # Gradient norms
                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    # RMSNorm scale statistics
                    rmsnorm_metrics = rmsnorm_scale_metrics(agent_state.params)
                    for k, v in rmsnorm_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    # Positional embedding metrics
                    spatial_pos = agent_state.params['network']['params']['spatial_pos_embed']
                    temporal_pos = agent_state.params['network']['params']['temporal_pos_embed']
                    pos_metrics = positional_embed_metrics(spatial_pos, temporal_pos)
                    for k, v in pos_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    # Actor/Critic head activation metrics
                    _, _, actor_activations = actor.apply(
                        agent_state.params['actor'], hidden, return_activations=True
                    )
                    actor_act_metrics = head_activation_metrics(actor_activations, "actor_relu")
                    for k, v in actor_act_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    _, critic_activations = critic.apply(
                        agent_state.params['critic'], hidden, return_activations=True
                    )
                    critic_act_metrics = head_activation_metrics(critic_activations, "critic_relu")
                    for k, v in critic_act_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
