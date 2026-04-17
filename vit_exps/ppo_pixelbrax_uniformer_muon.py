#!/usr/bin/env python
"""
PPO for PixelBrax environments with UniFormer backbone.

Uses:
- Adam for UniFormer encoder (backbone)
- Manifold MUON for actor/critic head matrices (2D+ params)
- Adam for actor/critic head vectors/scalars (biases, log_std)

UniFormer architecture based on:
- Paper: https://arxiv.org/pdf/2201.04676
- Reference: https://mmaction2.readthedocs.io/en/latest/_modules/mmaction/models/backbones/uniformer.html

Key components:
- Dynamic Position Embedding (DPE): 3x3x3 depthwise conv
- Local MHRA: PWConv-DWConv-PWConv (3D conv) for shallow layers
- Global MHRA: Self-attention for deep layers
- FFN: Linear-GELU-Linear
"""
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Tuple, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import distrax
from flax.linen.initializers import constant, orthogonal, lecun_normal
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
    """learning rate for encoder (Adam/AdamW)"""
    heads_muon_lr: float = 0.02
    """learning rate for actor/critic head matrices (MUON)"""
    heads_adam_lr: float = 3e-4
    """learning rate for actor/critic head vectors/scalars (Adam/AdamW)"""
    weight_decay: float = 0.0
    """weight decay for AdamW (0.0 uses Adam instead)"""

    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
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
    ent_coef: float = 0.01
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

    # Frame stacking
    frame_stack: int = 4
    """Number of frames to stack for temporal information"""

    # Action repeat
    action_repeat: int = 4
    """Number of times to repeat each action (frame skip)"""

    # UniFormer architecture arguments (simplified: 1 local + 1 global stage)
    uniformer_local_depth: int = 2
    """number of CBlock3D in local MHRA stage"""
    uniformer_global_depth: int = 1
    """number of SABlock3D in global MHRA stage"""
    uniformer_local_dim: int = 64
    """embedding dimension for local MHRA stage"""
    uniformer_global_dim: int = 128
    """embedding dimension for global MHRA stage"""
    uniformer_head_dim: int = 32
    """head dimension for global attention"""
    uniformer_mlp_ratio: float = 4.0
    """MLP expansion ratio"""
    uniformer_drop_path_rate: float = 0.1
    """stochastic depth drop rate"""
    uniformer_local_conv_size: Tuple[int, int, int] = (3, 5, 5)
    """kernel size (T, H, W) for local MHRA (3D conv) - temporal x spatial"""

    # Encoder output settings
    encoder_output_dim: int = 256
    """encoder output dimension"""
    encoder_output_scale: float = 0.1
    """scale factor for encoder output before tanh (prevents saturation)"""
    encoder_output_activation: str = "tanh"
    """output activation: 'tanh', 'layernorm', or 'none'"""

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


# =============================================================================
# UniFormer Components - True 3D Spatiotemporal Processing
# =============================================================================

class DynamicPositionEmbedding3D(nn.Module):
    """
    Dynamic Position Embedding (DPE) using 3D depthwise convolution.

    Uses 3x3x3 depthwise conv with zero padding to encode spatiotemporal
    positional information. Zero padding helps border tokens be aware of
    their absolute positions.

    Input: (B, T, H, W, C)
    Output: (B, T, H, W, C)
    """
    embed_dim: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)

    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W, C)
        # 3D depthwise conv for spatiotemporal position encoding
        out = nn.Conv(
            features=self.embed_dim,
            kernel_size=self.kernel_size,
            padding='SAME',
            feature_group_count=self.embed_dim,  # Depthwise
            kernel_init=lecun_normal(),
        )(x)
        return out


class LocalMHRA3D(nn.Module):
    """
    Local Multi-Head Relation Aggregator using 3D convolution.

    Implements the PWConv-DWConv-PWConv pattern from the paper:
    - PWConv (1x1x1): Pointwise conv for V projection
    - DWConv (txhxw): Depthwise conv for local spatiotemporal aggregation
    - PWConv (1x1x1): Pointwise conv for output projection

    The depthwise conv acts as local attention with learnable position-based
    weights a_n ∈ R^(t×h×w) within the neighborhood Ω_i^(t×h×w).

    Input: (B, T, H, W, C)
    Output: (B, T, H, W, C)
    """
    dim: int
    kernel_size: Tuple[int, int, int] = (3, 5, 5)  # Temporal x Spatial

    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W, C)

        # V projection (1x1x1 pointwise conv)
        v = nn.Conv(
            features=self.dim,
            kernel_size=(1, 1, 1),
            kernel_init=lecun_normal(),
        )(x)

        # Local aggregation (depthwise 3D conv - acts like local attention)
        # This learns the local affinity matrix a_n^(i-j) for j in neighborhood
        v = nn.Conv(
            features=self.dim,
            kernel_size=self.kernel_size,
            padding='SAME',
            feature_group_count=self.dim,  # Depthwise
            kernel_init=lecun_normal(),
        )(v)

        # Output projection (1x1x1 pointwise conv)
        out = nn.Conv(
            features=self.dim,
            kernel_size=(1, 1, 1),
            kernel_init=lecun_normal(),
        )(v)

        return out


class GlobalMHRA3D(nn.Module):
    """
    Global Multi-Head Relation Aggregator using self-attention.

    Performs self-attention across ALL spatiotemporal tokens (T × H × W).
    Each token can attend to every other token across all frames,
    enabling long-range temporal dependencies.

    Attention: A(Xi, Xj) = softmax(Q(Xi)^T K(Xj) / sqrt(d)) V(Xj)
    where i, j ∈ {1, ..., T×H×W}

    Input: (B, T, H, W, C)
    Output: (B, T, H, W, C)
    """
    dim: int
    num_heads: int
    head_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # Flatten spatiotemporal dimensions: (B, T, H, W, C) -> (B, T*H*W, C)
        N = T * H * W  # Total number of tokens
        x_flat = x.reshape(B, N, C)

        # QKV projection
        qkv = nn.Dense(
            features=3 * self.num_heads * self.head_dim,
            kernel_init=lecun_normal(),
        )(x_flat)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention over all T*H*W tokens
        scale = self.head_dim ** -0.5
        attn = jnp.einsum('bhid,bhjd->bhij', q, k) * scale  # (B, heads, N, N)
        attn = nn.softmax(attn, axis=-1)

        # Apply attention to values
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)  # (B, heads, N, head_dim)
        out = jnp.transpose(out, (0, 2, 1, 3))  # (B, N, heads, head_dim)
        out = out.reshape(B, N, self.num_heads * self.head_dim)

        # Output projection
        out = nn.Dense(
            features=self.dim,
            kernel_init=lecun_normal(),
        )(out)

        # Reshape back to spatiotemporal: (B, T*H*W, C) -> (B, T, H, W, C)
        out = out.reshape(B, T, H, W, self.dim)

        return out


class FFN(nn.Module):
    """
    Feed-Forward Network with GELU activation.

    Structure: Linear -> GELU -> Linear
    Applied independently to each token.
    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, kernel_init=lecun_normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim, kernel_init=lecun_normal())(x)
        return x


class CBlock3D(nn.Module):
    """
    Convolutional Block (Local MHRA) for shallow UniFormer stages.

    Uses 3D convolutions for local spatiotemporal relation aggregation.
    The local MHRA captures patterns within a t×h×w neighborhood.

    Structure:
        X = DPE(X_in) + X_in
        Y = LocalMHRA(Norm(X)) + X
        Z = FFN(Norm(Y)) + Y

    Input: (B, T, H, W, C)
    Output: (B, T, H, W, C)
    """
    dim: int
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    local_conv_size: Tuple[int, int, int] = (3, 5, 5)

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (B, T, H, W, C)

        # DPE: Dynamic Position Embedding (3x3x3 depthwise conv)
        x = x + DynamicPositionEmbedding3D(embed_dim=self.dim)(x)

        # Local MHRA with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = LocalMHRA3D(dim=self.dim, kernel_size=self.local_conv_size)(x)

        # Stochastic depth (drop path)
        if train and self.drop_path > 0:
            keep_prob = 1.0 - self.drop_path
            mask = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape=(x.shape[0], 1, 1, 1, 1)
            )
            x = x * mask / keep_prob

        x = residual + x

        # FFN with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        mlp_hidden = int(self.dim * self.mlp_ratio)
        x = FFN(hidden_dim=mlp_hidden, out_dim=self.dim)(x)

        if train and self.drop_path > 0:
            keep_prob = 1.0 - self.drop_path
            mask = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape=(x.shape[0], 1, 1, 1, 1)
            )
            x = x * mask / keep_prob

        x = residual + x

        return x


class SABlock3D(nn.Module):
    """
    Self-Attention Block (Global MHRA) for deep UniFormer stages.

    Uses global self-attention across ALL T×H×W spatiotemporal tokens.
    This enables each position to attend to all other positions across
    all frames, capturing long-range temporal dependencies.

    Structure:
        X = DPE(X_in) + X_in
        Y = GlobalMHRA(Norm(X)) + X
        Z = FFN(Norm(Y)) + Y

    Input: (B, T, H, W, C)
    Output: (B, T, H, W, C)
    """
    dim: int
    num_heads: int
    head_dim: int = 32
    mlp_ratio: float = 4.0
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (B, T, H, W, C)

        # DPE: Dynamic Position Embedding (3x3x3 depthwise conv)
        x = x + DynamicPositionEmbedding3D(embed_dim=self.dim)(x)

        # Global MHRA with pre-norm (attention over T*H*W tokens)
        residual = x
        x = nn.LayerNorm()(x)
        x = GlobalMHRA3D(dim=self.dim, num_heads=self.num_heads, head_dim=self.head_dim)(x)

        # Stochastic depth
        if train and self.drop_path > 0:
            keep_prob = 1.0 - self.drop_path
            mask = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape=(x.shape[0], 1, 1, 1, 1)
            )
            x = x * mask / keep_prob

        x = residual + x

        # FFN with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        mlp_hidden = int(self.dim * self.mlp_ratio)
        x = FFN(hidden_dim=mlp_hidden, out_dim=self.dim)(x)

        if train and self.drop_path > 0:
            keep_prob = 1.0 - self.drop_path
            mask = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape=(x.shape[0], 1, 1, 1, 1)
            )
            x = x * mask / keep_prob

        x = residual + x

        return x


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding layer using strided convolution.

    Reduces spatial resolution while preserving (or reducing) temporal dimension.
    First stage typically uses stride (2, 4, 4) for aggressive spatial reduction.
    Later stages use stride (1, 2, 2) to preserve temporal resolution.

    Input: (B, T, H, W, C)
    Output: (B, T', H', W', embed_dim)
    """
    embed_dim: int
    kernel_size: Tuple[int, int, int] = (1, 4, 4)
    stride: Tuple[int, int, int] = (1, 4, 4)

    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W, C)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='VALID',
            kernel_init=lecun_normal(),
        )(x)
        x = nn.LayerNorm()(x)
        return x


class UniFormerEncoder(nn.Module):
    """
    Simplified UniFormer encoder for pixel-based RL.

    Architecture (2 stages only):
    - LOCAL MHRA: PatchEmbed (84→7) + CBlock3D × local_depth
    - GLOBAL MHRA: PatchEmbed (channel proj) + SABlock3D × global_depth
    - Output: Flatten → Dense → 256

    For 84×84 input with 8 frames:
    - Local: (B, 8, 84, 84, 3) → (B, 8, 7, 7, 64)
    - Global: (B, 8, 7, 7, 64) → (B, 8, 7, 7, 128) with 392-token attention
    - Output: (B, 256)
    """
    local_depth: int = 2
    global_depth: int = 1
    local_dim: int = 64
    global_dim: int = 128
    head_dim: int = 32
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    local_conv_size: Tuple[int, int, int] = (3, 5, 5)
    out_dim: int = 256
    num_frames: int = 4
    output_scale: float = 0.1
    output_activation: str = "tanh"

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0

        B, H, W, C_total = x.shape
        T = self.num_frames
        C = C_total // T

        # Reshape to 3D: (B, H, W, T*C) → (B, T, H, W, C)
        x = x.reshape(B, H, W, T, C)
        x = jnp.transpose(x, (0, 3, 1, 2, 4))

        # Drop path scheduling
        total_blocks = self.local_depth + self.global_depth
        dpr = [v.item() for v in np.linspace(0, self.drop_path_rate, total_blocks)]
        block_idx = 0

        # ============================================================
        # LOCAL MHRA: 3D conv for local spatiotemporal patterns
        # PatchEmbed: 84×84 → 7×7 (kernel 12, stride 12)
        # ============================================================
        x = PatchEmbed3D(
            embed_dim=self.local_dim,
            kernel_size=(1, 12, 12),
            stride=(1, 12, 12),
        )(x)  # (B, T, 7, 7, local_dim)

        for _ in range(self.local_depth):
            x = CBlock3D(
                dim=self.local_dim,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[block_idx] if train else 0.0,
                local_conv_size=self.local_conv_size,
            )(x, train=train)
            block_idx += 1

        # ============================================================
        # GLOBAL MHRA: Self-attention over all T×7×7 tokens
        # PatchEmbed: channel projection only
        # ============================================================
        x = PatchEmbed3D(
            embed_dim=self.global_dim,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
        )(x)  # (B, T, 7, 7, global_dim)

        num_heads = max(1, self.global_dim // self.head_dim)
        for _ in range(self.global_depth):
            x = SABlock3D(
                dim=self.global_dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[block_idx] if train else 0.0,
            )(x, train=train)
            block_idx += 1

        # ============================================================
        # OUTPUT HEAD: Flatten → Dense → 256
        # ============================================================
        x = nn.LayerNorm()(x)
        x = x.reshape(B, -1)  # (B, T*7*7*global_dim)

        x = nn.Dense(
            features=self.out_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)

        if self.output_activation == "tanh":
            x = nn.LayerNorm()(x)
            x = nn.tanh(self.output_scale * x)
        elif self.output_activation == "layernorm":
            x = nn.LayerNorm()(x)

        return x


class Critic(nn.Module):
    """Value network with 2 hidden layers using Swish activation."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.swish(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.swish(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Continuous action actor with 2 hidden layers using Swish activation."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.swish(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.swish(x)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)
        actor_logstd = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,)
        )
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
class FrameStack:
    """Frame stacking buffer for temporal information in pixel-based RL."""
    frames: jnp.ndarray  # Shape: (n_envs, num_frames, H, W, C)

    @classmethod
    def create(cls, n_envs: int, num_frames: int, obs_shape: tuple):
        """Initialize frame stack with zeros."""
        h, w, c = obs_shape
        frames = jnp.zeros((n_envs, num_frames, h, w, c), dtype=jnp.uint8)
        return cls(frames=frames)

    def reset(self, obs: jnp.ndarray, done: jnp.ndarray = None):
        """Reset frame stack for environments that are done."""
        n_envs = obs.shape[0]
        num_frames = self.frames.shape[1]
        new_frames = jnp.broadcast_to(
            obs[:, None, :, :, :],
            (n_envs, num_frames, obs.shape[1], obs.shape[2], obs.shape[3])
        )

        if done is None:
            return self.replace(frames=new_frames)
        else:
            frames = jnp.where(
                done[:, None, None, None, None],
                new_frames,
                self.frames
            )
            return self.replace(frames=frames)

    def push(self, obs: jnp.ndarray):
        """Add a new frame to the stack, shifting out the oldest."""
        new_frames = jnp.concatenate([
            self.frames[:, 1:, :, :, :],
            obs[:, None, :, :, :]
        ], axis=1)
        return self.replace(frames=new_frames)

    def get_stacked(self) -> jnp.ndarray:
        """Get stacked observation by concatenating frames along channel dimension."""
        n_envs, num_frames, h, w, c = self.frames.shape
        frames_transposed = jnp.transpose(self.frames, (0, 2, 3, 1, 4))
        return frames_transposed.reshape(n_envs, h, w, c * num_frames)


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
#  Debug Metrics for Encoder Representations
# --------------------------------------------------------

def encoder_repr_metrics(hidden: jnp.ndarray) -> dict:
    """Compute metrics for encoder representation health."""
    metrics = {}

    metrics["repr/mean"] = jnp.mean(hidden)
    metrics["repr/std"] = jnp.std(hidden)
    metrics["repr/min"] = jnp.min(hidden)
    metrics["repr/max"] = jnp.max(hidden)

    unit_means = jnp.mean(hidden, axis=0)
    unit_stds = jnp.std(hidden, axis=0)

    metrics["repr/unit_mean_avg"] = jnp.mean(unit_means)
    metrics["repr/unit_std_avg"] = jnp.mean(unit_stds)
    metrics["repr/unit_std_min"] = jnp.min(unit_stds)
    metrics["repr/unit_std_max"] = jnp.max(unit_stds)

    dead_threshold = 0.01
    dead_units = jnp.mean(unit_stds < dead_threshold)
    metrics["repr/dead_units_frac"] = dead_units

    saturated_high = jnp.mean(jnp.abs(hidden) > 0.95)
    metrics["repr/saturated_frac"] = saturated_high

    active_units = jnp.mean(unit_stds > dead_threshold)
    metrics["repr/active_units_frac"] = active_units

    sample_norms = jnp.linalg.norm(hidden, axis=-1)
    metrics["repr/norm_mean"] = jnp.mean(sample_norms)
    metrics["repr/norm_std"] = jnp.std(sample_norms)
    metrics["repr/norm_min"] = jnp.min(sample_norms)
    metrics["repr/norm_max"] = jnp.max(sample_norms)

    sparsity = jnp.mean(jnp.abs(hidden) < 0.01)
    metrics["repr/sparsity"] = sparsity

    return metrics


def compute_grad_norms(grads: dict) -> dict:
    """Compute gradient norms for encoder, actor, and critic."""
    def tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))

    metrics = {}
    if 'network' in grads:
        metrics["grads/encoder_norm"] = tree_norm(grads['network'])
    if 'actor' in grads:
        metrics["grads/actor_norm"] = tree_norm(grads['actor'])
    if 'critic' in grads:
        metrics["grads/critic_norm"] = tree_norm(grads['critic'])

    return metrics


# --------------------------------------------------------
#  Optimizer: Adam for encoder, MUON for head matrices, Adam for head vectors
# --------------------------------------------------------

def create_encoder_adam_heads_muon_optimizer(
    encoder_lr,
    heads_muon_lr: float,
    heads_adam_lr,
    muon_dual_lr: float = 0.01,
    muon_dual_steps: int = 5,
    muon_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    max_grad_norm: float = 0.5,
    actor_muon_max_grad_norm: float = 1.0,
    critic_muon_max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
):
    """
    Create optimizer that uses:
    - Adam/AdamW for encoder (all params)
    - Manifold MUON for actor head matrices (2D+ with min dim > 1)
    - Manifold MUON for critic head matrices (2D+ with min dim > 1)
    - Adam/AdamW for actor/critic head vectors/scalars (biases, log_std, etc.)
    """

    if weight_decay > 0:
        adam_opt = lambda lr: optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr, eps=adam_eps, weight_decay=weight_decay
        )
    else:
        adam_opt = lambda lr: optax.inject_hyperparams(optax.adam)(
            learning_rate=lr, eps=adam_eps
        )

    encoder_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(encoder_lr),
    )

    heads_adam_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(heads_adam_lr),
    )

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

    transforms = {
        'encoder': encoder_tx,
        'actor_muon': actor_muon_tx,
        'critic_muon': critic_muon_tx,
        'heads_adam': heads_adam_tx,
    }

    def label_fn(params):
        def _label(path, param):
            if path[0] == 'network':
                return 'encoder'
            else:
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
        frame_stack=1,  # Get single RGB frames; we handle stacking ourselves
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
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

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
    print("PPO with UniFormer Backbone + Manifold MUON")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")

    print(f"\nUniFormer config (simplified 2-stage):")
    print(f"  LOCAL MHRA:  {args.uniformer_local_depth} CBlock3D, dim={args.uniformer_local_dim}")
    print(f"  GLOBAL MHRA: {args.uniformer_global_depth} SABlock3D, dim={args.uniformer_global_dim}")
    print(f"  head_dim: {args.uniformer_head_dim}")
    print(f"  mlp_ratio: {args.uniformer_mlp_ratio}")
    print(f"  drop_path_rate: {args.uniformer_drop_path_rate}")
    print(f"  local_conv_size (T,H,W): {args.uniformer_local_conv_size}")
    print(f"  num_frames (T): {args.frame_stack}")
    print(f"  output_dim: {args.encoder_output_dim}")
    print(f"  output_scale: {args.encoder_output_scale}")
    print(f"  output_activation: {args.encoder_output_activation}")

    adam_type = "AdamW" if args.weight_decay > 0 else "Adam"
    print(f"\nOptimizer config:")
    print(f"  encoder_lr ({adam_type}): {args.encoder_lr}")
    print(f"  heads_muon_lr (MUON for matrices): {args.heads_muon_lr}")
    print(f"  heads_adam_lr ({adam_type} for vectors): {args.heads_adam_lr}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  muon_dual_lr: {args.muon_dual_lr}")
    print(f"  muon_dual_steps: {args.muon_dual_steps}")
    print(f"  max_grad_norm ({adam_type}): {args.max_grad_norm}")
    print(f"  actor_muon_max_grad_norm: {args.actor_muon_max_grad_norm}")
    print(f"  critic_muon_max_grad_norm: {args.critic_muon_max_grad_norm}")
    print(f"  anneal_lr: {args.anneal_lr}")
    print("=" * 60)

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset(reset_rng)
    raw_obs_shape = init_env_state.pixels.shape[1:]  # (H, W, C)
    obs_shape = (raw_obs_shape[0], raw_obs_shape[1], raw_obs_shape[2] * args.frame_stack)
    print(f"raw_obs_shape: {raw_obs_shape}")
    print(f"frame_stack: {args.frame_stack}")
    print(f"action_repeat: {args.action_repeat}")
    print(f"obs_shape (with {args.frame_stack} stacked frames): {obs_shape}")

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

    # Initialize simplified UniFormer encoder (1 local + 1 global stage)
    network = UniFormerEncoder(
        local_depth=args.uniformer_local_depth,
        global_depth=args.uniformer_global_depth,
        local_dim=args.uniformer_local_dim,
        global_dim=args.uniformer_global_dim,
        head_dim=args.uniformer_head_dim,
        mlp_ratio=args.uniformer_mlp_ratio,
        drop_path_rate=args.uniformer_drop_path_rate,
        local_conv_size=args.uniformer_local_conv_size,
        out_dim=args.encoder_output_dim,
        num_frames=args.frame_stack,
        output_scale=args.encoder_output_scale,
        output_activation=args.encoder_output_activation,
    )
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init({'params': network_key, 'dropout': network_key}, dummy_obs, train=False)
    dummy_hidden = network.apply(network_params, dummy_obs, train=False)

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

    encoder_params = count_params(all_params, 'network')
    actor_params = count_params(all_params, 'actor')
    critic_params = count_params(all_params, 'critic')
    total_params = encoder_params + actor_params + critic_params

    actor_muon, actor_adam = count_by_type(all_params, 'actor')
    critic_muon, critic_adam = count_by_type(all_params, 'critic')

    print(f"\nParameter breakdown:")
    print(f"  Encoder/UniFormer (Adam): {encoder_params:,}")
    print(f"  Actor total: {actor_params:,} (MUON: {actor_muon:,}, Adam: {actor_adam:,})")
    print(f"  Critic total: {critic_params:,} (MUON: {critic_muon:,}, Adam: {critic_adam:,})")
    print(f"  Total: {total_params:,}")

    # Create learning rate schedules for Adam optimizers
    def make_linear_schedule(base_lr):
        """Linear annealing schedule for Adam optimizers."""
        def schedule(count):
            frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
            return base_lr * frac
        return schedule

    if args.anneal_lr:
        encoder_lr = make_linear_schedule(args.encoder_lr)
        heads_adam_lr = make_linear_schedule(args.heads_adam_lr)
        print(f"\nLearning rate annealing: ENABLED (linear decay for Adam)")
    else:
        encoder_lr = args.encoder_lr
        heads_adam_lr = args.heads_adam_lr
        print(f"\nLearning rate annealing: DISABLED")

    # Create optimizer
    tx = create_encoder_adam_heads_muon_optimizer(
        encoder_lr=encoder_lr,
        heads_muon_lr=args.heads_muon_lr,
        heads_adam_lr=heads_adam_lr,
        muon_dual_lr=args.muon_dual_lr,
        muon_dual_steps=args.muon_dual_steps,
        muon_msign_steps=args.muon_msign_steps,
        adam_eps=1e-5,
        max_grad_norm=args.max_grad_norm,
        actor_muon_max_grad_norm=args.actor_muon_max_grad_norm,
        critic_muon_max_grad_norm=args.critic_muon_max_grad_norm,
        weight_decay=args.weight_decay,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    # JIT compile network applications (no dropout during inference)
    @jax.jit
    def network_apply_inference(params, x):
        return network.apply(params, x, train=False)

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and return updated key."""
        hidden = network_apply_inference(agent_state.params['network'], next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params['actor'], hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic.apply(agent_state.params['critic'], hidden)

        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, logprob, value.squeeze(-1), key

    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
        dropout_key: jax.random.PRNGKey,
    ):
        """Calculate value, logprob of supplied action, and entropy (with training dropout)."""
        hidden = network.apply(params['network'], x, train=True, rngs={'dropout': dropout_key})
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
            network_apply_inference(agent_state.params['network'], next_obs)
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

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values, aug_key, dropout_key):
        # Apply random shift augmentation if enabled
        if args.use_augmentation:
            x = random_shift(aug_key, x, pad=args.augment_pad)

        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a, dropout_key)
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
            key, subkey, aug_key, dropout_key = jax.random.split(key, 4)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            aug_keys = jax.random.split(aug_key, args.num_minibatches)
            dropout_keys = jax.random.split(dropout_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state = carry
                minibatch, mb_aug_key, mb_dropout_key = inputs
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                    mb_dropout_key,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, (shuffled_storage, aug_keys, dropout_keys)
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

    # Initialize frame stack with initial observation
    frame_stack = FrameStack.create(args.n_envs, args.frame_stack, raw_obs_shape)
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
                opt_step = iteration * args.update_epochs * args.num_minibatches
                if args.anneal_lr:
                    frac = 1.0 - (opt_step // (args.num_minibatches * args.update_epochs)) / args.num_updates
                    encoder_lr_current = args.encoder_lr * frac
                    heads_adam_lr_current = args.heads_adam_lr * frac
                else:
                    encoder_lr_current = args.encoder_lr
                    heads_adam_lr_current = args.heads_adam_lr

                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/cumulative_episodic_return": cumulative_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/encoder_lr": encoder_lr_current,
                    "charts/heads_adam_lr": heads_adam_lr_current,
                    "charts/heads_muon_lr": args.heads_muon_lr,
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                }

                # Debug metrics for encoder representations
                if args.debug_repr:
                    sample_obs = storage.obs[0, :256]
                    hidden = network_apply_inference(agent_state.params['network'], sample_obs)

                    repr_metrics = encoder_repr_metrics(hidden)
                    for k, v in repr_metrics.items():
                        log_dict[k] = float(v)

                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(v)

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
