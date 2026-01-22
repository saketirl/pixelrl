#!/usr/bin/env python
"""
PPO for PixelBrax environments using Factorized Temporal CRATE.

This implementation uses a two-stage factorized attention approach:
1. Spatial CRATE: Attends over patches within each frame (shared weights)
2. Temporal CRATE: Attends over frame representations across time

Architecture:
    Input: (B, T, H, W, C_stacked)
           ↓
    Patchify each frame: (B, T, num_patches, patch_dim)
           ↓
    Spatial CRATE (per frame): (B, T, num_patches+1, embed_dim) → pool → (B, T, embed_dim)
           ↓
    Temporal CRATE: (B, T+1, embed_dim) → pool → (B, embed_dim)
           ↓
    Actor/Critic heads

Uses CRATE architecture (ISTA feedforward, simplified Q=K=V attention)
Based on: https://github.com/Ma-Lab-Berkeley/CRATE
"""
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import distrax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax
from pixelbrax.env_utils import make_pixel_brax

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
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
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    encoder_lr_scale: float = 1.0
    """scale factor for encoder learning rate (e.g., 0.1 = 10x slower than heads)"""
    weight_decay: float = 0.0
    """weight decay (L2 regularization) coefficient"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    warmup_steps: int = 10000
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
    """the maximum norm for the gradient clipping"""
    max_action: float = 1.0
    """maximum action value for clipping"""
    log_interval: int = 10
    """logging interval (in updates)"""

    # Debug/analysis flags
    debug_repr: bool = False
    """Enable lightweight representation health metrics logging"""

    # Hierarchical frame stacking
    temporal_stack: int = 8
    """Number of temporal tokens in the sequence"""
    channel_stack: int = 4
    """Number of frames stacked per token (channel-wise)"""

    # Action repeat
    action_repeat: int = 4
    """Number of times to repeat each action (frame skip)"""

    # Factorized CRATE architecture
    patch_size: int = 14
    """Size of each patch (patch_size x patch_size)"""
    embed_dim: int = 256
    """embedding dimension for CRATE"""
    spatial_depth: int = 2
    """number of spatial transformer layers (per-frame)"""
    temporal_depth: int = 2
    """number of temporal transformer layers (across frames)"""
    num_heads: int = 4
    """number of attention heads"""
    ista_step_size: float = 0.1
    """step size for ISTA gradient update"""
    ista_lambda: float = 0.1
    """sparsity regularization for ISTA"""
    emb_dropout: float = 0.0
    """dropout after embedding"""
    attn_dropout: float = 0.0
    """dropout in attention"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


# --------------------------------------------------------
#  CRATE Building Blocks
# --------------------------------------------------------

class ISTAFeedForward(nn.Module):
    """
    ISTA-based feedforward layer from CRATE.
    Uses gradient descent update with ReLU thresholding instead of standard MLP.
    Includes learnable bias for additional flexibility.
    """
    dim: int
    step_size: float = 0.1
    lambd: float = 0.1

    @nn.compact
    def __call__(self, x):
        weight = self.param(
            "weight",
            nn.initializers.kaiming_uniform(),
            (self.dim, self.dim)
        )
        bias = self.param(
            "bias",
            nn.initializers.zeros,
            (self.dim,)
        )
        x1 = x @ weight.T
        grad_1 = x1 @ weight
        grad_2 = x @ weight
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
        output = nn.relu(x + grad_update + bias)
        return output


class CRATEAttention(nn.Module):
    """
    CRATE-style attention with single projection (Q=K=V).
    """
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        B, N, _ = x.shape
        inner_dim = self.dim_head * self.heads

        qkv = nn.Dense(
            inner_dim,
            use_bias=False,
            kernel_init=orthogonal(1.0),
        )(x)

        w = qkv.reshape(B, N, self.heads, self.dim_head)
        w = jnp.transpose(w, (0, 2, 1, 3))

        scale = self.dim_head ** -0.5
        dots = jnp.matmul(w, jnp.transpose(w, (0, 1, 3, 2))) * scale
        attn = nn.softmax(dots, axis=-1)
        attn = nn.Dropout(self.dropout, deterministic=deterministic)(attn)

        out = jnp.matmul(attn, w)
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(B, N, inner_dim)

        out = nn.Dense(
            self.dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(out)
        out = nn.Dropout(self.dropout, deterministic=deterministic)(out)

        return out


class CRATEBlock(nn.Module):
    """
    CRATE transformer block matching original implementation.

    Structure (from https://github.com/Ma-Lab-Berkeley/CRATE):
        grad_x = PreNorm(Attention)(x) + x   # Attention with residual
        x = PreNorm(FeedForward)(grad_x)     # ISTA operates on residual sum

    The residual is added BEFORE feedforward, not after.
    """
    dim: int
    heads: int
    dim_head: int
    dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # -------------------
        # Attention (PreNorm + Residual)
        # -------------------
        y = nn.LayerNorm()(x)
        y = CRATEAttention(
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
        )(y, deterministic=deterministic)
        x = x + y

        # -------------------
        # ISTA FeedForward (PreNorm + Residual)
        # -------------------
        y = nn.LayerNorm()(x)
        z = ISTAFeedForward(
            dim=self.dim,
            step_size=self.ista_step_size,
            lambd=self.ista_lambda,  # = 0
        )(y)

        # Swish instead of ReLU
        z = z * nn.sigmoid(z)

        # Residual on same state
        x = x + z
        return x


# --------------------------------------------------------
#  Factorized CRATE Encoder
# --------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    CRATE-style patch embedding.
    Converts (H, W, C) → (num_patches, embed_dim)

    Uses: Rearrange → LayerNorm → Linear → LayerNorm
    """
    patch_size: int = 14
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, H, W, C) - input image
        Returns:
            (B, num_patches, embed_dim) - patch tokens
        """
        B, H, W, C = x.shape
        p = self.patch_size
        num_patches_h = H // p
        num_patches_w = W // p
        num_patches = num_patches_h * num_patches_w
        patch_dim = p * p * C

        # Rearrange into patches: (B, H, W, C) → (B, num_patches, patch_dim)
        # Equivalent to einops: 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)'
        x = x.reshape(B, num_patches_h, p, num_patches_w, p, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))  # (B, nh, nw, p, p, C)
        x = x.reshape(B, num_patches, patch_dim)

        # LayerNorm → Linear → LayerNorm (CRATE style)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.embed_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        x = nn.LayerNorm()(x)

        return x


class SpatialCRATE(nn.Module):
    """
    Spatial CRATE encoder for a single frame.
    Processes patches within one frame using self-attention.
    """
    patch_size: int = 14
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 4
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: (B, H, W, C) - single frame (normalized)
        Returns:
            (B, embed_dim) - frame representation (CLS token)
        """
        B, H, W, C = x.shape
        dim_head = self.embed_dim // self.num_heads
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Patch embedding
        x = PatchEmbed(patch_size=self.patch_size, embed_dim=self.embed_dim)(x)
        # x: (B, num_patches, embed_dim)

        # CLS token
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1 + num_patches, embed_dim)

        # Positional embeddings
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1 + num_patches, self.embed_dim)
        )
        x = x + pos_embed

        # Embedding dropout
        x = nn.Dropout(self.emb_dropout, deterministic=deterministic)(x)

        # CRATE transformer blocks
        for _ in range(self.depth):
            x = CRATEBlock(
                dim=self.embed_dim,
                heads=self.num_heads,
                dim_head=dim_head,
                dropout=self.attn_dropout,
                ista_step_size=self.ista_step_size,
                ista_lambda=self.ista_lambda,
            )(x, deterministic=deterministic)

        # Return CLS token as frame representation
        x = nn.LayerNorm()(x)
        return x[:, 0]  # (B, embed_dim)


class TemporalCRATE(nn.Module):
    """
    Temporal CRATE encoder.
    Processes frame representations across time using self-attention.
    """
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 4
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: (B, T, embed_dim) - sequence of frame representations
        Returns:
            (B, embed_dim) - temporal representation (CLS token)
        """
        B, T, _ = x.shape
        dim_head = self.embed_dim // self.num_heads

        # CLS token for temporal sequence
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1 + T, embed_dim)

        # Temporal positional embeddings
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1 + T, self.embed_dim)
        )
        x = x + pos_embed

        # Embedding dropout
        x = nn.Dropout(self.emb_dropout, deterministic=deterministic)(x)

        # CRATE transformer blocks
        for _ in range(self.depth):
            x = CRATEBlock(
                dim=self.embed_dim,
                heads=self.num_heads,
                dim_head=dim_head,
                dropout=self.attn_dropout,
                ista_step_size=self.ista_step_size,
                ista_lambda=self.ista_lambda,
            )(x, deterministic=deterministic)

        # Return CLS token as final representation
        x = nn.LayerNorm()(x)
        return x[:, 0]  # (B, embed_dim)


class FactorizedCRATEEncoder(nn.Module):
    """
    Factorized Spatial-Temporal CRATE encoder.

    Two-stage attention:
    1. Spatial: Attends over patches within each frame (shared weights across T)
    2. Temporal: Attends over frame representations across time

    This factorization reduces attention cost from O((T*P)^2) to O(P^2) + O(T^2)
    where T = temporal tokens, P = spatial patches per frame.
    """
    temporal_stack: int = 8
    channel_stack: int = 4
    patch_size: int = 14
    embed_dim: int = 256
    spatial_depth: int = 2
    temporal_depth: int = 2
    num_heads: int = 4
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: (B, T, H, W, C_stacked) - batch of frame sequences
               T = temporal_stack (number of temporal tokens)
               C_stacked = C * channel_stack (channel-stacked frames per token)

        Returns:
            (B, embed_dim) - encoded representation
        """
        B, T, H, W, C_stacked = x.shape

        # Normalize pixel values
        x = x.astype(jnp.float32) / 255.0

        # Stage 1: Spatial CRATE (shared across temporal dimension)
        # Reshape: (B, T, H, W, C) → (B*T, H, W, C)
        x = x.reshape(B * T, H, W, C_stacked)

        # Apply spatial encoder to each frame
        spatial_encoder = SpatialCRATE(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.spatial_depth,
            num_heads=self.num_heads,
            emb_dropout=self.emb_dropout,
            attn_dropout=self.attn_dropout,
            ista_step_size=self.ista_step_size,
            ista_lambda=self.ista_lambda,
        )
        x = spatial_encoder(x, deterministic=deterministic)
        # x: (B*T, embed_dim)

        # Reshape back: (B*T, embed_dim) → (B, T, embed_dim)
        x = x.reshape(B, T, self.embed_dim)

        # Stage 2: Temporal CRATE
        temporal_encoder = TemporalCRATE(
            embed_dim=self.embed_dim,
            depth=self.temporal_depth,
            num_heads=self.num_heads,
            emb_dropout=self.emb_dropout,
            attn_dropout=self.attn_dropout,
            ista_step_size=self.ista_step_size,
            ista_lambda=self.ista_lambda,
        )
        x = temporal_encoder(x, deterministic=deterministic)
        # x: (B, embed_dim)

        return x


# --------------------------------------------------------
#  Actor-Critic Heads
# --------------------------------------------------------

class Critic(nn.Module):
    """Value network with 2 hidden layers."""
    @nn.compact
    def __call__(self, x, return_activations: bool = False):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        h1 = nn.tanh(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h1)
        h2 = nn.tanh(x)
        out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h2)
        if return_activations:
            return out, (h1, h2)
        return out


class Actor(nn.Module):
    """Continuous action actor with Gaussian distribution."""
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x, return_activations: bool = False):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        h1 = nn.tanh(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h1)
        h2 = nn.tanh(x)
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


def tanh_saturation_metrics(activations: tuple, name: str) -> dict:
    """
    Compute tanh saturation metrics.
    Saturation occurs when |tanh(x)| > 0.95 (close to -1 or +1).
    """
    metrics = {}
    for i, h in enumerate(activations):
        layer_name = f"{name}/layer{i+1}"
        abs_h = jnp.abs(h)
        metrics[f"{layer_name}/mean"] = jnp.mean(h)
        metrics[f"{layer_name}/std"] = jnp.std(h)
        metrics[f"{layer_name}/saturated_frac"] = jnp.mean(abs_h > 0.95)  # Fraction saturated
        metrics[f"{layer_name}/saturated_pos"] = jnp.mean(h > 0.95)  # Saturated positive
        metrics[f"{layer_name}/saturated_neg"] = jnp.mean(h < -0.95)  # Saturated negative
        metrics[f"{layer_name}/abs_mean"] = jnp.mean(abs_h)
    return metrics


# --------------------------------------------------------
#  Storage and Statistics
# --------------------------------------------------------

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
    Hierarchical frame stacking buffer.
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
        n_envs, total_frames, H, W, C = self.frames.shape
        x = self.frames.reshape(n_envs, self.temporal_stack, self.channel_stack, H, W, C)
        x = jnp.transpose(x, (0, 1, 3, 4, 2, 5))
        x = x.reshape(n_envs, self.temporal_stack, H, W, self.channel_stack * C)
        return x


@flax.struct.dataclass
class RewardNormalizer:
    """Reward normalization using discounted returns."""
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
        normalized = rewards / jnp.sqrt(self.return_rms_var + epsilon)
        return jnp.clip(normalized, -clip, clip)


# --------------------------------------------------------
#  Debug Metrics
# --------------------------------------------------------

def repr_health(hidden: jnp.ndarray) -> dict:
    """Compute lightweight representation health metrics."""
    dim_std = jnp.std(hidden, axis=0)
    norms = jnp.linalg.norm(hidden, axis=-1)
    return {
        "repr/mean": jnp.mean(hidden),
        "repr/std": jnp.std(hidden),
        "repr/norm_mean": jnp.mean(norms),
        "repr/norm_std": jnp.std(norms),
        "repr/dead_dims": jnp.mean(dim_std < 0.01),
        "repr/dim_std_min": jnp.min(dim_std),
        "repr/dim_std_max": jnp.max(dim_std),
        "repr/sparsity": jnp.mean(jnp.abs(hidden) < 0.01),
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


# --------------------------------------------------------
#  Environment
# --------------------------------------------------------

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
    )
    try:
        action_dim = envs.action_size
    except AttributeError:
        action_dim = envs.env.action_size
    return envs, action_dim


# --------------------------------------------------------
#  Main Training Loop
# --------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.env_name}__factorized_crate__{args.seed}__{int(time.time())}"

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
    print("Factorized CRATE PPO")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")

    total_frames = args.temporal_stack * args.channel_stack
    num_patches = (args.hw // args.patch_size) ** 2
    print(f"\nFactorized CRATE config:")
    print(f"  temporal_stack: {args.temporal_stack}")
    print(f"  channel_stack: {args.channel_stack}")
    print(f"  total_frames: {total_frames}")
    print(f"  patch_size: {args.patch_size}")
    print(f"  num_patches per frame: {num_patches}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"  spatial_depth: {args.spatial_depth}")
    print(f"  temporal_depth: {args.temporal_depth}")
    print(f"  num_heads: {args.num_heads}")
    print(f"\nAttention cost comparison:")
    print(f"  Unfactorized: O(({args.temporal_stack}×{num_patches})²) = {(args.temporal_stack * num_patches)**2:,}")
    print(f"  Factorized: O({num_patches+1}²) + O({args.temporal_stack+1}²) = {(num_patches+1)**2 + (args.temporal_stack+1)**2:,}")
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

    def lr_schedule(count):
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
        return args.learning_rate * warmup_ratio * anneal_ratio

    # Initialize Factorized CRATE network
    network = FactorizedCRATEEncoder(
        temporal_stack=args.temporal_stack,
        channel_stack=args.channel_stack,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        num_heads=args.num_heads,
        emb_dropout=args.emb_dropout,
        attn_dropout=args.attn_dropout,
        ista_step_size=args.ista_step_size,
        ista_lambda=args.ista_lambda,
    )
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    # Dummy input
    dummy_obs = jnp.zeros((1,) + obs_shape)
    print(f"dummy_obs shape: {dummy_obs.shape}")

    network_params = network.init(network_key, dummy_obs, deterministic=True)
    dummy_hidden = network.apply(network_params, dummy_obs, deterministic=True)
    print(f"encoder output shape: {dummy_hidden.shape}")

    # Initialize all params first
    actor_params = actor.init(actor_key, dummy_hidden)
    critic_params = critic.init(critic_key, dummy_hidden)
    all_params = {
        'network': network_params,
        'actor': actor_params,
        'critic': critic_params,
    }

    # Create optimizer with separate LR for encoder and weight decay
    use_schedule = args.anneal_lr or args.warmup_steps > 0

    def make_optimizer(lr_scale=1.0):
        """Create optimizer with optional LR scaling."""
        def scaled_lr_schedule(count):
            return lr_schedule(count) * lr_scale

        if args.weight_decay > 0:
            return optax.adamw(
                learning_rate=scaled_lr_schedule if use_schedule else args.learning_rate * lr_scale,
                eps=1e-5,
                weight_decay=args.weight_decay,
            )
        else:
            return optax.adam(
                learning_rate=scaled_lr_schedule if use_schedule else args.learning_rate * lr_scale,
                eps=1e-5,
            )

    # Different LR for encoder vs actor/critic
    # Create param_labels with same structure as frozen params
    def label_fn(params):
        """Label each param by its top-level key."""
        def _label(path, _):
            return path[0]  # 'network', 'actor', or 'critic'
        flat = flax.traverse_util.flatten_dict(params)
        labeled = {k: _label(k, v) for k, v in flat.items()}
        return flax.core.freeze(flax.traverse_util.unflatten_dict(labeled))

    param_labels = label_fn(flax.core.unfreeze(flax.core.freeze(all_params)))

    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.multi_transform(
            transforms={
                'network': make_optimizer(lr_scale=args.encoder_lr_scale),
                'actor': make_optimizer(lr_scale=1.0),
                'critic': make_optimizer(lr_scale=1.0),
            },
            param_labels=param_labels,
        ),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze(all_params),
        tx=tx,
    )

    print(f"encoder_lr_scale: {args.encoder_lr_scale}")
    print(f"weight_decay: {args.weight_decay}")

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(agent_state.params))
    print(f"Total parameters: {param_count:,}")

    network.apply = jax.jit(partial(network.apply, deterministic=True))
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
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

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        if args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + jnp.clip(newvalue - mb_values, -args.clip_eps, args.clip_eps)
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
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                agent_state = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        final_grads = jax.tree_map(lambda x: x[-1, -1], grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, final_grads, key

    # Start training
    global_step = 0
    start_time = time.time()

    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    env_state = envs.reset(reset_rngs)

    frame_stack = HierarchicalFrameStack.create(
        args.n_envs, args.temporal_stack, args.channel_stack, raw_obs_shape
    )
    frame_stack = frame_stack.reset(env_state.pixels)
    next_obs = frame_stack.get_stacked()
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

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
                lr = float(lr_schedule(opt_step))
                encoder_lr = lr * args.encoder_lr_scale

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
                    "charts/learning_rate": lr,
                    "charts/encoder_learning_rate": encoder_lr,
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

                if args.debug_repr:
                    hidden = network.apply(agent_state.params['network'], next_obs)
                    health_metrics = repr_health(hidden)
                    for k, v in health_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    # Tanh saturation metrics for actor and critic
                    _, _, actor_activations = actor.apply(
                        agent_state.params['actor'], hidden, return_activations=True
                    )
                    actor_sat_metrics = tanh_saturation_metrics(actor_activations, "actor_tanh")
                    for k, v in actor_sat_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    _, critic_activations = critic.apply(
                        agent_state.params['critic'], hidden, return_activations=True
                    )
                    critic_sat_metrics = tanh_saturation_metrics(critic_activations, "critic_tanh")
                    for k, v in critic_sat_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
