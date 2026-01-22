#!/usr/bin/env python
"""
PPO for PixelBrax environments using Factorized CRATE with Swish activations
and Manifold MUON optimizer.

Key differences from swish version:
- Uses Manifold MUON for all matrices (2D+ parameters) - keeps weights on Stiefel manifold
- Uses Adam for vectors and scalars (1D and 0D parameters) - biases, layer norms, log_std, etc.
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
import wandb
import tyro
import distrax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax
from pixelbrax.env_utils import make_pixel_brax

# Import manifold MUON optimizer
from manifold_muon_optax import manifold_muon

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_GIT"] = "true"
os.environ["WANDB_DISABLE_PACKAGE_CHECK"] = "true"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 0
    track: bool = False
    wandb_project_name: str = "benchmark"
    wandb_entity: str = None

    # Environment arguments
    env_name: str = "halfcheetah"
    backend: str = "spring"
    n_envs: int = 512
    hw: int = 84

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    # Learning rates for each component/param-type combination
    encoder_muon_lr: float = 3e-4
    encoder_adam_lr: float = 3e-4
    heads_muon_lr: float = 3e-4
    heads_adam_lr: float = 3e-4
    weight_decay: float = 0.0
    num_steps: int = 256
    anneal_lr: bool = False
    warmup_steps: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    max_action: float = 1.0
    log_interval: int = 10

    # Manifold MUON optimizer arguments
    muon_dual_lr: float = 0.01
    muon_dual_steps: int = 5
    muon_msign_steps: int = 5

    # Debug/analysis flags
    debug_repr: bool = False

    # Hierarchical frame stacking
    temporal_stack: int = 8
    channel_stack: int = 4

    # Action repeat
    action_repeat: int = 4

    # Factorized CRATE architecture
    patch_size: int = 14
    embed_dim: int = 256
    spatial_depth: int = 2
    temporal_depth: int = 2
    num_heads: int = 4
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0


# --------------------------------------------------------
#  Swish activation
# --------------------------------------------------------

def swish(x):
    """Swish activation: x * sigmoid(x)"""
    return x * nn.sigmoid(x)


# --------------------------------------------------------
#  ISTA with Swish activation
# --------------------------------------------------------

class ISTASwish(nn.Module):
    """
    ISTA-based layer with Swish activation instead of ReLU.
    Swish doesn't hard-zero negatives, preventing activation death.
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
        output = swish(x + grad_update + bias)
        return output


# --------------------------------------------------------
#  CRATE Building Blocks with Swish
# --------------------------------------------------------

class CRATEAttention(nn.Module):
    """CRATE-style attention with single projection (Q=K=V)."""
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
    CRATE transformer block with Swish ISTA.

    Structure:
        grad_x = PreNorm(Attention)(x) + x
        x = PreNorm(ISTASwish)(grad_x)
    """
    dim: int
    heads: int
    dim_head: int
    dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True, return_intermediates: bool = False):
        # Attention with residual
        y = nn.LayerNorm()(x)
        attn_out = CRATEAttention(
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
        )(y, deterministic=deterministic)
        grad_x = attn_out + x

        # ISTA with Swish (ISTASwish already applies swish internally)
        y = nn.LayerNorm()(grad_x)
        z = ISTASwish(
            dim=self.dim,
            step_size=self.ista_step_size,
            lambd=self.ista_lambda,
        )(y)

        # Residual connection
        x = x + z

        if return_intermediates:
            return x, attn_out
        return x


# --------------------------------------------------------
#  Factorized CRATE Encoder
# --------------------------------------------------------

class PatchEmbed(nn.Module):
    """CRATE-style patch embedding."""
    patch_size: int = 14
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        num_patches_h = H // p
        num_patches_w = W // p
        num_patches = num_patches_h * num_patches_w
        patch_dim = p * p * C

        x = x.reshape(B, num_patches_h, p, num_patches_w, p, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape(B, num_patches, patch_dim)

        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.embed_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        x = nn.LayerNorm()(x)

        return x


class SpatialCRATE(nn.Module):
    """Spatial CRATE encoder for a single frame."""
    patch_size: int = 14
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 4
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True, return_intermediates: bool = False):
        B, H, W, C = x.shape
        dim_head = self.embed_dim // self.num_heads
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        x = PatchEmbed(patch_size=self.patch_size, embed_dim=self.embed_dim)(x)

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1 + num_patches, self.embed_dim)
        )
        x = x + pos_embed

        x = nn.Dropout(self.emb_dropout, deterministic=deterministic)(x)

        attn_outputs = []
        for _ in range(self.depth):
            if return_intermediates:
                x, attn_out = CRATEBlock(
                    dim=self.embed_dim,
                    heads=self.num_heads,
                    dim_head=dim_head,
                    dropout=self.attn_dropout,
                    ista_step_size=self.ista_step_size,
                    ista_lambda=self.ista_lambda,
                )(x, deterministic=deterministic, return_intermediates=True)
                attn_outputs.append(attn_out)
            else:
                x = CRATEBlock(
                    dim=self.embed_dim,
                    heads=self.num_heads,
                    dim_head=dim_head,
                    dropout=self.attn_dropout,
                    ista_step_size=self.ista_step_size,
                    ista_lambda=self.ista_lambda,
                )(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)

        if return_intermediates:
            # Return: CLS output, full sequence (all tokens), attention outputs per layer
            return x[:, 0], x, attn_outputs
        return x[:, 0]


class TemporalCRATE(nn.Module):
    """Temporal CRATE encoder."""
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 4
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True, return_intermediates: bool = False):
        B, T, _ = x.shape
        dim_head = self.embed_dim // self.num_heads

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1 + T, self.embed_dim)
        )
        x = x + pos_embed

        x = nn.Dropout(self.emb_dropout, deterministic=deterministic)(x)

        attn_outputs = []
        for _ in range(self.depth):
            if return_intermediates:
                x, attn_out = CRATEBlock(
                    dim=self.embed_dim,
                    heads=self.num_heads,
                    dim_head=dim_head,
                    dropout=self.attn_dropout,
                    ista_step_size=self.ista_step_size,
                    ista_lambda=self.ista_lambda,
                )(x, deterministic=deterministic, return_intermediates=True)
                attn_outputs.append(attn_out)
            else:
                x = CRATEBlock(
                    dim=self.embed_dim,
                    heads=self.num_heads,
                    dim_head=dim_head,
                    dropout=self.attn_dropout,
                    ista_step_size=self.ista_step_size,
                    ista_lambda=self.ista_lambda,
                )(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)

        if return_intermediates:
            # Return: CLS output, full sequence (all tokens), attention outputs per layer
            return x[:, 0], x, attn_outputs
        return x[:, 0]


class FactorizedCRATEEncoder(nn.Module):
    """Factorized Spatial-Temporal CRATE encoder with Swish ISTA."""
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
    def __call__(self, x, deterministic: bool = True, return_intermediates: bool = False):
        B, T, H, W, C_stacked = x.shape

        x = x.astype(jnp.float32) / 255.0
        x = x.reshape(B * T, H, W, C_stacked)

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

        if return_intermediates:
            spatial_cls, spatial_seq, spatial_attn = spatial_encoder(
                x, deterministic=deterministic, return_intermediates=True
            )
        else:
            spatial_cls = spatial_encoder(x, deterministic=deterministic)

        x = spatial_cls.reshape(B, T, self.embed_dim)

        temporal_encoder = TemporalCRATE(
            embed_dim=self.embed_dim,
            depth=self.temporal_depth,
            num_heads=self.num_heads,
            emb_dropout=self.emb_dropout,
            attn_dropout=self.attn_dropout,
            ista_step_size=self.ista_step_size,
            ista_lambda=self.ista_lambda,
        )

        if return_intermediates:
            temporal_cls, temporal_seq, temporal_attn = temporal_encoder(
                x, deterministic=deterministic, return_intermediates=True
            )
            intermediates = {
                'spatial_cls': spatial_cls,  # (B*T, embed_dim)
                'spatial_seq': spatial_seq,  # (B*T, num_patches+1, embed_dim)
                'spatial_attn': spatial_attn,  # list of (B*T, num_patches+1, embed_dim)
                'temporal_cls': temporal_cls,  # (B, embed_dim)
                'temporal_seq': temporal_seq,  # (B, T+1, embed_dim)
                'temporal_attn': temporal_attn,  # list of (B, T+1, embed_dim)
            }
            return temporal_cls, intermediates

        x = temporal_encoder(x, deterministic=deterministic)
        return x


# --------------------------------------------------------
#  Actor-Critic Heads with ISTA Swish
# --------------------------------------------------------

class ISTACritic(nn.Module):
    """Value network with ISTA Swish layers instead of Dense+tanh."""
    hidden_dim: int = 256
    ista_step_size: float = 0.1
    ista_lambda: float = 0.0

    @nn.compact
    def __call__(self, x, return_activations: bool = False):
        # Project to hidden dim if needed
        if x.shape[-1] != self.hidden_dim:
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        # ISTA layer 1
        h1 = ISTASwish(dim=self.hidden_dim, step_size=self.ista_step_size, lambd=self.ista_lambda)(x)

        # ISTA layer 2
        h2 = ISTASwish(dim=self.hidden_dim, step_size=self.ista_step_size, lambd=self.ista_lambda)(h1)

        # Output
        out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h2)

        if return_activations:
            return out, (h1, h2)
        return out


class ISTAActor(nn.Module):
    """Continuous action actor with ISTA Swish layers instead of Dense+tanh."""
    action_dim: int
    hidden_dim: int = 256
    ista_step_size: float = 0.1
    ista_lambda: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x, return_activations: bool = False):
        # Project to hidden dim if needed
        if x.shape[-1] != self.hidden_dim:
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        # ISTA layer 1
        h1 = ISTASwish(dim=self.hidden_dim, step_size=self.ista_step_size, lambd=self.ista_lambda)(x)

        # ISTA layer 2
        h2 = ISTASwish(dim=self.hidden_dim, step_size=self.ista_step_size, lambd=self.ista_lambda)(h1)

        # Output
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


def swish_activation_metrics(activations: tuple, name: str) -> dict:
    """
    Compute activation metrics for Swish layers.
    Unlike tanh, Swish doesn't saturate, so we track different metrics.
    """
    metrics = {}
    for i, h in enumerate(activations):
        layer_name = f"{name}/layer{i+1}"
        metrics[f"{layer_name}/mean"] = jnp.mean(h)
        metrics[f"{layer_name}/std"] = jnp.std(h)
        metrics[f"{layer_name}/min"] = jnp.min(h)
        metrics[f"{layer_name}/max"] = jnp.max(h)
        metrics[f"{layer_name}/dead_frac"] = jnp.mean(jnp.abs(h) < 0.01)  # Near-zero activations
        metrics[f"{layer_name}/negative_frac"] = jnp.mean(h < 0)  # Fraction negative
    return metrics


def attention_vs_cls_metrics(intermediates: dict) -> dict:
    """
    Compute metrics comparing attention outputs vs CLS token outputs.

    Tracks:
    - Norms and statistics of attention outputs at each layer
    - Norms and statistics of CLS tokens
    - Ratio of attention to CLS norms (information flow)
    - Cosine similarity between CLS and mean of patch tokens
    """
    metrics = {}

    def compute_stats(x, prefix):
        """Compute standard statistics for a tensor."""
        norms = jnp.linalg.norm(x, axis=-1)
        return {
            f"{prefix}/mean": jnp.mean(x),
            f"{prefix}/std": jnp.std(x),
            f"{prefix}/norm_mean": jnp.mean(norms),
            f"{prefix}/norm_std": jnp.std(norms),
            f"{prefix}/norm_min": jnp.min(norms),
            f"{prefix}/norm_max": jnp.max(norms),
        }

    # Spatial CLS output stats
    spatial_cls = intermediates['spatial_cls']
    metrics.update(compute_stats(spatial_cls, "spatial_cls"))

    # Temporal CLS output stats (final encoder output)
    temporal_cls = intermediates['temporal_cls']
    metrics.update(compute_stats(temporal_cls, "temporal_cls"))

    # Spatial attention outputs per layer
    for i, attn_out in enumerate(intermediates['spatial_attn']):
        # attn_out shape: (B*T, num_patches+1, embed_dim)
        # CLS attention (first token)
        cls_attn = attn_out[:, 0, :]
        # Patch attention (rest of tokens)
        patch_attn = attn_out[:, 1:, :]

        metrics.update(compute_stats(cls_attn, f"spatial_attn_L{i}/cls"))
        metrics.update(compute_stats(patch_attn, f"spatial_attn_L{i}/patches"))

        # Ratio of CLS to patch attention norms
        cls_norm = jnp.linalg.norm(cls_attn, axis=-1)
        patch_norm = jnp.mean(jnp.linalg.norm(patch_attn, axis=-1), axis=-1)
        metrics[f"spatial_attn_L{i}/cls_to_patch_ratio"] = jnp.mean(cls_norm / (patch_norm + 1e-8))

    # Temporal attention outputs per layer
    for i, attn_out in enumerate(intermediates['temporal_attn']):
        # attn_out shape: (B, T+1, embed_dim)
        cls_attn = attn_out[:, 0, :]
        frame_attn = attn_out[:, 1:, :]

        metrics.update(compute_stats(cls_attn, f"temporal_attn_L{i}/cls"))
        metrics.update(compute_stats(frame_attn, f"temporal_attn_L{i}/frames"))

        # Ratio of CLS to frame attention norms
        cls_norm = jnp.linalg.norm(cls_attn, axis=-1)
        frame_norm = jnp.mean(jnp.linalg.norm(frame_attn, axis=-1), axis=-1)
        metrics[f"temporal_attn_L{i}/cls_to_frame_ratio"] = jnp.mean(cls_norm / (frame_norm + 1e-8))

    # Full sequence comparisons
    # Spatial: compare CLS to mean of patches
    spatial_seq = intermediates['spatial_seq']  # (B*T, num_patches+1, embed_dim)
    spatial_cls_final = spatial_seq[:, 0, :]
    spatial_patches_mean = jnp.mean(spatial_seq[:, 1:, :], axis=1)

    # Cosine similarity between CLS and mean patch
    cos_sim = jnp.sum(spatial_cls_final * spatial_patches_mean, axis=-1) / (
        jnp.linalg.norm(spatial_cls_final, axis=-1) * jnp.linalg.norm(spatial_patches_mean, axis=-1) + 1e-8
    )
    metrics["spatial/cls_patch_cosine_sim"] = jnp.mean(cos_sim)

    # Temporal: compare CLS to mean of frames
    temporal_seq = intermediates['temporal_seq']  # (B, T+1, embed_dim)
    temporal_cls_final = temporal_seq[:, 0, :]
    temporal_frames_mean = jnp.mean(temporal_seq[:, 1:, :], axis=1)

    cos_sim = jnp.sum(temporal_cls_final * temporal_frames_mean, axis=-1) / (
        jnp.linalg.norm(temporal_cls_final, axis=-1) * jnp.linalg.norm(temporal_frames_mean, axis=-1) + 1e-8
    )
    metrics["temporal/cls_frame_cosine_sim"] = jnp.mean(cos_sim)

    return metrics


def orthogonality_metrics(params: dict) -> dict:
    """
    Compute orthogonality metrics for weight matrices.
    MUON should keep matrices close to orthogonal.
    """
    metrics = {}

    def check_orthogonality(path, param):
        # Only check 2D matrices with min dimension > 1 (same as MUON condition)
        if param.ndim == 2 and min(param.shape) > 1:
            # For a matrix W, orthogonality means W^T @ W ≈ I (or W @ W^T ≈ I)
            if param.shape[0] <= param.shape[1]:
                # More columns than rows: check W @ W^T
                prod = param @ param.T
            else:
                # More rows than columns: check W^T @ W
                prod = param.T @ param

            eye = jnp.eye(prod.shape[0])
            deviation = jnp.mean((prod - eye) ** 2)
            return deviation
        return None

    flat_params = flax.traverse_util.flatten_dict(params)
    deviations = []
    for path, param in flat_params.items():
        dev = check_orthogonality(path, param)
        if dev is not None:
            deviations.append(dev)

    if deviations:
        metrics["muon/orthogonality_deviation_mean"] = jnp.mean(jnp.array(deviations))
        metrics["muon/orthogonality_deviation_max"] = jnp.max(jnp.array(deviations))

    return metrics


# --------------------------------------------------------
#  Optimizer: Manifold MUON for matrices, Adam for vectors/scalars
#  with separate learning rates for encoder vs actor/critic heads
# --------------------------------------------------------

def create_muon_matrices_adam_rest_optimizer(
    encoder_muon_lr: float,
    encoder_adam_lr: float,
    heads_muon_lr: float,
    heads_adam_lr: float,
    muon_dual_lr: float = 0.01,
    muon_dual_steps: int = 5,
    muon_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    weight_decay: float = 0.0,
    max_grad_norm: float = 0.5,
):
    """
    Create optimizer that uses:
    - Manifold MUON for all matrices (2D+ parameters with min dimension > 1)
    - Adam for vectors and scalars (1D and 0D parameters, or matrices with min dim <= 1)

    With 4 separate learning rates:
    - encoder_muon_lr: encoder matrices (MUON)
    - encoder_adam_lr: encoder vectors/scalars (Adam)
    - heads_muon_lr: actor/critic matrices (MUON)
    - heads_adam_lr: actor/critic vectors/scalars (Adam)
    """

    # Create transforms for each combination of component and param type
    def make_adam(lr):
        if weight_decay > 0:
            return optax.adamw(learning_rate=lr, eps=adam_eps, weight_decay=weight_decay)
        return optax.adam(learning_rate=lr, eps=adam_eps)

    def make_muon(lr):
        return manifold_muon(
            learning_rate=lr,
            dual_lr=muon_dual_lr,
            dual_steps=muon_dual_steps,
            msign_steps=muon_msign_steps,
            min_ndim=2,
        )

    # 4 transforms: encoder/heads x muon/adam
    transforms = {
        'encoder_muon': make_muon(encoder_muon_lr),
        'encoder_adam': make_adam(encoder_adam_lr),
        'heads_muon': make_muon(heads_muon_lr),
        'heads_adam': make_adam(heads_adam_lr),
    }

    # Label function: combines component (encoder/heads) and param type (muon/adam)
    def label_fn(params):
        def _label(path, param):
            # Determine component: 'network' -> encoder, 'actor'/'critic' -> heads
            is_encoder = path[0] == 'network'
            component = 'encoder' if is_encoder else 'heads'

            # Determine param type: matrices -> muon, vectors/scalars -> adam
            is_matrix = param.ndim >= 2 and min(param.shape) > 1
            param_type = 'muon' if is_matrix else 'adam'

            return f'{component}_{param_type}'

        flat = flax.traverse_util.flatten_dict(params)
        labeled = {k: _label(k, v) for k, v in flat.items()}
        return flax.core.freeze(flax.traverse_util.unflatten_dict(labeled))

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform(
            transforms=transforms,
            param_labels=label_fn,
        ),
    )


# --------------------------------------------------------
#  Environment
# --------------------------------------------------------

def make_pixelbrax_envs(args):
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
    run_name = f"{args.env_name}__crate_muon__{args.seed}__{int(time.time())}"

    if args.track:

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    print("=" * 60)
    print("CRATE PPO with Manifold MUON Optimizer")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
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
    print(f"  ista_step_size: {args.ista_step_size}")
    print(f"  ista_lambda: {args.ista_lambda}")
    print(f"  activation: Swish (all ISTA layers)")
    print(f"\nOptimizer config:")
    print(f"  Matrices (2D+ params): Manifold MUON")
    print(f"  Vectors/Scalars (1D/0D params): Adam")
    print(f"  encoder_muon_lr: {args.encoder_muon_lr}")
    print(f"  encoder_adam_lr: {args.encoder_adam_lr}")
    print(f"  heads_muon_lr: {args.heads_muon_lr}")
    print(f"  heads_adam_lr: {args.heads_adam_lr}")
    print(f"  muon_dual_lr: {args.muon_dual_lr}")
    print(f"  muon_dual_steps: {args.muon_dual_steps}")
    print(f"  muon_msign_steps: {args.muon_msign_steps}")
    print("=" * 60)

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset(reset_rng)
    raw_obs_shape = init_env_state.pixels.shape[1:]
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
        """Compute learning rate schedule (for logging only, uses heads_adam_lr as reference)."""
        update_step = count // (args.num_minibatches * args.update_epochs)
        if args.warmup_steps > 0:
            warmup_ratio = jnp.minimum(count / args.warmup_steps, 1.0)
        else:
            warmup_ratio = 1.0
        if args.anneal_lr:
            anneal_ratio = 1.0 - update_step / args.num_updates
        else:
            anneal_ratio = 1.0
        # Use heads_adam_lr as reference for logging
        return args.heads_adam_lr * warmup_ratio * anneal_ratio

    # Initialize network
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
    # ISTA Actor and Critic with Swish
    actor = ISTAActor(
        action_dim=action_dim,
        hidden_dim=256,
        ista_step_size=args.ista_step_size,
        ista_lambda=0.0,  # No sparsity in heads
    )
    critic = ISTACritic(
        hidden_dim=256,
        ista_step_size=args.ista_step_size,
        ista_lambda=0.0,  # No sparsity in heads
    )

    dummy_obs = jnp.zeros((1,) + obs_shape)
    print(f"dummy_obs shape: {dummy_obs.shape}")

    network_params = network.init(network_key, dummy_obs, deterministic=True)
    dummy_hidden = network.apply(network_params, dummy_obs, deterministic=True)
    print(f"encoder output shape: {dummy_hidden.shape}")

    # Initialize params
    actor_params = actor.init(actor_key, dummy_hidden)
    critic_params = critic.init(critic_key, dummy_hidden)
    all_params = {
        'network': network_params,
        'actor': actor_params,
        'critic': critic_params,
    }

    # Count params by component
    def count_component_params(params, key):
        if key in params:
            return sum(p.size for p in jax.tree_util.tree_leaves(params[key]))
        return 0

    encoder_params = count_component_params(all_params, 'network')
    actor_params = count_component_params(all_params, 'actor')
    critic_params = count_component_params(all_params, 'critic')
    total_params = encoder_params + actor_params + critic_params

    print(f"\nParameter count by component:")
    print(f"  Encoder params: {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
    print(f"  Actor params: {actor_params:,} ({100*actor_params/total_params:.1f}%)")
    print(f"  Critic params: {critic_params:,} ({100*critic_params/total_params:.1f}%)")
    print(f"  Total params: {total_params:,}")

    # Count params by shape (matrices vs vectors/scalars)
    def count_by_shape(params):
        flat = flax.traverse_util.flatten_dict(params)
        muon_count = sum(p.size for p in flat.values() if p.ndim >= 2 and min(p.shape) > 1)
        adam_count = sum(p.size for p in flat.values() if p.ndim < 2 or min(p.shape) <= 1)
        return muon_count, adam_count

    muon_params, adam_params = count_by_shape(all_params)
    print(f"\nParameter breakdown by optimizer:")
    print(f"  Matrix params (MUON): {muon_params:,} ({100*muon_params/total_params:.1f}%)")
    print(f"  Vector/Scalar params (Adam): {adam_params:,} ({100*adam_params/total_params:.1f}%)")

    # Create optimizer: MUON for matrices, Adam for vectors/scalars
    tx = create_muon_matrices_adam_rest_optimizer(
        encoder_muon_lr=args.encoder_muon_lr,
        encoder_adam_lr=args.encoder_adam_lr,
        heads_muon_lr=args.heads_muon_lr,
        heads_adam_lr=args.heads_adam_lr,
        muon_dual_lr=args.muon_dual_lr,
        muon_dual_steps=args.muon_dual_steps,
        muon_msign_steps=args.muon_msign_steps,
        adam_eps=1e-5,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze(all_params),
        tx=tx,
    )

    print(f"\nweight_decay: {args.weight_decay}")

    # Keep non-JIT versions for debug logging (return_activations/return_intermediates needs concrete bool)
    network_apply_debug = network.apply
    actor_apply_debug = actor.apply
    critic_apply_debug = critic.apply

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
                opt_step = iteration * args.update_epochs * args.num_minibatches
                lr = float(lr_schedule(opt_step))

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
                    # Get encoder output with intermediates for attention vs CLS tracking
                    hidden, intermediates = network_apply_debug(
                        agent_state.params['network'], next_obs,
                        deterministic=True, return_intermediates=True
                    )
                    health_metrics = repr_health(hidden)
                    for k, v in health_metrics.items():
                        log_dict[k] =float(jax.device_get(v))

                    # Attention vs CLS metrics
                    attn_cls_metrics = attention_vs_cls_metrics(intermediates)
                    for k, v in attn_cls_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    # Orthogonality metrics for MUON
                    orth_metrics = orthogonality_metrics(jax.device_get(agent_state.params))
                    for k, v in orth_metrics.items():
                        log_dict[k] = float(v)

                    # Swish activation metrics for actor and critic (use non-JIT versions)
                    _, _, actor_activations = actor_apply_debug(
                        agent_state.params['actor'], hidden, return_activations=True
                    )
                    actor_metrics = swish_activation_metrics(actor_activations, "actor_swish")
                    for k, v in actor_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                    _, critic_activations = critic_apply_debug(
                        agent_state.params['critic'], hidden, return_activations=True
                    )
                    critic_metrics = swish_activation_metrics(critic_activations, "critic_swish")
                    for k, v in critic_metrics.items():
                        log_dict[k] = float(jax.device_get(v))

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
