#!/usr/bin/env python
"""
PPO for PixelBrax environments with optional JEPA/LeJEPA auxiliary objectives.

Optimizer split:
- Adam/AdamW for encoder (network)
- Manifold MUON for actor/critic head matrices (2D+ params)
- Adam/AdamW for actor/critic vectors/scalars
- Adam/AdamW for JEPA heads (predictor/projector)
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

# Import manifold MUON optimizer
from manifold_muon_optax import manifold_muon, manifold_muon_per_head
from encoders import ViTConfig, build_encoder

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
    jepa_heads_lr: float = 3e-4
    """learning rate for JEPA heads (predictor/projector, Adam/AdamW)"""
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
    jepa_heads_max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping on JEPA heads (Adam)"""
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
    action_repeat: int = 1
    """Number of times to repeat each action (frame skip)"""

    # Debug/analysis flags
    debug_repr: bool = False
    """Toggle debug logging for encoder representations"""

    # Encoder architecture
    encoder_type: str = "cnn"
    """encoder architecture to use: 'cnn', 'mlp', or 'vit'"""
    encoder_tanh_scale: float = 0.5
    """Multiplier for encoder output before tanh (controls saturation)"""
    encoder_warmup_updates: int = 0
    """Warmup updates for encoder LR when annealing is enabled (0 disables warmup)."""
    vit_patch_size: int = 14
    """ViT patch size (pixels) for both height and width."""
    vit_hidden_size: int = 192
    """ViT token/embedding hidden dimension."""
    vit_mlp_dim: int = 768
    """ViT MLP expansion dimension in transformer blocks."""
    vit_num_heads: int = 3
    """ViT number of self-attention heads."""
    vit_num_layers: int = 4
    """ViT number of transformer encoder blocks."""
    vit_dropout_rate: float = 0.0
    """ViT dropout rate (kept deterministic unless encoder path is extended)."""
    vit_attention_dropout_rate: float = 0.0
    """ViT attention dropout rate (kept deterministic unless encoder path is extended)."""
    vit_use_cls_token: bool = False
    """If true, use CLS-token readout; otherwise mean-pool patch tokens."""
    vit_use_conv_stem: bool = True
    """If true, apply a lightweight CNN stem before ViT patch embedding."""
    vit_conv_stem_channels: int = 64
    """Channel width for the optional ViT CNN stem."""
    vit_conv_stem_kernel: int = 3
    """Kernel size for the optional ViT CNN stem convolutions."""
    vit_apply_output_tanh: bool = False
    """If true, apply tanh bottleneck on ViT 512-dim output."""
    vit_qk_stiefel: bool = False
    """If true, apply Stiefel-constrained updates to ViT attention Q/K kernels."""
    vit_qk_stiefel_lr: float = 1e-4
    """Learning rate for ViT Q/K Stiefel updates."""
    vit_qk_stiefel_dual_lr: float = 0.01
    """Dual-variable learning rate for ViT Q/K Stiefel updates."""
    vit_qk_stiefel_dual_steps: int = 5
    """Number of dual optimization steps for ViT Q/K Stiefel updates."""
    vit_qk_stiefel_msign_steps: int = 5
    """Number of matrix-sign iterations for ViT Q/K Stiefel updates."""
    vit_qk_stiefel_max_grad_norm: float = 0.5
    """Max grad norm clip for ViT Q/K Stiefel parameter group."""

    # JEPA auxiliary objective
    jepa_mode: str = "none"
    """JEPA mode: 'none', 'jepa' (original), 'le_jepa' (LeJEPA + SIGReg)"""
    jepa_lambda: float = 0.05
    """coefficient for JEPA loss"""
    jepa_ema_tau: float = 0.99
    """EMA coefficient for target encoder (higher = slower update)"""
    jepa_predictor_hidden: int = 512
    """hidden dimension of JEPA predictor MLP"""
    jepa_warmup_updates: int = 2
    """updates before JEPA loss is active"""
    jepa_rampup_updates: int = 10
    """updates to linearly ramp JEPA lambda to target"""
    sigreg_weight: float = 1.0
    """inner LeJEPA weight for SIGReg term"""
    sigreg_num_slices: int = 64
    """number of random 1D projections for SIGReg"""
    sigreg_t_points: int = 17
    """number of integration grid points for SIGReg characteristic-function matching"""
    sigreg_t_min: float = -5.0
    """minimum t-value for SIGReg integration grid"""
    sigreg_t_max: float = 5.0
    """maximum t-value for SIGReg integration grid"""
    proj_dim: int = 256
    """projection dimension used by LeJEPA and SIGReg"""
    lejepa_use_target_ema: bool = True
    """if True, use EMA target for network+projector in le_jepa mode"""
    jepa_mask_num_patches: int = 1
    """number of cutout patches per sample for JEPA context masking"""
    jepa_mask_area_low: float = 0.10
    """minimum cutout area fraction for JEPA context masking"""
    jepa_mask_area_high: float = 0.25
    """maximum cutout area fraction for JEPA context masking"""
    jepa_mask_ar_low: float = 0.5
    """minimum cutout aspect ratio for JEPA context masking"""
    jepa_mask_ar_high: float = 2.0
    """maximum cutout aspect ratio for JEPA context masking"""
    jepa_mask_fill_value: int = 0
    """fill value used for JEPA cutout masking"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


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


class JEPAPredictor(nn.Module):
    """Predictor head for JEPA/LeJEPA."""
    hidden_dim: int = 512
    output_dim: int = 512

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        return x


class Projector(nn.Module):
    """Projection head for LeJEPA/SIGReg space."""
    proj_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.proj_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


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
    """
    Compute metrics for encoder representation health.

    Args:
        hidden: Encoder output of shape (batch, hidden_dim)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic statistics
    metrics["repr/mean"] = jnp.mean(hidden)
    metrics["repr/std"] = jnp.std(hidden)
    metrics["repr/min"] = jnp.min(hidden)
    metrics["repr/max"] = jnp.max(hidden)

    # Per-unit statistics (across batch)
    unit_means = jnp.mean(hidden, axis=0)  # (hidden_dim,)
    unit_stds = jnp.std(hidden, axis=0)    # (hidden_dim,)

    metrics["repr/unit_mean_avg"] = jnp.mean(unit_means)
    metrics["repr/unit_std_avg"] = jnp.mean(unit_stds)
    metrics["repr/unit_std_min"] = jnp.min(unit_stds)
    metrics["repr/unit_std_max"] = jnp.max(unit_stds)

    # Dead units: units with very low variance (not learning)
    dead_threshold = 0.01
    dead_units = jnp.mean(unit_stds < dead_threshold)
    metrics["repr/dead_units_frac"] = dead_units

    # Saturated units: units always near boundaries (for tanh encoder output)
    saturated_high = jnp.mean(jnp.abs(hidden) > 0.95)
    metrics["repr/saturated_frac"] = saturated_high

    # Active units: units with reasonable variance
    active_units = jnp.mean(unit_stds > dead_threshold)
    metrics["repr/active_units_frac"] = active_units

    # Norms
    sample_norms = jnp.linalg.norm(hidden, axis=-1)  # (batch,)
    metrics["repr/norm_mean"] = jnp.mean(sample_norms)
    metrics["repr/norm_std"] = jnp.std(sample_norms)
    metrics["repr/norm_min"] = jnp.min(sample_norms)
    metrics["repr/norm_max"] = jnp.max(sample_norms)

    # Sparsity: fraction of near-zero activations
    sparsity = jnp.mean(jnp.abs(hidden) < 0.01)
    metrics["repr/sparsity"] = sparsity

    # Feature correlation (expensive, use subset)
    # Compute correlation between random pairs of features
    if hidden.shape[1] >= 2:
        # Normalize features
        hidden_centered = hidden - jnp.mean(hidden, axis=0, keepdims=True)
        hidden_norm = hidden_centered / (jnp.std(hidden_centered, axis=0, keepdims=True) + 1e-8)
        # Compute correlation matrix diagonal bands
        corr_matrix = jnp.corrcoef(hidden_norm.T)
        # Average absolute off-diagonal correlation
        mask = 1 - jnp.eye(corr_matrix.shape[0])
        avg_corr = jnp.sum(jnp.abs(corr_matrix) * mask) / (jnp.sum(mask) + 1e-8)
        metrics["repr/feature_correlation"] = avg_corr

    return metrics


def compute_grad_norms(grads: dict) -> dict:
    """Compute gradient norms for key model components."""
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
    if 'predictor' in grads:
        metrics["grads/predictor_norm"] = tree_norm(grads['predictor'])
    if 'projector' in grads:
        metrics["grads/projector_norm"] = tree_norm(grads['projector'])

    return metrics


def swish_activation_metrics(hidden: jnp.ndarray, name: str) -> dict:
    """
    Compute activation metrics for Swish layers in actor/critic.
    """
    metrics = {}
    norms = jnp.linalg.norm(hidden, axis=-1)

    metrics[f"{name}/mean"] = jnp.mean(hidden)
    metrics[f"{name}/std"] = jnp.std(hidden)
    metrics[f"{name}/norm_mean"] = jnp.mean(norms)
    metrics[f"{name}/dead_frac"] = jnp.mean(jnp.abs(hidden) < 0.01)
    metrics[f"{name}/negative_frac"] = jnp.mean(hidden < 0)

    return metrics


# --------------------------------------------------------
#  Optimizer: Adam for encoder, MUON for head matrices, Adam for head vectors
# --------------------------------------------------------

def create_encoder_adam_heads_muon_optimizer(
    encoder_lr,  # Can be float or schedule
    heads_muon_lr: float,
    heads_adam_lr,  # Can be float or schedule
    jepa_heads_lr,  # Can be float or schedule
    muon_dual_lr: float = 0.01,
    muon_dual_steps: int = 5,
    muon_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    max_grad_norm: float = 0.5,
    jepa_heads_max_grad_norm: float = 0.5,
    actor_muon_max_grad_norm: float = 1.0,
    critic_muon_max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    vit_qk_stiefel: bool = False,
    vit_qk_stiefel_lr: float = 1e-4,
    vit_qk_stiefel_dual_lr: float = 0.01,
    vit_qk_stiefel_dual_steps: int = 5,
    vit_qk_stiefel_msign_steps: int = 5,
    vit_qk_stiefel_max_grad_norm: float = 0.5,
):
    """
    Create optimizer that uses:
    - Adam/AdamW for encoder (all params) - supports lr schedule, with grad clipping
    - Optional per-head manifold MUON for ViT encoder attention Q/K kernels
    - Manifold MUON for actor head matrices (2D+ with min dim > 1) - with separate grad clipping
    - Manifold MUON for critic head matrices (2D+ with min dim > 1) - with separate grad clipping
    - Adam/AdamW for actor/critic head vectors/scalars (biases, log_std, etc.) - supports lr schedule, with grad clipping
    - Adam/AdamW for JEPA heads (predictor/projector), with separate clipping

    Uses AdamW when weight_decay > 0, otherwise uses Adam.
    """

    # Choose Adam or AdamW based on weight_decay
    if weight_decay > 0:
        adam_opt = lambda lr: optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr, eps=adam_eps, weight_decay=weight_decay
        )
    else:
        adam_opt = lambda lr: optax.inject_hyperparams(optax.adam)(
            learning_rate=lr, eps=adam_eps
        )

    # Adam/AdamW for encoder (with schedule support and grad clipping)
    encoder_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(encoder_lr),
    )

    # Optional Stiefel-constrained updates for ViT Q/K kernels (per-head)
    encoder_qk_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(vit_qk_stiefel_max_grad_norm),
        manifold_muon_per_head(
            learning_rate=vit_qk_stiefel_lr,
            dual_lr=vit_qk_stiefel_dual_lr,
            dual_steps=vit_qk_stiefel_dual_steps,
            msign_steps=vit_qk_stiefel_msign_steps,
            min_ndim=2,
        ),
    )

    # Adam/AdamW for head vectors/scalars (with schedule support and grad clipping)
    heads_adam_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(heads_adam_lr),
    )

    # Adam/AdamW for JEPA heads (with schedule support and grad clipping)
    jepa_heads_adam_tx = optax.chain(
        optax.clip_by_global_norm(jepa_heads_max_grad_norm),
        adam_opt(jepa_heads_lr),
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

    # 6 transforms: encoder, encoder_qk_stiefel, actor_muon (matrices), critic_muon
    # (matrices), heads_adam (actor/critic vectors/scalars), jepa_heads_adam
    # (predictor/projector)
    transforms = {
        'encoder': encoder_tx,
        'encoder_qk_stiefel': encoder_qk_stiefel_tx,
        'actor_muon': actor_muon_tx,
        'critic_muon': critic_muon_tx,
        'heads_adam': heads_adam_tx,
        'jepa_heads_adam': jepa_heads_adam_tx,
    }

    # Label function
    def label_fn(params):
        def _label(path, param):
            # path[0] is a top-level module key in params
            if path[0] == 'network':
                is_vit_qk_kernel = (
                    len(path) >= 7
                    and path[1] == 'params'
                    and 'Transformer' in path
                    and 'SelfAttention_0' in path
                    and path[-1] == 'kernel'
                    and path[-2] in ('query', 'key')
                )
                if vit_qk_stiefel and is_vit_qk_kernel:
                    return 'encoder_qk_stiefel'
                return 'encoder'
            if path[0] in ('predictor', 'projector'):
                return 'jepa_heads_adam'

            # For actor/critic heads, check if matrix or vector/scalar
            is_matrix = param.ndim >= 2 and min(param.shape) > 1
            if path[0] == 'actor':
                return 'actor_muon' if is_matrix else 'heads_adam'
            if path[0] == 'critic':
                return 'critic_muon' if is_matrix else 'heads_adam'

            # Fallback for any extra top-level params.
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


def ema_update(target_params, online_params, tau):
    """EMA update: target = tau * target + (1 - tau) * online."""
    return jax.tree_util.tree_map(
        lambda t, o: t * tau + o * (1.0 - tau),
        target_params,
        online_params,
    )


def l2_normalize(x, axis=-1, eps=1e-8):
    """L2 normalize along a specified axis."""
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)


def sigreg_loss(
    z: jnp.ndarray,
    key: jax.random.PRNGKey,
    num_slices: int,
    t_points: int,
    t_min: float,
    t_max: float,
    axis_name=None,
):
    """SIGReg via characteristic-function matching to N(0,1) under random 1D projections."""
    feat_dim = z.shape[-1]
    a = jax.random.normal(key, (feat_dim, num_slices))
    a = a / (jnp.linalg.norm(a, axis=0, keepdims=True) + 1e-8)

    y = z @ a
    t = jnp.linspace(t_min, t_max, t_points)
    yt = y[:, :, None] * t[None, None, :]

    ecf_real = jnp.mean(jnp.cos(yt), axis=0)
    ecf_imag = jnp.mean(jnp.sin(yt), axis=0)
    if axis_name is not None:
        ecf_real = jax.lax.pmean(ecf_real, axis_name=axis_name)
        ecf_imag = jax.lax.pmean(ecf_imag, axis_name=axis_name)

    target_real = jnp.exp(-0.5 * (t ** 2))[None, :]
    sq_err = (ecf_real - target_real) ** 2 + ecf_imag ** 2

    dt = (t_max - t_min) / jnp.maximum(t_points - 1, 1)
    trapz = 0.5 * dt * (sq_err[:, :-1] + sq_err[:, 1:]).sum(axis=-1)
    return trapz.mean()


def get_jepa_lambda_py(update_step, target_lambda, warmup_updates, rampup_updates):
    """Host-side warmup + linear-ramp schedule for JEPA coefficient."""
    if update_step < warmup_updates:
        return 0.0
    if update_step < warmup_updates + rampup_updates:
        return target_lambda * (update_step - warmup_updates) / rampup_updates
    return target_lambda


def sample_cutout_mask(
    key: jax.random.PRNGKey,
    batch_size: int,
    h: int,
    w: int,
    num_patches: int = 1,
    area_low: float = 0.10,
    area_high: float = 0.25,
    ar_low: float = 0.5,
    ar_high: float = 2.0,
) -> jnp.ndarray:
    """
    Sample a cutout mask shared across stacked channels.

    Returns:
        Boolean mask of shape (B, H, W, 1), where True means "masked".
    """
    k_area, k_ar, k_pos = jax.random.split(key, 3)
    kx, ky = jax.random.split(k_pos)

    area = jax.random.uniform(k_area, (batch_size, num_patches), minval=area_low, maxval=area_high)
    log_ar = jax.random.uniform(
        k_ar,
        (batch_size, num_patches),
        minval=jnp.log(ar_low),
        maxval=jnp.log(ar_high),
    )
    ar = jnp.exp(log_ar)

    patch_area = area * float(h * w)
    ph = jnp.clip(jnp.sqrt(patch_area / ar).astype(jnp.int32), 1, h)
    pw = jnp.clip(jnp.sqrt(patch_area * ar).astype(jnp.int32), 1, w)

    y0 = jax.random.randint(ky, (batch_size, num_patches), 0, jnp.maximum(1, h - ph + 1))
    x0 = jax.random.randint(kx, (batch_size, num_patches), 0, jnp.maximum(1, w - pw + 1))

    ys = jnp.arange(h)[None, None, :, None]
    xs = jnp.arange(w)[None, None, None, :]
    y0e = y0[:, :, None, None]
    x0e = x0[:, :, None, None]
    phe = ph[:, :, None, None]
    pwe = pw[:, :, None, None]

    patch_mask = (ys >= y0e) & (ys < (y0e + phe)) & (xs >= x0e) & (xs < (x0e + pwe))
    return jnp.any(patch_mask, axis=1)[:, :, :, None]


def apply_cutout_mask(obs: jnp.ndarray, mask: jnp.ndarray, fill_value: int = 0) -> jnp.ndarray:
    """
    Apply cutout mask to observations.

    Args:
        obs: (B, H, W, Cstack)
        mask: (B, H, W, 1) where True means "masked"
    """
    fill = jnp.array(fill_value, dtype=obs.dtype)
    return jnp.where(mask, fill, obs)


if __name__ == "__main__":
    args = tyro.cli(Args)
    valid_jepa_modes = {"none", "jepa", "le_jepa"}
    if args.jepa_mode not in valid_jepa_modes:
        raise ValueError(f"Invalid jepa_mode='{args.jepa_mode}'. Expected one of {sorted(valid_jepa_modes)}")
    args.use_jepa = (args.jepa_mode == "jepa")
    args.use_le_jepa = (args.jepa_mode == "le_jepa")

    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.env_name}__{args.exp_name}__{args.jepa_mode}__{args.seed}__{int(time.time())}"

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
    key, network_key, actor_key, critic_key, predictor_key, projector_key = jax.random.split(key, 6)

    # Environment setup
    print("=" * 60)
    print("PPO with Manifold MUON + Optional JEPA/LeJEPA")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")

    adam_type = "AdamW" if args.weight_decay > 0 else "Adam"
    print(f"\nOptimizer config:")
    print(f"  encoder_lr ({adam_type}): {args.encoder_lr}")
    print(f"  heads_muon_lr (MUON for actor/critic matrices): {args.heads_muon_lr}")
    print(f"  heads_adam_lr ({adam_type} for actor/critic vectors): {args.heads_adam_lr}")
    print(f"  jepa_heads_lr ({adam_type} for predictor/projector): {args.jepa_heads_lr}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  muon_dual_lr: {args.muon_dual_lr}")
    print(f"  muon_dual_steps: {args.muon_dual_steps}")
    print(f"  max_grad_norm ({adam_type}): {args.max_grad_norm}")
    print(f"  jepa_heads_max_grad_norm: {args.jepa_heads_max_grad_norm}")
    print(f"  actor_muon_max_grad_norm: {args.actor_muon_max_grad_norm}")
    print(f"  critic_muon_max_grad_norm: {args.critic_muon_max_grad_norm}")
    print(f"  encoder_type: {args.encoder_type}")
    print(f"  encoder_tanh_scale: {args.encoder_tanh_scale}")
    print(f"  encoder_warmup_updates: {args.encoder_warmup_updates}")
    if args.encoder_type.lower() == "vit":
        print(f"  vit_patch_size: {args.vit_patch_size}")
        print(f"  vit_hidden_size: {args.vit_hidden_size}")
        print(f"  vit_mlp_dim: {args.vit_mlp_dim}")
        print(f"  vit_num_heads: {args.vit_num_heads}")
        print(f"  vit_num_layers: {args.vit_num_layers}")
        print(f"  vit_dropout_rate: {args.vit_dropout_rate}")
        print(f"  vit_attention_dropout_rate: {args.vit_attention_dropout_rate}")
        print(f"  vit_use_cls_token: {args.vit_use_cls_token}")
        print(f"  vit_use_conv_stem: {args.vit_use_conv_stem}")
        print(f"  vit_conv_stem_channels: {args.vit_conv_stem_channels}")
        print(f"  vit_conv_stem_kernel: {args.vit_conv_stem_kernel}")
        print(f"  vit_apply_output_tanh: {args.vit_apply_output_tanh}")
        print(f"  vit_qk_stiefel: {args.vit_qk_stiefel}")
        print(f"  vit_qk_stiefel_lr: {args.vit_qk_stiefel_lr}")
        print(f"  vit_qk_stiefel_dual_lr: {args.vit_qk_stiefel_dual_lr}")
        print(f"  vit_qk_stiefel_dual_steps: {args.vit_qk_stiefel_dual_steps}")
        print(f"  vit_qk_stiefel_msign_steps: {args.vit_qk_stiefel_msign_steps}")
        print(f"  vit_qk_stiefel_max_grad_norm: {args.vit_qk_stiefel_max_grad_norm}")
    print(f"  anneal_lr: {args.anneal_lr}")
    print(f"  jepa_mode: {args.jepa_mode}")
    if args.use_jepa or args.use_le_jepa:
        print(f"  jepa_lambda: {args.jepa_lambda}")
        print(f"  jepa_ema_tau: {args.jepa_ema_tau}")
        print(f"  jepa_warmup_updates: {args.jepa_warmup_updates}")
        print(f"  jepa_rampup_updates: {args.jepa_rampup_updates}")
    if args.use_le_jepa:
        print(f"  proj_dim: {args.proj_dim}")
        print(f"  sigreg_weight: {args.sigreg_weight}")
        print(f"  sigreg_num_slices: {args.sigreg_num_slices}")
        print(f"  sigreg_t_points: {args.sigreg_t_points}")
        print(f"  sigreg_t_min: {args.sigreg_t_min}")
        print(f"  sigreg_t_max: {args.sigreg_t_max}")
        print(f"  lejepa_use_target_ema: {args.lejepa_use_target_ema}")
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

    # Initialize networks
    vit_config = ViTConfig(
        patch_size=args.vit_patch_size,
        hidden_size=args.vit_hidden_size,
        mlp_dim=args.vit_mlp_dim,
        num_heads=args.vit_num_heads,
        num_layers=args.vit_num_layers,
        dropout_rate=args.vit_dropout_rate,
        attention_dropout_rate=args.vit_attention_dropout_rate,
        use_cls_token=args.vit_use_cls_token,
        use_conv_stem=args.vit_use_conv_stem,
        conv_stem_channels=args.vit_conv_stem_channels,
        conv_stem_kernel=args.vit_conv_stem_kernel,
        apply_output_tanh=args.vit_apply_output_tanh,
    )
    network = build_encoder(
        args.encoder_type,
        args.encoder_tanh_scale,
        vit_config=vit_config,
    )
    actor = Actor(action_dim=action_dim)
    critic = Critic()
    projector = Projector(proj_dim=args.proj_dim)
    predictor_output_dim = args.proj_dim if args.use_le_jepa else 512
    predictor = JEPAPredictor(hidden_dim=args.jepa_predictor_hidden, output_dim=predictor_output_dim)

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)

    params_dict = {
        "network": network_params,
        "actor": actor.init(actor_key, dummy_hidden),
        "critic": critic.init(critic_key, dummy_hidden),
    }

    if args.use_jepa or args.use_le_jepa:
        if args.use_le_jepa:
            projector_params = projector.init(projector_key, dummy_hidden)
            dummy_proj = projector.apply(projector_params, dummy_hidden)
            predictor_params = predictor.init(predictor_key, dummy_proj)
            params_dict["projector"] = projector_params
        else:
            predictor_params = predictor.init(predictor_key, dummy_hidden)
        params_dict["predictor"] = predictor_params

    all_params = flax.core.freeze(params_dict)

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

    encoder_params = count_params(all_params, "network")
    actor_params = count_params(all_params, "actor")
    critic_params = count_params(all_params, "critic")
    predictor_params_count = count_params(all_params, "predictor")
    projector_params_count = count_params(all_params, "projector")
    total_params = (
        encoder_params
        + actor_params
        + critic_params
        + predictor_params_count
        + projector_params_count
    )

    actor_muon, actor_adam = count_by_type(all_params, "actor")
    critic_muon, critic_adam = count_by_type(all_params, "critic")

    print(f"\nParameter breakdown:")
    print(f"  Encoder (Adam): {encoder_params:,}")
    print(f"  Actor total: {actor_params:,} (MUON: {actor_muon:,}, Adam: {actor_adam:,})")
    print(f"  Critic total: {critic_params:,} (MUON: {critic_muon:,}, Adam: {critic_adam:,})")
    if args.use_jepa or args.use_le_jepa:
        print(f"  Predictor (Adam): {predictor_params_count:,}")
    if args.use_le_jepa:
        print(f"  Projector (Adam): {projector_params_count:,}")
    print(f"  Total: {total_params:,}")

    # Create learning rate schedules for Adam optimizers
    steps_per_update = args.num_minibatches * args.update_epochs

    def lr_from_update(base_lr: float, update_idx: int, warmup_updates: int = 0):
        """Warmup + linear decay learning-rate helper."""
        update_idx = jnp.asarray(update_idx, dtype=jnp.float32)
        warmup = float(max(0, warmup_updates))
        decay_updates = float(max(1, args.num_updates - warmup_updates))

        warm_lr = base_lr * ((update_idx + 1.0) / jnp.maximum(1.0, warmup))
        decay_idx = jnp.maximum(0.0, update_idx - warmup)
        frac = 1.0 - jnp.minimum(1.0, decay_idx / decay_updates)
        decay_lr = base_lr * frac

        if warmup_updates <= 0:
            return decay_lr
        return jnp.where(update_idx < warmup, warm_lr, decay_lr)

    def make_linear_schedule(base_lr, warmup_updates: int = 0):
        """Schedule for Adam optimizers; optional warmup before linear decay."""
        def schedule(count):
            update_idx = count // steps_per_update
            return lr_from_update(base_lr, update_idx, warmup_updates)

        return schedule

    if args.anneal_lr:
        encoder_lr = make_linear_schedule(args.encoder_lr, args.encoder_warmup_updates)
        heads_adam_lr = make_linear_schedule(args.heads_adam_lr, warmup_updates=0)
        jepa_heads_lr = make_linear_schedule(args.jepa_heads_lr, warmup_updates=0)
        if args.encoder_warmup_updates > 0:
            print(
                f"\nLearning rate annealing: ENABLED "
                f"(encoder warmup={args.encoder_warmup_updates} updates + linear decay)"
            )
        else:
            print("\nLearning rate annealing: ENABLED (linear decay for Adam)")
    else:
        encoder_lr = args.encoder_lr
        heads_adam_lr = args.heads_adam_lr
        jepa_heads_lr = args.jepa_heads_lr
        print("\nLearning rate annealing: DISABLED")

    # Create optimizer
    tx = create_encoder_adam_heads_muon_optimizer(
        encoder_lr=encoder_lr,
        heads_muon_lr=args.heads_muon_lr,
        heads_adam_lr=heads_adam_lr,
        jepa_heads_lr=jepa_heads_lr,
        muon_dual_lr=args.muon_dual_lr,
        muon_dual_steps=args.muon_dual_steps,
        muon_msign_steps=args.muon_msign_steps,
        adam_eps=1e-5,
        max_grad_norm=args.max_grad_norm,
        jepa_heads_max_grad_norm=args.jepa_heads_max_grad_norm,
        actor_muon_max_grad_norm=args.actor_muon_max_grad_norm,
        critic_muon_max_grad_norm=args.critic_muon_max_grad_norm,
        weight_decay=args.weight_decay,
        vit_qk_stiefel=args.vit_qk_stiefel,
        vit_qk_stiefel_lr=args.vit_qk_stiefel_lr,
        vit_qk_stiefel_dual_lr=args.vit_qk_stiefel_dual_lr,
        vit_qk_stiefel_dual_steps=args.vit_qk_stiefel_dual_steps,
        vit_qk_stiefel_msign_steps=args.vit_qk_stiefel_msign_steps,
        vit_qk_stiefel_max_grad_norm=args.vit_qk_stiefel_max_grad_norm,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    # Target params for EMA teacher branches
    if args.use_le_jepa:
        target_params = flax.core.freeze({
            "network": agent_state.params["network"],
            "projector": agent_state.params["projector"],
        })
    elif args.use_jepa:
        target_params = agent_state.params["network"]
    else:
        target_params = agent_state.params["network"]

    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    if args.use_jepa or args.use_le_jepa:
        predictor.apply = jax.jit(predictor.apply)
    if args.use_le_jepa:
        projector.apply = jax.jit(projector.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and return updated key."""
        hidden = network.apply(agent_state.params["network"], next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params["actor"], hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic.apply(agent_state.params["critic"], hidden)

        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        hidden = network.apply(params["network"], x)
        actor_mean, actor_logstd = actor.apply(params["actor"], hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic.apply(params["critic"], hidden).squeeze(-1)

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
            agent_state.params["critic"],
            network.apply(agent_state.params["network"], next_obs),
        ).squeeze(-1)

        advantages = jnp.zeros((args.n_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def compute_jepa_loss(params, target_network_params, obs, jepa_key):
        """Compute JEPA loss for a minibatch."""
        batch_size, h, w, _ = obs.shape
        mask = sample_cutout_mask(
            jepa_key,
            batch_size,
            h,
            w,
            num_patches=args.jepa_mask_num_patches,
            area_low=args.jepa_mask_area_low,
            area_high=args.jepa_mask_area_high,
            ar_low=args.jepa_mask_ar_low,
            ar_high=args.jepa_mask_ar_high,
        )
        context_obs = apply_cutout_mask(obs, mask, fill_value=args.jepa_mask_fill_value)
        target_obs = obs

        context_embed = network.apply(params["network"], context_obs)
        pred_embed = predictor.apply(params["predictor"], context_embed)

        target_embed = network.apply(target_network_params, target_obs)

        pred_norm = l2_normalize(pred_embed)
        target_norm = l2_normalize(target_embed)
        return jnp.mean((pred_norm - target_norm) ** 2)

    def compute_le_jepa_loss(params, target_params, obs, mask_key, sig_key):
        """Compute LeJEPA loss: masked prediction + SIGReg regularization."""
        batch_size, h, w, _ = obs.shape
        mask = sample_cutout_mask(
            mask_key,
            batch_size,
            h,
            w,
            num_patches=args.jepa_mask_num_patches,
            area_low=args.jepa_mask_area_low,
            area_high=args.jepa_mask_area_high,
            ar_low=args.jepa_mask_ar_low,
            ar_high=args.jepa_mask_ar_high,
        )
        context_obs = apply_cutout_mask(obs, mask, fill_value=args.jepa_mask_fill_value)
        target_obs = obs

        h_ctx = network.apply(params["network"], context_obs)
        z_ctx = projector.apply(params["projector"], h_ctx)
        z_pred = predictor.apply(params["predictor"], z_ctx)

        if args.lejepa_use_target_ema:
            h_tgt = network.apply(target_params["network"], target_obs)
            z_tgt = projector.apply(target_params["projector"], h_tgt)
        else:
            h_tgt = network.apply(params["network"], target_obs)
            z_tgt = projector.apply(params["projector"], h_tgt)
        z_tgt = jax.lax.stop_gradient(z_tgt)

        pred_norm = l2_normalize(z_pred)
        tgt_norm = l2_normalize(z_tgt)
        pred_loss = jnp.mean((pred_norm - tgt_norm) ** 2)

        sig_loss = sigreg_loss(
            z_ctx,
            sig_key,
            args.sigreg_num_slices,
            args.sigreg_t_points,
            args.sigreg_t_min,
            args.sigreg_t_max,
        )
        total_le_jepa = pred_loss + args.sigreg_weight * sig_loss

        z_mean = jnp.mean(z_ctx)
        z_var_per_dim = jnp.var(z_ctx, axis=0)
        z_var_mean = jnp.mean(z_var_per_dim)
        z_var_min = jnp.min(z_var_per_dim)
        z_norm_mean = jnp.mean(jnp.linalg.norm(z_ctx, axis=-1))
        diag = jnp.array([z_mean, z_var_mean, z_var_min, z_norm_mean], dtype=jnp.float32)
        return pred_loss, sig_loss, total_le_jepa, diag

    def ppo_jepa_loss(
        params,
        target_params,
        x,
        a,
        logp,
        mb_advantages,
        mb_returns,
        mb_values,
        aug_key,
        mask_key,
        sig_key,
        jepa_lambda_current,
    ):
        """Combined PPO + JEPA/LeJEPA loss function."""
        if args.use_augmentation:
            x = random_shift(aug_key, x, pad=args.augment_pad)

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
        ppo_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        le_zero_diag = jnp.zeros((4,), dtype=jnp.float32)
        if args.use_jepa:
            j_loss = jax.lax.cond(
                jepa_lambda_current > 0,
                lambda: compute_jepa_loss(params, target_params, x, mask_key),
                lambda: 0.0,
            )
        else:
            j_loss = 0.0

        if args.use_le_jepa:
            le_pred_loss, le_sig_loss, le_total_loss, le_diag = jax.lax.cond(
                jepa_lambda_current > 0,
                lambda: compute_le_jepa_loss(params, target_params, x, mask_key, sig_key),
                lambda: (0.0, 0.0, 0.0, le_zero_diag),
            )
        else:
            le_pred_loss, le_sig_loss, le_total_loss, le_diag = 0.0, 0.0, 0.0, le_zero_diag

        total_loss = ppo_loss
        if args.use_jepa:
            total_loss = total_loss + jepa_lambda_current * j_loss
        if args.use_le_jepa:
            total_loss = total_loss + jepa_lambda_current * le_total_loss

        return total_loss, (
            pg_loss,
            v_loss,
            entropy_loss,
            j_loss,
            le_pred_loss,
            le_sig_loss,
            le_total_loss,
            le_diag,
            jax.lax.stop_gradient(approx_kl),
        )

    ppo_jepa_loss_grad_fn = jax.value_and_grad(ppo_jepa_loss, has_aux=True)

    @jax.jit
    def update_ppo_jepa(
        agent_state: TrainState,
        target_params,
        storage: Storage,
        key: jax.random.PRNGKey,
        jepa_lambda_current: float,
    ):
        """PPO update with optional JEPA/LeJEPA auxiliary objective."""
        def update_epoch(carry, unused_inp):
            agent_state, target_params, key, last_grads = carry
            key, subkey, aug_key, mask_parent_key, sig_parent_key = jax.random.split(key, 5)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            aug_keys = jax.random.split(aug_key, args.num_minibatches)
            mask_keys = jax.random.split(mask_parent_key, args.num_minibatches)
            sig_keys = jax.random.split(sig_parent_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state, target_params, _ = carry
                minibatch, mb_aug_key, mb_mask_key, mb_sig_key = inputs

                (
                    loss,
                    (
                        pg_loss,
                        v_loss,
                        entropy_loss,
                        j_loss,
                        le_pred_loss,
                        le_sig_loss,
                        le_total_loss,
                        le_diag,
                        approx_kl,
                    ),
                ), grads = ppo_jepa_loss_grad_fn(
                    agent_state.params,
                    jax.lax.stop_gradient(target_params),
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                    mb_mask_key,
                    mb_sig_key,
                    jepa_lambda_current,
                )

                grad_norm = optax.global_norm(grads)
                agent_state = agent_state.apply_gradients(grads=grads)

                if args.use_le_jepa and args.lejepa_use_target_ema:
                    target_params = jax.lax.cond(
                        jepa_lambda_current > 0,
                        lambda: flax.core.freeze(
                            {
                                "network": ema_update(
                                    target_params["network"],
                                    agent_state.params["network"],
                                    args.jepa_ema_tau,
                                ),
                                "projector": ema_update(
                                    target_params["projector"],
                                    agent_state.params["projector"],
                                    args.jepa_ema_tau,
                                ),
                            }
                        ),
                        lambda: target_params,
                    )
                elif args.use_jepa:
                    target_params = jax.lax.cond(
                        jepa_lambda_current > 0,
                        lambda: ema_update(
                            target_params,
                            agent_state.params["network"],
                            args.jepa_ema_tau,
                        ),
                        lambda: target_params,
                    )

                return (agent_state, target_params, grads), (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    j_loss,
                    le_pred_loss,
                    le_sig_loss,
                    le_total_loss,
                    le_diag,
                    approx_kl,
                    grad_norm,
                )

            (
                (agent_state, target_params, last_grads),
                (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    j_loss,
                    le_pred_loss,
                    le_sig_loss,
                    le_total_loss,
                    le_diag,
                    approx_kl,
                    grad_norm,
                ),
            ) = jax.lax.scan(
                update_minibatch,
                (agent_state, target_params, last_grads),
                (shuffled_storage, aug_keys, mask_keys, sig_keys),
            )
            return (agent_state, target_params, key, last_grads), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                j_loss,
                le_pred_loss,
                le_sig_loss,
                le_total_loss,
                le_diag,
                approx_kl,
                grad_norm,
            )

        init_grads = jax.tree_util.tree_map(jnp.zeros_like, agent_state.params)
        (
            (agent_state, target_params, key, final_grads),
            (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                j_loss,
                le_pred_loss,
                le_sig_loss,
                le_total_loss,
                le_diag,
                approx_kl,
                grad_norm,
            ),
        ) = jax.lax.scan(
            update_epoch,
            (agent_state, target_params, key, init_grads),
            (),
            length=args.update_epochs,
        )
        return (
            agent_state,
            target_params,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            j_loss,
            le_pred_loss,
            le_sig_loss,
            le_total_loss,
            le_diag,
            approx_kl,
            grad_norm,
            final_grads,
            key,
        )

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
        next_done_local = env_state.done.astype(jnp.bool_)

        fs = fs.push(raw_obs)
        fs = fs.reset(raw_obs, next_done_local)
        next_obs_local = fs.get_stacked()

        reward_norm = reward_norm.update(raw_reward, next_done_local.astype(jnp.float32))
        reward = reward_norm.normalize(raw_reward)

        new_episode_return = episode_stats.episode_returns + raw_reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=jnp.where(next_done_local, 0.0, new_episode_return),
            episode_lengths=jnp.where(next_done_local, 0, new_episode_length).astype(jnp.int32),
            returned_episode_returns=jnp.where(
                next_done_local,
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                next_done_local,
                new_episode_length,
                episode_stats.returned_episode_lengths,
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
        return (
            agent_state,
            episode_stats,
            reward_norm,
            fs,
            env_state,
            next_obs_local,
            next_done_local,
            key,
        ), storage

    def rollout(agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key, max_steps):
        (
            agent_state,
            episode_stats,
            reward_norm,
            fs,
            env_state,
            next_obs,
            next_done,
            key,
        ), storage = jax.lax.scan(
            step_once,
            (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key),
            jnp.arange(max_steps),
        )
        return agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, storage, key

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("\nStarting training...")
    cumulative_episodic_return = 0.0
    for iteration in range(1, args.num_updates + 1):
        iteration_time_start = time.time()
        (
            agent_state,
            episode_stats,
            reward_normalizer,
            frame_stack,
            env_state,
            next_obs,
            next_done,
            storage,
            key,
        ) = rollout(
            agent_state,
            episode_stats,
            reward_normalizer,
            frame_stack,
            env_state,
            next_obs,
            next_done,
            key,
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)

        if args.use_jepa or args.use_le_jepa:
            jepa_lambda_current = get_jepa_lambda_py(
                iteration,
                args.jepa_lambda,
                args.jepa_warmup_updates,
                args.jepa_rampup_updates,
            )
        else:
            jepa_lambda_current = 0.0

        (
            agent_state,
            target_params,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            j_loss,
            le_pred_loss,
            le_sig_loss,
            le_total_loss,
            le_diag,
            approx_kl,
            grad_norm,
            final_grads,
            key,
        ) = update_ppo_jepa(
            agent_state,
            target_params,
            storage,
            key,
            jepa_lambda_current,
        )

        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            cumulative_episodic_return += avg_episodic_return
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))

            base_msg = (
                f"update={iteration} step={global_step} "
                f"ep_return={avg_episodic_return:.1f} "
                f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                f"loss={loss[-1, -1].item():.4f} "
                f"SPS={sps}"
            )
            if args.use_jepa:
                base_msg += (
                    f" jepa_loss={j_loss[-1, -1].item():.4f}"
                    f" jepa_lambda={jepa_lambda_current:.4f}"
                )
            elif args.use_le_jepa:
                base_msg += (
                    f" le_total={le_total_loss[-1, -1].item():.4f}"
                    f" jepa_lambda={jepa_lambda_current:.4f}"
                )
            print(base_msg)

            if args.track:
                opt_step = iteration * args.update_epochs * args.num_minibatches
                if args.anneal_lr:
                    update_idx = opt_step // steps_per_update
                    encoder_lr_current = lr_from_update(
                        args.encoder_lr,
                        update_idx,
                        args.encoder_warmup_updates,
                    )
                    heads_adam_lr_current = lr_from_update(
                        args.heads_adam_lr,
                        update_idx,
                        warmup_updates=0,
                    )
                    jepa_heads_lr_current = lr_from_update(
                        args.jepa_heads_lr,
                        update_idx,
                        warmup_updates=0,
                    )
                else:
                    encoder_lr_current = args.encoder_lr
                    heads_adam_lr_current = args.heads_adam_lr
                    jepa_heads_lr_current = args.jepa_heads_lr

                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/cumulative_episodic_return": cumulative_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/encoder_lr": float(encoder_lr_current),
                    "charts/heads_adam_lr": float(heads_adam_lr_current),
                    "charts/heads_muon_lr": args.heads_muon_lr,
                    "charts/jepa_heads_lr": float(jepa_heads_lr_current),
                    "charts/vit_qk_stiefel_lr": (
                        args.vit_qk_stiefel_lr if args.vit_qk_stiefel else 0.0
                    ),
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/grad_norm": grad_norm[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                }

                if args.use_jepa:
                    log_dict["jepa/loss"] = j_loss[-1, -1].item()
                    log_dict["jepa/lambda"] = jepa_lambda_current
                if args.use_le_jepa:
                    log_dict["le_jepa/pred_loss"] = le_pred_loss[-1, -1].item()
                    log_dict["le_jepa/sigreg_loss"] = le_sig_loss[-1, -1].item()
                    log_dict["le_jepa/total"] = le_total_loss[-1, -1].item()
                    log_dict["le_jepa/z_mean"] = le_diag[-1, -1, 0].item()
                    log_dict["le_jepa/z_var_mean"] = le_diag[-1, -1, 1].item()
                    log_dict["le_jepa/z_var_min"] = le_diag[-1, -1, 2].item()
                    log_dict["le_jepa/z_norm_mean"] = le_diag[-1, -1, 3].item()
                    log_dict["le_jepa/lambda"] = jepa_lambda_current

                # Debug metrics for encoder representations
                if args.debug_repr:
                    sample_obs = storage.obs[0, :256]
                    hidden = network.apply(agent_state.params["network"], sample_obs)

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
