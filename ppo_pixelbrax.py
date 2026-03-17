#!/usr/bin/env python
"""
PPO for PixelBrax environments.

Optimizer split:
- Adam/AdamW for encoder (network)
- Manifold MUON for actor/critic head matrices (2D+ params)
- Adam/AdamW for actor/critic vectors/scalars
"""
import os
import random
import time
from dataclasses import dataclass
from functools import partial

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
from encoders import (
    ViTConfig,
    build_encoder,
    MIMJEPAPredictor,
    BottleneckProjection,
    LatentTransition,
    LinearLatentTransition,
)

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinism
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


class CRATEFeedForward(nn.Module):
    """CRATE-style FeedForward layer implementing an ISTA step.

    Computes: output = swish(x + step_size * (W.T @ x - W.T @ W @ x))

    This implements a gradient descent step for sparse coding with dictionary W.
    Reference: https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/model/crate.py
    """
    dim: int
    step_size: float = 0.1

    @nn.compact
    def __call__(self, x):
        # Weight matrix W of shape (dim, dim), initialized with Kaiming uniform
        weight = self.param(
            "weight",
            nn.initializers.kaiming_uniform(),
            (self.dim, self.dim)
        )

        # Compute D^T * D * x (W @ x then W.T @ result)
        x1 = x @ weight.T  # (batch, dim) @ (dim, dim) -> (batch, dim)
        grad_1 = x1 @ weight  # (batch, dim) @ (dim, dim) -> (batch, dim)

        # Compute D^T * x
        grad_2 = x @ weight  # (batch, dim) @ (dim, dim) -> (batch, dim)

        # Compute gradient update: step_size * (D^T * x - D^T * D * x)
        # lambda is set to 0.0 so we omit the - step_size * lambda term
        grad_update = self.step_size * (grad_2 - grad_1)

        # Apply swish activation (instead of ReLU in original CRATE)
        output = nn.swish(x + grad_update)
        return output


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
    env_name: str = "inverted_pendulum"
    """the name of the environment"""
    backend: str = "generalized"
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
    use_heads_muon: bool = True
    """if True, use manifold MUON for actor/critic weight matrices; if False, use Adam for all head params"""
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

    # Checkpoint saving
    save_checkpoint: bool = False
    """If true, save model params to disk at end of training"""
    checkpoint_dir: str = "checkpoints"
    """Directory for saved checkpoints"""

    # Online linear probe
    probe_interval: int = 0
    """Run linear probe every N global steps (0 disables). E.g. 100000."""
    probe_n_eval_steps: int = 400
    """Number of random-action rollout steps for probe data collection."""
    probe_output_dir: str = "probe_results"
    """Directory for probe plots (subdirectory per probe call is created)."""
    probe_save_plots: bool = False
    """If true, generate probe/additive plots to disk and log them to wandb when tracking."""

    # Encoder architecture
    encoder_type: str = "cnn"
    """encoder architecture to use: 'cnn', 'stiefel_cnn', 'mlp', 'vit', 'hybrid_vit', 'drq_vit', or 'scott'"""
    encoder_tanh_scale: float = 0.5
    """Multiplier for encoder output before tanh (controls saturation)"""
    stiefel_latent_dim: int = 32
    """Latent dimension for stiefel_cnn projection output."""
    encoder_warmup_updates: int = 0
    """Warmup updates for encoder LR when annealing is enabled (0 disables warmup)."""
    vit_patch_size: int = 14
    """ViT patch size (pixels) for both height and width."""
    vit_hidden_size: int = 192
    """ViT token/embedding hidden dimension."""
    vit_proj_dim: int = 512
    """RL projection head output dimension for ViT/HybridViT encoders."""
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
    """If true, apply tanh bottleneck on ViT encoder output projection."""
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
    hybrid_vit_stem_c1: int = 24
    """HybridViT conv stem stage-1 output channels."""
    hybrid_vit_stem_c2: int = 48
    """HybridViT conv stem stage-2 output channels."""
    hybrid_vit_stem_c3: int = 96
    """HybridViT conv stem stage-3 output channels."""
    hybrid_vit_stem_c4: int = 192
    """HybridViT conv stem stage-4 output channels."""
    drq_vit_stem_channels: int = 32
    """DrQViT conv stem channel width (all 4 stem layers use this)."""
    drq_vit_token_downsample: int = 1
    """Optional patch-like downsample factor after DrQ bridge conv (1 disables)."""
    drq_vit_apply_output_tanh: bool = False
    """If true, apply tanh to DrQViT encoder output projection."""

    # SCOTT encoder args
    scott_use_swiglu: bool = True
    """Use SwiGLU FFN in SCOTT transformer blocks."""
    scott_num_register_tokens: int = 0
    """Number of learnable register tokens for SCOTT encoder."""
    scott_dropout_rate: float = 0.0
    """SCOTT transformer dropout rate."""
    scott_attention_dropout_rate: float = 0.0
    """SCOTT attention dropout rate."""
    scott_stochastic_depth_rate: float = 0.0
    """SCOTT stochastic depth rate."""

    # MIM-JEPA auxiliary loss
    mim_jepa: bool = False
    """Enable MIM-JEPA self-supervised auxiliary loss (requires scott encoder)."""
    mim_jepa_lambda: float = 0.05
    """Coefficient for MIM-JEPA reconstruction loss."""
    mim_jepa_ema_tau: float = 0.996
    """EMA momentum for target encoder (closer to 1 = slower update)."""
    mim_jepa_mask_ratio: float = 0.6
    """Fraction of tokens to mask in MIM-JEPA context encoder."""
    mim_jepa_predictor_depth: int = 2
    """Number of transformer layers in MIM-JEPA predictor."""
    mim_jepa_warmup_updates: int = 2
    """Number of PPO updates before MIM-JEPA lambda starts ramping up."""
    mim_jepa_rampup_updates: int = 10
    """Number of updates to linearly ramp MIM-JEPA lambda from 0 to mim_jepa_lambda."""

    # LeJEPA auxiliary loss
    lejepa: bool = False
    """Enable LeJEPA auxiliary loss on top of the CNN encoder."""
    lejepa_lambda: float = 0.1
    """Mixing coefficient between agreement and SIGReg terms."""
    lejepa_aux_coef: float = 1e-3
    """Outer coefficient scaling LeJEPA against PPO."""
    lejepa_proj_dim: int = 128
    """Projector output dimension for LeJEPA."""
    lejepa_num_views: int = 2
    """Number of augmented views of each observation used by LeJEPA."""
    lejepa_num_slices: int = 64
    """Number of random projection slices used by SIGReg."""
    lejepa_warmup_updates: int = 10
    """Number of PPO updates before LeJEPA starts ramping up."""
    lejepa_rampup_updates: int = 50
    """Number of PPO updates to linearly ramp LeJEPA to lejepa_aux_coef."""
    lejepa_t_points: int = 17
    """Number of evaluation points for the 1D characteristic-function grid."""
    lejepa_t_min: float = -5.0
    """Minimum t-value for SIGReg's characteristic-function grid."""
    lejepa_t_max: float = 5.0
    """Maximum t-value for SIGReg's characteristic-function grid."""
    lejepa_view_mode: str = "shift"
    """LeJEPA view generation mode: shift, shift_noise, or temporal."""
    lejepa_use_layernorm: bool = True
    """If true, apply LayerNorm in the LeJEPA projector."""
    lejepa_sigreg_mode: str = "epps_pulley"
    """SIGReg implementation: epps_pulley or legacy."""

    # Simplified latent bottleneck + dynamics auxiliary
    use_latent_bottleneck: bool = False
    """If true, project the encoder feature into a deterministic latent before actor/critic."""
    latent_dim: int = 32
    """Deterministic bottleneck dimension used by actor/critic and latent auxiliaries."""
    latent_transition_type: str = "mlp"
    """Latent transition type: 'mlp' (existing) or 'linear' (A z + B u + b)."""
    use_latent_dynamics: bool = False
    """If true, train a deterministic one-step latent transition on rollout data."""
    latent_dyn_coef: float = 1e-3
    """Target coefficient for the latent dynamics auxiliary loss."""
    latent_dyn_warmup_updates: int = 10
    """Number of PPO updates before the latent dynamics coefficient starts ramping."""
    latent_dyn_rampup_updates: int = 50
    """Number of PPO updates to linearly ramp latent dynamics to latent_dyn_coef."""
    latent_cov_coef: float = 0.0
    """Optional covariance decorrelation penalty on z_t."""
    latent_var_coef: float = 0.0
    """Optional variance-floor anti-collapse penalty on z_t."""
    latent_whiten_coef: float = 0.0
    """Optional whitening penalty for z_t to match identity covariance."""
    latent_subspace_coef: float = 0.0
    """Optional subspace projection residual penalty for stiefel_cnn latent map."""
    cnn_sigreg_coef: float = 0.0
    """SIGReg coefficient on pre-tanh CNN latents (vanilla cnn only)."""
    cnn_sigreg_warmup_updates: int = 0
    """Number of PPO updates before CNN SIGReg starts ramping up."""
    cnn_sigreg_rampup_updates: int = 50
    """Number of PPO updates to linearly ramp CNN SIGReg to cnn_sigreg_coef."""
    cnn_sigreg_num_slices: int = 64
    """Number of random projection slices used by CNN SIGReg."""
    cnn_sigreg_t_points: int = 17
    """Number of evaluation points for CNN SIGReg characteristic-function grid."""
    cnn_sigreg_t_min: float = -5.0
    """Minimum t-value for CNN SIGReg's characteristic-function grid."""
    cnn_sigreg_t_max: float = 5.0
    """Maximum t-value for CNN SIGReg's characteristic-function grid."""
    log_latent_grad_alignment: bool = False
    """If true, periodically log PPO-vs-latent-dynamics encoder gradient alignment."""
    latent_grad_alignment_interval: int = 10
    """Update interval for expensive latent gradient alignment logging."""

    # CRATE head architecture
    use_crate_head: bool = False
    """If true, use CRATE-style FeedForward as final hidden layer in actor/critic."""
    crate_step_size: float = 0.1
    """Step size for CRATE FeedForward ISTA update."""

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


class LeJEPAProjector(nn.Module):
    """Project encoder latents into the LeJEPA auxiliary space."""

    proj_dim: int
    use_layernorm: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.proj_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        return x


class CRATECritic(nn.Module):
    """Value network with CRATE FeedForward as final hidden layer."""
    crate_step_size: float = 0.1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.swish(x)
        x = CRATEFeedForward(dim=256, step_size=self.crate_step_size)(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class CRATEActor(nn.Module):
    """Continuous action actor with CRATE FeedForward as final hidden layer."""
    action_dim: int
    crate_step_size: float = 0.1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.swish(x)
        x = CRATEFeedForward(dim=256, step_size=self.crate_step_size)(x)
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
    step_idx: jnp.array
    env_idx: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array
    next_obs: jnp.array
    next_dones: jnp.array


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

def encoder_repr_metrics(hidden: jnp.ndarray, prefix: str = "repr") -> dict:
    """
    Compute metrics for encoder representation health.

    Args:
        hidden: Encoder output of shape (batch, hidden_dim)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic statistics
    metrics[f"{prefix}/mean"] = jnp.mean(hidden)
    metrics[f"{prefix}/std"] = jnp.std(hidden)
    metrics[f"{prefix}/min"] = jnp.min(hidden)
    metrics[f"{prefix}/max"] = jnp.max(hidden)

    # Per-unit statistics (across batch)
    unit_means = jnp.mean(hidden, axis=0)  # (hidden_dim,)
    unit_stds = jnp.std(hidden, axis=0)    # (hidden_dim,)

    metrics[f"{prefix}/unit_mean_avg"] = jnp.mean(unit_means)
    metrics[f"{prefix}/unit_std_avg"] = jnp.mean(unit_stds)
    metrics[f"{prefix}/unit_std_min"] = jnp.min(unit_stds)
    metrics[f"{prefix}/unit_std_max"] = jnp.max(unit_stds)

    # Dead units: units with very low variance (not learning)
    dead_threshold = 0.01
    dead_units = jnp.mean(unit_stds < dead_threshold)
    metrics[f"{prefix}/dead_units_frac"] = dead_units

    # Saturated units: units always near boundaries (for tanh encoder output)
    saturated_high = jnp.mean(jnp.abs(hidden) > 0.95)
    metrics[f"{prefix}/saturated_frac"] = saturated_high

    # Active units: units with reasonable variance
    active_units = jnp.mean(unit_stds > dead_threshold)
    metrics[f"{prefix}/active_units_frac"] = active_units

    # Norms
    sample_norms = jnp.linalg.norm(hidden, axis=-1)  # (batch,)
    metrics[f"{prefix}/norm_mean"] = jnp.mean(sample_norms)
    metrics[f"{prefix}/norm_std"] = jnp.std(sample_norms)
    metrics[f"{prefix}/norm_min"] = jnp.min(sample_norms)
    metrics[f"{prefix}/norm_max"] = jnp.max(sample_norms)

    # Sparsity: fraction of near-zero activations
    sparsity = jnp.mean(jnp.abs(hidden) < 0.01)
    metrics[f"{prefix}/sparsity"] = sparsity

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
        metrics[f"{prefix}/feature_correlation"] = avg_corr

    return metrics


def compute_grad_norms(grads: dict) -> dict:
    """Compute gradient norms for key model components."""
    metrics = {}
    if 'network' in grads:
        metrics["grads/encoder_norm"] = tree_norm(
            latent_param_tree(grads, 'bottleneck' in grads)
        )
    if 'lejepa_head' in grads:
        metrics["grads/lejepa_head_norm"] = tree_norm(grads['lejepa_head'])
    if 'latent_transition' in grads:
        metrics["grads/latent_transition_norm"] = tree_norm(grads['latent_transition'])
    if 'actor' in grads:
        metrics["grads/actor_norm"] = tree_norm(grads['actor'])
    if 'critic' in grads:
        metrics["grads/critic_norm"] = tree_norm(grads['critic'])

    return metrics


# --------------------------------------------------------
#  Optimizer: Adam for encoder, MUON for head matrices, Adam for head vectors
# --------------------------------------------------------

def create_optimizer(
    encoder_lr,  # Can be float or schedule
    heads_muon_lr: float,
    heads_adam_lr,  # Can be float or schedule
    muon_dual_lr: float = 0.01,
    muon_dual_steps: int = 5,
    muon_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    max_grad_norm: float = 0.5,
    actor_muon_max_grad_norm: float = 1.0,
    critic_muon_max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    vit_qk_stiefel: bool = False,
    vit_qk_stiefel_lr: float = 1e-4,
    vit_qk_stiefel_dual_lr: float = 0.01,
    vit_qk_stiefel_dual_steps: int = 5,
    vit_qk_stiefel_msign_steps: int = 5,
    vit_qk_stiefel_max_grad_norm: float = 0.5,
    use_stiefel_encoder_muon: bool = True,
    stiefel_encoder_muon_lr: float = 0.02,
    stiefel_encoder_muon_max_grad_norm: float = 0.5,
    use_heads_muon: bool = True,
):
    """
    Create optimizer that uses:
    - Adam/AdamW for encoder (all params) - supports lr schedule, with grad clipping
    - Optional per-head manifold MUON for ViT encoder attention Q/K kernels
    - Optional manifold MUON for Stiefel projection kernel
    - Manifold MUON for actor head matrices (2D+ with min dim > 1) - with separate grad clipping
    - Manifold MUON for critic head matrices (2D+ with min dim > 1) - with separate grad clipping
    - Adam/AdamW for actor/critic head vectors/scalars (biases, log_std, etc.) - supports lr schedule, with grad clipping

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

    # Optional manifold MUON for Stiefel projection kernel
    encoder_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(stiefel_encoder_muon_max_grad_norm),
        manifold_muon(
            learning_rate=stiefel_encoder_muon_lr,
            dual_lr=muon_dual_lr,
            dual_steps=muon_dual_steps,
            msign_steps=muon_msign_steps,
            min_ndim=2,
        ),
    )

    # Adam/AdamW for head vectors/scalars (with schedule support and grad clipping)
    heads_adam_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(heads_adam_lr),
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

    # Predictor Adam (same settings as encoder Adam)
    predictor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(encoder_lr),
    )

    lejepa_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(encoder_lr),
    )

    latent_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(encoder_lr),
    )

    # 9 transforms: encoder, encoder_qk_stiefel, encoder_stiefel,
    # actor_muon (matrices), critic_muon (matrices), heads_adam,
    # predictor_adam, lejepa_adam, latent_adam
    transforms = {
        'encoder': encoder_tx,
        'encoder_qk_stiefel': encoder_qk_stiefel_tx,
        'encoder_stiefel': encoder_stiefel_tx,
        'actor_muon': actor_muon_tx,
        'critic_muon': critic_muon_tx,
        'heads_adam': heads_adam_tx,
        'predictor_adam': predictor_tx,
        'lejepa_adam': lejepa_tx,
        'latent_adam': latent_tx,
    }

    # Label function
    def label_fn(params):
        def _label(path, param):
            # path[0] is a top-level module key in params
            if path[0] == 'network':
                is_stiefel_projection_kernel = (
                    use_stiefel_encoder_muon
                    and len(path) >= 4
                    and path[1] == 'params'
                    and path[-2] == 'stiefel_projection'
                    and path[-1] == 'kernel'
                )
                if is_stiefel_projection_kernel:
                    return 'encoder_stiefel'

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
            if path[0] == 'predictor':
                return 'predictor_adam'
            if path[0] == 'lejepa_head':
                return 'lejepa_adam'
            if path[0] in ('bottleneck', 'latent_transition'):
                return 'latent_adam'
            # For actor/critic heads, check if matrix or vector/scalar
            is_matrix = param.ndim >= 2 and min(param.shape) > 1
            if path[0] == 'actor':
                return 'actor_muon' if (is_matrix and use_heads_muon) else 'heads_adam'
            if path[0] == 'critic':
                return 'critic_muon' if (is_matrix and use_heads_muon) else 'heads_adam'

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


def sample_block_mask(
    key: jax.random.PRNGKey,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: float,
) -> jnp.ndarray:
    """Generate upstream-style blockwise keep masks. Returns (B, N) float32."""
    total = grid_h * grid_w
    num_mask = jnp.asarray(jnp.ceil(mask_ratio * total), dtype=jnp.int32)
    max_iters = max(8, total * 4)
    log_aspect_lo = np.log(0.3)
    log_aspect_hi = np.log(3.3)

    def single_mask(k):
        def cond_fn(state):
            _, masked, step, _ = state
            return jnp.logical_and(masked < num_mask, step < max_iters)

        def body_fn(state):
            mask, masked, step, rng = state
            rng, area_key, aspect_key, top_key, left_key = jax.random.split(rng, 5)
            remaining = jnp.maximum(1, num_mask - masked)
            area_lo = jnp.minimum(remaining, jnp.asarray(16, dtype=jnp.int32))
            area = jax.random.randint(area_key, (), 1, remaining + 1)
            area = jnp.maximum(area, area_lo)
            aspect = jnp.exp(jax.random.uniform(aspect_key, (), minval=log_aspect_lo, maxval=log_aspect_hi))
            h = jnp.maximum(1, jnp.round(jnp.sqrt(area * aspect)).astype(jnp.int32))
            w = jnp.maximum(1, jnp.round(jnp.sqrt(area / aspect)).astype(jnp.int32))
            h = jnp.minimum(h, grid_h)
            w = jnp.minimum(w, grid_w)
            top = jax.random.randint(top_key, (), 0, grid_h - h + 1)
            left = jax.random.randint(left_key, (), 0, grid_w - w + 1)

            row_ids = jnp.arange(grid_h)[:, None]
            col_ids = jnp.arange(grid_w)[None, :]
            rect = (
                (row_ids >= top)
                & (row_ids < top + h)
                & (col_ids >= left)
                & (col_ids < left + w)
            )
            delta = jnp.sum(jnp.logical_and(rect, jnp.logical_not(mask)))
            accept = jnp.logical_and(delta > 0, delta <= remaining)
            mask = jnp.where(accept, jnp.logical_or(mask, rect), mask)
            masked = masked + jnp.where(accept, delta, 0)
            return mask, masked, step + 1, rng

        init = (jnp.zeros((grid_h, grid_w), dtype=bool), jnp.int32(0), jnp.int32(0), k)
        mask, masked, _, rng = jax.lax.while_loop(cond_fn, body_fn, init)

        remaining = num_mask - masked
        flat_mask = mask.reshape(-1)
        scores = jax.random.uniform(rng, (total,), minval=0.0, maxval=1.0)
        scores = jnp.where(flat_mask, 2.0, scores)
        order = jnp.argsort(scores)
        pick = jnp.arange(total) < remaining
        extra_mask = jnp.zeros(total, dtype=bool).at[order].set(pick)
        flat_mask = jnp.logical_or(flat_mask, extra_mask)
        return 1.0 - flat_mask.astype(jnp.float32)

    return jax.vmap(single_mask)(jax.random.split(key, batch_size))


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


def random_shift_with_noise(
    key: jax.random.PRNGKey,
    x: jnp.ndarray,
    pad: int = 4,
    noise_std: float = 2.0,
    brightness_delta: float = 0.05,
) -> jnp.ndarray:
    """Random shift plus mild additive noise and brightness jitter."""
    shift_key, noise_key, bright_key = jax.random.split(key, 3)
    shifted = random_shift(shift_key, x, pad=pad).astype(jnp.float32)
    brightness = jax.random.uniform(
        bright_key, (shifted.shape[0], 1, 1, 1), minval=1.0 - brightness_delta, maxval=1.0 + brightness_delta
    )
    noise = noise_std * jax.random.normal(noise_key, shifted.shape)
    augmented = shifted * brightness + noise
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.clip(jnp.rint(augmented), 0.0, 255.0).astype(x.dtype)
    return augmented.astype(x.dtype)


def empirical_char_diff(samples: jnp.ndarray, t_values: jnp.ndarray) -> jnp.ndarray:
    """Characteristic-function distance to a standard 1D Gaussian."""
    phases = samples[:, :, None] * t_values[None, None, :]
    real_part = jnp.mean(jnp.cos(phases), axis=0)
    imag_part = jnp.mean(jnp.sin(phases), axis=0)
    target_real = jnp.exp(-0.5 * jnp.square(t_values))[None, :]
    return jnp.mean(jnp.square(real_part - target_real) + jnp.square(imag_part))


def sigreg_epps_pulley(
    samples: jnp.ndarray,
    directions: jnp.ndarray,
    t_values: jnp.ndarray,
) -> jnp.ndarray:
    """Epps-Pulley-style SIGReg with Gaussian weighting over random 1D projections."""
    projected = samples @ directions.T
    phases = projected[:, :, None] * t_values[None, None, :]
    ecf_real = jnp.mean(jnp.cos(phases), axis=0)
    ecf_imag = jnp.mean(jnp.sin(phases), axis=0)
    target_real = jnp.exp(-0.5 * jnp.square(t_values))[None, :]
    squared_error = jnp.square(ecf_real - target_real) + jnp.square(ecf_imag)
    weights = jnp.exp(-0.5 * jnp.square(t_values))[None, :]
    integrand = squared_error * weights
    integrated = jnp.trapezoid(integrand, t_values, axis=-1)
    return samples.shape[0] * jnp.mean(integrated)


def tree_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))


def tree_dot(tree_a, tree_b) -> jnp.ndarray:
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    if not leaves_a or not leaves_b:
        return jnp.array(0.0, dtype=jnp.float32)
    return sum(jnp.sum(a * b) for a, b in zip(leaves_a, leaves_b))


def latent_covariance_loss(z_t: jnp.ndarray) -> jnp.ndarray:
    """Decorrelation loss on latent dimensions across the minibatch."""
    batch_size = z_t.shape[0]
    if batch_size <= 1:
        return jnp.array(0.0, dtype=jnp.float32)
    z_centered = z_t - z_t.mean(axis=0, keepdims=True)
    cov = (z_centered.T @ z_centered) / jnp.maximum(1.0, batch_size - 1)
    off_diag = cov * (1.0 - jnp.eye(cov.shape[0], dtype=cov.dtype))
    return jnp.sum(jnp.square(off_diag)) / z_t.shape[-1]


def latent_variance_loss(z_t: jnp.ndarray) -> jnp.ndarray:
    """Penalize collapsed latent dimensions with low variance."""
    std = jnp.sqrt(jnp.var(z_t, axis=0) + 1e-4)
    return jnp.mean(jnp.maximum(1.0 - std, 0.0))


def latent_whiten_loss(z_t: jnp.ndarray) -> jnp.ndarray:
    """Match latent covariance to identity for conditioning control."""
    batch_size = z_t.shape[0]
    if batch_size <= 1:
        return jnp.array(0.0, dtype=jnp.float32)
    z_centered = z_t - jnp.mean(z_t, axis=0, keepdims=True)
    cov = (z_centered.T @ z_centered) / jnp.maximum(1.0, batch_size - 1)
    ident = jnp.eye(cov.shape[0], dtype=cov.dtype)
    return jnp.mean(jnp.square(cov - ident))


def latent_stiefel_projection_loss(h_t: jnp.ndarray, network_params: flax.core.FrozenDict) -> jnp.ndarray:
    """Penalize Stiefel latent features for not lying in the learned subspace."""
    try:
        kernel = network_params["network"]["params"]["stiefel_projection"]["kernel"]
    except (TypeError, KeyError):
        return jnp.array(0.0, dtype=jnp.float32)

    z_t = h_t @ kernel
    h_t_hat = z_t @ kernel.T
    return jnp.mean(jnp.sum(jnp.square(h_t - h_t_hat), axis=-1))


def latent_cnn_sigreg_loss(h_t: jnp.ndarray, sigreg_key: jax.random.PRNGKey) -> jnp.ndarray:
    """SIGReg loss on pre-tanh CNN latents against standard normal projections."""
    directions = jax.random.normal(
        sigreg_key,
        (args.cnn_sigreg_num_slices, h_t.shape[-1]),
    )
    directions = directions / (jnp.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8)
    t_values = jnp.linspace(
        args.cnn_sigreg_t_min,
        args.cnn_sigreg_t_max,
        num=args.cnn_sigreg_t_points,
    )
    return sigreg_epps_pulley(h_t, directions, t_values)


def latent_param_tree(tree, include_bottleneck: bool):
    """Select encoder-side params for alignment checks."""
    selected = {"network": tree["network"]}
    if include_bottleneck and "bottleneck" in tree:
        selected["bottleneck"] = tree["bottleneck"]
    return selected


def latent_auxiliary_enabled() -> bool:
    """Return whether any latent-side auxiliary terms are in use."""
    return bool(
        args.use_latent_dynamics
        or args.latent_cov_coef > 0.0
        or args.latent_var_coef > 0.0
        or args.latent_whiten_coef > 0.0
        or (args.latent_subspace_coef > 0.0 and args.encoder_type.lower() == "stiefel_cnn")
        or (args.cnn_sigreg_coef > 0.0 and args.encoder_type.lower() == "cnn")
    )


def latent_path_is_active() -> bool:
    """Whether actor/critic are driven by a dedicated latent manifold."""
    return bool(args.use_latent_bottleneck or args.encoder_type.lower() == "stiefel_cnn")


def latent_transition_matrix_metrics(transition_params) -> dict:
    """Return singular-value diagnostics for linear latent transition matrices."""
    if transition_params is None:
        return {
            "latent_transition/F_sigma_max": 0.0,
            "latent_transition/F_sigma_min": 0.0,
            "latent_transition/F_condition_number": 0.0,
            "latent_transition/G_sigma_max": 0.0,
            "latent_transition/G_sigma_min": 0.0,
            "latent_transition/G_condition_number": 0.0,
        }
    try:
        f_matrix = np.array(jax.device_get(transition_params["params"]["F"]))
        g_matrix = np.array(jax.device_get(transition_params["params"]["G"]))
    except (TypeError, KeyError):
        return {
            "latent_transition/F_sigma_max": 0.0,
            "latent_transition/F_sigma_min": 0.0,
            "latent_transition/F_condition_number": 0.0,
            "latent_transition/G_sigma_max": 0.0,
            "latent_transition/G_sigma_min": 0.0,
            "latent_transition/G_condition_number": 0.0,
        }

    if f_matrix.size == 0:
        f_sigma_max = 0.0
        f_sigma_min = 0.0
        f_cond = 0.0
    else:
        f_sv = np.linalg.svd(f_matrix, full_matrices=False, compute_uv=False)
        f_sigma_max = float(f_sv[0]) if f_sv.size > 0 else 0.0
        f_sigma_min = float(f_sv[-1]) if f_sv.size > 0 else 0.0
        f_cond = float(f_sigma_max / (f_sigma_min + 1e-12)) if f_sv.size > 0 else 0.0

    if g_matrix.size == 0:
        g_sigma_max = 0.0
        g_sigma_min = 0.0
        g_cond = 0.0
    else:
        g_sv = np.linalg.svd(g_matrix, full_matrices=False, compute_uv=False)
        g_sigma_max = float(g_sv[0]) if g_sv.size > 0 else 0.0
        g_sigma_min = float(g_sv[-1]) if g_sv.size > 0 else 0.0
        g_cond = float(g_sigma_max / (g_sigma_min + 1e-12)) if g_sv.size > 0 else 0.0

    return {
        "latent_transition/F_sigma_max": f_sigma_max,
        "latent_transition/F_sigma_min": f_sigma_min,
        "latent_transition/F_condition_number": f_cond,
        "latent_transition/G_sigma_max": g_sigma_max,
        "latent_transition/G_sigma_min": g_sigma_min,
        "latent_transition/G_condition_number": g_cond,
    }


def ramped_aux_coef(
    iteration: int,
    target_coef: float,
    warmup_updates: int,
    rampup_updates: int,
) -> float:
    """Warmup at zero, then linearly ramp to target_coef."""
    if iteration <= warmup_updates:
        return 0.0
    if iteration <= warmup_updates + rampup_updates:
        ramp_frac = (iteration - warmup_updates) / max(1, rampup_updates)
        return float(target_coef) * ramp_frac
    return float(target_coef)


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
    print("PPO with Manifold MUON")
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
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  muon_dual_lr: {args.muon_dual_lr}")
    print(f"  muon_dual_steps: {args.muon_dual_steps}")
    print(f"  max_grad_norm ({adam_type}): {args.max_grad_norm}")
    print(f"  actor_muon_max_grad_norm: {args.actor_muon_max_grad_norm}")
    print(f"  critic_muon_max_grad_norm: {args.critic_muon_max_grad_norm}")
    print(f"  encoder_type: {args.encoder_type}")
    print(f"  encoder_tanh_scale: {args.encoder_tanh_scale}")
    print(f"  encoder_warmup_updates: {args.encoder_warmup_updates}")
    print(f"  stiefel_latent_dim: {args.stiefel_latent_dim}")
    if args.encoder_type.lower() == "stiefel_cnn":
        print(f"  latent_transition_type: {args.latent_transition_type}")
    print(f"  use_latent_bottleneck: {args.use_latent_bottleneck}")
    if args.use_latent_bottleneck:
        print(f"  latent_dim: {args.latent_dim}")
    print(f"  use_latent_dynamics: {args.use_latent_dynamics}")
    if args.use_latent_dynamics:
        print(f"  latent_transition_type: {args.latent_transition_type}")
        print(f"  latent_dyn_coef: {args.latent_dyn_coef}")
        print(f"  latent_dyn_warmup_updates: {args.latent_dyn_warmup_updates}")
        print(f"  latent_dyn_rampup_updates: {args.latent_dyn_rampup_updates}")
    print(f"  latent_cov_coef: {args.latent_cov_coef}")
    print(f"  latent_var_coef: {args.latent_var_coef}")
    print(f"  latent_whiten_coef: {args.latent_whiten_coef}")
    print(f"  latent_subspace_coef: {args.latent_subspace_coef}")
    print(f"  cnn_sigreg_coef: {args.cnn_sigreg_coef}")
    print(f"  cnn_sigreg_warmup_updates: {args.cnn_sigreg_warmup_updates}")
    print(f"  cnn_sigreg_rampup_updates: {args.cnn_sigreg_rampup_updates}")
    print(f"  cnn_sigreg_num_slices: {args.cnn_sigreg_num_slices}")
    print(f"  cnn_sigreg_t_points: {args.cnn_sigreg_t_points}")
    print(f"  cnn_sigreg_t_min: {args.cnn_sigreg_t_min}")
    print(f"  cnn_sigreg_t_max: {args.cnn_sigreg_t_max}")
    print(f"  log_latent_grad_alignment: {args.log_latent_grad_alignment}")
    print(f"  latent_grad_alignment_interval: {args.latent_grad_alignment_interval}")
    print(f"  lejepa: {args.lejepa}")
    if args.encoder_type.lower() == "vit":
        print(f"  vit_patch_size: {args.vit_patch_size}")
        print(f"  vit_hidden_size: {args.vit_hidden_size}")
        print(f"  vit_proj_dim: {args.vit_proj_dim}")
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
    if args.encoder_type.lower() == "hybrid_vit":
        print(f"  vit_hidden_size: {args.vit_hidden_size}")
        print(f"  vit_proj_dim: {args.vit_proj_dim}")
        print(f"  vit_mlp_dim: {args.vit_mlp_dim}")
        print(f"  vit_num_heads: {args.vit_num_heads}")
        print(f"  vit_num_layers: {args.vit_num_layers}")
        print(f"  hybrid_vit_stem_c1: {args.hybrid_vit_stem_c1}")
        print(f"  hybrid_vit_stem_c2: {args.hybrid_vit_stem_c2}")
        print(f"  hybrid_vit_stem_c3: {args.hybrid_vit_stem_c3}")
        print(f"  hybrid_vit_stem_c4: {args.hybrid_vit_stem_c4}")
    if args.encoder_type.lower() == "drq_vit":
        print(f"  vit_hidden_size: {args.vit_hidden_size}")
        print(f"  vit_proj_dim: {args.vit_proj_dim}")
        print(f"  vit_mlp_dim: {args.vit_mlp_dim}")
        print(f"  vit_num_heads: {args.vit_num_heads}")
        print(f"  vit_num_layers: {args.vit_num_layers}")
        print(f"  drq_vit_stem_channels: {args.drq_vit_stem_channels}")
        print(f"  drq_vit_token_downsample: {args.drq_vit_token_downsample}")
        print(f"  drq_vit_apply_output_tanh: {args.drq_vit_apply_output_tanh}")
    if args.encoder_type.lower() == "scott":
        print(f"  vit_hidden_size: {args.vit_hidden_size}")
        print(f"  vit_proj_dim: {args.vit_proj_dim}")
        print(f"  vit_mlp_dim: {args.vit_mlp_dim}")
        print(f"  vit_num_heads: {args.vit_num_heads}")
        print(f"  vit_num_layers: {args.vit_num_layers}")
        print(f"  scott_use_swiglu: {args.scott_use_swiglu}")
        print(f"  scott_num_register_tokens: {args.scott_num_register_tokens}")
        print(f"  scott_dropout_rate: {args.scott_dropout_rate}")
        print(f"  scott_attention_dropout_rate: {args.scott_attention_dropout_rate}")
        print(f"  scott_stochastic_depth_rate: {args.scott_stochastic_depth_rate}")
    if args.lejepa:
        print(f"  lejepa_lambda: {args.lejepa_lambda}")
        print(f"  lejepa_aux_coef: {args.lejepa_aux_coef}")
        print(f"  lejepa_proj_dim: {args.lejepa_proj_dim}")
        print(f"  lejepa_num_views: {args.lejepa_num_views}")
        print(f"  lejepa_num_slices: {args.lejepa_num_slices}")
        print(f"  lejepa_warmup_updates: {args.lejepa_warmup_updates}")
        print(f"  lejepa_rampup_updates: {args.lejepa_rampup_updates}")
        print(f"  lejepa_t_points: {args.lejepa_t_points}")
        print(f"  lejepa_t_min: {args.lejepa_t_min}")
        print(f"  lejepa_t_max: {args.lejepa_t_max}")
        print(f"  lejepa_view_mode: {args.lejepa_view_mode}")
        print(f"  lejepa_use_layernorm: {args.lejepa_use_layernorm}")
        print(f"  lejepa_sigreg_mode: {args.lejepa_sigreg_mode}")
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
    if args.mim_jepa and args.encoder_type.lower() != "scott":
        raise ValueError("--mim-jepa requires --encoder-type scott")
    if args.lejepa and args.encoder_type.lower() != "cnn":
        raise ValueError("--lejepa requires --encoder-type cnn in the first pass")
    if args.lejepa and args.mim_jepa:
        raise ValueError("--lejepa cannot be combined with --mim-jepa")
    if args.lejepa_num_views < 2:
        raise ValueError("--lejepa-num-views must be at least 2")
    if args.lejepa_num_slices < 1:
        raise ValueError("--lejepa-num-slices must be positive")
    if args.lejepa_t_points < 1:
        raise ValueError("--lejepa-t-points must be positive")
    if args.lejepa_aux_coef < 0:
        raise ValueError("--lejepa-aux-coef must be non-negative")
    if args.lejepa_warmup_updates < 0:
        raise ValueError("--lejepa-warmup-updates must be non-negative")
    if args.lejepa_rampup_updates < 0:
        raise ValueError("--lejepa-rampup-updates must be non-negative")
    if args.lejepa_view_mode not in {"shift", "shift_noise", "temporal"}:
        raise ValueError("--lejepa-view-mode must be one of: shift, shift_noise, temporal")
    if args.lejepa_sigreg_mode not in {"legacy", "epps_pulley"}:
        raise ValueError("--lejepa-sigreg-mode must be one of: legacy, epps_pulley")
    if args.lejepa and args.lejepa_view_mode == "temporal" and args.lejepa_num_views != 2:
        raise ValueError("--lejepa-view-mode temporal currently requires --lejepa-num-views 2")
    if args.latent_transition_type not in {"mlp", "linear"}:
        raise ValueError("--latent-transition-type must be one of: mlp, linear")

    supports_latent_transitions = args.use_latent_bottleneck or args.encoder_type.lower() == "stiefel_cnn"
    if args.use_latent_dynamics and not supports_latent_transitions:
        raise ValueError(
            "--use-latent-dynamics currently supports --encoder-type stiefel_cnn or --use-latent-bottleneck"
        )

    if args.use_latent_bottleneck and args.latent_dim < 1:
        raise ValueError("--latent-dim must be positive")
    if args.latent_dyn_coef < 0:
        raise ValueError("--latent-dyn-coef must be non-negative")
    if args.latent_cov_coef < 0:
        raise ValueError("--latent-cov-coef must be non-negative")
    if args.latent_var_coef < 0:
        raise ValueError("--latent-var-coef must be non-negative")
    if args.latent_whiten_coef < 0:
        raise ValueError("--latent-whiten-coef must be non-negative")
    if args.latent_subspace_coef < 0:
        raise ValueError("--latent-subspace-coef must be non-negative")
    if args.latent_subspace_coef > 0 and args.encoder_type.lower() != "stiefel_cnn":
        raise ValueError("--latent-subspace-coef currently applies only when --encoder-type stiefel_cnn")
    if args.cnn_sigreg_coef > 0 and args.encoder_type.lower() != "cnn":
        raise ValueError("--cnn-sigreg-coef currently applies only when --encoder-type cnn")
    if args.cnn_sigreg_coef < 0:
        raise ValueError("--cnn-sigreg-coef must be non-negative")
    if args.cnn_sigreg_warmup_updates < 0:
        raise ValueError("--cnn-sigreg-warmup-updates must be non-negative")
    if args.cnn_sigreg_rampup_updates < 0:
        raise ValueError("--cnn-sigreg-rampup-updates must be non-negative")
    if args.cnn_sigreg_num_slices < 1:
        raise ValueError("--cnn-sigreg-num-slices must be positive")
    if args.cnn_sigreg_t_points < 1:
        raise ValueError("--cnn-sigreg-t-points must be positive")
    if args.latent_dyn_warmup_updates < 0:
        raise ValueError("--latent-dyn-warmup-updates must be non-negative")
    if args.latent_dyn_rampup_updates < 0:
        raise ValueError("--latent-dyn-rampup-updates must be non-negative")
    if args.latent_grad_alignment_interval < 1:
        raise ValueError("--latent-grad-alignment-interval must be at least 1")
    if args.use_latent_bottleneck and args.encoder_type.lower() not in ("cnn", "stiefel_cnn"):
        raise ValueError("--use-latent-bottleneck currently supports --encoder-type cnn or stiefel_cnn")
    if args.use_latent_bottleneck and args.mim_jepa:
        raise ValueError("--use-latent-bottleneck and --mim-jepa are mutually exclusive")

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
        stem_c1=args.hybrid_vit_stem_c1,
        stem_c2=args.hybrid_vit_stem_c2,
        stem_c3=args.hybrid_vit_stem_c3,
        stem_c4=args.hybrid_vit_stem_c4,
        drq_stem_channels=args.drq_vit_stem_channels,
        drq_token_downsample=args.drq_vit_token_downsample,
        drq_apply_output_tanh=args.drq_vit_apply_output_tanh,
        proj_dim=args.vit_proj_dim,
        scott_use_swiglu=args.scott_use_swiglu,
        scott_num_register_tokens=args.scott_num_register_tokens,
        scott_dropout_rate=args.scott_dropout_rate,
        scott_attention_dropout_rate=args.scott_attention_dropout_rate,
        scott_stochastic_depth_rate=args.scott_stochastic_depth_rate,
    )
    network = build_encoder(
        args.encoder_type,
        args.encoder_tanh_scale,
        vit_config=vit_config,
        stiefel_latent_dim=args.stiefel_latent_dim,
    )
    bottleneck = None
    latent_transition = None
    latent_transition_dim = None
    if args.use_crate_head:
        actor = CRATEActor(action_dim=action_dim, crate_step_size=args.crate_step_size)
        critic = CRATECritic(crate_step_size=args.crate_step_size)
        print(f"Using CRATE heads with step_size={args.crate_step_size}")
    else:
        actor = Actor(action_dim=action_dim)
        critic = Critic()
    lejepa_head = None

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)
    actor_input = dummy_hidden
    bottleneck_params = None
    if args.use_latent_bottleneck:
        key, bottleneck_key = jax.random.split(key)
        bottleneck = BottleneckProjection(latent_dim=args.latent_dim)
        bottleneck_params = bottleneck.init(bottleneck_key, dummy_hidden)
        actor_input = bottleneck.apply(bottleneck_params, dummy_hidden)
        latent_transition_dim = args.latent_dim
    elif args.encoder_type.lower() == "stiefel_cnn":
        latent_transition_dim = args.stiefel_latent_dim
    scott_dummy_tokens = None
    scott_num_patches = None
    scott_grid_size = None
    if args.encoder_type.lower() == "scott":
        scott_dummy_tokens = network.apply(network_params, dummy_obs, return_tokens=True)
        scott_num_patches = int(scott_dummy_tokens.shape[1])
        scott_grid_size = int(round(np.sqrt(scott_num_patches)))
        if scott_grid_size * scott_grid_size != scott_num_patches:
            raise ValueError(
                f"SCOTT produced a non-square token grid: {scott_num_patches} tokens"
            )
        print(f"scott_num_patches: {scott_num_patches}")
        print(f"scott_grid_size: {scott_grid_size}x{scott_grid_size}")

    params_dict = {
        "network": network_params,
        "actor": actor.init(actor_key, actor_input),
        "critic": critic.init(critic_key, actor_input),
    }
    if bottleneck_params is not None:
        params_dict["bottleneck"] = bottleneck_params

    if args.lejepa:
        key, lejepa_key = jax.random.split(key)
        lejepa_head = LeJEPAProjector(
            proj_dim=args.lejepa_proj_dim,
            use_layernorm=args.lejepa_use_layernorm,
        )
        params_dict["lejepa_head"] = lejepa_head.init(lejepa_key, dummy_hidden)
        print(f"lejepa_proj_dim: {args.lejepa_proj_dim}")
        print(f"lejepa_num_views: {args.lejepa_num_views}")

    # MIM-JEPA: predictor + target encoder
    target_params = None
    predictor = None
    if args.mim_jepa:
        key, pred_key = jax.random.split(key)
        predictor = MIMJEPAPredictor(
            embed_dim=args.vit_hidden_size,
            num_register_tokens=args.scott_num_register_tokens,
            backbone_depth=args.vit_num_layers,
            num_layers=args.mim_jepa_predictor_depth,
            num_heads=args.vit_num_heads,
            use_swiglu=args.scott_use_swiglu,
            mlp_dim=args.vit_mlp_dim,
            dropout_rate=args.scott_dropout_rate,
            attention_dropout_rate=args.scott_attention_dropout_rate,
            stochastic_depth_rate=args.scott_stochastic_depth_rate,
        )
        predictor_params = predictor.init(pred_key, scott_dummy_tokens)
        params_dict["predictor"] = predictor_params
        print(f"mim_jepa_mask_type: blockwise")
        print(f"mim_jepa_ema_tau: {args.mim_jepa_ema_tau}")

    if args.use_latent_dynamics:
        key, transition_key = jax.random.split(key)
        if latent_transition_dim is None:
            raise ValueError("latent_transition_dim must be set when use_latent_dynamics is enabled")

        if args.latent_transition_type == "linear":
            latent_transition = LinearLatentTransition(
                latent_dim=latent_transition_dim,
                action_dim=action_dim,
            )
        elif args.latent_transition_type == "mlp":
            latent_transition = LatentTransition(
                latent_dim=latent_transition_dim,
                action_dim=action_dim,
            )
        else:
            raise ValueError(f"Unsupported latent transition type: {args.latent_transition_type}")

        dummy_latent = jnp.zeros((1, latent_transition_dim))
        dummy_action = jnp.zeros((1, action_dim))
        params_dict["latent_transition"] = latent_transition.init(
            transition_key, dummy_latent, dummy_action
        )

    all_params = flax.core.freeze(params_dict)
    if args.mim_jepa:
        target_params = all_params["network"]

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
    lejepa_params_count = count_params(all_params, "lejepa_head")
    total_params = (
        encoder_params
        + actor_params
        + critic_params
        + predictor_params_count
        + lejepa_params_count
    )

    actor_muon, actor_adam = count_by_type(all_params, "actor")
    critic_muon, critic_adam = count_by_type(all_params, "critic")

    print(f"\nParameter breakdown:")
    print(f"  Encoder (Adam): {encoder_params:,}")
    print(f"  Actor total: {actor_params:,} (MUON: {actor_muon:,}, Adam: {actor_adam:,})")
    print(f"  Critic total: {critic_params:,} (MUON: {critic_muon:,}, Adam: {critic_adam:,})")
    if args.mim_jepa:
        print(f"  Predictor (Adam): {predictor_params_count:,}")
    if args.lejepa:
        print(f"  LeJEPA projector (Adam): {lejepa_params_count:,}")
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
        print("\nLearning rate annealing: DISABLED")

    # Create optimizer
    tx = create_optimizer(
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
        vit_qk_stiefel=args.vit_qk_stiefel,
        vit_qk_stiefel_lr=args.vit_qk_stiefel_lr,
        vit_qk_stiefel_dual_lr=args.vit_qk_stiefel_dual_lr,
        vit_qk_stiefel_dual_steps=args.vit_qk_stiefel_dual_steps,
        vit_qk_stiefel_msign_steps=args.vit_qk_stiefel_msign_steps,
        vit_qk_stiefel_max_grad_norm=args.vit_qk_stiefel_max_grad_norm,
        use_stiefel_encoder_muon=(args.encoder_type.lower() == "stiefel_cnn"),
        stiefel_encoder_muon_lr=args.heads_muon_lr,
        stiefel_encoder_muon_max_grad_norm=args.actor_muon_max_grad_norm,
        use_heads_muon=args.use_heads_muon,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    network_apply = network.apply  # raw apply before JIT wrapping
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    if args.mim_jepa:
        predictor.apply = jax.jit(predictor.apply)
    if args.lejepa:
        lejepa_head.apply = jax.jit(lejepa_head.apply)
    if args.use_latent_bottleneck:
        bottleneck.apply = jax.jit(bottleneck.apply)
    if args.use_latent_dynamics:
        latent_transition.apply = jax.jit(latent_transition.apply)

    # MIM-JEPA loss helpers (defined only when mim_jepa=True; closed over network/predictor/args)
    if args.mim_jepa:
        _mim_grid_size = scott_grid_size
        _mim_num_patches = scott_num_patches

        def compute_mim_jepa_loss(params, target_params, obs, mask_key):
            """SmoothL1 reconstruction loss on masked token positions."""
            B = obs.shape[0]
            keep_masks = sample_block_mask(
                mask_key, B, _mim_grid_size, _mim_grid_size, args.mim_jepa_mask_ratio
            )
            target_tokens = network_apply(target_params, obs, return_tokens=True)
            target_tokens = jax.lax.stop_gradient(target_tokens)
            t_mean = jnp.mean(target_tokens, axis=-1, keepdims=True)
            t_var = jnp.var(target_tokens, axis=-1, keepdims=True)
            target_tokens = (target_tokens - t_mean) / jnp.sqrt(t_var + 1e-5)
            context_tokens = network_apply(
                params["network"], obs, return_tokens=True, masks=keep_masks
            )
            pred_tokens = predictor.apply(params["predictor"], context_tokens)
            drop_masks = 1.0 - keep_masks  # (B, N) — 1 where masked
            diff = pred_tokens - target_tokens
            abs_diff = jnp.abs(diff)
            huber = jnp.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
            D = pred_tokens.shape[-1]
            huber = huber * drop_masks[:, :, None]  # (B, N, D) — zero at kept positions
            loss_per_sample = (huber.sum(axis=-1) / D).sum(axis=-1) / (
                drop_masks.sum(axis=-1) + 1e-8
            )
            return loss_per_sample.mean()

    if args.lejepa:
        lejepa_t_values = jnp.linspace(
            args.lejepa_t_min,
            args.lejepa_t_max,
            args.lejepa_t_points,
            dtype=jnp.float32,
        )
        temporal_offsets = jnp.array([1, 2, 3], dtype=jnp.int32)

        def sample_temporal_partner_obs(
            rollout_obs: jnp.ndarray,
            rollout_dones: jnp.ndarray,
            step_idx: jnp.ndarray,
            env_idx: jnp.ndarray,
            temporal_key: jax.random.PRNGKey,
        ) -> jnp.ndarray:
            """Sample same-env nearby rollout observations without crossing episode boundaries."""
            num_steps = rollout_obs.shape[0]
            offset_key, direction_key = jax.random.split(temporal_key)
            offset_idx = jax.random.randint(
                offset_key, step_idx.shape, 0, temporal_offsets.shape[0]
            )
            signed_offsets = temporal_offsets[offset_idx]
            direction = jax.random.bernoulli(
                direction_key, 0.5, step_idx.shape
            ).astype(jnp.int32) * 2 - 1
            candidate_step = step_idx + direction * signed_offsets
            candidate_step = jnp.clip(candidate_step, 0, num_steps - 1)

            def sample_one(t0, t1, env):
                lo = jnp.minimum(t0, t1)
                hi = jnp.maximum(t0, t1)
                env_dones = rollout_dones[:, env]
                between_mask = jnp.logical_and(
                    jnp.arange(num_steps) > lo,
                    jnp.arange(num_steps) <= hi,
                )
                valid = jnp.logical_and(
                    t0 != t1,
                    jnp.logical_not(jnp.any(jnp.logical_and(between_mask, env_dones))),
                )
                return jax.lax.cond(
                    valid,
                    lambda _: rollout_obs[t1, env],
                    lambda _: rollout_obs[t0, env],
                    operand=None,
                )

            return jax.vmap(sample_one)(step_idx, candidate_step, env_idx)

        def make_lejepa_views(
            obs: jnp.ndarray,
            aux_key: jax.random.PRNGKey,
            rollout_obs: jnp.ndarray,
            rollout_dones: jnp.ndarray,
            step_idx: jnp.ndarray,
            env_idx: jnp.ndarray,
        ) -> jnp.ndarray:
            """Build LeJEPA views according to the configured RL-safe view mode."""
            if args.lejepa_view_mode == "temporal":
                if args.lejepa_num_views != 2:
                    raise ValueError("--lejepa-view-mode temporal currently requires --lejepa-num-views 2")
                first_key, second_key = jax.random.split(aux_key)
                first_view = random_shift(first_key, obs, pad=args.augment_pad)
                temporal_obs = sample_temporal_partner_obs(
                    rollout_obs, rollout_dones, step_idx, env_idx, second_key
                )
                second_view = random_shift(second_key, temporal_obs, pad=args.augment_pad)
                return jnp.stack([first_view, second_view], axis=0)

            view_keys = jax.random.split(aux_key, args.lejepa_num_views)
            if args.lejepa_view_mode == "shift_noise":
                return jax.vmap(
                    lambda k: random_shift_with_noise(k, obs, pad=args.augment_pad)
                )(view_keys)
            return jax.vmap(lambda k: random_shift(k, obs, pad=args.augment_pad))(view_keys)

        def compute_lejepa_loss(
            params,
            obs,
            aux_key,
            rollout_obs,
            rollout_dones,
            step_idx,
            env_idx,
        ):
            """Agreement-plus-SIGReg loss on projected CNN latents."""
            view_key, slice_key = jax.random.split(aux_key)
            views = make_lejepa_views(
                obs,
                view_key,
                rollout_obs,
                rollout_dones,
                step_idx,
                env_idx,
            )
            hidden_views = jax.vmap(
                lambda view_obs: network_apply(params["network"], view_obs)
            )(views)
            proj_views = jax.vmap(
                lambda hidden: lejepa_head.apply(params["lejepa_head"], hidden)
            )(hidden_views)

            centers = jnp.mean(proj_views, axis=0, keepdims=True)
            sim_loss = jnp.mean(jnp.square(proj_views - centers))

            directions = jax.random.normal(
                slice_key,
                (args.lejepa_num_slices, args.lejepa_proj_dim),
            )
            directions = directions / (
                jnp.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8
            )
            if args.lejepa_sigreg_mode == "legacy":
                projected = jnp.einsum("vbk,sk->vbs", proj_views, directions)
                sigreg_loss = jnp.mean(
                    jax.vmap(lambda samples: empirical_char_diff(samples, lejepa_t_values))(projected)
                )
            else:
                sigreg_loss = jnp.mean(
                    jax.vmap(lambda samples: sigreg_epps_pulley(samples, directions, lejepa_t_values))(proj_views)
                )

            total_loss = (
                (1.0 - args.lejepa_lambda) * sim_loss
                + args.lejepa_lambda * sigreg_loss
            )
            return total_loss, sim_loss, sigreg_loss

    def compute_latent_losses(
        params,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        cnn_sigreg_key: jax.random.PRNGKey = None,
        next_obs: jnp.ndarray = None,
        next_dones: jnp.ndarray = None,
        include_cnn_sigreg: bool = True,
    ):
        """Compute optional latent auxiliary losses.

        Returns (z_t, latent_dyn_loss, latent_cov_loss,
                 latent_var_loss, latent_whiten_loss, latent_subspace_loss,
                 latent_cnn_sigreg_loss).
        """
        return_pre_tanh = args.encoder_type.lower() == "cnn" and args.cnn_sigreg_coef > 0.0
        h_t, z_t = encode_features(params, obs, return_preproj=True, return_pre_tanh=return_pre_tanh)
        latent_dyn_loss_raw = jnp.array(0.0, dtype=jnp.float32)
        latent_cov_loss = latent_covariance_loss(z_t)
        latent_var_loss = latent_variance_loss(z_t)
        latent_whiten_loss_val = latent_whiten_loss(z_t)

        latent_subspace_loss = jnp.array(0.0, dtype=jnp.float32)
        if args.latent_subspace_coef > 0.0 and args.encoder_type.lower() == "stiefel_cnn":
            latent_subspace_loss = latent_stiefel_projection_loss(h_t, params)

        if args.use_latent_dynamics:
            if next_obs is None or next_dones is None:
                latent_dyn_loss_raw = jnp.array(0.0, dtype=jnp.float32)
            else:
                _, z_next = encode_features(params, next_obs)
                z_next_pred = latent_transition.apply(
                    params["latent_transition"], z_t, actions
                )
                not_done = 1.0 - next_dones.astype(jnp.float32)
                pred_error = (z_next_pred - z_next) * not_done[:, None]
                latent_dyn_loss_raw = jnp.mean(jnp.sum(jnp.square(pred_error), axis=-1))

        latent_cnn_sigreg_raw = jnp.array(0.0, dtype=jnp.float32)
        if return_pre_tanh and args.cnn_sigreg_coef > 0.0 and include_cnn_sigreg:
            if cnn_sigreg_key is None:
                raise ValueError("cnn_sigreg_key must be provided when cnn_sigreg_coef > 0")
            latent_cnn_sigreg_raw = latent_cnn_sigreg_loss(h_t, cnn_sigreg_key)

        return (
            z_t,
            latent_dyn_loss_raw,
            latent_cov_loss,
            latent_var_loss,
            latent_whiten_loss_val,
            latent_subspace_loss,
            latent_cnn_sigreg_raw,
        )

    def encode_features(params, obs, return_preproj: bool = False, return_pre_tanh: bool = False):
        is_stiefel = args.encoder_type.lower() == "stiefel_cnn"
        is_cnn = args.encoder_type.lower() == "cnn"
        if is_stiefel and (return_preproj or args.use_latent_bottleneck):
            h_t, z_t = network_apply(
                params["network"],
                obs,
                return_preproj=True,
            )
        elif is_cnn and (return_pre_tanh or return_preproj):
            h_t = network_apply(
                params["network"],
                obs,
                return_pre_tanh=True,
            )
            z_t = jnp.tanh(h_t)
        else:
            h_t = network_apply(params["network"], obs)
            z_t = h_t

        if args.use_latent_bottleneck:
            z_t = bottleneck.apply(params["bottleneck"], h_t)

        return h_t, z_t

    def actor_critic_input(params, obs):
        _, features = encode_features(params, obs)
        return features

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and return updated key."""
        features = actor_critic_input(agent_state.params, next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params["actor"], features)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic.apply(agent_state.params["critic"], features)

        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        features = actor_critic_input(params, x)
        actor_mean, actor_logstd = actor.apply(params["actor"], features)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic.apply(params["critic"], features).squeeze(-1)

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
            actor_critic_input(agent_state.params, next_obs),
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

    def ppo_objective_loss(
        params,
        x,
        a,
        logp,
        mb_advantages,
        mb_returns,
        mb_values,
        aug_key,
        *_unused,
    ):
        """Pure PPO objective without auxiliary losses."""
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
        ppo_only_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return ppo_only_loss, (
            pg_loss,
            v_loss,
            entropy_loss,
            jax.lax.stop_gradient(approx_kl),
        )

    def lejepa_raw_loss(
        params,
        orig_x,
        step_idx,
        env_idx,
        rollout_obs,
        rollout_dones,
        mask_key,
    ):
        """Raw LeJEPA loss before outer coefficient scaling."""
        if not args.lejepa:
            return (
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
            )
        return compute_lejepa_loss(
            params,
            orig_x,
            mask_key,
            rollout_obs,
            rollout_dones,
            step_idx,
            env_idx,
        )

    def maybe_augment_obs(obs, aug_key):
        return random_shift(aug_key, obs, pad=args.augment_pad) if args.use_augmentation else obs

    def ppo_loss(
        params,
        x,
        a,
        logp,
        mb_advantages,
        mb_returns,
        mb_values,
        aug_key,
        step_idx,
        env_idx,
        rollout_obs,
        rollout_dones,
        target_params=None,
        mask_key=None,
        mim_lambda=jnp.float32(0.0),
        lejepa_aux_coef=jnp.float32(0.0),
        latent_dyn_coef_active=jnp.float32(0.0),
        cnn_sigreg_coef_active=jnp.float32(0.0),
        mb_next_obs=None,
        mb_next_dones=None,
        cnn_sigreg_key=None,
    ):
        """PPO loss function with optional MIM-JEPA and LeJEPA auxiliaries."""
        orig_x = x
        ppo_x = maybe_augment_obs(x, aug_key)
        ppo_only_loss, (
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = ppo_objective_loss(
            params,
            ppo_x,
            a,
            logp,
            mb_advantages,
            mb_returns,
            mb_values,
            aug_key,
        )
        total_loss = ppo_only_loss

        if args.mim_jepa:
            mim_loss = compute_mim_jepa_loss(params, target_params, ppo_x, mask_key)
            total_loss = total_loss + mim_lambda * mim_loss
        else:
            mim_loss = jnp.float32(0.0)

        if args.lejepa:
            lejepa_loss_raw, lejepa_sim_loss, lejepa_sigreg_loss = lejepa_raw_loss(
                params,
                orig_x,
                step_idx,
                env_idx,
                rollout_obs,
                rollout_dones,
                mask_key,
            )
            lejepa_loss = lejepa_aux_coef * lejepa_loss_raw
            total_loss = total_loss + lejepa_loss
        else:
            lejepa_loss_raw = jnp.float32(0.0)
            lejepa_loss = jnp.float32(0.0)
            lejepa_sim_loss = jnp.float32(0.0)
            lejepa_sigreg_loss = jnp.float32(0.0)

        if latent_auxiliary_enabled() or args.use_latent_dynamics:
            (
                z_t,
                latent_dyn_loss_raw,
                latent_cov_loss,
                latent_var_loss,
                latent_whiten_loss_val,
                latent_subspace_loss,
                latent_cnn_sigreg_loss_raw,
            ) = compute_latent_losses(
                params,
                orig_x,
                a,
                cnn_sigreg_key,
                mb_next_obs,
                mb_next_dones,
            )
            latent_dyn_loss = latent_dyn_coef_active * latent_dyn_loss_raw
            latent_cnn_sigreg_loss = cnn_sigreg_coef_active * latent_cnn_sigreg_loss_raw
            if args.use_latent_dynamics:
                total_loss = total_loss + latent_dyn_loss
            if args.cnn_sigreg_coef > 0.0:
                total_loss = total_loss + latent_cnn_sigreg_loss
            if args.latent_cov_coef > 0.0:
                total_loss = total_loss + args.latent_cov_coef * latent_cov_loss
            if args.latent_var_coef > 0.0:
                total_loss = total_loss + args.latent_var_coef * latent_var_loss
            if args.latent_whiten_coef > 0.0:
                total_loss = total_loss + args.latent_whiten_coef * latent_whiten_loss_val
            if args.latent_subspace_coef > 0.0:
                total_loss = total_loss + args.latent_subspace_coef * latent_subspace_loss
        else:
                latent_dyn_loss = jnp.float32(0.0)
                latent_dyn_loss_raw = jnp.float32(0.0)
                latent_cov_loss = jnp.float32(0.0)
                latent_var_loss = jnp.float32(0.0)
                latent_whiten_loss_val = jnp.float32(0.0)
                latent_subspace_loss = jnp.float32(0.0)
                latent_cnn_sigreg_loss = jnp.float32(0.0)
                latent_cnn_sigreg_loss_raw = jnp.float32(0.0)

        return total_loss, (
            ppo_only_loss,
            pg_loss,
            v_loss,
            entropy_loss,
            mim_loss,
            lejepa_loss,
            lejepa_loss_raw,
            lejepa_sim_loss,
            lejepa_sigreg_loss,
            latent_dyn_loss,
            latent_dyn_loss_raw,
            latent_cov_loss,
            latent_var_loss,
            latent_whiten_loss_val,
            latent_subspace_loss,
            latent_cnn_sigreg_loss,
            latent_cnn_sigreg_loss_raw,
            jax.lax.stop_gradient(approx_kl),
        )

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    ppo_only_grad_fn = jax.grad(
        lambda params, x, a, logp, mb_advantages, mb_returns, mb_values, aug_key:
        ppo_objective_loss(
            params, x, a, logp, mb_advantages, mb_returns, mb_values, aug_key
        )[0]
    )
    lejepa_raw_grad_fn = jax.grad(
        lambda params, orig_x, step_idx, env_idx, rollout_obs, rollout_dones, mask_key:
        lejepa_raw_loss(
            params, orig_x, step_idx, env_idx, rollout_obs, rollout_dones, mask_key
        )[0]
    )
    latent_dyn_grad_fn = jax.grad(
        lambda params, obs, actions, next_obs, next_dones, latent_dyn_coef_active:
        latent_dyn_coef_active * compute_latent_losses(
            params, obs, actions, None, next_obs, next_dones, include_cnn_sigreg=False
        )[1]
    )

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
        target_params=None,
        mim_lambda: jnp.ndarray = jnp.float32(0.0),
        lejepa_aux_coef: jnp.ndarray = jnp.float32(0.0),
        latent_dyn_coef_active: jnp.ndarray = jnp.float32(0.0),
        cnn_sigreg_coef_active: jnp.ndarray = jnp.float32(0.0),
        compute_latent_grad_alignment: jnp.ndarray = jnp.array(False),
    ):
        """PPO update, optionally with MIM-JEPA auxiliary loss."""
        def update_epoch(carry, unused_inp):
            agent_state, key, last_grads = carry
            key, subkey, aug_key, mim_key, cnn_sigreg_key = jax.random.split(key, 5)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            aug_keys = jax.random.split(aug_key, args.num_minibatches)
            mim_keys = jax.random.split(mim_key, args.num_minibatches)
            cnn_sigreg_keys = jax.random.split(cnn_sigreg_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state, _ = carry
                minibatch, mb_aug_key, mb_mim_key, mb_cnn_sigreg_key = inputs

                (
                    loss,
                    (
                        ppo_only_loss,
                        pg_loss,
                        v_loss,
                        entropy_loss,
                        mim_loss,
                        lejepa_loss,
                        lejepa_loss_raw,
                        lejepa_sim_loss,
                        lejepa_sigreg_loss,
                        latent_dyn_loss,
                        latent_dyn_loss_raw,
                        latent_cov_loss,
                        latent_var_loss,
                        latent_whiten_loss_val,
                        latent_subspace_loss,
                        latent_cnn_sigreg_loss,
                        latent_cnn_sigreg_loss_raw,
                        approx_kl,
                    ),
                ), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                    minibatch.step_idx,
                    minibatch.env_idx,
                    storage.obs,
                    storage.dones,
                    target_params,
                    mb_mim_key,
                    mim_lambda,
                    lejepa_aux_coef,
                    latent_dyn_coef_active,
                    cnn_sigreg_coef_active,
                    minibatch.next_obs,
                    minibatch.next_dones,
                    mb_cnn_sigreg_key,
                )

                grad_norm = optax.global_norm(grads)
                should_compute_alignment = (
                    compute_latent_grad_alignment
                    & jnp.asarray(latent_path_is_active())
                    & jnp.asarray(args.use_latent_dynamics)
                )

                def compute_alignment_metrics(_):
                    ppo_encoder_grads_full = ppo_only_grad_fn(
                        agent_state.params,
                        maybe_augment_obs(minibatch.obs, mb_aug_key),
                        minibatch.actions,
                        minibatch.logprobs,
                        minibatch.advantages,
                        minibatch.returns,
                        minibatch.values,
                        mb_aug_key,
                    )
                    ppo_encoder_grads = latent_param_tree(
                        ppo_encoder_grads_full, args.use_latent_bottleneck
                    )
                    latent_dyn_grads_full = latent_dyn_grad_fn(
                        agent_state.params,
                        minibatch.obs,
                        minibatch.actions,
                        minibatch.next_obs,
                        minibatch.next_dones,
                        latent_dyn_coef_active,
                    )
                    latent_dyn_grads = latent_param_tree(
                        latent_dyn_grads_full, args.use_latent_bottleneck
                    )
                    encoder_ppo_norm = tree_norm(ppo_encoder_grads)
                    encoder_latent_dyn_norm = tree_norm(latent_dyn_grads)
                    cosine_denom = encoder_ppo_norm * encoder_latent_dyn_norm
                    encoder_latent_dyn_cosine = jnp.where(
                        cosine_denom > 0,
                        tree_dot(ppo_encoder_grads, latent_dyn_grads) / (cosine_denom + 1e-8),
                        0.0,
                    )
                    return (
                        encoder_ppo_norm,
                        encoder_latent_dyn_norm,
                        encoder_latent_dyn_cosine,
                    )

                encoder_ppo_norm, encoder_latent_dyn_norm, encoder_latent_dyn_cosine = jax.lax.cond(
                    should_compute_alignment,
                    compute_alignment_metrics,
                    lambda _: (
                        jnp.float32(0.0),
                        jnp.float32(0.0),
                        jnp.float32(0.0),
                    ),
                    operand=None,
                )
                if args.lejepa:
                    lejepa_encoder_grads = lejepa_raw_grad_fn(
                        agent_state.params,
                        minibatch.obs,
                        minibatch.step_idx,
                        minibatch.env_idx,
                        storage.obs,
                        storage.dones,
                        mb_mim_key,
                    )
                    lejepa_encoder_grads = latent_param_tree(
                        lejepa_encoder_grads, False
                    )["network"]
                    encoder_lejepa_norm = tree_norm(lejepa_encoder_grads)
                    ppo_network_grads = ppo_only_grad_fn(
                        agent_state.params,
                        maybe_augment_obs(minibatch.obs, mb_aug_key),
                        minibatch.actions,
                        minibatch.logprobs,
                        minibatch.advantages,
                        minibatch.returns,
                        minibatch.values,
                        mb_aug_key,
                    )["network"]
                    ppo_network_norm = tree_norm(ppo_network_grads)
                    encoder_ppo_norm = jnp.where(
                        should_compute_alignment,
                        encoder_ppo_norm,
                        ppo_network_norm,
                    )
                    cosine_denom = ppo_network_norm * encoder_lejepa_norm
                    encoder_grad_cosine = jnp.where(
                        cosine_denom > 0,
                        tree_dot(ppo_network_grads, lejepa_encoder_grads) / (cosine_denom + 1e-8),
                        0.0,
                    )
                else:
                    encoder_lejepa_norm = jnp.float32(0.0)
                    encoder_grad_cosine = jnp.float32(0.0)
                agent_state = agent_state.apply_gradients(grads=grads)

                return (agent_state, grads), (
                    loss,
                    ppo_only_loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    mim_loss,
                    lejepa_loss,
                    lejepa_loss_raw,
                    lejepa_sim_loss,
                    lejepa_sigreg_loss,
                    latent_dyn_loss,
                    latent_dyn_loss_raw,
                    latent_cov_loss,
                    latent_var_loss,
                    latent_whiten_loss_val,
                    latent_subspace_loss,
                    latent_cnn_sigreg_loss,
                    latent_cnn_sigreg_loss_raw,
                    approx_kl,
                    grad_norm,
                    encoder_ppo_norm,
                    encoder_latent_dyn_norm,
                    encoder_latent_dyn_cosine,
                    encoder_lejepa_norm,
                    encoder_grad_cosine,
                )

            (
                (agent_state, last_grads),
                (
                    loss,
                    ppo_only_loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    mim_loss,
                    lejepa_loss,
                    lejepa_loss_raw,
                    lejepa_sim_loss,
                    lejepa_sigreg_loss,
                    latent_dyn_loss,
                    latent_dyn_loss_raw,
                    latent_cov_loss,
                    latent_var_loss,
                    latent_whiten_loss_val,
                    latent_subspace_loss,
                    latent_cnn_sigreg_loss,
                    latent_cnn_sigreg_loss_raw,
                    approx_kl,
                    grad_norm,
                    encoder_ppo_norm,
                    encoder_latent_dyn_norm,
                    encoder_latent_dyn_cosine,
                    encoder_lejepa_norm,
                    encoder_grad_cosine,
                ),
            ) = jax.lax.scan(
                update_minibatch,
                (agent_state, last_grads),
                (shuffled_storage, aug_keys, mim_keys, cnn_sigreg_keys),
            )
            return (agent_state, key, last_grads), (
                loss,
                ppo_only_loss,
                pg_loss,
                v_loss,
                entropy_loss,
                mim_loss,
                lejepa_loss,
                lejepa_loss_raw,
                lejepa_sim_loss,
                lejepa_sigreg_loss,
                latent_dyn_loss,
                latent_dyn_loss_raw,
                latent_cov_loss,
                latent_var_loss,
                latent_whiten_loss_val,
                latent_subspace_loss,
                latent_cnn_sigreg_loss,
                latent_cnn_sigreg_loss_raw,
                approx_kl,
                grad_norm,
                encoder_ppo_norm,
                encoder_latent_dyn_norm,
                encoder_latent_dyn_cosine,
                encoder_lejepa_norm,
                encoder_grad_cosine,
            )

        init_grads = jax.tree_util.tree_map(jnp.zeros_like, agent_state.params)
        (
            (agent_state, key, final_grads),
            (
                loss,
                ppo_only_loss,
                pg_loss,
                v_loss,
                entropy_loss,
                mim_loss,
                lejepa_loss,
                lejepa_loss_raw,
                lejepa_sim_loss,
                lejepa_sigreg_loss,
                latent_dyn_loss,
                latent_dyn_loss_raw,
                latent_cov_loss,
                latent_var_loss,
                latent_whiten_loss_val,
                latent_subspace_loss,
                latent_cnn_sigreg_loss,
                latent_cnn_sigreg_loss_raw,
                approx_kl,
                grad_norm,
                encoder_ppo_norm,
                encoder_latent_dyn_norm,
                encoder_latent_dyn_cosine,
                encoder_lejepa_norm,
                encoder_grad_cosine,
            ),
        ) = jax.lax.scan(
            update_epoch,
            (agent_state, key, init_grads),
            (),
            length=args.update_epochs,
        )
        return (
            agent_state,
            loss,
            ppo_only_loss,
            pg_loss,
            v_loss,
            entropy_loss,
            mim_loss,
            lejepa_loss,
            lejepa_loss_raw,
            lejepa_sim_loss,
            lejepa_sigreg_loss,
            latent_dyn_loss,
            latent_dyn_loss_raw,
            latent_cov_loss,
            latent_var_loss,
            latent_whiten_loss_val,
            latent_subspace_loss,
            latent_cnn_sigreg_loss,
            latent_cnn_sigreg_loss_raw,
            approx_kl,
            grad_norm,
            encoder_ppo_norm,
            encoder_latent_dyn_norm,
            encoder_latent_dyn_cosine,
            encoder_lejepa_norm,
            encoder_grad_cosine,
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
            step_idx=jnp.full((args.n_envs,), step, dtype=jnp.int32),
            env_idx=jnp.arange(args.n_envs, dtype=jnp.int32),
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
            next_obs=next_obs_local,
            next_dones=next_done_local,
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

    def collect_probe_plot_images(output_dir: str) -> dict:
        """Load probe plot PNGs as wandb.Image for logging, if tracking is enabled."""
        if not args.track or not args.probe_save_plots or output_dir is None:
            return {}

        plot_files = {
            "probe/plots/r2_vs_k": f"{args.env_name}_r2_vs_k.png",
            "probe/plots/scree": f"{args.env_name}_scree.png",
            "probe/plots/corr_heatmap": f"{args.env_name}_corr_heatmap.png",
            "additive/plots/parity": f"{args.env_name}_additive_parity.png",
            "additive/plots/cosine_gram": f"{args.env_name}_cosine_gram.png",
            "additive/plots/raw_gram": f"{args.env_name}_raw_gram.png",
            "additive/plots/component_norms": f"{args.env_name}_component_norms.png",
        }

        image_logs = {}
        for log_key, file_name in plot_files.items():
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                image_logs[log_key] = wandb.Image(file_path)
        return image_logs

    # Track next probe threshold as an absolute step count so LCM cadence issues
    # don't arise when global_step jumps by (num_steps * n_envs) each iteration.
    next_probe_step = args.probe_interval if args.probe_interval > 0 else float("inf")
    # Probe metrics are buffered here and flushed into the main wandb.log call
    # so we never call wandb.log twice at the same step.
    pending_probe_metrics: dict = {}

    # Probe at initialisation (step 0) — establishes random-encoder baseline.
    if args.probe_interval > 0:
        from linear_probe import run_online_probe
        import pickle
        probe_init_output = os.path.join(args.probe_output_dir, "step_0") if args.probe_output_dir else None
        probe_init_metrics = run_online_probe(
            network=network,
            network_params=agent_state.params["network"],
            bottleneck=bottleneck if args.use_latent_bottleneck else None,
            bottleneck_params=agent_state.params.get("bottleneck") if args.use_latent_bottleneck else None,
            envs=envs,
            args_dict=vars(args),
            n_eval_envs=args.n_envs,
            n_eval_steps=args.probe_n_eval_steps,
            seed=args.seed,
            output_dir=probe_init_output if args.probe_save_plots else None,
        )
        print(f"[probe] step=0 (init) metrics={probe_init_metrics}")
        if args.save_checkpoint:
            ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            params_bytes = flax.serialization.to_bytes(agent_state.params)
            ckpt_path = os.path.join(ckpt_dir, "params_step0.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({"params_bytes": params_bytes, "args": vars(args),
                             "global_step": 0}, f)
            print(f"Checkpoint saved to {ckpt_path}")
        if args.track:
            probe_init_media = collect_probe_plot_images(probe_init_output)
            wandb.log({**probe_init_metrics, **probe_init_media}, step=0)

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

        # Compute MIM-JEPA lambda with warmup + linear ramp-up
        if args.mim_jepa:
            mim_lambda_current = ramped_aux_coef(
                iteration,
                args.mim_jepa_lambda,
                args.mim_jepa_warmup_updates,
                args.mim_jepa_rampup_updates,
            )
            mim_lambda_jax = jnp.float32(mim_lambda_current)
        else:
            mim_lambda_jax = jnp.float32(0.0)

        if args.lejepa:
            lejepa_aux_coef_current = ramped_aux_coef(
                iteration,
                args.lejepa_aux_coef,
                args.lejepa_warmup_updates,
                args.lejepa_rampup_updates,
            )
            lejepa_aux_coef_jax = jnp.float32(lejepa_aux_coef_current)
        else:
            lejepa_aux_coef_current = 0.0
            lejepa_aux_coef_jax = jnp.float32(0.0)

        if args.use_latent_dynamics:
            latent_dyn_coef_current = ramped_aux_coef(
                iteration,
                args.latent_dyn_coef,
                args.latent_dyn_warmup_updates,
                args.latent_dyn_rampup_updates,
            )
            latent_dyn_coef_jax = jnp.float32(latent_dyn_coef_current)
        else:
            latent_dyn_coef_current = 0.0
            latent_dyn_coef_jax = jnp.float32(0.0)
        if args.cnn_sigreg_coef > 0.0 and args.encoder_type.lower() == "cnn":
            cnn_sigreg_coef_current = ramped_aux_coef(
                iteration,
                args.cnn_sigreg_coef,
                args.cnn_sigreg_warmup_updates,
                args.cnn_sigreg_rampup_updates,
            )
            cnn_sigreg_coef_jax = jnp.float32(cnn_sigreg_coef_current)
        else:
            cnn_sigreg_coef_current = 0.0
            cnn_sigreg_coef_jax = jnp.float32(0.0)
        compute_latent_grad_alignment = (
            args.log_latent_grad_alignment
            and args.use_latent_dynamics
            and (iteration % args.latent_grad_alignment_interval == 0)
        )

        (
            agent_state,
            loss,
            ppo_only_loss,
            pg_loss,
            v_loss,
            entropy_loss,
            mim_loss,
            lejepa_loss,
            lejepa_loss_raw,
            lejepa_sim_loss,
            lejepa_sigreg_loss,
            latent_dyn_loss,
            latent_dyn_loss_raw,
            latent_cov_loss,
            latent_var_loss,
            latent_whiten_loss_val,
            latent_subspace_loss,
            latent_cnn_sigreg_loss,
            latent_cnn_sigreg_loss_raw,
            approx_kl,
            grad_norm,
            encoder_ppo_norm,
            encoder_latent_dyn_norm,
            encoder_latent_dyn_cosine,
            encoder_lejepa_norm,
            encoder_grad_cosine,
            final_grads,
            key,
        ) = update_ppo(
            agent_state,
            storage,
            key,
            target_params=target_params,
            mim_lambda=mim_lambda_jax,
            lejepa_aux_coef=lejepa_aux_coef_jax,
            latent_dyn_coef_active=latent_dyn_coef_jax,
            cnn_sigreg_coef_active=cnn_sigreg_coef_jax,
            compute_latent_grad_alignment=jnp.array(compute_latent_grad_alignment),
        )

        # EMA update of target encoder (after optimizer step)
        if args.mim_jepa:
            tau = args.mim_jepa_ema_tau
            target_params = jax.tree_util.tree_map(
                lambda t, o: tau * t + (1.0 - tau) * o,
                target_params,
                agent_state.params["network"],
            )

        # Online linear probe — fire when global_step crosses the next threshold.
        # Using >= avoids the LCM cadence problem that % would cause when
        # global_step jumps by (num_steps * n_envs = 1280) each iteration.
        if global_step >= next_probe_step:
            from linear_probe import run_online_probe
            probe_output = os.path.join(args.probe_output_dir, f"step_{global_step}")
            probe_metrics = run_online_probe(
                network=network,
                network_params=agent_state.params["network"],
                bottleneck=bottleneck if args.use_latent_bottleneck else None,
                bottleneck_params=agent_state.params.get("bottleneck") if args.use_latent_bottleneck else None,
                envs=envs,
                args_dict=vars(args),
                n_eval_envs=args.n_envs,
                n_eval_steps=args.probe_n_eval_steps,
                seed=args.seed + iteration,
                output_dir=probe_output if args.probe_save_plots else None,
            )
            print(f"[probe] step={global_step} metrics={probe_metrics}")
            # Buffer into pending_probe_metrics; flushed in the main wandb.log
            # call below so we never call wandb.log twice at the same step.
            pending_probe_metrics.update(probe_metrics)
            if args.track:
                pending_probe_metrics.update(collect_probe_plot_images(probe_output))

            # Save a checkpoint at each probe point so the probe can be re-run
            # offline on any training snapshot without retraining.
            if args.save_checkpoint:
                import pickle
                ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
                os.makedirs(ckpt_dir, exist_ok=True)
                params_bytes = flax.serialization.to_bytes(agent_state.params)
                ckpt_path = os.path.join(ckpt_dir, f"params_step{global_step}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"params_bytes": params_bytes, "args": vars(args),
                                 "global_step": global_step}, f)
                print(f"Checkpoint saved to {ckpt_path}")

            next_probe_step += args.probe_interval

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
                else:
                    encoder_lr_current = args.encoder_lr
                    heads_adam_lr_current = args.heads_adam_lr

                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/cumulative_episodic_return": cumulative_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/encoder_lr": float(encoder_lr_current),
                    "charts/heads_adam_lr": float(heads_adam_lr_current),
                    "charts/heads_muon_lr": args.heads_muon_lr,
                    "charts/vit_qk_stiefel_lr": (
                        args.vit_qk_stiefel_lr if args.vit_qk_stiefel else 0.0
                    ),
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/ppo_only_loss": ppo_only_loss[-1, -1].item(),
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/grad_norm": grad_norm[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                    "charts/latent_dim": float(args.stiefel_latent_dim if args.encoder_type.lower() == "stiefel_cnn" else (args.latent_dim if args.use_latent_bottleneck else 0)),
                    "charts/latent_dyn_coef_active": float(latent_dyn_coef_current),
                    "charts/cnn_sigreg_coef_active": float(cnn_sigreg_coef_current),
                }
                if args.mim_jepa:
                    log_dict["losses/mim_jepa_loss"] = mim_loss[-1, -1].item()
                    log_dict["charts/mim_jepa_lambda"] = float(mim_lambda_current)
                    log_dict["charts/mim_jepa_ema_tau"] = float(args.mim_jepa_ema_tau)
                    log_dict["charts/mim_jepa_mask_type"] = 1.0
                if args.lejepa:
                    log_dict["losses/lejepa_loss"] = lejepa_loss[-1, -1].item()
                    log_dict["losses/lejepa_loss_raw"] = lejepa_loss_raw[-1, -1].item()
                    log_dict["losses/lejepa_sim_loss"] = lejepa_sim_loss[-1, -1].item()
                    log_dict["losses/lejepa_sigreg_loss"] = lejepa_sigreg_loss[-1, -1].item()
                    log_dict["charts/lejepa_lambda"] = float(args.lejepa_lambda)
                    log_dict["charts/lejepa_aux_coef_active"] = float(lejepa_aux_coef_current)
                    log_dict["grads/encoder_ppo_norm"] = encoder_ppo_norm[-1, -1].item()
                    log_dict["grads/encoder_lejepa_norm"] = encoder_lejepa_norm[-1, -1].item()
                    log_dict["grads/encoder_ppo_lejepa_cosine"] = encoder_grad_cosine[-1, -1].item()
                if args.cnn_sigreg_coef > 0.0:
                    log_dict["losses/cnn_sigreg_loss"] = latent_cnn_sigreg_loss[-1, -1].item()
                    log_dict["losses/cnn_sigreg_loss_raw"] = latent_cnn_sigreg_loss_raw[-1, -1].item()
                    log_dict["losses/cnn_sigreg_coef"] = float(cnn_sigreg_coef_current)
                if latent_path_is_active():
                    log_dict["losses/latent_dyn"] = latent_dyn_loss_raw[-1, -1].item()
                    log_dict["losses/latent_cov"] = latent_cov_loss[-1, -1].item()
                    log_dict["losses/latent_var"] = latent_var_loss[-1, -1].item()
                    log_dict["losses/latent_whiten"] = latent_whiten_loss_val[-1, -1].item()
                    log_dict["losses/latent_subspace"] = latent_subspace_loss[-1, -1].item()
                    if compute_latent_grad_alignment:
                        log_dict["grads/encoder_ppo_norm"] = encoder_ppo_norm[-1, -1].item()
                        log_dict["grads/encoder_latent_dyn_norm"] = encoder_latent_dyn_norm[-1, -1].item()
                        log_dict["grads/encoder_ppo_latent_dyn_cosine"] = (
                            encoder_latent_dyn_cosine[-1, -1].item()
                        )
                if args.encoder_type.lower() == "scott":
                    log_dict["charts/scott_num_patches"] = float(scott_num_patches)
                    log_dict["charts/scott_grid_size"] = float(scott_grid_size)
                if "latent_transition" in agent_state.params:
                    log_dict.update(latent_transition_matrix_metrics(agent_state.params["latent_transition"]))

                sample_obs = None
                hidden = None
                if latent_path_is_active() or args.debug_repr:
                    sample_obs = storage.obs[0, :256]
                    hidden = network.apply(agent_state.params["network"], sample_obs)

                if latent_path_is_active():
                    if args.encoder_type.lower() == "stiefel_cnn":
                        latent_z = hidden
                    else:
                        latent_z = bottleneck.apply(agent_state.params["bottleneck"], hidden)
                    latent_metrics = encoder_repr_metrics(latent_z, prefix="latent")
                    for k, v in latent_metrics.items():
                        log_dict[k] = float(v)

                # Debug metrics for encoder representations
                if args.debug_repr:
                    repr_metrics = encoder_repr_metrics(hidden)
                    for k, v in repr_metrics.items():
                        log_dict[k] = float(v)

                    if args.lejepa:
                        proj_hidden = lejepa_head.apply(agent_state.params["lejepa_head"], hidden)
                        proj_metrics = encoder_repr_metrics(proj_hidden)
                        for k, v in proj_metrics.items():
                            log_dict[f"lejepa/{k.split('/', 1)[1]}"] = float(v)

                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(v)

                # Flush any buffered probe metrics into the same wandb step.
                if pending_probe_metrics:
                    log_dict.update(pending_probe_metrics)
                    pending_probe_metrics = {}

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.save_checkpoint:
        import pickle
        ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        params_bytes = flax.serialization.to_bytes(agent_state.params)
        ckpt_path = os.path.join(ckpt_dir, "params.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params_bytes": params_bytes, "args": vars(args)}, f)
        print(f"Checkpoint saved to {ckpt_path}")

    if args.track:
        wandb.finish()
