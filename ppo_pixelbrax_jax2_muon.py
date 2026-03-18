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
from manifold_muon_optax import manifold_muon
from encoders import build_encoder
from sigreg import sigreg_loss, sigreg_loss_masked

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
    use_encoder_final_muon: bool = False
    """if True, use manifold MUON for the encoder's final Dense kernel only"""
    encoder_muon_lr: float = 0.02
    """learning rate for the encoder's final Dense kernel when using MUON"""
    encoder_muon_max_grad_norm: float = 1.0
    """maximum norm for gradient clipping on the encoder final Dense MUON branch"""
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
    """encoder architecture to use: 'cnn', 'sigreg_cnn', 'innovation_cnn', or 'mlp'"""
    encoder_tanh_scale: float = 0.5
    """Multiplier for encoder output before tanh (controls saturation)"""
    encoder_warmup_updates: int = 0
    """Warmup updates for encoder LR when annealing is enabled (0 disables warmup)."""
    encoder_use_crate_block: bool = False
    """If true, insert a CRATE block inside the encoder before the SIGReg projector."""
    encoder_crate_step_size: float = 0.1
    """Step size for the encoder CRATE block when enabled."""

    # SIGReg regularization
    sigreg_mode: str = "off"
    """SIGReg mode to use: 'off' or 'projected'."""
    sigreg_coef: float = 0.0
    """Maximum coefficient for the SIGReg auxiliary loss."""
    sigreg_proj_dim: int = 64
    """Projected subspace dimension used by the SIGReg CNN encoder."""
    sigreg_num_slices: int = 16
    """Number of random projections used by SIGReg."""
    sigreg_num_t: int = 8
    """Number of evaluation points used by SIGReg."""
    sigreg_t_max: float = 5.0
    """Maximum absolute t value for the SIGReg characteristic-function grid."""
    sigreg_warmup_updates: int = 0
    """Number of PPO updates before enabling SIGReg."""
    sigreg_ramp_updates: int = 0
    """Number of PPO updates used to ramp SIGReg to its full coefficient."""

    # Innovation-aware auxiliary regularization
    innovation_coef: float = 0.0
    """Maximum coefficient for the innovation-aware auxiliary loss."""
    innovation_proj_dim: int = 64
    """Projected subspace dimension used by the innovation-aware CNN encoder."""
    innovation_num_slices: int = 16
    """Number of random projections used by the innovation-aware residual SIGReg."""
    innovation_num_t: int = 8
    """Number of evaluation points used by the innovation-aware residual SIGReg."""
    innovation_t_max: float = 5.0
    """Maximum absolute t value for the innovation-aware characteristic-function grid."""
    innovation_warmup_updates: int = 0
    """Number of PPO updates before enabling the innovation-aware auxiliary loss."""
    innovation_ramp_updates: int = 0
    """Number of PPO updates used to ramp innovation-aware regularization to its full coefficient."""

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


class InnovationDynamics(nn.Module):
    """Linear latent dynamics model for innovation-aware auxiliary loss."""
    action_dim: int
    proj_dim: int

    @nn.compact
    def __call__(self, u_t, action):
        state_term = nn.Dense(
            self.proj_dim,
            use_bias=False,
            kernel_init=orthogonal(1.0),
            name="state_transition",
        )(u_t)
        action_term = nn.Dense(
            self.proj_dim,
            use_bias=False,
            kernel_init=orthogonal(1.0),
            name="action_transition",
        )(action)
        return state_term + action_term


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    next_obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    next_dones: jnp.array
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


def cnn_dense_metrics(
    dense_pre_ln: jnp.ndarray,
    hidden: jnp.ndarray,
    network_params: dict,
    network_grads: dict,
) -> dict:
    """Compute scalar debug metrics for the CNN encoder's final Dense layer."""
    eps = 1e-8
    metrics = {}

    metrics["cnn_dense/pre_ln_mean"] = jnp.mean(dense_pre_ln)
    metrics["cnn_dense/pre_ln_std"] = jnp.std(dense_pre_ln)
    metrics["cnn_dense/pre_ln_abs_mean"] = jnp.mean(jnp.abs(dense_pre_ln))
    metrics["cnn_dense/pre_ln_max_abs"] = jnp.max(jnp.abs(dense_pre_ln))
    metrics["cnn_dense/pre_ln_l2_mean"] = jnp.mean(jnp.linalg.norm(dense_pre_ln, axis=-1))

    metrics["cnn_dense/tanh_sat_frac_0.95"] = jnp.mean(jnp.abs(hidden) > 0.95)
    metrics["cnn_dense/tanh_sat_frac_0.99"] = jnp.mean(jnp.abs(hidden) > 0.99)

    hidden_centered = hidden - jnp.mean(hidden, axis=0, keepdims=True)
    feature_var = jnp.var(hidden_centered, axis=0)
    metrics["cnn_dense/feature_var_min"] = jnp.min(feature_var)
    metrics["cnn_dense/feature_var_max"] = jnp.max(feature_var)
    metrics["cnn_dense/feature_var_ratio"] = jnp.max(feature_var) / (jnp.min(feature_var) + eps)

    batch_denom = float(max(hidden.shape[0] - 1, 1))
    cov = (hidden_centered.T @ hidden_centered) / batch_denom
    feature_std = jnp.sqrt(feature_var + eps)
    corr = cov / (feature_std[:, None] * feature_std[None, :] + eps)
    corr_mask = 1.0 - jnp.eye(corr.shape[0], dtype=corr.dtype)
    metrics["cnn_dense/mean_abs_corr"] = (
        jnp.sum(jnp.abs(corr) * corr_mask) / (jnp.sum(corr_mask) + eps)
    )

    eigvals = jnp.clip(jnp.linalg.eigvalsh(cov), a_min=0.0)
    eig_sum = jnp.sum(eigvals)
    eig_sq_sum = jnp.sum(jnp.square(eigvals))
    top_eig = jnp.max(eigvals)
    metrics["cnn_dense/participation_ratio"] = (eig_sum ** 2) / (eig_sq_sum + eps)
    metrics["cnn_dense/top_eig_fraction"] = top_eig / (eig_sum + eps)
    metrics["cnn_dense/stable_rank"] = eig_sum / (top_eig + eps)

    dense_kernel = network_params["params"]["Dense_0"]["kernel"]
    dense_kernel_grad = network_grads["params"]["Dense_0"]["kernel"]
    singular_values = jnp.linalg.svd(dense_kernel, compute_uv=False)
    sigma_max = jnp.max(singular_values)
    metrics["cnn_dense/weight_mean"] = jnp.mean(dense_kernel)
    metrics["cnn_dense/weight_std"] = jnp.std(dense_kernel)
    metrics["cnn_dense/weight_spectral_norm"] = sigma_max
    metrics["cnn_dense/weight_stable_rank"] = (
        jnp.sum(jnp.square(singular_values)) / (jnp.square(sigma_max) + eps)
    )
    metrics["cnn_dense/weight_grad_norm"] = jnp.linalg.norm(dense_kernel_grad)

    return metrics


def projected_latent_metrics(projected: jnp.ndarray, prefix: str = "sigreg_proj") -> dict:
    """Compute simple anisotropy metrics on an auxiliary projected latent."""
    eps = 1e-8
    metrics = {}
    centered = projected - jnp.mean(projected, axis=0, keepdims=True)
    feature_var = jnp.var(centered, axis=0)
    cov = (centered.T @ centered) / float(max(projected.shape[0] - 1, 1))
    feature_std = jnp.sqrt(feature_var + eps)
    corr = cov / (feature_std[:, None] * feature_std[None, :] + eps)
    corr_mask = 1.0 - jnp.eye(corr.shape[0], dtype=corr.dtype)
    eigvals = jnp.clip(jnp.linalg.eigvalsh(cov), a_min=0.0)
    eig_sum = jnp.sum(eigvals)
    eig_sq_sum = jnp.sum(jnp.square(eigvals))
    top_eig = jnp.max(eigvals)

    metrics[f"{prefix}/mean_abs_corr"] = (
        jnp.sum(jnp.abs(corr) * corr_mask) / (jnp.sum(corr_mask) + eps)
    )
    metrics[f"{prefix}/top_eig_fraction"] = top_eig / (eig_sum + eps)
    metrics[f"{prefix}/participation_ratio"] = (eig_sum ** 2) / (eig_sq_sum + eps)
    metrics[f"{prefix}/stable_rank"] = eig_sum / (top_eig + eps)
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
    encoder_muon_lr: float = 0.02,
    encoder_muon_max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    use_heads_muon: bool = True,
    use_encoder_final_muon: bool = False,
):
    """
    Create optimizer that uses:
    - Adam/AdamW for encoder (all params except optional final Dense kernel MUON branch) - supports lr schedule, with grad clipping
    - Manifold MUON for the encoder final Dense kernel only when enabled
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

    # Adam/AdamW for head vectors/scalars (with schedule support and grad clipping)
    heads_adam_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(heads_adam_lr),
    )

    # MUON for encoder final Dense kernel only (with separate grad clipping)
    encoder_final_muon_tx = optax.chain(
        optax.clip_by_global_norm(encoder_muon_max_grad_norm),
        manifold_muon(
            learning_rate=encoder_muon_lr,
            dual_lr=muon_dual_lr,
            dual_steps=muon_dual_steps,
            msign_steps=muon_msign_steps,
            min_ndim=2,
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

    # 4 transforms: encoder, actor_muon (matrices),
    # critic_muon (matrices), heads_adam (actor/critic vectors/scalars)
    transforms = {
        'encoder': encoder_tx,
        'encoder_final_muon': encoder_final_muon_tx,
        'actor_muon': actor_muon_tx,
        'critic_muon': critic_muon_tx,
        'heads_adam': heads_adam_tx,
    }

    # Label function
    def label_fn(params):
        def _label(path, param):
            # path[0] is a top-level module key in params
            if path[0] == 'network':
                is_encoder_final_dense_kernel = path == ('network', 'params', 'Dense_0', 'kernel')
                if use_encoder_final_muon and is_encoder_final_dense_kernel:
                    return 'encoder_final_muon'
                return 'encoder'
            if path[0] == 'innovation':
                return 'encoder'
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
    key, network_key, innovation_key, actor_key, critic_key = jax.random.split(key, 5)

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
    print(f"  use_encoder_final_muon: {args.use_encoder_final_muon}")
    print(f"  encoder_muon_lr: {args.encoder_muon_lr}")
    print(f"  encoder_muon_max_grad_norm: {args.encoder_muon_max_grad_norm}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  muon_dual_lr: {args.muon_dual_lr}")
    print(f"  muon_dual_steps: {args.muon_dual_steps}")
    print(f"  max_grad_norm ({adam_type}): {args.max_grad_norm}")
    print(f"  actor_muon_max_grad_norm: {args.actor_muon_max_grad_norm}")
    print(f"  critic_muon_max_grad_norm: {args.critic_muon_max_grad_norm}")
    print(f"  encoder_type: {args.encoder_type}")
    print(f"  encoder_tanh_scale: {args.encoder_tanh_scale}")
    print(f"  encoder_warmup_updates: {args.encoder_warmup_updates}")
    print(f"  encoder_use_crate_block: {args.encoder_use_crate_block}")
    print(f"  encoder_crate_step_size: {args.encoder_crate_step_size}")
    print(f"  sigreg_mode: {args.sigreg_mode}")
    print(f"  sigreg_coef: {args.sigreg_coef}")
    print(f"  sigreg_proj_dim: {args.sigreg_proj_dim}")
    print(f"  sigreg_num_slices: {args.sigreg_num_slices}")
    print(f"  sigreg_num_t: {args.sigreg_num_t}")
    print(f"  sigreg_t_max: {args.sigreg_t_max}")
    print(f"  sigreg_warmup_updates: {args.sigreg_warmup_updates}")
    print(f"  sigreg_ramp_updates: {args.sigreg_ramp_updates}")
    print(f"  innovation_coef: {args.innovation_coef}")
    print(f"  innovation_proj_dim: {args.innovation_proj_dim}")
    print(f"  innovation_num_slices: {args.innovation_num_slices}")
    print(f"  innovation_num_t: {args.innovation_num_t}")
    print(f"  innovation_t_max: {args.innovation_t_max}")
    print(f"  innovation_warmup_updates: {args.innovation_warmup_updates}")
    print(f"  innovation_ramp_updates: {args.innovation_ramp_updates}")
    print(f"  anneal_lr: {args.anneal_lr}")
    print("=" * 60)

    encoder_type = args.encoder_type.lower()
    if encoder_type not in {"cnn", "sigreg_cnn", "innovation_cnn", "mlp"}:
        raise ValueError(
            f"Unsupported encoder_type='{args.encoder_type}'. Expected one of: ['cnn', 'sigreg_cnn', 'innovation_cnn', 'mlp']"
        )

    sigreg_mode = args.sigreg_mode.lower()
    if sigreg_mode not in {"off", "projected"}:
        raise ValueError(
            f"Unsupported sigreg_mode='{args.sigreg_mode}'. Expected one of: ['off', 'projected']"
        )
    if sigreg_mode == "projected" and encoder_type != "sigreg_cnn":
        raise ValueError("Projected SIGReg requires encoder_type='sigreg_cnn'.")
    if args.innovation_coef > 0 and encoder_type != "innovation_cnn":
        raise ValueError("Innovation-aware regularization requires encoder_type='innovation_cnn'.")

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
    network = build_encoder(
        encoder_type,
        args.encoder_tanh_scale,
        sigreg_proj_dim=args.sigreg_proj_dim,
        innovation_proj_dim=args.innovation_proj_dim,
        use_crate_block=args.encoder_use_crate_block,
        crate_step_size=args.encoder_crate_step_size,
    )
    if args.use_crate_head:
        actor = CRATEActor(action_dim=action_dim, crate_step_size=args.crate_step_size)
        critic = CRATECritic(crate_step_size=args.crate_step_size)
        print(f"Using CRATE heads with step_size={args.crate_step_size}")
    else:
        actor = Actor(action_dim=action_dim)
        critic = Critic()

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    innovation_model = None
    dummy_hidden = network.apply(network_params, dummy_obs)

    params_dict = {
        "network": network_params,
    }
    if encoder_type == "innovation_cnn":
        innovation_model = InnovationDynamics(
            action_dim=action_dim,
            proj_dim=args.innovation_proj_dim,
        )
        dummy_debug = network.apply(network_params, dummy_obs, return_intermediates=True)
        params_dict["innovation"] = innovation_model.init(
            innovation_key,
            dummy_debug["u"],
            jnp.zeros((1, action_dim), dtype=jnp.float32),
        )
    params_dict["actor"] = actor.init(actor_key, dummy_hidden)
    params_dict["critic"] = critic.init(critic_key, dummy_hidden)

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
    innovation_params = count_params(all_params, "innovation")
    actor_params = count_params(all_params, "actor")
    critic_params = count_params(all_params, "critic")
    total_params = encoder_params + innovation_params + actor_params + critic_params

    actor_muon, actor_adam = count_by_type(all_params, "actor")
    critic_muon, critic_adam = count_by_type(all_params, "critic")
    encoder_final_muon_params = 0
    if args.use_encoder_final_muon:
        try:
            encoder_final_muon_params = all_params["network"]["params"]["Dense_0"]["kernel"].size
        except KeyError:
            encoder_final_muon_params = 0
    encoder_adam_params = encoder_params - encoder_final_muon_params

    print(f"\nParameter breakdown:")
    print(
        f"  Encoder total: {encoder_params:,} "
        f"(MUON final dense: {encoder_final_muon_params:,}, Adam rest: {encoder_adam_params:,})"
    )
    if innovation_params > 0:
        print(f"  Innovation dynamics (Adam): {innovation_params:,}")
    print(f"  Actor total: {actor_params:,} (MUON: {actor_muon:,}, Adam: {actor_adam:,})")
    print(f"  Critic total: {critic_params:,} (MUON: {critic_muon:,}, Adam: {critic_adam:,})")
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


    def sigreg_coef_from_update(
        base_coef: float,
        update_idx: int,
        warmup_updates: int = 0,
        ramp_updates: int = 0,
    ):
        """Warmup + linear ramp helper for the SIGReg coefficient."""
        if base_coef <= 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        update_idx = jnp.asarray(update_idx, dtype=jnp.float32)
        warmup = float(max(0, warmup_updates))
        ramp = float(max(0, ramp_updates))
        if ramp_updates <= 0:
            return jnp.where(update_idx < warmup, 0.0, base_coef)
        ramp_progress = jnp.clip((update_idx - warmup + 1.0) / jnp.maximum(1.0, ramp), 0.0, 1.0)
        return jnp.where(update_idx < warmup, 0.0, base_coef * ramp_progress)

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
        encoder_muon_lr=args.encoder_muon_lr,
        encoder_muon_max_grad_norm=args.encoder_muon_max_grad_norm,
        weight_decay=args.weight_decay,
        use_heads_muon=args.use_heads_muon,
        use_encoder_final_muon=args.use_encoder_final_muon,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    network_debug_apply = jax.jit(network.apply, static_argnames=("return_intermediates",))
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    if innovation_model is not None:
        innovation_model.apply = jax.jit(innovation_model.apply)


    def encode_with_intermediates(network_params, obs):
        if encoder_type not in {"cnn", "sigreg_cnn", "innovation_cnn"}:
            raise ValueError(f"return_intermediates is not supported for encoder_type={encoder_type}")
        return network_debug_apply(network_params, obs, return_intermediates=True)

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

    

    def ppo_loss(
        params,
        x,
        next_x,
        next_done,
        a,
        logp,
        mb_advantages,
        mb_returns,
        mb_values,
        aug_key,
        sigreg_coef_current,
        innovation_coef_current,
    ):
        """PPO loss function."""
        if args.use_augmentation:
            aug_key, next_aug_key, aux_key = jax.random.split(aug_key, 3)
            x = random_shift(aug_key, x, pad=args.augment_pad)
            next_x = random_shift(next_aug_key, next_x, pad=args.augment_pad)
        else:
            aux_key = aug_key

        sigreg_total = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_re = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_im = jnp.asarray(0.0, dtype=jnp.float32)
        innovation_total = jnp.asarray(0.0, dtype=jnp.float32)
        innovation_re = jnp.asarray(0.0, dtype=jnp.float32)
        innovation_im = jnp.asarray(0.0, dtype=jnp.float32)

        if encoder_type == "sigreg_cnn":
            encoder_debug = encode_with_intermediates(params["network"], x)
            hidden = encoder_debug["hidden"]
            if sigreg_mode == "projected":
                sigreg_total, sigreg_re, sigreg_im = sigreg_loss(
                    encoder_debug["u"],
                    aux_key,
                    num_slices=args.sigreg_num_slices,
                    num_t=args.sigreg_num_t,
                    t_max=args.sigreg_t_max,
                )
        elif encoder_type == "innovation_cnn":
            encoder_debug = encode_with_intermediates(params["network"], x)
            hidden = encoder_debug["hidden"]
            aux_key, innovation_key = jax.random.split(aux_key)
            next_encoder_debug = encode_with_intermediates(params["network"], next_x)
            predicted_next_u = innovation_model.apply(
                params["innovation"],
                encoder_debug["u"],
                a,
            )
            residual = next_encoder_debug["u"] - predicted_next_u
            innovation_mask = 1.0 - next_done.astype(jnp.float32)
            innovation_total, innovation_re, innovation_im = sigreg_loss_masked(
                residual,
                innovation_mask,
                innovation_key,
                num_slices=args.innovation_num_slices,
                num_t=args.innovation_num_t,
                t_max=args.innovation_t_max,
            )
        else:
            hidden = network.apply(params["network"], x)

        actor_mean, actor_logstd = actor.apply(params["actor"], hidden)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        newlogprob = pi.log_prob(a)
        entropy = pi.entropy()
        newvalue = critic.apply(params["critic"], hidden).squeeze(-1)

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
        total_loss = (
            pg_loss
            - args.ent_coef * entropy_loss
            + v_loss * args.vf_coef
            + sigreg_coef_current * sigreg_total
            + innovation_coef_current * innovation_total
        )

        return total_loss, (
            pg_loss,
            v_loss,
            entropy_loss,
            jax.lax.stop_gradient(approx_kl),
            jax.lax.stop_gradient(sigreg_total),
            jax.lax.stop_gradient(sigreg_re),
            jax.lax.stop_gradient(sigreg_im),
            jax.lax.stop_gradient(innovation_total),
            jax.lax.stop_gradient(innovation_re),
            jax.lax.stop_gradient(innovation_im),
        )

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
        sigreg_coef_current: jnp.ndarray,
        innovation_coef_current: jnp.ndarray,
    ):
        """PPO update."""
        def update_epoch(carry, unused_inp):
            agent_state, key, last_grads = carry
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
                agent_state, _ = carry
                minibatch, mb_aug_key = inputs

                (
                    loss,
                    (
                        pg_loss,
                        v_loss,
                        entropy_loss,
                        approx_kl,
                        sigreg_total,
                        sigreg_re,
                        sigreg_im,
                        innovation_total,
                        innovation_re,
                        innovation_im,
                    ),
                ), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.next_obs,
                    minibatch.next_dones,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                    sigreg_coef_current,
                    innovation_coef_current,
                )

                grad_norm = optax.global_norm(grads)
                agent_state = agent_state.apply_gradients(grads=grads)

                return (agent_state, grads), (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    sigreg_total,
                    sigreg_re,
                    sigreg_im,
                    innovation_total,
                    innovation_re,
                    innovation_im,
                    grad_norm,
                )

            (
                (agent_state, last_grads),
                (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    sigreg_total,
                    sigreg_re,
                    sigreg_im,
                    innovation_total,
                    innovation_re,
                    innovation_im,
                    grad_norm,
                ),
            ) = jax.lax.scan(
                update_minibatch,
                (agent_state, last_grads),
                (shuffled_storage, aug_keys),
            )
            return (agent_state, key, last_grads), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                sigreg_total,
                sigreg_re,
                sigreg_im,
                innovation_total,
                innovation_re,
                innovation_im,
                grad_norm,
            )

        init_grads = jax.tree_util.tree_map(jnp.zeros_like, agent_state.params)
        (
            (agent_state, key, final_grads),
            (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                sigreg_total,
                sigreg_re,
                sigreg_im,
                innovation_total,
                innovation_re,
                innovation_im,
                grad_norm,
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
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            sigreg_total,
            sigreg_re,
            sigreg_im,
            innovation_total,
            innovation_re,
            innovation_im,
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
            next_obs=next_obs_local,
            actions=action,
            logprobs=logprob,
            dones=done,
            next_dones=next_done_local,
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


        sigreg_coef_current = sigreg_coef_from_update(
            args.sigreg_coef,
            iteration - 1,
            args.sigreg_warmup_updates,
            args.sigreg_ramp_updates,
        )
        innovation_coef_current = sigreg_coef_from_update(
            args.innovation_coef,
            iteration - 1,
            args.innovation_warmup_updates,
            args.innovation_ramp_updates,
        )
        (
            agent_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            sigreg_total,
            sigreg_re,
            sigreg_im,
            innovation_total,
            innovation_re,
            innovation_im,
            grad_norm,
            final_grads,
            key,
        ) = update_ppo(
            agent_state,
            storage,
            key,
            sigreg_coef_current,
            innovation_coef_current,
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
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/grad_norm": grad_norm[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                    "losses/sigreg_total": sigreg_total[-1, -1].item(),
                    "losses/sigreg_re": sigreg_re[-1, -1].item(),
                    "losses/sigreg_im": sigreg_im[-1, -1].item(),
                    "losses/sigreg_coef": float(sigreg_coef_current),
                    "losses/innovation_total": innovation_total[-1, -1].item(),
                    "losses/innovation_re": innovation_re[-1, -1].item(),
                    "losses/innovation_im": innovation_im[-1, -1].item(),
                    "losses/innovation_coef": float(innovation_coef_current),
                }

                # Debug metrics for encoder representations
                if args.debug_repr:
                    sample_obs = storage.obs[0, :256]
                    if encoder_type in {"cnn", "sigreg_cnn", "innovation_cnn"}:
                        cnn_debug = encode_with_intermediates(
                            agent_state.params["network"],
                            sample_obs,
                        )
                        hidden = cnn_debug["hidden"]
                        dense_metrics = cnn_dense_metrics(
                            cnn_debug["dense_pre_ln"],
                            hidden,
                            agent_state.params["network"],
                            final_grads["network"],
                        )
                        for k, v in dense_metrics.items():
                            log_dict[k] = float(v)
                        if encoder_type == "sigreg_cnn":
                            proj_metrics = projected_latent_metrics(cnn_debug["u"], prefix="sigreg_proj")
                            for k, v in proj_metrics.items():
                                log_dict[k] = float(v)
                        if encoder_type == "innovation_cnn":
                            proj_metrics = projected_latent_metrics(cnn_debug["u"], prefix="innovation_proj")
                            for k, v in proj_metrics.items():
                                log_dict[k] = float(v)
                    else:
                        hidden = network.apply(agent_state.params["network"], sample_obs)

                    repr_metrics = encoder_repr_metrics(hidden)
                    for k, v in repr_metrics.items():
                        log_dict[k] = float(v)

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
