#!/usr/bin/env python
"""
PPO for PixelBrax environments with JEPA (Joint-Embedding Predictive Architecture) auxiliary objective.
Adapted from CleanRL's PPO Atari implementation for continuous control with pixel observations.

JEPA auxiliary objective: Train the CNN encoder to predict the target encoder's embedding
of an unmasked frame from the context encoder's embedding of a masked frame.
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
    learning_rate: float = 3e-5
    """the learning rate of the optimizer"""
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
    """the maximum norm for the gradient clipping"""
    max_action: float = 1.0
    """maximum action value for clipping"""
    log_interval: int = 1
    """logging interval (in updates)"""

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

    # JEPA auxiliary objective
    jepa_mode: str = "jepa"
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


class Network(nn.Module):
    """CNN encoder for pixel observations with LayerNorm for stability."""

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) - already in NHWC format from PixelBrax
        x = x.astype(jnp.float32) / 255.0

        # Conv layers with LayerNorm for stable training
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
        x = nn.tanh(x)  # tanh for bounded features, helps with stability
        return x


class Critic(nn.Module):
    """Value network with 2 hidden layers for sufficient capacity."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Continuous action actor with 2 hidden layers using Gaussian distribution."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
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
    """Predictor head for JEPA: maps context embedding to predicted target embedding."""
    hidden_dim: int = 512
    output_dim: int = 512  # Match Network output

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


# JEPA Helper Functions

def ema_update(target_params, online_params, tau):
    """EMA update: target = tau * target + (1 - tau) * online"""
    return jax.tree_util.tree_map(
        lambda t, o: t * tau + o * (1.0 - tau),
        target_params, online_params,
    )


def l2_normalize(x, axis=-1, eps=1e-8):
    """L2 normalize along specified axis."""
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
    """
    SIGReg via characteristic-function matching to N(0,1) under random 1D projections.

    Args:
        z: (B, D) projected embeddings.
        key: random key for projection sampling.
        num_slices: number of random projection directions.
        t_points: number of grid points in [t_min, t_max].
        t_min/t_max: integration range for CF discrepancy.
        axis_name: optional pmap axis name for cross-device averaging.
    """
    feat_dim = z.shape[-1]
    a = jax.random.normal(key, (feat_dim, num_slices))
    a = a / (jnp.linalg.norm(a, axis=0, keepdims=True) + 1e-8)

    # y: (B, S) where S=num_slices
    y = z @ a
    t = jnp.linspace(t_min, t_max, t_points)
    yt = y[:, :, None] * t[None, None, :]  # (B, S, T)

    ecf_real = jnp.mean(jnp.cos(yt), axis=0)  # (S, T)
    ecf_imag = jnp.mean(jnp.sin(yt), axis=0)  # (S, T)
    if axis_name is not None:
        ecf_real = jax.lax.pmean(ecf_real, axis_name=axis_name)
        ecf_imag = jax.lax.pmean(ecf_imag, axis_name=axis_name)

    target_real = jnp.exp(-0.5 * (t ** 2))[None, :]  # (1, T)
    sq_err = (ecf_real - target_real) ** 2 + ecf_imag ** 2  # (S, T)

    dt = (t_max - t_min) / jnp.maximum(t_points - 1, 1)
    trapz = 0.5 * dt * (sq_err[:, :-1] + sq_err[:, 1:]).sum(axis=-1)  # (S,)
    return trapz.mean()


def get_jepa_lambda(update_step, target_lambda, warmup_updates, rampup_updates):
    """
    Warmup then linear rampup schedule for JEPA loss coefficient.
    JAX version for potential use inside JIT (uses jax.lax.cond).

    Args:
        update_step: Current update step (1-indexed)
        target_lambda: Target JEPA lambda value
        warmup_updates: Number of updates before JEPA loss is active
        rampup_updates: Number of updates to linearly ramp JEPA lambda

    Returns:
        Current JEPA lambda value
    """
    return jax.lax.cond(
        update_step < warmup_updates,
        lambda: 0.0,
        lambda: jax.lax.cond(
            update_step < warmup_updates + rampup_updates,
            lambda: target_lambda * (update_step - warmup_updates) / rampup_updates,
            lambda: target_lambda,
        ),
    )


def get_jepa_lambda_py(update_step, target_lambda, warmup_updates, rampup_updates):
    """
    Pure Python schedule for JEPA loss coefficient.
    Use this for host-side computations (logging, passing to JIT).

    Args:
        update_step: Current update step (1-indexed)
        target_lambda: Target JEPA lambda value
        warmup_updates: Number of updates before JEPA loss is active
        rampup_updates: Number of updates to linearly ramp JEPA lambda

    Returns:
        Current JEPA lambda value (float)
    """
    if update_step < warmup_updates:
        return 0.0
    elif update_step < warmup_updates + rampup_updates:
        return target_lambda * (update_step - warmup_updates) / rampup_updates
    return target_lambda


# Using FrozenDict for params instead of a dataclass for Flax compatibility
# Params structure: {'network': ..., 'actor': ..., 'critic': ..., 'predictor': ...}


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
    """
    Frame stacking buffer for temporal information in pixel-based RL.
    Stores the last N frames and provides stacked observations.
    """
    frames: jnp.ndarray  # Shape: (n_envs, num_frames, H, W, C)

    @classmethod
    def create(cls, n_envs: int, num_frames: int, obs_shape: tuple):
        """Initialize frame stack with zeros."""
        h, w, c = obs_shape
        frames = jnp.zeros((n_envs, num_frames, h, w, c), dtype=jnp.uint8)
        return cls(frames=frames)

    def reset(self, obs: jnp.ndarray, done: jnp.ndarray = None):
        """
        Reset frame stack for environments that are done.
        If done is None, reset all environments with the given observation.

        Args:
            obs: New observation of shape (n_envs, H, W, C)
            done: Boolean mask of shape (n_envs,) indicating which envs to reset
        """
        # Stack the same observation num_frames times for reset
        n_envs = obs.shape[0]
        num_frames = self.frames.shape[1]
        new_frames = jnp.broadcast_to(
            obs[:, None, :, :, :],
            (n_envs, num_frames, obs.shape[1], obs.shape[2], obs.shape[3])
        )

        if done is None:
            return self.replace(frames=new_frames)
        else:
            # Only reset environments that are done
            frames = jnp.where(
                done[:, None, None, None, None],
                new_frames,
                self.frames
            )
            return self.replace(frames=frames)

    def push(self, obs: jnp.ndarray):
        """
        Add a new frame to the stack, shifting out the oldest.

        Args:
            obs: New observation of shape (n_envs, H, W, C)

        Returns:
            Updated FrameStack
        """
        # Shift frames left (drop oldest) and add new frame at the end
        new_frames = jnp.concatenate([
            self.frames[:, 1:, :, :, :],
            obs[:, None, :, :, :]
        ], axis=1)
        return self.replace(frames=new_frames)

    def get_stacked(self) -> jnp.ndarray:
        """
        Get stacked observation by concatenating frames along channel dimension.

        Returns:
            Stacked observation of shape (n_envs, H, W, C * num_frames)
        """
        # Reshape from (n_envs, num_frames, H, W, C) to (n_envs, H, W, C * num_frames)
        n_envs, num_frames, h, w, c = self.frames.shape
        # Transpose to (n_envs, H, W, num_frames, C) then reshape
        frames_transposed = jnp.transpose(self.frames, (0, 2, 3, 1, 4))
        return frames_transposed.reshape(n_envs, h, w, c * num_frames)


@flax.struct.dataclass
class RewardNormalizer:
    """
    Reward normalization using discounted returns (CleanRL style).
    Normalizes rewards by the standard deviation of discounted returns,
    which is more stable for continuous control.
    """
    return_rms_mean: jnp.array  # Running mean of returns (unused but tracked)
    return_rms_var: jnp.array   # Running variance of returns
    return_rms_count: jnp.array
    discounted_return: jnp.array  # Track discounted returns per environment
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
        """
        Update running statistics with a batch of rewards.
        Uses discounted returns for variance estimation (like gym.wrappers.NormalizeReward).
        """
        # Update discounted returns: R_t = r_t + gamma * R_{t-1} * (1 - done)
        new_discounted_return = rewards + self.gamma * self.discounted_return * (1.0 - dones)

        # Update running statistics of returns
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

        # Reset discounted return where episodes ended
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
    """
    DrQ-style random shift augmentation.
    Pads the image and then takes a random crop back to original size.

    Args:
        key: JAX random key
        x: Image tensor of shape (B, H, W, C)
        pad: Padding size (default 4 pixels)

    Returns:
        Augmented image tensor of same shape
    """
    b, h, w, c = x.shape

    # Pad the image with edge values
    x_padded = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Generate random crop offsets for each image in batch
    key1, key2 = jax.random.split(key)
    crop_h = jax.random.randint(key1, (b,), 0, 2 * pad + 1)
    crop_w = jax.random.randint(key2, (b,), 0, 2 * pad + 1)

    # Use vmap to apply random crops to each image
    def crop_single(x_pad, ch, cw):
        return jax.lax.dynamic_slice(x_pad, (ch, cw, 0), (h, w, c))

    return jax.vmap(crop_single)(x_padded, crop_h, crop_w)


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
    Sample tube-cutout mask shared across all stacked channels.

    Returns:
        Boolean mask of shape (B, H, W, 1), where True means "masked out".
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
    args.exp_name = args.jepa_mode
    args.use_jepa = (args.jepa_mode == "jepa")
    args.use_le_jepa = (args.jepa_mode == "le_jepa")
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    # Mark Slurm array runs explicitly so W&B names show sweep context.
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_array_task_id is not None:
        run_name = f"{args.env_name}__{args.exp_name}__sweep__a{slurm_array_task_id}__{args.seed}__{int(time.time())}"
    else:
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
    key, network_key, actor_key, critic_key, predictor_key, projector_key = jax.random.split(key, 6)

    # Environment setup
    print("JAX devices:", jax.devices())
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")
    print(f"jepa_mode: {args.jepa_mode}")
    if args.use_jepa:
        print(f"jepa_lambda: {args.jepa_lambda}")
        print(f"jepa_ema_tau: {args.jepa_ema_tau}")
        print(f"jepa_warmup_updates: {args.jepa_warmup_updates}")
        print(f"jepa_rampup_updates: {args.jepa_rampup_updates}")
    if args.use_le_jepa:
        print(f"jepa_lambda: {args.jepa_lambda}")
        print(f"jepa_ema_tau: {args.jepa_ema_tau}")
        print(f"jepa_warmup_updates: {args.jepa_warmup_updates}")
        print(f"jepa_rampup_updates: {args.jepa_rampup_updates}")
        print(f"proj_dim: {args.proj_dim}")
        print(f"sigreg_weight: {args.sigreg_weight}")
        print(f"sigreg_num_slices: {args.sigreg_num_slices}")
        print(f"sigreg_t_points: {args.sigreg_t_points}")
        print(f"sigreg_t_min: {args.sigreg_t_min}")
        print(f"sigreg_t_max: {args.sigreg_t_max}")
        print(f"lejepa_use_target_ema: {args.lejepa_use_target_ema}")

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset(reset_rng)
    raw_obs_shape = init_env_state.pixels.shape[1:]  # (H, W, C)
    # Stacked observation shape: channels are multiplied by frame_stack
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

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    # Initialize networks
    network = Network()
    actor = Actor(action_dim=action_dim)
    critic = Critic()
    projector = Projector(proj_dim=args.proj_dim)
    predictor_output_dim = args.proj_dim if args.use_le_jepa else 512
    predictor = JEPAPredictor(hidden_dim=args.jepa_predictor_hidden, output_dim=predictor_output_dim)

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)

    # Initialize predictor with the relevant input space.
    if args.use_le_jepa:
        projector_params = projector.init(projector_key, dummy_hidden)
        dummy_proj = projector.apply(projector_params, dummy_hidden)
        predictor_params = predictor.init(predictor_key, dummy_proj)
    else:
        projector_params = None
        predictor_params = predictor.init(predictor_key, dummy_hidden)

    params_dict = {
        'network': network_params,
        'actor': actor.init(actor_key, dummy_hidden),
        'critic': critic.init(critic_key, dummy_hidden),
        'predictor': predictor_params,
    }
    if args.use_le_jepa:
        params_dict['projector'] = projector_params

    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze(params_dict),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                eps=1e-5
            ),
        ),
    )

    # Target params - separate from TrainState (no optimizer)
    # Initialize with online params
    if args.use_le_jepa:
        target_params = flax.core.freeze({
            'network': agent_state.params['network'],
            'projector': agent_state.params['projector'],
        })
    else:
        target_params = agent_state.params['network']

    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
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
        hidden = network.apply(agent_state.params['network'], next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params['actor'], hidden)

        # Create Gaussian distribution
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)

        value = critic.apply(agent_state.params['critic'], hidden).squeeze(-1)

        # Clip action
        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, logprob, value, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        hidden = network.apply(params['network'], x)
        actor_mean, actor_logstd = actor.apply(params['actor'], hidden)

        # Create Gaussian distribution
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
        hidden = network.apply(agent_state.params['network'], next_obs)
        next_value = critic.apply(
            agent_state.params['critic'], hidden
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

    def compute_jepa_loss(params, target_network_params, obs, jepa_key):
        """
        Compute JEPA loss for a minibatch.

        Masks out one random frame from the stacked frames for each sample.

        Args:
            params: Online network params (network + predictor)
            target_network_params: EMA target network params
            obs: Stacked observations (B, H, W, C*num_frames)
            jepa_key: Random key for mask generation

        Returns:
            JEPA loss scalar
        """
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

        # Target uses the unmasked full observation
        target_obs = obs

        # Encode context with online encoder
        context_embed = network.apply(params['network'], context_obs)

        # Predict target embedding
        pred_embed = predictor.apply(params['predictor'], context_embed)

        # Encode target with EMA encoder (stop gradient applied in caller)
        target_embed = network.apply(target_network_params, target_obs)

        # L2 normalize embeddings
        pred_norm = l2_normalize(pred_embed)
        target_norm = l2_normalize(target_embed)

        # L2 distance loss
        jepa_loss = jnp.mean((pred_norm - target_norm) ** 2)

        return jepa_loss

    def compute_le_jepa_loss(params, target_params, obs, mask_key, sig_key):
        """
        Compute LeJEPA loss: masked prediction + SIGReg isotropy regularization.

        Returns:
            pred_loss, sig_loss, total_le_jepa_loss, diag(z_mean, z_var_mean, z_var_min, z_norm_mean)
        """
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

        # Online branch.
        h_ctx = network.apply(params['network'], context_obs)
        z_ctx = projector.apply(params['projector'], h_ctx)
        z_pred = predictor.apply(params['predictor'], z_ctx)

        # Target branch (EMA target by default for stability).
        if args.lejepa_use_target_ema:
            h_tgt = network.apply(target_params['network'], target_obs)
            z_tgt = projector.apply(target_params['projector'], h_tgt)
        else:
            h_tgt = network.apply(params['network'], target_obs)
            z_tgt = projector.apply(params['projector'], h_tgt)
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

    def ppo_jepa_loss(params, target_params, x, a, logp, mb_advantages,
                      mb_returns, mb_values, aug_key, mask_key, sig_key,
                      jepa_lambda_current):
        """Combined PPO + JEPA / LeJEPA loss function."""
        # Augmentation key for observations
        aug_key_obs = aug_key

        # Apply random shift augmentation if enabled
        if args.use_augmentation:
            x = random_shift(aug_key_obs, x, pad=args.augment_pad)

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
        ppo_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        # JEPA losses
        le_zero_diag = jnp.zeros((4,), dtype=jnp.float32)
        if args.use_jepa:
            def compute_jepa():
                return compute_jepa_loss(params, target_params, x, mask_key)
            def skip_jepa():
                return 0.0
            j_loss = jax.lax.cond(
                jepa_lambda_current > 0,
                compute_jepa,
                skip_jepa,
            )
        else:
            j_loss = 0.0
        if args.use_le_jepa:
            def compute_le_jepa():
                return compute_le_jepa_loss(params, target_params, x, mask_key, sig_key)
            def skip_le_jepa():
                return 0.0, 0.0, 0.0, le_zero_diag
            le_pred_loss, le_sig_loss, le_total_loss, le_diag = jax.lax.cond(
                jepa_lambda_current > 0,
                compute_le_jepa,
                skip_le_jepa,
            )
        else:
            le_pred_loss, le_sig_loss, le_total_loss, le_diag = 0.0, 0.0, 0.0, le_zero_diag

        # Total loss
        total_loss = ppo_loss
        if args.use_jepa:
            total_loss = total_loss + jepa_lambda_current * j_loss
        if args.use_le_jepa:
            total_loss = total_loss + jepa_lambda_current * le_total_loss

        return total_loss, (pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag,
                           jax.lax.stop_gradient(approx_kl))

    ppo_jepa_loss_grad_fn = jax.value_and_grad(ppo_jepa_loss, has_aux=True)

    @jax.jit
    def update_ppo_jepa(
        agent_state: TrainState,
        target_params,
        storage: Storage,
        key: jax.random.PRNGKey,
        jepa_lambda_current: float,
    ):
        """PPO update with JEPA / LeJEPA auxiliary objective."""
        def update_epoch(carry, unused_inp):
            agent_state, target_params, key = carry
            key, subkey, aug_key, mask_parent_key, sig_parent_key = jax.random.split(key, 5)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            # Generate keys for each minibatch (augmentation, masking, SIGReg projections).
            aug_keys = jax.random.split(aug_key, args.num_minibatches)
            mask_keys = jax.random.split(mask_parent_key, args.num_minibatches)
            sig_keys = jax.random.split(sig_parent_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state, target_params = carry
                minibatch, mb_aug_key, mb_mask_key, mb_sig_key = inputs

                (loss, (pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag, approx_kl)), grads = ppo_jepa_loss_grad_fn(
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

                # EMA update target networks
                if args.use_le_jepa and args.lejepa_use_target_ema:
                    # LeJEPA: EMA update encoder + projector.
                    target_params = jax.lax.cond(
                        jepa_lambda_current > 0,
                        lambda: flax.core.freeze({
                            'network': ema_update(target_params['network'], agent_state.params['network'], args.jepa_ema_tau),
                            'projector': ema_update(target_params['projector'], agent_state.params['projector'], args.jepa_ema_tau),
                        }),
                        lambda: target_params,
                    )
                elif args.use_jepa:
                    # JEPA: EMA update encoder only
                    target_params = jax.lax.cond(
                        jepa_lambda_current > 0,
                        lambda: ema_update(target_params, agent_state.params['network'], args.jepa_ema_tau),
                        lambda: target_params,
                    )

                return (agent_state, target_params), (loss, pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag, approx_kl, grad_norm)

            (agent_state, target_params), (loss, pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag, approx_kl, grad_norm) = jax.lax.scan(
                update_minibatch, (agent_state, target_params), (shuffled_storage, aug_keys, mask_keys, sig_keys)
            )
            return (agent_state, target_params, key), (loss, pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag, approx_kl, grad_norm)

        (agent_state, target_params, key), (loss, pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag, approx_kl, grad_norm) = jax.lax.scan(
            update_epoch, (agent_state, target_params, key), (), length=args.update_epochs
        )
        return (agent_state, target_params, loss, pg_loss, v_loss, entropy_loss, j_loss, le_pred_loss, le_sig_loss, le_total_loss, le_diag, approx_kl,
                grad_norm, key)

    # Start the game
    global_step = 0
    start_time = time.time()

    # Reset environment
    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    env_state = envs.reset(reset_rngs)

    # Initialize frame stack with initial observation
    frame_stack = FrameStack.create(args.n_envs, args.frame_stack, raw_obs_shape)
    frame_stack = frame_stack.reset(env_state.pixels)  # Fill all frames with initial obs
    next_obs = frame_stack.get_stacked()  # Get stacked observation
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    # Initialize reward normalizer (discounted return-based, like CleanRL)
    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    def step_once(carry, step):
        agent_state, episode_stats, reward_norm, fs, env_state, obs, done, key = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        # Step environment
        env_state = envs.step(env_state, action)
        raw_obs = env_state.pixels  # Raw single-frame observation
        raw_reward = env_state.reward
        next_done = env_state.done.astype(jnp.bool_)  # Ensure bool type

        # Update frame stack: push new frame, then reset for done envs
        fs = fs.push(raw_obs)
        fs = fs.reset(raw_obs, next_done)  # Reset done envs to copies of new obs
        next_obs = fs.get_stacked()  # Get stacked observation

        # Update reward normalizer and normalize reward (discounted return-based)
        reward_norm = reward_norm.update(raw_reward, next_done.astype(jnp.float32))
        reward = reward_norm.normalize(raw_reward)

        # Update episode statistics (use raw reward for tracking true returns)
        new_episode_return = episode_stats.episode_returns + raw_reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            # Use jnp.where to preserve dtypes
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
            next_obs=next_obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            next_dones=next_done,
            values=value,
            rewards=reward,  # Normalized reward for PPO training
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

    print("Starting training...")
    cumulative_episodic_return = 0.0
    for iteration in range(1, args.num_updates + 1):
        iteration_time_start = time.time()
        agent_state, episode_stats, reward_normalizer, frame_stack, env_state, next_obs, next_done, storage, key = rollout(
            agent_state, episode_stats, reward_normalizer, frame_stack, env_state, next_obs, next_done, key
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)

        # Compute loss schedule lambdas OUTSIDE jit to avoid recompilation
        jepa_lambda_current = get_jepa_lambda_py(
            iteration, args.jepa_lambda,
            args.jepa_warmup_updates, args.jepa_rampup_updates
        ) if (args.use_jepa or args.use_le_jepa) else 0.0

        # PPO + JEPA / LeJEPA update
        (agent_state, target_params, loss, pg_loss, v_loss, entropy_loss,
         jepa_loss_val, le_pred_loss_val, le_sig_loss_val, le_total_loss_val, le_diag_val,
         approx_kl, grad_norm_val, key) = update_ppo_jepa(
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

            if args.use_jepa:
                print(
                    f"update={iteration} step={global_step} "
                    f"ep_return={avg_episodic_return:.1f} "
                    f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                    f"loss={loss[-1, -1].item():.4f} "
                    f"jepa_loss={jepa_loss_val[-1, -1].item():.4f} "
                    f"jepa_lambda={jepa_lambda_current:.4f} "
                    f"SPS={sps}"
                )
            elif args.use_le_jepa:
                print(
                    f"update={iteration} step={global_step} "
                    f"ep_return={avg_episodic_return:.1f} "
                    f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                    f"loss={loss[-1, -1].item():.4f} "
                    f"le_pred={le_pred_loss_val[-1, -1].item():.4f} "
                    f"le_sig={le_sig_loss_val[-1, -1].item():.4f} "
                    f"le_total={le_total_loss_val[-1, -1].item():.4f} "
                    f"jepa_lambda={jepa_lambda_current:.4f} "
                    f"SPS={sps}"
                )
            else:
                print(
                    f"update={iteration} step={global_step} "
                    f"ep_return={avg_episodic_return:.1f} "
                    f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                    f"loss={loss[-1, -1].item():.4f} "
                    f"SPS={sps}"
                )

            if args.track:
                lr = agent_state.opt_state[1].hyperparams["learning_rate"].item()
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
                    "losses/grad_norm": grad_norm_val[-1, -1].item(),
                }

                if args.use_jepa:
                    log_dict["jepa/loss"] = jepa_loss_val[-1, -1].item()
                    log_dict["jepa/lambda"] = jepa_lambda_current
                if args.use_le_jepa:
                    log_dict["le_jepa/pred_loss"] = le_pred_loss_val[-1, -1].item()
                    log_dict["le_jepa/sigreg_loss"] = le_sig_loss_val[-1, -1].item()
                    log_dict["le_jepa/total"] = le_total_loss_val[-1, -1].item()
                    log_dict["le_jepa/z_mean"] = le_diag_val[-1, -1, 0].item()
                    log_dict["le_jepa/z_var_mean"] = le_diag_val[-1, -1, 1].item()
                    log_dict["le_jepa/z_var_min"] = le_diag_val[-1, -1, 2].item()
                    log_dict["le_jepa/z_norm_mean"] = le_diag_val[-1, -1, 3].item()
                    log_dict["le_jepa/lambda"] = jepa_lambda_current

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
