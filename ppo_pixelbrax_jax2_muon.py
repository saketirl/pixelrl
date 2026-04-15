#!/usr/bin/env python
"""
PPO for PixelBrax environments - CleanRL style with Manifold MUON optimizer.

Uses:
- Adam for encoder (CNN)
- Manifold MUON for actor/critic head matrices (2D+ params)
- Adam for actor/critic head vectors/scalars (biases, log_std)
"""
import os
import json
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

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
from pixelbrax.continual_dynamics import (
    apply_dynamics_task,
    continual_log_metrics,
    load_continual_dynamics_config,
    make_dynamics_task,
    resolve_schedule_seed,
    validate_continual_dynamics_config,
)
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
    use_heads_muon: bool = True
    """if toggled, use MUON for actor/critic head matrices"""

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
    action_repeat: int = 1
    """Number of times to repeat each action (frame skip)"""

    # Continual dynamics
    continual_dynamics_config: Optional[str] = None
    """Path to a YAML continual dynamics config"""

    # Debug/analysis flags
    debug_repr: bool = False
    """Toggle debug logging for encoder representations"""

    # Encoder architecture
    encoder_tanh_scale: float = 0.5
    """Multiplier for encoder output before tanh (controls saturation)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


class Network(nn.Module):
    """CNN encoder for pixel observations with LayerNorm for stability."""
    tanh_scale: float = 0.5

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
        x = nn.tanh(self.tanh_scale * x)  # tanh for bounded features, helps with stability
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

    def reset_discounted_return(self):
        """Reset per-env discounted returns while preserving running moments."""
        return self.replace(
            discounted_return=jnp.zeros_like(self.discounted_return),
        )


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
    muon_dual_lr: float = 0.01,
    muon_dual_steps: int = 5,
    muon_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    max_grad_norm: float = 0.5,
    actor_muon_max_grad_norm: float = 1.0,
    critic_muon_max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    use_heads_muon: bool = True,
):
    """
    Create optimizer that uses:
    - Adam/AdamW for encoder (all params) - supports lr schedule, with grad clipping
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
                if not use_heads_muon:
                    return 'heads_adam'
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
    print("PPO with Manifold MUON for Actor/Critic Heads")
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
    print(f"  use_heads_muon: {args.use_heads_muon}")
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

    base_sys = envs.base_sys
    current_sys = base_sys
    continual_config = None
    current_task = None
    task_start_step = 0
    updates_per_task = 0
    schedule_seed = args.seed

    if args.continual_dynamics_config is not None:
        loaded_config = load_continual_dynamics_config(args.continual_dynamics_config)
        if loaded_config.enabled:
            updates_per_task = validate_continual_dynamics_config(
                loaded_config,
                env_name=args.env_name,
                backend=args.backend,
                n_envs=args.n_envs,
                num_steps=args.num_steps,
                base_sys=base_sys,
            )
            continual_config = loaded_config
            schedule_seed = resolve_schedule_seed(continual_config, args.seed)
            current_task = make_dynamics_task(
                continual_config,
                base_sys,
                task_index=0,
                schedule_seed=schedule_seed,
            )
            current_sys = apply_dynamics_task(base_sys, current_task)
            print("\nContinual dynamics: ENABLED")
            print(f"  config: {continual_config.path}")
            print(f"  switch_every_env_steps: {continual_config.switch_every_env_steps}")
            print(f"  updates_per_task: {updates_per_task}")
            print(f"  schedule_seed: {schedule_seed}")
            print(
                "  task=0 step=0 default=True values="
                f"{json.dumps(current_task.values, sort_keys=True)}"
            )
        else:
            print("\nContinual dynamics: config disabled; using fixed dynamics")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset_with_sys(current_sys, reset_rng)
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
    network = Network(tanh_scale=args.encoder_tanh_scale)
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)

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
    if not args.use_heads_muon:
        actor_adam += actor_muon
        critic_adam += critic_muon
        actor_muon = 0
        critic_muon = 0

    print(f"\nParameter breakdown:")
    print(f"  Encoder (Adam): {encoder_params:,}")
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
        use_heads_muon=args.use_heads_muon,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

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
    env_state = envs.reset_with_sys(current_sys, reset_rngs)

    # Initialize frame stack with initial observation
    frame_stack = FrameStack.create(args.n_envs, args.frame_stack, raw_obs_shape)
    frame_stack = frame_stack.reset(env_state.pixels)
    next_obs = frame_stack.get_stacked()
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    # Initialize reward normalizer
    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    def rollout(current_sys, agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key, max_steps):
        def step_once(carry, step):
            agent_state, episode_stats, reward_norm, fs, env_state, obs, done, key = carry
            action, logprob, value, key = get_action_and_value(agent_state, obs, key)

            env_state = envs.step_with_sys(current_sys, env_state, action)
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

        (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key), storage = jax.lax.scan(
            step_once, (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key), jnp.arange(max_steps)
        )
        return agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, storage, key

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("\nStarting training...")
    cumulative_episodic_return = 0.0
    for iteration in range(1, args.num_updates + 1):
        if (
            continual_config is not None
            and iteration > 1
            and (iteration - 1) % updates_per_task == 0
        ):
            task_index = (iteration - 1) // updates_per_task
            current_task = make_dynamics_task(
                continual_config,
                base_sys,
                task_index=task_index,
                schedule_seed=schedule_seed,
            )
            current_sys = apply_dynamics_task(base_sys, current_task)
            task_start_step = global_step

            key, reset_key = jax.random.split(key)
            reset_rngs = jax.random.split(reset_key, args.n_envs)
            env_state = envs.reset_with_sys(current_sys, reset_rngs)
            frame_stack = frame_stack.reset(env_state.pixels)
            next_obs = frame_stack.get_stacked()
            next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)
            episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            )
            reward_normalizer = reward_normalizer.reset_discounted_return()

            print(
                f"continual_switch task={task_index} step={global_step} "
                f"default={current_task.is_default} values="
                f"{json.dumps(current_task.values, sort_keys=True)}"
            )

            if args.track:
                import wandb
                wandb.log(
                    continual_log_metrics(
                        current_task,
                        switch_every_env_steps=continual_config.switch_every_env_steps,
                        task_start_step=task_start_step,
                        global_step=global_step,
                    ),
                    step=global_step,
                )

        iteration_time_start = time.time()
        agent_state, episode_stats, reward_normalizer, frame_stack, env_state, next_obs, next_done, storage, key = rollout(
            current_sys, agent_state, episode_stats, reward_normalizer, frame_stack, env_state, next_obs, next_done, key
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
                # Get actual learning rates from optimizer state
                # opt_state structure: (clip_state, multi_transform_state)
                # multi_transform_state has inner states for each transform
                opt_step = iteration * args.update_epochs * args.num_minibatches
                if args.anneal_lr:
                    # Calculate scheduled LR
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
                    "charts/heads_muon_lr": args.heads_muon_lr if args.use_heads_muon else 0.0,
                    "charts/use_heads_muon": int(args.use_heads_muon),
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                }

                if continual_config is not None:
                    log_dict.update(
                        continual_log_metrics(
                            current_task,
                            switch_every_env_steps=continual_config.switch_every_env_steps,
                            task_start_step=task_start_step,
                            global_step=global_step,
                        )
                    )

                # Debug metrics for encoder representations
                if args.debug_repr:
                    # Compute encoder output for a sample of observations
                    sample_obs = storage.obs[0, :256]  # First step, up to 256 envs
                    hidden = network.apply(agent_state.params['network'], sample_obs)

                    # Encoder representation metrics
                    repr_metrics = encoder_repr_metrics(hidden)
                    for k, v in repr_metrics.items():
                        log_dict[k] = float(v)

                    # Gradient norms
                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(v)

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
