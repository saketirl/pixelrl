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
    """JEPA mode: 'none', 'jepa' (original), 'td_jepa' (successor features)"""
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

    # TD-JEPA (successor features) hyperparameters
    sf_dim: int = 256
    """dimensionality of successor/reward features"""
    beta_sf: float = 0.1
    """coefficient for SF TD loss"""
    beta_r: float = 1.0
    """coefficient for reward regression loss"""
    sf_ema_tau: float = 0.995
    """EMA coefficient for TD-JEPA target networks"""
    sf_warmup_updates: int = 2
    """updates before SF losses activate"""
    sf_rampup_updates: int = 10
    """updates to linearly ramp SF loss coefficients"""
    refit_w_only: bool = False
    """freeze all params except w for reward-change adaptation"""

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


class PsiHead(nn.Module):
    """Reward features: hidden -> psi (sf_dim)."""
    sf_dim: int = 256
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(self.sf_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


class MHead(nn.Module):
    """Successor features: hidden -> m (sf_dim)."""
    sf_dim: int = 256
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(self.sf_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
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
    raw_rewards: jnp.array


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.use_jepa = (args.jepa_mode == "jepa")
    args.use_td_jepa = (args.jepa_mode == "td_jepa")
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
    key, network_key, actor_key, critic_key, predictor_key = jax.random.split(key, 5)

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
    if args.use_td_jepa:
        print(f"sf_dim: {args.sf_dim}")
        print(f"beta_sf: {args.beta_sf}")
        print(f"beta_r: {args.beta_r}")
        print(f"sf_ema_tau: {args.sf_ema_tau}")
        print(f"sf_warmup_updates: {args.sf_warmup_updates}")
        print(f"sf_rampup_updates: {args.sf_rampup_updates}")
    if args.refit_w_only:
        print("REFIT-W-ONLY MODE: all params frozen except w")

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
    predictor = JEPAPredictor(hidden_dim=args.jepa_predictor_hidden, output_dim=512)

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)

    # Initialize predictor with dummy hidden state
    predictor_params = predictor.init(predictor_key, dummy_hidden)

    params_dict = {
        'network': network_params,
        'actor': actor.init(actor_key, dummy_hidden),
        'critic': critic.init(critic_key, dummy_hidden),
        'predictor': predictor_params,
    }

    # TD-JEPA: initialize PsiHead, MHead, and w
    if args.use_td_jepa:
        psi_head = PsiHead(sf_dim=args.sf_dim)
        m_head = MHead(sf_dim=args.sf_dim)
        key, psi_key, m_key, w_key = jax.random.split(key, 4)
        psi_params = psi_head.init(psi_key, dummy_hidden)
        m_params = m_head.init(m_key, dummy_hidden)
        w_init = jax.random.normal(w_key, (args.sf_dim,)) * 0.01
        params_dict['psi_head'] = psi_params
        params_dict['m_head'] = m_params
        params_dict['w'] = {'w': w_init}
    else:
        # Create module references for consistent code paths (unused but needed for closures)
        psi_head = PsiHead(sf_dim=args.sf_dim)
        m_head = MHead(sf_dim=args.sf_dim)

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
    if args.use_td_jepa:
        target_params = flax.core.freeze({
            'network': agent_state.params['network'],
            'psi_head': agent_state.params['psi_head'],
            'm_head': agent_state.params['m_head'],
        })
    else:
        target_params = agent_state.params['network']

    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    predictor.apply = jax.jit(predictor.apply)
    if args.use_td_jepa:
        psi_head.apply = jax.jit(psi_head.apply)
        m_head.apply = jax.jit(m_head.apply)

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
        batch_size = obs.shape[0]
        total_channels = obs.shape[-1]
        channels_per_frame = 3
        num_frames = total_channels // channels_per_frame

        # Pick a random frame to mask for each sample in the batch
        frame_indices = jax.random.randint(jepa_key, (batch_size,), 0, num_frames)

        # Create channel mask: 1.0 for channels to keep, 0.0 for channels to mask
        # Shape: (num_frames, channels_per_frame) -> broadcast to (B, num_frames, channels_per_frame)
        frame_ids = jnp.arange(num_frames)[None, :]  # (1, num_frames)
        mask_per_frame = (frame_ids != frame_indices[:, None])  # (B, num_frames) - True = keep
        # Expand to per-channel: (B, num_frames, 1) -> (B, num_frames, channels_per_frame)
        channel_mask = jnp.repeat(mask_per_frame[:, :, None], channels_per_frame, axis=-1)
        # Flatten to (B, total_channels)
        channel_mask = channel_mask.reshape(batch_size, total_channels)
        # Reshape for broadcasting: (B, 1, 1, total_channels)
        channel_mask = channel_mask[:, None, None, :]

        # Apply mask: zero out the masked frame's channels
        obs_float = obs.astype(jnp.float32)
        context_obs = obs_float * channel_mask
        context_obs = context_obs.astype(obs.dtype)

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

    def compute_sf_losses(params, target_params, obs, next_obs, rewards, next_dones, aug_key):
        """Compute successor feature TD loss and reward regression loss."""
        # Augment next_obs independently if augmentation enabled
        if args.use_augmentation:
            next_obs = random_shift(aug_key, next_obs, pad=args.augment_pad)

        # Encode
        hidden = network.apply(params['network'], obs)
        hidden_next_tgt = network.apply(target_params['network'], next_obs)

        # Online heads
        psi = psi_head.apply(params['psi_head'], hidden)      # (B, sf_dim)
        m = m_head.apply(params['m_head'], hidden)             # (B, sf_dim)
        w = params['w']['w']                                    # (sf_dim,)

        # Target m on next obs
        m_tgt_next = m_head.apply(target_params['m_head'], hidden_next_tgt)

        # SF TD target: psi(o_t) + gamma * (1 - d_t) * sg[m_tgt(o_{t+1})]
        nonterminal = (1.0 - next_dones.astype(jnp.float32))[:, None]
        sf_target = psi + args.gamma * nonterminal * jax.lax.stop_gradient(m_tgt_next)

        # SF TD loss: ||m(o_t) - sf_target||^2  (psi gets grads, m_tgt_next is already stopped)
        sf_td_loss = jnp.mean(jnp.sum((m - sf_target) ** 2, axis=-1))

        # Reward regression loss: (w . psi(o_t) - r_t)^2
        pred_reward = jnp.dot(psi, w)
        reward_loss = jnp.mean((pred_reward - rewards) ** 2)

        # Diagnostics
        reward_r2 = 1.0 - jnp.sum((pred_reward - rewards)**2) / (jnp.sum((rewards - rewards.mean())**2) + 1e-8)
        sf_td_error = jnp.mean(jnp.sqrt(jnp.sum((m - sf_target)**2, axis=-1)))
        w_norm = jnp.linalg.norm(w)
        psi_norm = jnp.mean(jnp.linalg.norm(psi, axis=-1))
        m_norm = jnp.mean(jnp.linalg.norm(m, axis=-1))

        return sf_td_loss, reward_loss, (reward_r2, sf_td_error, w_norm, psi_norm, m_norm)

    def ppo_jepa_loss(params, target_params, x, a, logp, mb_advantages,
                      mb_returns, mb_values, aug_key, jepa_key,
                      jepa_lambda_current, next_x, mb_rewards, mb_raw_rewards,
                      mb_next_dones, sf_lambda_current):
        """Combined PPO + JEPA + TD-JEPA loss function."""
        # Split aug_key for obs vs next_obs augmentation
        aug_key_obs, aug_key_next = jax.random.split(aug_key)

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

        # JEPA loss (conditional on lambda > 0 and use_jepa flag)
        if args.use_jepa:
            def compute_jepa():
                return compute_jepa_loss(params, target_params, x, jepa_key)
            def skip_jepa():
                return 0.0
            j_loss = jax.lax.cond(
                jepa_lambda_current > 0,
                compute_jepa,
                skip_jepa,
            )
        else:
            j_loss = 0.0

        # SF losses: reward regression always on, SF TD gated by sf_lambda_current
        sf_zero_diag = (0.0, 0.0, 0.0, 0.0, 0.0)
        if args.use_td_jepa:
            # Always compute SF losses (reward regression needs to train from the start)
            sf_td_loss, reward_loss, sf_diag = compute_sf_losses(
                params, target_params, x, next_x, mb_raw_rewards, mb_next_dones, aug_key_next
            )
        else:
            sf_td_loss, reward_loss, sf_diag = 0.0, 0.0, sf_zero_diag

        # Total loss
        total_loss = ppo_loss
        if args.use_jepa:
            total_loss = total_loss + jepa_lambda_current * j_loss
        if args.use_td_jepa:
            # Reward regression always active; SF TD warmed up via sf_lambda_current
            total_loss = total_loss + args.beta_r * reward_loss + sf_lambda_current * args.beta_sf * sf_td_loss

        return total_loss, (pg_loss, v_loss, entropy_loss, j_loss,
                           jax.lax.stop_gradient(approx_kl),
                           sf_td_loss, reward_loss, sf_diag)

    ppo_jepa_loss_grad_fn = jax.value_and_grad(ppo_jepa_loss, has_aux=True)

    @jax.jit
    def update_ppo_jepa(
        agent_state: TrainState,
        target_params,
        storage: Storage,
        key: jax.random.PRNGKey,
        jepa_lambda_current: float,
        sf_lambda_current: float,
    ):
        """PPO update with JEPA / TD-JEPA auxiliary objective."""
        def update_epoch(carry, unused_inp):
            agent_state, target_params, key = carry
            key, subkey, aug_key, jepa_key = jax.random.split(key, 4)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            # Generate keys for each minibatch (for augmentation and JEPA)
            aug_keys = jax.random.split(aug_key, args.num_minibatches)
            jepa_keys = jax.random.split(jepa_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state, target_params = carry
                minibatch, mb_aug_key, mb_jepa_key = inputs

                (loss, (pg_loss, v_loss, entropy_loss, j_loss, approx_kl,
                        sf_td_loss, reward_loss, sf_diag)), grads = ppo_jepa_loss_grad_fn(
                    agent_state.params,
                    jax.lax.stop_gradient(target_params),
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                    mb_jepa_key,
                    jepa_lambda_current,
                    minibatch.next_obs,
                    minibatch.rewards,
                    minibatch.raw_rewards,
                    minibatch.next_dones,
                    sf_lambda_current,
                )
                if args.refit_w_only:
                    # Zero all gradients except w — only reward regression trains w
                    def _mask_grad(path, g):
                        # path is a tuple of DictKey/etc; keep only leaves under 'w'
                        if path and str(path[0].key) == 'w':
                            return g
                        return jnp.zeros_like(g)
                    grads = jax.tree_util.tree_map_with_path(_mask_grad, grads)

                grad_norm = optax.global_norm(grads)
                agent_state = agent_state.apply_gradients(grads=grads)

                # EMA update target networks
                if args.use_td_jepa and not args.refit_w_only:
                    # TD-JEPA: EMA update encoder + psi_head + m_head
                    target_params = jax.lax.cond(
                        sf_lambda_current > 0,
                        lambda: flax.core.freeze({
                            'network': ema_update(target_params['network'], agent_state.params['network'], args.sf_ema_tau),
                            'psi_head': ema_update(target_params['psi_head'], agent_state.params['psi_head'], args.sf_ema_tau),
                            'm_head': ema_update(target_params['m_head'], agent_state.params['m_head'], args.sf_ema_tau),
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

                return (agent_state, target_params), (loss, pg_loss, v_loss, entropy_loss, j_loss, approx_kl,
                                                       sf_td_loss, reward_loss, sf_diag, grad_norm)

            (agent_state, target_params), (loss, pg_loss, v_loss, entropy_loss, j_loss, approx_kl,
                                            sf_td_loss, reward_loss, sf_diag, grad_norm) = jax.lax.scan(
                update_minibatch, (agent_state, target_params), (shuffled_storage, aug_keys, jepa_keys)
            )
            return (agent_state, target_params, key), (loss, pg_loss, v_loss, entropy_loss, j_loss, approx_kl,
                                                        sf_td_loss, reward_loss, sf_diag, grad_norm)

        (agent_state, target_params, key), (loss, pg_loss, v_loss, entropy_loss, j_loss, approx_kl,
                                             sf_td_loss, reward_loss, sf_diag, grad_norm) = jax.lax.scan(
            update_epoch, (agent_state, target_params, key), (), length=args.update_epochs
        )
        return (agent_state, target_params, loss, pg_loss, v_loss, entropy_loss, j_loss, approx_kl,
                sf_td_loss, reward_loss, sf_diag, grad_norm, key)

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
            raw_rewards=raw_reward,  # Raw reward for SF reward regression
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
        ) if args.use_jepa else 0.0

        sf_lambda_current = get_jepa_lambda_py(
            iteration, 1.0,
            args.sf_warmup_updates, args.sf_rampup_updates
        ) if args.use_td_jepa else 0.0

        # PPO + JEPA / TD-JEPA update
        (agent_state, target_params, loss, pg_loss, v_loss, entropy_loss,
         jepa_loss_val, approx_kl, sf_td_loss_val, reward_loss_val,
         sf_diag_val, grad_norm_val, key) = update_ppo_jepa(
            agent_state,
            target_params,
            storage,
            key,
            jepa_lambda_current,
            sf_lambda_current,
        )

        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            cumulative_episodic_return += avg_episodic_return
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))

            if args.use_td_jepa:
                print(
                    f"update={iteration} step={global_step} "
                    f"ep_return={avg_episodic_return:.1f} "
                    f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                    f"loss={loss[-1, -1].item():.4f} "
                    f"sf_td={sf_td_loss_val[-1, -1].item():.4f} "
                    f"r_loss={reward_loss_val[-1, -1].item():.4f} "
                    f"r2={sf_diag_val[0][-1, -1].item():.3f} "
                    f"w_norm={sf_diag_val[2][-1, -1].item():.3f} "
                    f"grad_norm={grad_norm_val[-1, -1].item():.3f} "
                    f"sf_lam={sf_lambda_current:.4f} "
                    f"SPS={sps}"
                )
            elif args.use_jepa:
                print(
                    f"update={iteration} step={global_step} "
                    f"ep_return={avg_episodic_return:.1f} "
                    f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                    f"loss={loss[-1, -1].item():.4f} "
                    f"jepa_loss={jepa_loss_val[-1, -1].item():.4f} "
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

                if args.use_td_jepa:
                    log_dict["td_jepa/sf_td_loss"] = sf_td_loss_val[-1, -1].item()
                    log_dict["td_jepa/reward_loss"] = reward_loss_val[-1, -1].item()
                    log_dict["td_jepa/reward_r2"] = sf_diag_val[0][-1, -1].item()
                    log_dict["td_jepa/sf_td_error_mean"] = sf_diag_val[1][-1, -1].item()
                    log_dict["td_jepa/w_norm"] = sf_diag_val[2][-1, -1].item()
                    log_dict["td_jepa/psi_norm_mean"] = sf_diag_val[3][-1, -1].item()
                    log_dict["td_jepa/m_norm_mean"] = sf_diag_val[4][-1, -1].item()
                    log_dict["td_jepa/sf_lambda"] = sf_lambda_current

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
