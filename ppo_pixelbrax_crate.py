#!/usr/bin/env python
"""
PPO for PixelBrax environments using Hierarchical Temporal CRATE.

This implementation combines CNN-style and transformer-style temporal processing:
- Channel stacking: Groups of frames are stacked along channels (local temporal, like CNNs)
- Temporal transformer: Channel-stacked groups become tokens in a sequence (global temporal)

Total frames = temporal_stack × channel_stack
Example: temporal_stack=4, channel_stack=2 → 8 total frames
         → 4 transformer tokens, each with 2 frames channel-stacked

Uses CRATE architecture (ISTA feedforward, simplified attention)
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
# Using default Flax initializers (lecun_normal) to match PyTorch/CRATE defaults
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
    learning_rate: float = 1e-4
    """the learning rate of the optimizer (lower for transformers)"""
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
    log_interval: int = 10
    """logging interval (in updates)"""

    # Data augmentation (DrQ-style)
    use_augmentation: bool = False
    """Toggle random shift data augmentation (DrQ-style)"""
    augment_pad: int = 4
    """Padding size for random shift augmentation"""

    # Hierarchical frame stacking
    # Total frames = temporal_stack × channel_stack
    temporal_stack: int = 4
    """Number of temporal tokens in the sequence"""
    channel_stack: int = 2
    """Number of frames stacked per token (channel-wise, like CNNs)"""

    # Action repeat
    action_repeat: int = 1
    """Number of times to repeat each action (frame skip)"""

    # Temporal CRATE-specific arguments
    embed_dim: int = 256
    """embedding dimension for CRATE"""
    depth: int = 6
    """number of transformer layers"""
    num_heads: int = 4
    """number of attention heads"""
    ista_step_size: float = 0.1
    """step size for ISTA gradient update"""
    ista_lambda: float = 0.1
    """sparsity regularization for ISTA"""
    emb_dropout: float = 0.0
    """dropout after frame embedding"""
    attn_dropout: float = 0.0
    """dropout in attention"""
    pool: str = "cls"
    """pooling type: 'cls' or 'mean'"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


# --------------------------------------------------------
#  Temporal CRATE Architecture (JAX/Flax)
# --------------------------------------------------------

class ISTAFeedForward(nn.Module):
    """
    ISTA-based feedforward layer from CRATE.
    Uses gradient descent update with ReLU thresholding instead of standard MLP.

    Update rule: x = ReLU(x + step_size * (W^T @ x - W^T @ W @ x) - step_size * lambda)
    """
    dim: int
    step_size: float = 0.1
    lambd: float = 0.1

    @nn.compact
    def __call__(self, x):
        # Learnable weight matrix W (kaiming_uniform like original CRATE)
        weight = self.param(
            "weight",
            nn.initializers.he_uniform(),
            (self.dim, self.dim)
        )

        # Forward: x1 = W @ x
        x1 = x @ weight.T

        # Gradient computation for sparse coding objective
        # grad_1 = W^T @ W @ x
        grad_1 = x1 @ weight
        # grad_2 = W^T @ x
        grad_2 = x @ weight

        # ISTA gradient update with ReLU thresholding
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        # ReLU acts as soft thresholding for sparsity
        output = nn.relu(x + grad_update)
        return output


class CRATEAttention(nn.Module):
    """
    CRATE-style attention with single projection (Q=K=V).
    This is a key simplification in CRATE's white-box design.
    """
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        B, N, _ = x.shape
        inner_dim = self.dim_head * self.heads

        # Single projection for Q, K, V (CRATE's simplification)
        # Uses default Flax init (lecun_normal, similar to PyTorch default)
        qkv = nn.Dense(inner_dim, use_bias=False)(x)

        # Reshape to (B, heads, N, dim_head)
        w = qkv.reshape(B, N, self.heads, self.dim_head)
        w = jnp.transpose(w, (0, 2, 1, 3))  # (B, heads, N, dim_head)

        # Self-attention: W @ W^T (same projection for Q and K)
        scale = self.dim_head ** -0.5
        dots = jnp.matmul(w, jnp.transpose(w, (0, 1, 3, 2))) * scale

        # Softmax attention
        attn = nn.softmax(dots, axis=-1)
        attn = nn.Dropout(self.dropout, deterministic=deterministic)(attn)

        # Apply attention to values
        out = jnp.matmul(attn, w)

        # Reshape back
        out = jnp.transpose(out, (0, 2, 1, 3))  # (B, N, heads, dim_head)
        out = out.reshape(B, N, inner_dim)

        # Output projection (default init)
        out = nn.Dense(self.dim)(out)
        out = nn.Dropout(self.dropout, deterministic=deterministic)(out)

        return out


class CRATEBlock(nn.Module):
    """Single CRATE transformer block with pre-norm."""
    dim: int
    heads: int
    dim_head: int
    dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Pre-norm attention with residual
        y = nn.LayerNorm()(x)
        y = CRATEAttention(
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
        )(y, deterministic=deterministic)
        grad_x = y + x  # Residual connection

        # Pre-norm ISTA feedforward
        y = nn.LayerNorm()(grad_x)
        x = ISTAFeedForward(
            dim=self.dim,
            step_size=self.ista_step_size,
            lambd=self.ista_lambda,
        )(y)

        return x


class TemporalCRATEEncoder(nn.Module):
    """
    Hierarchical Temporal CRATE encoder for pixel observations.

    Combines CNN-style channel stacking with temporal transformer:
    - Groups of frames are channel-stacked (local temporal info, like CNNs)
    - Channel-stacked groups become tokens in a temporal sequence
    - Temporal transformer attends across groups (global temporal info)

    Total frames = temporal_stack × channel_stack
    Example: temporal_stack=4, channel_stack=2 → 8 total frames
             → 4 tokens, each with 2 frames channel-stacked
    """
    temporal_stack: int = 4
    channel_stack: int = 2
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1
    pool: str = "cls"

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: (B, T, H, W, C_stacked) - batch of frame sequences
               T = temporal_stack (number of tokens)
               C_stacked = C * channel_stack (channel-stacked frames per token)

        Returns:
            (B, 512) - encoded representation
        """
        B, T, H, W, C_stacked = x.shape
        token_dim = H * W * C_stacked  # Each token: flattened channel-stacked frames
        dim_head = self.embed_dim // self.num_heads

        # Normalize pixel values
        x = x.astype(jnp.float32) / 255.0

        # Flatten each token: (B, T, H, W, C_stacked) → (B, T, H*W*C_stacked)
        x = x.reshape(B, T, token_dim)

        # Token embedding (like CRATE's patch embedding but for channel-stacked frames)
        # LayerNorm → Linear → LayerNorm (default init like original CRATE)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.LayerNorm()(x)

        # CLS token for sequence pooling (torch.randn = normal(0, 1))
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1 + T, embed_dim)

        # Temporal positional embeddings (torch.randn = normal(0, 1))
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=1.0),
            (1, 1 + T, self.embed_dim)
        )
        x = x + pos_embed

        # Embedding dropout
        x = nn.Dropout(self.emb_dropout, deterministic=deterministic)(x)

        # CRATE transformer blocks (temporal attention)
        for _ in range(self.depth):
            x = CRATEBlock(
                dim=self.embed_dim,
                heads=self.num_heads,
                dim_head=dim_head,
                dropout=self.attn_dropout,
                ista_step_size=self.ista_step_size,
                ista_lambda=self.ista_lambda,
            )(x, deterministic=deterministic)

        # Pooling
        if self.pool == "mean":
            x = x[:, 1:, :].mean(axis=1)  # Mean pool over temporal tokens
        else:
            x = x[:, 0]  # CLS token

        # Final layer norm and projection (default init)
        x = nn.LayerNorm()(x)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return x


class Critic(nn.Module):
    """Value network with 2 hidden layers (default init like CRATE)."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        return nn.Dense(1)(x)


class Actor(nn.Module):
    """Continuous action actor with Gaussian distribution (default init like CRATE)."""
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        actor_mean = nn.Dense(self.action_dim)(x)
        actor_logstd = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,)
        )
        # Clamp log_std to prevent extreme values
        actor_logstd = jnp.clip(actor_logstd, self.log_std_min, self.log_std_max)
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
    """
    Hierarchical frame stacking buffer that combines:
    - Channel stacking (CNN-style, local temporal info)
    - Temporal tokens (transformer-style, global temporal info)

    Stores: (n_envs, total_frames, H, W, C)
    Returns: (n_envs, temporal_stack, H, W, C * channel_stack)

    Total frames = temporal_stack × channel_stack
    """
    frames: jnp.ndarray  # Shape: (n_envs, total_frames, H, W, C)
    temporal_stack: int = flax.struct.field(pytree_node=False)
    channel_stack: int = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, n_envs: int, temporal_stack: int, channel_stack: int, obs_shape: tuple):
        h, w, c = obs_shape
        total_frames = temporal_stack * channel_stack
        frames = jnp.zeros((n_envs, total_frames, h, w, c), dtype=jnp.uint8)
        return cls(frames=frames, temporal_stack=temporal_stack, channel_stack=channel_stack)

    def reset(self, obs: jnp.ndarray, done: jnp.ndarray = None):
        """Reset frame stack for environments that are done."""
        n_envs = obs.shape[0]
        total_frames = self.frames.shape[1]
        # Fill all frames with the same observation on reset
        new_frames = jnp.broadcast_to(
            obs[:, None, :, :, :],
            (n_envs, total_frames, obs.shape[1], obs.shape[2], obs.shape[3])
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
        """Add new frame, shifting out the oldest."""
        new_frames = jnp.concatenate([
            self.frames[:, 1:, :, :, :],
            obs[:, None, :, :, :]
        ], axis=1)
        return self.replace(frames=new_frames)

    def get_stacked(self) -> jnp.ndarray:
        """
        Return frames as hierarchical structure:
        (B, temporal_stack, H, W, C * channel_stack)

        Groups consecutive frames into channel-stacked tokens.
        Example with temporal_stack=4, channel_stack=2:
          frames [f0,f1,f2,f3,f4,f5,f6,f7] →
          token0: [f0,f1] stacked → (H, W, 6)
          token1: [f2,f3] stacked → (H, W, 6)
          token2: [f4,f5] stacked → (H, W, 6)
          token3: [f6,f7] stacked → (H, W, 6)
        """
        n_envs, total_frames, H, W, C = self.frames.shape
        # Reshape: (B, total_frames, H, W, C) → (B, temporal_stack, channel_stack, H, W, C)
        x = self.frames.reshape(n_envs, self.temporal_stack, self.channel_stack, H, W, C)
        # Reorder to stack channels: (B, temporal_stack, H, W, channel_stack, C)
        x = jnp.transpose(x, (0, 1, 3, 4, 2, 5))
        # Merge channel dimensions: (B, temporal_stack, H, W, channel_stack * C)
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
#  Environment and Augmentation
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


def random_shift(key: jax.random.PRNGKey, x: jnp.ndarray, pad: int = 4) -> jnp.ndarray:
    """
    DrQ-style random shift augmentation for hierarchical temporal frames.
    x: (B, T, H, W, C_stacked) where C_stacked = C * channel_stack
    """
    B, T, H, W, C_stacked = x.shape
    # Reshape to apply same shift to all tokens in sequence
    x_flat = x.reshape(B * T, H, W, C_stacked)

    x_padded = jnp.pad(x_flat, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Same crop for all tokens in a batch element
    key1, key2 = jax.random.split(key)
    crop_h = jax.random.randint(key1, (B,), 0, 2 * pad + 1)
    crop_w = jax.random.randint(key2, (B,), 0, 2 * pad + 1)

    # Repeat crops for each token
    crop_h = jnp.repeat(crop_h, T)
    crop_w = jnp.repeat(crop_w, T)

    def crop_single(x_pad, ch, cw):
        return jax.lax.dynamic_slice(x_pad, (ch, cw, 0), (H, W, C_stacked))

    x_cropped = jax.vmap(crop_single)(x_padded, crop_h, crop_w)
    return x_cropped.reshape(B, T, H, W, C_stacked)


# --------------------------------------------------------
#  Main Training Loop
# --------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.env_name}__temporal_crate__{args.seed}__{int(time.time())}"

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
    print("Temporal CRATE PPO")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")
    total_frames = args.temporal_stack * args.channel_stack
    print(f"\nHierarchical Temporal CRATE config:")
    print(f"  temporal_stack: {args.temporal_stack} (transformer tokens)")
    print(f"  channel_stack: {args.channel_stack} (frames per token)")
    print(f"  total_frames: {total_frames}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"  depth: {args.depth}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  ista_step_size: {args.ista_step_size}")
    print(f"  ista_lambda: {args.ista_lambda}")
    print(f"  pool: {args.pool}")
    print("=" * 60)

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset(reset_rng)
    raw_obs_shape = init_env_state.pixels.shape[1:]  # (H, W, C)
    H, W, C = raw_obs_shape

    # Hierarchical observation shape: (temporal_stack, H, W, C * channel_stack)
    C_stacked = C * args.channel_stack
    obs_shape = (args.temporal_stack, H, W, C_stacked)
    token_dim = H * W * C_stacked

    print(f"raw_obs_shape: {raw_obs_shape}")
    print(f"temporal_stack: {args.temporal_stack}")
    print(f"channel_stack: {args.channel_stack}")
    print(f"total_frames: {total_frames}")
    print(f"obs_shape (T, H, W, C_stacked): {obs_shape}")
    print(f"token_dim (H*W*C_stacked): {token_dim}")
    print(f"sequence length: {args.temporal_stack} tokens + 1 CLS = {args.temporal_stack + 1}")

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    # Initialize Hierarchical Temporal CRATE network
    network = TemporalCRATEEncoder(
        temporal_stack=args.temporal_stack,
        channel_stack=args.channel_stack,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        emb_dropout=args.emb_dropout,
        attn_dropout=args.attn_dropout,
        ista_step_size=args.ista_step_size,
        ista_lambda=args.ista_lambda,
        pool=args.pool,
    )
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    # Dummy input: (B, temporal_stack, H, W, C_stacked)
    dummy_obs = jnp.zeros((1,) + obs_shape)
    print(f"dummy_obs shape: {dummy_obs.shape}")

    network_params = network.init(network_key, dummy_obs, deterministic=True)
    dummy_hidden = network.apply(network_params, dummy_obs, deterministic=True)
    print(f"encoder output shape: {dummy_hidden.shape}")

    # Create optimizer (Adam)
    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
            eps=1e-5
        ),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze({
            'network': network_params,
            'actor': actor.init(actor_key, dummy_hidden),
            'critic': critic.init(critic_key, dummy_hidden),
        }),
        tx=tx,
    )

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
        """next_obs: (B, temporal_stack, H, W, C_stacked)"""
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
        """x: (B, temporal_stack, H, W, C_stacked)"""
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

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

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
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # Start training
    global_step = 0
    start_time = time.time()

    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    env_state = envs.reset(reset_rngs)

    # Initialize hierarchical frame stack (returns (B, T, H, W, C_stacked))
    frame_stack = HierarchicalFrameStack.create(
        args.n_envs, args.temporal_stack, args.channel_stack, raw_obs_shape
    )
    frame_stack = frame_stack.reset(env_state.pixels)
    next_obs = frame_stack.get_stacked()  # (B, temporal_stack, H, W, C * channel_stack)
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    def step_once(carry, step):
        agent_state, episode_stats, reward_norm, fs, env_state, obs, done, key = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        env_state = envs.step(env_state, action)
        raw_obs = env_state.pixels  # (B, H, W, C)
        raw_reward = env_state.reward
        next_done = env_state.done.astype(jnp.bool_)

        # Update frame stack
        fs = fs.push(raw_obs)
        fs = fs.reset(raw_obs, next_done)
        next_obs = fs.get_stacked()  # (B, temporal_stack, H, W, C * channel_stack)

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
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
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
                lr = agent_state.opt_state[1].hyperparams["learning_rate"].item()

                # Get log_std values from actor params
                log_std = jax.device_get(agent_state.params['actor']['params']['log_std'])
                log_std_mean = float(np.mean(log_std))
                log_std_min = float(np.min(log_std))
                log_std_max = float(np.max(log_std))
                std_mean = float(np.mean(np.exp(log_std)))

                wandb.log({
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
                }, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        wandb.finish()
