#!/usr/bin/env python
"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) for PixelBrax environments.
Implements TD3 with pixel observations using JAX/Flax.

Key differences from DDPG:
1. Twin Q-networks: Uses two critics and takes minimum for target computation
2. Delayed policy updates: Actor is updated less frequently than critics
3. Target policy smoothing: Noise added to target actions for regularization
"""

import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import wandb

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax
from pixelbrax.env_utils import make_pixel_brax


# --------------------------------------------------------
#  Shared encoder and Actor / Critic networks
# --------------------------------------------------------

class Encoder(nn.Module):
    """Shared CNN encoder for pixel observations with LayerNorm for stability."""

    @nn.compact
    def __call__(self, obs):
        # obs: (B, H, W, C) uint8 or float
        x = obs.astype(jnp.float32) / 255.0

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        return x  # (B, 256)


class Actor(nn.Module):
    """Actor network that takes encoder features as input."""
    action_dim: int
    max_action: float

    @nn.compact
    def __call__(self, features):
        # features: (B, 256) from shared encoder
        x = nn.Dense(256)(features)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        # TD3: continuous actions, tanh squashed and scaled
        return self.max_action * nn.tanh(x)


class Critic(nn.Module):
    """Single Q-network that takes encoder features and actions as input."""
    @nn.compact
    def __call__(self, features, actions):
        # features: (B, 256) from shared encoder, actions: (B, A)
        x = jnp.concatenate([features, actions], axis=-1)

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)  # (B,)


# --------------------------------------------------------
#  Replay buffer (host-side, numpy)
# --------------------------------------------------------

class ReplayBuffer:
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        capacity: int,
    ):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ):
        """
        obs, next_obs: (n_envs, *obs_shape)
        actions: (n_envs, action_dim)
        rewards, dones: (n_envs,)
        """
        n_envs = obs.shape[0]
        idxs = (self.ptr + np.arange(n_envs)) % self.capacity

        self.obs[idxs] = obs
        self.next_obs[idxs] = next_obs
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones

        self.ptr = (self.ptr + n_envs) % self.capacity
        self.size = min(self.size + n_envs, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs],
        )
        return batch


# --------------------------------------------------------
#  Config
# --------------------------------------------------------

@dataclass
class Config:
    env_name: str = "halfcheetah"
    backend: str = "spring"
    n_envs: int = 16
    hw: int = 84  # pixel height/width
    total_timesteps: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 256
    start_timesteps: int = 25_000
    exploration_noise: float = 0.1
    max_action: float = 1.0
    seed: int = 0
    train_freq: int = 1  # gradient steps per env step
    log_interval: int = 1000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    # TD3-specific parameters
    policy_noise: float = 0.2  # noise added to target policy
    noise_clip: float = 0.5  # range to clip target policy noise
    policy_frequency: int = 2  # delayed policy updates


# --------------------------------------------------------
#  PixelBrax env wrapper
# --------------------------------------------------------

def make_pixelbrax_envs(cfg: Config):
    """
    Create PixelBrax environments.
    """
    envs, _, _ = make_pixel_brax(
        backend=cfg.backend,
        env_name=cfg.env_name,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        hw=cfg.hw,
        distractor=None,
        video_path="datasets/DAVIS",
        video_set="train",
        return_float32=False,
    )

    try:
        action_dim = envs.action_size
    except AttributeError:
        action_dim = envs.env.action_size

    return envs, action_dim


# --------------------------------------------------------
#  JAX training pieces
# --------------------------------------------------------

def soft_update(target_params, online_params, tau):
    """Polyak averaging for target network updates."""
    return jax.tree_util.tree_map(
        lambda tp, p: tp * (1.0 - tau) + p * tau,
        target_params,
        online_params,
    )


@jax.jit
def td3_critic_step(
    encoder_state,
    actor_state,
    critic1_state,
    critic2_state,
    encoder_target_params,
    actor_target_params,
    critic1_target_params,
    critic2_target_params,
    batch,
    gamma: float,
    policy_noise: float,
    noise_clip: float,
    max_action: float,
    key: jax.random.PRNGKey,
):
    """
    TD3 critic update step with shared encoder.
    Uses clipped double Q-learning with target policy smoothing.
    Encoder is updated along with critics.
    """
    obs = batch["obs"]          # (B, H, W, C)
    actions = batch["actions"]  # (B, A)
    rewards = batch["rewards"]  # (B,)
    next_obs = batch["next_obs"]
    dones = batch["dones"]      # (B,)

    # Target policy smoothing: add clipped noise to target actions
    key, noise_key = jax.random.split(key)
    clipped_noise = (
        jnp.clip(
            jax.random.normal(noise_key, actions.shape) * policy_noise,
            -noise_clip,
            noise_clip,
        )
        * max_action
    )

    # Encode next_obs with target encoder for target Q computation
    next_features_target = encoder_state.apply_fn(
        {"params": encoder_target_params},
        next_obs,
    )

    # Get target actions from target actor and add smoothing noise
    next_actions = actor_state.apply_fn(
        {"params": actor_target_params},
        next_features_target,
    )
    next_actions = jnp.clip(next_actions + clipped_noise, -max_action, max_action)

    # Clipped double Q-learning: use minimum of two target Q-values
    next_q1 = critic1_state.apply_fn(
        {"params": critic1_target_params},
        next_features_target,
        next_actions,
    )
    next_q2 = critic2_state.apply_fn(
        {"params": critic2_target_params},
        next_features_target,
        next_actions,
    )
    next_q = jnp.minimum(next_q1, next_q2)
    target_q = rewards + gamma * (1.0 - dones) * next_q

    def combined_critic_loss_fn(encoder_params, critic1_params, critic2_params):
        # Encode observations with online encoder
        features = encoder_state.apply_fn({"params": encoder_params}, obs)

        # Compute Q-values
        q1_vals = critic1_state.apply_fn({"params": critic1_params}, features, actions)
        q2_vals = critic2_state.apply_fn({"params": critic2_params}, features, actions)

        # MSE loss for both critics
        critic1_loss = jnp.mean((q1_vals - target_q) ** 2)
        critic2_loss = jnp.mean((q2_vals - target_q) ** 2)

        return critic1_loss + critic2_loss, (critic1_loss, critic2_loss)

    # Compute gradients for encoder and both critics jointly
    (total_loss, (critic1_loss, critic2_loss)), grads = jax.value_and_grad(
        combined_critic_loss_fn, argnums=(0, 1, 2), has_aux=True
    )(encoder_state.params, critic1_state.params, critic2_state.params)

    encoder_grads, critic1_grads, critic2_grads = grads

    # Apply gradients
    encoder_state_new = encoder_state.apply_gradients(grads=encoder_grads)
    critic1_state_new = critic1_state.apply_gradients(grads=critic1_grads)
    critic2_state_new = critic2_state.apply_gradients(grads=critic2_grads)

    return (
        encoder_state_new,
        critic1_state_new,
        critic2_state_new,
        critic1_loss,
        critic2_loss,
        key,
    )


@jax.jit
def td3_actor_step(
    encoder_state,
    actor_state,
    critic1_state,
    critic2_state,
    encoder_target_params,
    actor_target_params,
    critic1_target_params,
    critic2_target_params,
    batch,
    tau: float,
):
    """
    TD3 actor update step (delayed) with shared encoder.
    Also performs soft update of all target networks.
    """
    obs = batch["obs"]

    # Encode observations (stop gradient through encoder for actor update)
    features = jax.lax.stop_gradient(
        encoder_state.apply_fn({"params": encoder_state.params}, obs)
    )

    def actor_loss_fn(actor_params):
        # Policy gradient: maximize Q(s, pi(s)) using critic1
        cur_actions = actor_state.apply_fn({"params": actor_params}, features)
        q_vals = critic1_state.apply_fn(
            {"params": critic1_state.params},
            features,
            cur_actions,
        )
        return -jnp.mean(q_vals)

    actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
    actor_state_new = actor_state.apply_gradients(grads=actor_grads)

    # Soft update all target networks (including encoder)
    encoder_target_params_new = soft_update(encoder_target_params, encoder_state.params, tau)
    actor_target_params_new = soft_update(actor_target_params, actor_state_new.params, tau)
    critic1_target_params_new = soft_update(critic1_target_params, critic1_state.params, tau)
    critic2_target_params_new = soft_update(critic2_target_params, critic2_state.params, tau)

    return (
        actor_state_new,
        encoder_target_params_new,
        actor_target_params_new,
        critic1_target_params_new,
        critic2_target_params_new,
        actor_loss,
    )


# --------------------------------------------------------
#  Main training loop
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="halfcheetah")
    parser.add_argument("--backend", type=str, default="spring")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--hw", type=int, default=84)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--start-timesteps", type=int, default=25_000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--max-action", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    # TD3-specific arguments
    parser.add_argument("--policy-noise", type=float, default=0.2,
                        help="Noise added to target policy during critic update")
    parser.add_argument("--noise-clip", type=float, default=0.5,
                        help="Range to clip target policy noise")
    parser.add_argument("--policy-frequency", type=int, default=2,
                        help="Frequency of delayed policy updates")
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()

    print("JAX devices:", jax.devices())
    print("env name:", args.env_name)
    print("total timesteps:", args.total_timesteps)
    print("batch size:", args.batch_size)
    print("start timesteps:", args.start_timesteps)
    print("exploration noise:", args.exploration_noise)
    print("max action:", args.max_action)
    print("seed:", args.seed)
    print("log interval:", args.log_interval)
    print("actor lr:", args.actor_lr)
    print("critic lr:", args.critic_lr)
    print("policy noise:", args.policy_noise)
    print("noise clip:", args.noise_clip)
    print("policy frequency:", args.policy_frequency)

    cfg = Config(
        env_name=args.env_name,
        backend=args.backend,
        n_envs=args.n_envs,
        hw=args.hw,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        start_timesteps=args.start_timesteps,
        exploration_noise=args.exploration_noise,
        max_action=args.max_action,
        seed=args.seed,
        log_interval=args.log_interval,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_frequency=args.policy_frequency,
    )

    if args.track:
        wandb.init(
            project="benchmark",
            config=vars(cfg),
            name=f"{cfg.env_name}-pixels-td3",
        )

    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # --- envs ---
    envs, action_dim = make_pixelbrax_envs(cfg)

    # Brax-style reset: state.pixels has shape (n_envs, H, W, C)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.n_envs)
    print("reset_keys:", reset_keys.shape, reset_keys.dtype)
    state = envs.reset(reset_keys)
    obs = np.array(state.pixels)  # move to host
    obs_shape = obs.shape[1:]
    print("obs shape at reset:", np.array(state.pixels).shape)

    # --- algo state ---
    encoder_module = Encoder()
    actor_module = Actor(action_dim=action_dim, max_action=cfg.max_action)
    critic1_module = Critic()
    critic2_module = Critic()

    key, encoder_key, actor_key, critic1_key, critic2_key = jax.random.split(key, 5)

    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
    dummy_features = jnp.zeros((1, 256), dtype=jnp.float32)  # Encoder output dim
    dummy_act = jnp.zeros((1, action_dim), dtype=jnp.float32)

    encoder_params = encoder_module.init(encoder_key, dummy_obs)["params"]
    actor_params = actor_module.init(actor_key, dummy_features)["params"]
    critic1_params = critic1_module.init(critic1_key, dummy_features, dummy_act)["params"]
    critic2_params = critic2_module.init(critic2_key, dummy_features, dummy_act)["params"]

    # Use same learning rate for encoder as critics (trained together)
    encoder_tx = optax.adam(cfg.critic_lr)
    actor_tx = optax.adam(cfg.actor_lr)
    critic_tx = optax.adam(cfg.critic_lr)

    encoder_state = train_state.TrainState.create(
        apply_fn=encoder_module.apply,
        params=encoder_params,
        tx=encoder_tx,
    )

    actor_state = train_state.TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_params,
        tx=actor_tx,
    )

    critic1_state = train_state.TrainState.create(
        apply_fn=critic1_module.apply,
        params=critic1_params,
        tx=critic_tx,
    )

    critic2_state = train_state.TrainState.create(
        apply_fn=critic2_module.apply,
        params=critic2_params,
        tx=critic_tx,
    )

    # Target networks start equal to online params
    encoder_target_params = encoder_state.params
    actor_target_params = actor_state.params
    critic1_target_params = critic1_state.params
    critic2_target_params = critic2_state.params

    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_dim=action_dim,
        capacity=cfg.buffer_size,
    )

    # Per-environment episode tracking
    episode_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    episode_lengths = np.zeros(cfg.n_envs, dtype=np.int32)
    # Store completed episode stats for logging
    completed_episode_returns = []
    completed_episode_lengths = []
    
    global_step = 0
    train_step = 0
    t0 = time.time()

    critic1_loss_val = 0.0
    critic2_loss_val = 0.0
    actor_loss_val = 0.0

    while global_step < cfg.total_timesteps:
        # --- act in env ---
        if global_step < cfg.start_timesteps:
            actions = np.random.uniform(
                low=-cfg.max_action,
                high=cfg.max_action,
                size=(cfg.n_envs, action_dim),
            ).astype(np.float32)
        else:
            obs_jax = jnp.array(obs)
            # Encode observations, then get actions
            features = encoder_state.apply_fn(
                {"params": encoder_state.params},
                obs_jax,
            )
            actions_jax = actor_state.apply_fn(
                {"params": actor_state.params},
                features,
            )
            actions = np.array(actions_jax)
            # Add exploration noise
            actions += cfg.exploration_noise * np.random.randn(*actions.shape)
            actions = np.clip(actions, -cfg.max_action, cfg.max_action)

        # Step env: Brax-style
        state = envs.step(state, actions)
        next_obs = np.array(state.pixels)
        rewards = np.array(state.reward)
        dones = np.array(state.done).astype(np.float32)

        # Track per-env episodic stats
        episode_rewards += rewards
        episode_lengths += 1
        
        # Check for completed episodes and record their stats
        done_mask = dones > 0.5
        if np.any(done_mask):
            completed_episode_returns.extend(episode_rewards[done_mask].tolist())
            completed_episode_lengths.extend(episode_lengths[done_mask].tolist())
            # Reset stats for environments that finished
            episode_rewards[done_mask] = 0.0
            episode_lengths[done_mask] = 0

        # Store transitions
        replay_buffer.add_batch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        obs = next_obs
        global_step += cfg.n_envs

        # --- training ---
        if replay_buffer.size >= cfg.batch_size and global_step >= cfg.start_timesteps:
            for _ in range(cfg.train_freq):
                batch = replay_buffer.sample(cfg.batch_size)
                batch = {k: jnp.array(v) for k, v in batch.items()}

                # Update encoder and critics
                key, subkey = jax.random.split(key)
                (
                    encoder_state,
                    critic1_state,
                    critic2_state,
                    critic1_loss_val,
                    critic2_loss_val,
                    key,
                ) = td3_critic_step(
                    encoder_state,
                    actor_state,
                    critic1_state,
                    critic2_state,
                    encoder_target_params,
                    actor_target_params,
                    critic1_target_params,
                    critic2_target_params,
                    batch,
                    gamma=cfg.gamma,
                    policy_noise=cfg.policy_noise,
                    noise_clip=cfg.noise_clip,
                    max_action=cfg.max_action,
                    key=subkey,
                )

                train_step += 1

                # Delayed policy updates
                if train_step % cfg.policy_frequency == 0:
                    (
                        actor_state,
                        encoder_target_params,
                        actor_target_params,
                        critic1_target_params,
                        critic2_target_params,
                        actor_loss_val,
                    ) = td3_actor_step(
                        encoder_state,
                        actor_state,
                        critic1_state,
                        critic2_state,
                        encoder_target_params,
                        actor_target_params,
                        critic1_target_params,
                        critic2_target_params,
                        batch,
                        tau=cfg.tau,
                    )

        # --- logging ---
        if global_step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = global_step / max(elapsed, 1e-6)

            # Use completed episode stats if available
            if len(completed_episode_returns) > 0:
                mean_ep_ret = np.mean(completed_episode_returns)
                mean_ep_len = np.mean(completed_episode_lengths)
                num_episodes = len(completed_episode_returns)
                # Clear completed stats for next logging window
                completed_episode_returns.clear()
                completed_episode_lengths.clear()
            else:
                # No episodes completed in this window
                mean_ep_ret = 0.0
                mean_ep_len = 0.0
                num_episodes = 0

            log_data = {
                "global_step": global_step,
                "charts/episodic_return": mean_ep_ret,
                "charts/episodic_length": mean_ep_len,
                "charts/num_episodes": num_episodes,
                "charts/steps_per_second": steps_per_sec,
                "charts/buffer_size": replay_buffer.size,
            }

            if args.track:
                wandb.log(log_data, step=global_step)

                wandb.log({
                    "losses/critic1_loss": float(critic1_loss_val),
                    "losses/critic2_loss": float(critic2_loss_val),
                    "losses/actor_loss": float(actor_loss_val),
                }, step=global_step)

            print(
                f"step={global_step} "
                f"return={mean_ep_ret:.1f} "
                f"len={mean_ep_len:.1f} "
                f"episodes={num_episodes} "
                f"sps={steps_per_sec:.0f}"
            )

    print("Training finished.")

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
