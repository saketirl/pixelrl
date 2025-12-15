#!/usr/bin/env python

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
from dataclasses import asdict





import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax  # pip install . in the pixelbrax repo
from pixelbrax.env_utils import make_pixel_brax



# --------------------------------------------------------
#  Actor / Critic networks for pixel observations
# --------------------------------------------------------

class PixelEncoder(nn.Module):
    """Simple CNN like Atari-style encoders."""
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C) uint8 or float
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)
        return x


class Actor(nn.Module):
    action_dim: int
    max_action: float

    @nn.compact
    def __call__(self, obs):
        # obs: (B, H, W, C) uint8 or float
        x = obs.astype(jnp.float32) / 255.0

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        # DDPG: continuous actions, tanh squashed and scaled
        return self.max_action * nn.tanh(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs, actions):
        # obs: (B, H, W, C), actions: (B, A)
        x = obs.astype(jnp.float32) / 255.0

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, actions], axis=-1)

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
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 256
    start_timesteps: int = 25_000
    exploration_noise: float = 0.1
    max_action: float = 1.0
    seed: int = 0
    train_freq: int = 1  # gradient steps per env step
    policy_delay: int = 1  # DDPG: 1, TD3-style delay would be >1
    log_interval: int = 1000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3



# --------------------------------------------------------
#  PixelBrax env wrapper (you may need to tweak this)
# --------------------------------------------------------

def make_pixelbrax_envs(cfg: Config):
    """
    PixelBrax README shows:

        envs, _, _ = pixelbrax.make(
            backend="spring",
            env_name="halfcheetah",
            n_envs=100,
            seed=0,
            hw=84,
            distractor=None,
            video_path="datasets/DAVIS",
            video_set="train",
            return_float32=False,
        )

    This helper assumes envs follows the usual Brax API:
      - state = envs.reset(rng)
      - state = envs.step(state, actions)
      - state.obs, state.reward, state.done

    You might need to adjust names if PixelBrax changed them.
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

    # Guessing Brax-style attributes
    try:
        action_dim = envs.action_size
    except AttributeError:
        # fall back: some brax env wrappers attach .action_size on .env
        action_dim = envs.env.action_size

    return envs, action_dim


# --------------------------------------------------------
#  JAX training pieces
# --------------------------------------------------------

def create_train_states(
    rng: jax.random.PRNGKey,
    obs_shape: Tuple[int, ...],
    action_dim: int,
    cfg: Config,
):
    dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

    actor = Actor(action_dim=action_dim, max_action=cfg.max_action)
    critic = Critic(action_dim=action_dim)

    rng_actor, rng_critic = jax.random.split(rng)

    actor_params = actor.init(rng_actor, dummy_obs)
    critic_params = critic.init(rng_critic, dummy_obs, dummy_action)

    actor_tx = optax.adam(cfg.learning_rate)
    critic_tx = optax.adam(cfg.learning_rate)

    actor_state = TrainState.create(
        apply_fn=actor.apply, params=actor_params, tx=actor_tx
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply, params=critic_params, tx=critic_tx
    )

    # target networks start equal to main networks
    actor_target_params = actor_params
    critic_target_params = critic_params

    return actor_state, critic_state, actor_target_params, critic_target_params


def soft_update(target_params, online_params, tau):
    return jax.tree.map(lambda t, s: t * (1.0 - tau) + s * tau, target_params, online_params)


@jax.jit
def ddpg_train_step(
    actor_state,
    critic_state,
    actor_target_params,
    critic_target_params,
    batch,
    gamma: float,
    tau: float,
):
    obs = batch["obs"]          # (B, H, W, C)
    actions = batch["actions"]  # (B, A)
    rewards = batch["rewards"]  # (B,)
    next_obs = batch["next_obs"]
    dones = batch["dones"]      # (B,)

    def critic_loss_fn(critic_params):
        # current Q(s,a)
        q_vals = critic_state.apply_fn(
            {"params": critic_params},
            obs,
            actions,
        )

        # target Q(s', pi_target(s'))
        next_actions = actor_state.apply_fn(
            {"params": actor_target_params},
            next_obs,
        )
        next_q_vals = critic_state.apply_fn(
            {"params": critic_target_params},
            next_obs,
            next_actions,
        )

        target_q = rewards + gamma * (1.0 - dones) * next_q_vals
        loss = jnp.mean((q_vals - target_q) ** 2)
        return loss

    critic_grads = jax.grad(critic_loss_fn)(critic_state.params)
    critic_state_new = critic_state.apply_gradients(grads=critic_grads)
    critic_loss_val = critic_loss_fn(critic_state_new.params)

    def actor_loss_fn(actor_params):
        # policy gradient: maximize Q(s, pi(s)) == minimize -Q
        cur_actions = actor_state.apply_fn(
            {"params": actor_params},
            obs,
        )
        q_vals = critic_state_new.apply_fn(
            {"params": critic_state_new.params},
            obs,
            cur_actions,
        )
        return -jnp.mean(q_vals)

    actor_grads = jax.grad(actor_loss_fn)(actor_state.params)
    actor_state_new = actor_state.apply_gradients(grads=actor_grads)
    actor_loss_val = actor_loss_fn(actor_state_new.params)

    # soft update targets
    def soft_update(target_params, online_params):
        return jax.tree_util.tree_map(
            lambda tp, p: tp * (1.0 - tau) + p * tau,
            target_params,
            online_params,
        )

    actor_target_params_new = soft_update(actor_target_params, actor_state_new.params)
    critic_target_params_new = soft_update(
        critic_target_params, critic_state_new.params
    )

    return (
        actor_state_new,
        critic_state_new,
        actor_target_params_new,
        critic_target_params_new,
        critic_loss_val,
        actor_loss_val,
    )




@jax.jit
def policy_apply(actor_params, apply_fn, obs):
    return apply_fn(actor_params, obs)


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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
    parser.add_argument("--track", type=bool, default=False)
    args = parser.parse_args()

    print("JAX devices:", jax.devices())

    cfg = Config(
        env_name=args.env_name,
        backend=args.backend,
        n_envs=args.n_envs,
        hw=args.hw,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
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
        critic_lr=args.critic_lr
    )
    if args.track:
        wandb.init(
            project="benchmark",
            config=vars(cfg),
            name=f"{cfg.env_name}-pixels-ddpg",
        )

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )


    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # --- envs ---
    envs, action_dim = make_pixelbrax_envs(cfg)

    # Brax-style reset: state.pixels has shape (n_envs, H, W, C)
    key = jax.random.PRNGKey(cfg.seed)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.n_envs)
    print("reset_keys:", reset_keys.shape, reset_keys.dtype)
    state = envs.reset(reset_keys)
    obs = np.array(state.pixels)  # move to host
    obs_shape = obs.shape[1:]
    print("obs shape at reset:", np.array(state.pixels).shape)

    # --- algo state ---
    key, subkey = jax.random.split(key)
    actor_module = Actor(action_dim=action_dim, max_action=cfg.max_action)
    critic_module = Critic()

    key, actor_key, critic_key = jax.random.split(key, 3)

    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
    dummy_act = jnp.zeros((1, action_dim), dtype=jnp.float32)

    actor_params = actor_module.init(actor_key, dummy_obs)["params"]
    critic_params = critic_module.init(critic_key, dummy_obs, dummy_act)["params"]

    actor_tx = optax.adam(cfg.actor_lr)
    critic_tx = optax.adam(cfg.critic_lr)

    actor_state = train_state.TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_params,
        tx=actor_tx,
    )

    critic_state = train_state.TrainState.create(
        apply_fn=critic_module.apply,
        params=critic_params,
        tx=critic_tx,
    )

    # target networks start equal to online params
    actor_target_params = actor_state.params
    critic_target_params = critic_state.params

    replay_buffer = ReplayBuffer(obs_shape=obs_shape, action_dim=action_dim, capacity=cfg.buffer_size)

    episode_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    episode_lengths = np.zeros(cfg.n_envs, dtype=np.int32)
    global_step = 0
    t0 = time.time()

    critic_loss_val = 0.0
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
            obs_jax = jnp.array(obs)  # obs is state.pixels, (n_envs,H,W,9)
            actions_jax = actor_state.apply_fn(
                {"params": actor_state.params},
                obs_jax,
            )
            actions = np.array(actions_jax)
            actions += cfg.exploration_noise * np.random.randn(*actions.shape)
            actions = np.clip(actions, -cfg.max_action, cfg.max_action)

        # step env: Brax-style
        state = envs.step(state, actions)
        next_obs = np.array(state.pixels)
        rewards = np.array(state.reward)  # (n_envs,)
        dones = np.array(state.done).astype(np.float32)

        # log per-env episodic stats (simple)
        episode_rewards += rewards
        episode_lengths += 1

        # store transitions
        replay_buffer.add_batch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        obs = next_obs
        global_step += cfg.n_envs

        # reset finished envs if PixelBrax/Brax doesn't auto-reset them
        # If envs already does auto-reset, you can delete this block.
        if dones.any():
            # for Brax, envs.step usually already resets where done==True.
            # If not, you'd need to manually call reset for those env indices.
            pass

        # --- training ---
        if replay_buffer.size >= cfg.batch_size and global_step >= cfg.start_timesteps:
            for _ in range(cfg.train_freq):
                batch = replay_buffer.sample(cfg.batch_size)
                # move to device
                batch = {k: jnp.array(v) for k, v in batch.items()}
                (
                    actor_state,
                    critic_state,
                    actor_target_params,
                    critic_target_params,
                    critic_loss_val,
                    actor_loss_val,
                ) = ddpg_train_step(
                    actor_state,
                    critic_state,
                    actor_target_params,
                    critic_target_params,
                    batch,
                    gamma=cfg.gamma,
                    tau=cfg.tau,
                )

        # --- logging ---
        if global_step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = global_step / max(elapsed, 1e-6)

            mean_ep_ret = np.mean(episode_rewards)
            mean_ep_len = np.mean(episode_lengths)

            log_data = {
                "global_step": global_step,
                "charts/episodic_return": mean_ep_ret,
                "charts/episodic_length": mean_ep_len,
                "charts/steps_per_second": steps_per_sec,
                "charts/buffer_size": replay_buffer.size,
            }

            wandb.log(log_data, step=global_step)

            wandb.log({
                "losses/critic_loss": float(critic_loss_val),
                "losses/actor_loss": float(actor_loss_val),
            }, step=global_step)

            print(
                f"step={global_step} "
                f"return={mean_ep_ret:.1f} "
                f"len={mean_ep_len:.1f} "
                f"sps={steps_per_sec:.0f}"
            )

            # reset episode stats (rolling summary)
            episode_rewards[:] = 0.0
            episode_lengths[:] = 0

    print("Training finished.")


if __name__ == "__main__":
    main()
