#!/usr/bin/env python

import argparse
import time
from typing import NamedTuple, Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import wandb

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax  # pip install . in the pixelbrax repo
from pixelbrax.env_utils import make_pixel_brax


# --------------------------------------------------------
#  Actor/Critic network for pixel observations
# --------------------------------------------------------

class PixelActorCritic(nn.Module):
    """CNN-based Actor-Critic for pixel observations."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) uint8 or float
        x = x.astype(jnp.float32) / 255.0

        # CNN encoder (shared between actor and critic)
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        # Shared dense layer
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        # Actor head
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        # Critic head
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    # Episode tracking
    episode_return: jnp.ndarray  # cumulative return at this step
    episode_length: jnp.ndarray  # cumulative length at this step


# --------------------------------------------------------
#  PixelBrax env wrapper
# --------------------------------------------------------

def make_pixelbrax_envs(config):
    """
    Create PixelBrax environments.
    Returns envs and action_dim.
    """
    envs, _, _ = make_pixel_brax(
        backend=config["BACKEND"],
        env_name=config["ENV_NAME"],
        n_envs=config["NUM_ENVS"],
        seed=config["SEED"],
        hw=config["HW"],
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
#  Pure JAX PPO Training
# --------------------------------------------------------

def make_train(config, envs, action_dim, obs_shape):
    """Create the pure JAX training function."""

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    network = PixelActorCritic(action_dim=action_dim)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
        network_params = network.init(_rng, dummy_obs)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_state = envs.reset(reset_rng)
        obs = env_state.pixels  # (n_envs, H, W, C)

        # Initialize episode tracking
        episode_returns = jnp.zeros(config["NUM_ENVS"])
        episode_lengths = jnp.zeros(config["NUM_ENVS"])

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            train_state, env_state, last_obs, episode_returns, episode_lengths, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, episode_returns, episode_lengths, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Clip action
                action = jnp.clip(action, -config["MAX_ACTION"], config["MAX_ACTION"])

                # STEP ENV
                env_state = envs.step(env_state, action)
                next_obs = env_state.pixels
                reward = env_state.reward
                done = env_state.done

                # Update episode tracking
                episode_returns = episode_returns + reward
                episode_lengths = episode_lengths + 1

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs,
                    episode_returns, episode_lengths
                )

                # Reset episode stats where done
                episode_returns = jnp.where(done, 0.0, episode_returns)
                episode_lengths = jnp.where(done, 0, episode_lengths)

                runner_state = (train_state, env_state, next_obs, episode_returns, episode_lengths, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, episode_returns, episode_lengths, rng), None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, episode_returns, episode_lengths, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Compute metrics for logging
            total_loss = loss_info[0].mean()
            value_loss = loss_info[1][0].mean()
            actor_loss = loss_info[1][1].mean()
            entropy = loss_info[1][2].mean()

            # Compute global step
            global_step = (update_idx + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]

            # Extract completed episode returns
            # traj_batch.done: (num_steps, num_envs)
            # traj_batch.episode_return: (num_steps, num_envs) - return at moment of done
            done_mask = traj_batch.done  # episodes that completed
            completed_returns = jnp.where(done_mask, traj_batch.episode_return, jnp.nan)
            completed_lengths = jnp.where(done_mask, traj_batch.episode_length, jnp.nan)

            # Count completed episodes and compute mean (handle case with no completions)
            num_completed = done_mask.sum()
            mean_episode_return = jnp.nanmean(completed_returns)
            mean_episode_length = jnp.nanmean(completed_lengths)

            metric = {
                "global_step": global_step,
                "mean_episode_return": mean_episode_return,
                "mean_episode_length": mean_episode_length,
                "num_completed_episodes": num_completed,
                "mean_reward": traj_batch.reward.mean(),
                "mean_value": traj_batch.value.mean(),
                "total_loss": total_loss,
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "entropy": entropy,
                "update_idx": update_idx,
            }

            # Logging callback
            def callback(metric):
                update_idx = metric["update_idx"]
                if update_idx % config["LOG_INTERVAL"] == 0:
                    global_step = int(metric["global_step"])
                    mean_ep_return = float(metric["mean_episode_return"])
                    mean_ep_length = float(metric["mean_episode_length"])
                    num_completed = int(metric["num_completed_episodes"])

                    print(
                        f"update={update_idx} step={global_step} "
                        f"ep_return={mean_ep_return:.1f} "
                        f"ep_len={mean_ep_length:.0f} "
                        f"completed={num_completed} "
                        f"loss={float(metric['total_loss']):.4f}"
                    )

                    if config.get("TRACK"):
                        wandb.log({
                            "global_step": global_step,
                            "charts/episodic_return": mean_ep_return,
                            "charts/episodic_length": mean_ep_length,
                            "charts/num_completed_episodes": num_completed,
                            "charts/mean_reward": float(metric["mean_reward"]),
                            "charts/mean_value": float(metric["mean_value"]),
                            "losses/total_loss": float(metric["total_loss"]),
                            "losses/actor_loss": float(metric["actor_loss"]),
                            "losses/value_loss": float(metric["value_loss"]),
                            "losses/entropy": float(metric["entropy"]),
                        }, step=global_step)

            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, episode_returns, episode_lengths, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, episode_returns, episode_lengths, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="halfcheetah")
    parser.add_argument("--backend", type=str, default="spring")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--hw", type=int, default=84)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-action", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()

    print("JAX devices:", jax.devices())
    print('env name:', args.env_name)
    print('total timesteps:', args.total_timesteps)
    print('num steps per rollout:', args.num_steps)
    print('num envs:', args.n_envs)
    print('learning rate:', args.learning_rate)
    print('seed:', args.seed)

    config = {
        "ENV_NAME": args.env_name,
        "BACKEND": args.backend,
        "NUM_ENVS": args.n_envs,
        "HW": args.hw,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_STEPS": args.num_steps,
        "NUM_MINIBATCHES": args.num_minibatches,
        "UPDATE_EPOCHS": args.update_epochs,
        "LR": args.learning_rate,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "MAX_ACTION": args.max_action,
        "SEED": args.seed,
        "LOG_INTERVAL": args.log_interval,
        "ANNEAL_LR": args.anneal_lr,
        "TRACK": args.track,
    }

    # Compute derived values
    num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
    minibatch_size = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]

    print(f"num_updates: {num_updates}")
    print(f"minibatch_size: {minibatch_size}")

    if args.track:
        wandb.init(
            project="benchmark",
            config=config,
            name=f"{config['ENV_NAME']}-pixels-ppo",
        )

    np.random.seed(config["SEED"])

    # Create environments
    envs, action_dim = make_pixelbrax_envs(config)
    print(f"action_dim: {action_dim}")

    # Get observation shape from a test reset
    test_rng = jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
    test_state = envs.reset(test_rng)
    obs_shape = test_state.pixels.shape[1:]  # (H, W, C)
    print(f"obs_shape: {obs_shape}")

    # Create training function
    train_fn = make_train(config, envs, action_dim, obs_shape)

    # JIT compile
    print("JIT compiling training function...")
    t0 = time.time()
    train_jit = jax.jit(train_fn)

    # Run training
    rng = jax.random.PRNGKey(config["SEED"])
    print("Starting training...")
    out = train_jit(rng)

    # Block until done
    jax.block_until_ready(out)

    elapsed = time.time() - t0
    total_steps = config["TOTAL_TIMESTEPS"]
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {total_steps / elapsed:.0f}")

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
