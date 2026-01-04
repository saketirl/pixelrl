#!/usr/bin/env python
"""
PPO for PixelBrax environments with pixel observations.
Adapted from pure JAX PPO implementation to work with pixelbrax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import argparse
import time
import wandb

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax
from pixelbrax.env_utils import make_pixel_brax


class PixelActorCritic(nn.Module):
    """CNN-based Actor-Critic for pixel observations."""
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        # x: (B, H, W, C) uint8 or float - normalize to [0, 1]
        x = x.astype(jnp.float32) / 255.0
        
        # CNN encoder
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        
        x = x.reshape((x.shape[0], -1))  # flatten
        
        # Shared dense layer
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        
        # Actor head
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic head
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # Create pixelbrax environment
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

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = PixelActorCritic(
            action_dim, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        # Dummy obs for initialization: (1, H, W, C)
        init_x = jnp.zeros((1, config["HW"], config["HW"], 9))  # 3 frames * 3 channels
        network_params = network.init(_rng, init_x)
        
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
        obsv = env_state.pixels  # (NUM_ENVS, H, W, C)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                # Clip actions
                action = jnp.clip(action, -1.0, 1.0)

                # STEP ENV
                env_state = envs.step(env_state, action)
                obsv = env_state.pixels
                reward = env_state.reward
                done = env_state.done
                
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
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
                def _update_minbatch(train_state, batch_info):
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
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # Collect metrics
            metric = {
                "reward": traj_batch.reward,
                "done": traj_batch.done,
            }
            
            rng = update_state[-1]
            if config.get("DEBUG"):
                def callback(metric):
                    # Calculate episode returns from rewards and dones
                    rewards = metric["reward"]  # (NUM_STEPS, NUM_ENVS)
                    dones = metric["done"]
                    
                    # Sum rewards per step across all envs
                    mean_reward = jnp.mean(rewards)
                    num_dones = jnp.sum(dones)
                    
                    jax.debug.print(
                        "mean_step_reward={mean_reward}, num_episode_ends={num_dones}",
                        mean_reward=mean_reward,
                        num_dones=num_dones,
                    )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="halfcheetah")
    parser.add_argument("--backend", type=str, default="spring")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--hw", type=int, default=84)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anneal-lr", action="store_true", default=False)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track", action="store_true", default=False)
    args = parser.parse_args()
    
    print("JAX devices:", jax.devices())
    
    config = {
        "LR": args.lr,
        "NUM_ENVS": args.n_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": int(args.total_timesteps),
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ACTIVATION": args.activation,
        "ENV_NAME": args.env_name,
        "BACKEND": args.backend,
        "HW": args.hw,
        "SEED": args.seed,
        "ANNEAL_LR": args.anneal_lr,
        "DEBUG": args.debug,
    }
    
    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    if args.track:
        wandb.init(
            project="benchmark",
            config=config,
            name=f"{config['ENV_NAME']}-pixels-ppo",
        )
    
    rng = jax.random.PRNGKey(args.seed)
    
    t0 = time.time()
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    
    print("Compiling...")
    out = train_jit(rng)
    
    # Block until computation is done
    jax.block_until_ready(out)
    
    elapsed = time.time() - t0
    total_steps = config["TOTAL_TIMESTEPS"]
    print(f"Training finished in {elapsed:.1f}s")
    print(f"Steps per second: {total_steps / elapsed:.0f}")
    
    # Log final metrics
    if args.track:
        metrics = out["metrics"]
        # metrics["reward"] has shape (NUM_UPDATES, NUM_STEPS, NUM_ENVS)
        mean_rewards = jnp.mean(metrics["reward"], axis=(1, 2))  # (NUM_UPDATES,)
        
        for i, mean_rew in enumerate(mean_rewards):
            step = (i + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
            wandb.log({
                "charts/mean_step_reward": float(mean_rew),
                "global_step": step,
            })
        
        wandb.finish()
