#!/usr/bin/env python
"""
RePPO (Regularized PPO) for PixelBrax environments.
Combines PPO with SAC-style entropy regularization and KL constraints.

Key features:
1. Learnable entropy temperature (SAC-style)
2. KL constraint with Lagrangian multiplier
3. Target actor for KL divergence computation
4. Importance weighting for exploration
5. N-step lambda returns with soft rewards
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

# Fix OOM issues
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
    learning_rate: float = 3e-4
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
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    max_action: float = 1.0
    """maximum action value for clipping"""
    log_interval: int = 10
    """logging interval (in updates)"""

    # RePPO specific arguments
    ent_coef: float = 0.01
    """initial entropy coefficient (learnable)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    kl_coef: float = 0.1
    """initial KL constraint coefficient (Lagrangian, learnable)"""
    kl_target: float = 0.01
    """target KL divergence for constraint"""
    target_entropy_scale: float = -1.0
    """target entropy as multiple of action dim (negative means auto)"""
    polyak: float = 0.005
    """polyak averaging coefficient for target actor"""
    exploration_noise_min: float = 1.0
    """minimum exploration noise scale"""
    exploration_noise_max: float = 2.0
    """maximum exploration noise scale"""
    use_kl_constraint: bool = True
    """whether to use KL constraint"""
    use_entropy_constraint: bool = True
    """whether to use entropy constraint"""
    actor_min_std: float = 0.1
    """minimum std for actor distribution"""

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
        x = x.astype(jnp.float32) / 255.0

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
        x = nn.tanh(x)
        return x


class Critic(nn.Module):
    """Value network."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Continuous action actor with Gaussian distribution."""
    action_dim: int
    min_std: float = 0.1

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


class TemperatureParams(nn.Module):
    """Learnable temperature and Lagrangian parameters."""
    ent_init: float = 0.01
    kl_init: float = 0.1

    @nn.compact
    def __call__(self):
        log_temp = self.param("log_temperature", nn.initializers.constant(jnp.log(self.ent_init)), ())
        log_lagrangian = self.param("log_lagrangian", nn.initializers.constant(jnp.log(self.kl_init)), ())
        return jnp.exp(log_temp), jnp.exp(log_lagrangian)


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
    importance_weights: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


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
    )

    try:
        action_dim = envs.action_size
    except AttributeError:
        action_dim = envs.env.action_size

    return envs, action_dim


def soft_update(target_params, online_params, tau):
    """Polyak averaging for target network updates."""
    return jax.tree_util.tree_map(
        lambda t, o: t * (1.0 - tau) + o * tau,
        target_params,
        online_params,
    )


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
    key, network_key, actor_key, critic_key, temp_key = jax.random.split(key, 5)

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
    print(f"kl_target: {args.kl_target}")
    print(f"ent_coef (initial): {args.ent_coef}")
    print(f"kl_coef (initial): {args.kl_coef}")

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    # Compute target entropy
    if args.target_entropy_scale < 0:
        target_entropy = -action_dim  # Standard heuristic
    else:
        target_entropy = args.target_entropy_scale * action_dim
    print(f"target_entropy: {target_entropy}")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    init_env_state = envs.reset(reset_rng)
    obs_shape = init_env_state.pixels.shape[1:]
    print(f"obs_shape: {obs_shape}")

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
    actor = Actor(action_dim=action_dim, min_std=args.actor_min_std)
    critic = Critic()
    temp_params_module = TemperatureParams(ent_init=args.ent_coef, kl_init=args.kl_coef)

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)

    # Main agent state (network + actor + critic)
    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze({
            'network': network_params,
            'actor': actor.init(actor_key, dummy_hidden),
            'critic': critic.init(critic_key, dummy_hidden),
        }),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                eps=1e-5
            ),
        ),
    )

    # Temperature state (entropy temp + KL Lagrangian)
    temp_state = TrainState.create(
        apply_fn=temp_params_module.apply,
        params=temp_params_module.init(temp_key),
        tx=optax.adam(learning_rate=args.learning_rate * 0.1),
    )

    # Target actor params (for KL computation)
    target_actor_params = agent_state.params['actor']

    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
        exploration_scale: jnp.ndarray,
    ):
        """Sample action with exploration noise scaling."""
        hidden = network.apply(agent_state.params['network'], next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params['actor'], hidden)

        # Base std for policy
        base_std = jnp.exp(actor_logstd) + args.actor_min_std
        # Exploration std (scaled per environment)
        explore_std = base_std * exploration_scale

        # Sample with exploration noise
        key, subkey = jax.random.split(key)
        pi_explore = distrax.MultivariateNormalDiag(actor_mean, explore_std)
        action = pi_explore.sample(seed=subkey)

        # Compute log prob under base policy (for importance weighting)
        pi_base = distrax.MultivariateNormalDiag(actor_mean, base_std)
        base_logprob = pi_base.log_prob(action)
        explore_logprob = pi_explore.log_prob(action)

        # Importance weight: pi_base / pi_explore
        importance_weight = jnp.clip(base_logprob - explore_logprob, -2.0, 0.0)

        value = critic.apply(agent_state.params['critic'], hidden)

        # Clip action
        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, base_logprob, value.squeeze(-1), importance_weight, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        target_actor_params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob, entropy, and KL divergence."""
        hidden = network.apply(params['network'], x)
        actor_mean, actor_logstd = actor.apply(params['actor'], hidden)

        # Current policy
        std = jnp.exp(actor_logstd) + args.actor_min_std
        pi = distrax.MultivariateNormalDiag(actor_mean, std)

        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic.apply(params['critic'], hidden).squeeze(-1)

        # Target policy for KL computation
        target_mean, target_logstd = actor.apply(target_actor_params, hidden)
        target_std = jnp.exp(target_logstd) + args.actor_min_std
        pi_target = distrax.MultivariateNormalDiag(target_mean, target_std)

        # KL divergence: KL(pi_target || pi_current) - forward KL
        # This encourages the new policy to stay close to the target
        kl_div = pi_target.kl_divergence(pi)

        return logprob, entropy, value, kl_div

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward, importance_weight = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        # Apply importance weighting to lambda
        effective_lambda = gae_lambda * jnp.exp(importance_weight)
        advantages = delta + gamma * effective_lambda * nextnonterminal * advantages
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
            compute_gae_once, advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards, storage.importance_weights),
            reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def reppo_loss(params, target_actor_params, temp_params, x, a, logp, mb_advantages,
                   mb_returns, mb_values, mb_importance_weights):
        newlogprob, entropy, newvalue, kl_div = get_action_and_value2(
            params, target_actor_params, x, a
        )

        # Get temperature and Lagrangian
        temperature, lagrangian = temp_params_module.apply(temp_params)

        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss with clipping
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

        # Entropy loss (SAC-style with temperature)
        entropy_mean = entropy.mean()
        entropy_loss = -temperature * entropy_mean

        # KL constraint loss (with Lagrangian)
        kl_mean = kl_div.mean()
        if args.use_kl_constraint:
            kl_loss = lagrangian * kl_mean
        else:
            kl_loss = 0.0

        # Total loss
        loss = pg_loss + entropy_loss + v_loss * args.vf_coef + kl_loss

        return loss, (pg_loss, v_loss, entropy_mean, kl_mean, temperature, lagrangian,
                      jax.lax.stop_gradient(approx_kl))

    def temp_loss_fn(temp_params, entropy_mean, kl_mean):
        """Update temperature and Lagrangian based on constraints."""
        temperature, lagrangian = temp_params_module.apply(temp_params)

        # Entropy constraint: increase temp if entropy too low
        if args.use_entropy_constraint:
            entropy_target_loss = temperature * jax.lax.stop_gradient(entropy_mean - target_entropy)
        else:
            entropy_target_loss = 0.0

        # KL constraint: increase Lagrangian if KL too high
        if args.use_kl_constraint:
            kl_target_loss = -lagrangian * jax.lax.stop_gradient(kl_mean - args.kl_target)
        else:
            kl_target_loss = 0.0

        return entropy_target_loss + kl_target_loss

    reppo_loss_grad_fn = jax.value_and_grad(reppo_loss, has_aux=True)
    temp_loss_grad_fn = jax.value_and_grad(temp_loss_fn)

    @jax.jit
    def update_reppo(
        agent_state: TrainState,
        temp_state: TrainState,
        target_actor_params: flax.core.FrozenDict,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, temp_state, target_actor_params, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                agent_state, temp_state, target_actor_params = carry

                (loss, (pg_loss, v_loss, entropy_mean, kl_mean, temperature, lagrangian, approx_kl)), grads = reppo_loss_grad_fn(
                    agent_state.params,
                    target_actor_params,
                    temp_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    minibatch.importance_weights,
                )
                agent_state = agent_state.apply_gradients(grads=grads)

                # Update temperature params
                temp_loss, temp_grads = temp_loss_grad_fn(
                    temp_state.params, entropy_mean, kl_mean
                )
                temp_state = temp_state.apply_gradients(grads=temp_grads)

                return (agent_state, temp_state, target_actor_params), (
                    loss, pg_loss, v_loss, entropy_mean, kl_mean, temperature, lagrangian, approx_kl
                )

            (agent_state, temp_state, target_actor_params), metrics = jax.lax.scan(
                update_minibatch, (agent_state, temp_state, target_actor_params), shuffled_storage
            )

            # Soft update target actor
            target_actor_params = soft_update(
                target_actor_params, agent_state.params['actor'], args.polyak
            )

            return (agent_state, temp_state, target_actor_params, key), metrics

        (agent_state, temp_state, target_actor_params, key), all_metrics = jax.lax.scan(
            update_epoch, (agent_state, temp_state, target_actor_params, key), (), length=args.update_epochs
        )

        # Extract final metrics
        loss, pg_loss, v_loss, entropy, kl, temperature, lagrangian, approx_kl = all_metrics

        return agent_state, temp_state, target_actor_params, loss, pg_loss, v_loss, entropy, kl, temperature, lagrangian, approx_kl, key

    # Compute exploration noise scales per environment
    exploration_scales = jnp.linspace(
        args.exploration_noise_min,
        args.exploration_noise_max,
        args.n_envs
    )[:, None]  # (n_envs, 1)

    # Start the game
    global_step = 0
    start_time = time.time()

    # Reset environment
    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    env_state = envs.reset(reset_rngs)
    next_obs = env_state.pixels
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    def step_once(carry, step):
        agent_state, episode_stats, env_state, obs, done, key, exploration_scales = carry
        action, logprob, value, importance_weight, key = get_action_and_value(
            agent_state, obs, key, exploration_scales
        )

        # Step environment
        env_state = envs.step(env_state, action)
        next_obs = env_state.pixels
        reward = env_state.reward
        next_done = env_state.done.astype(jnp.bool_)

        # Update episode statistics
        new_episode_return = episode_stats.episode_returns + reward
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
            importance_weights=importance_weight,
        )
        return (agent_state, episode_stats, env_state, next_obs, next_done, key, exploration_scales), storage

    def rollout(agent_state, episode_stats, env_state, next_obs, next_done, key, exploration_scales, max_steps):
        (agent_state, episode_stats, env_state, next_obs, next_done, key, _), storage = jax.lax.scan(
            step_once, (agent_state, episode_stats, env_state, next_obs, next_done, key, exploration_scales), jnp.arange(max_steps)
        )
        return agent_state, episode_stats, env_state, next_obs, next_done, storage, key

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("Starting training...")
    for iteration in range(1, args.num_updates + 1):
        iteration_time_start = time.time()
        agent_state, episode_stats, env_state, next_obs, next_done, storage, key = rollout(
            agent_state, episode_stats, env_state, next_obs, next_done, key, exploration_scales
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, temp_state, target_actor_params, loss, pg_loss, v_loss, entropy, kl, temperature, lagrangian, approx_kl, key = update_reppo(
            agent_state,
            temp_state,
            target_actor_params,
            storage,
            key,
        )

        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))

            print(
                f"update={iteration} step={global_step} "
                f"ep_return={avg_episodic_return:.1f} "
                f"ep_len={avg_episodic_length:.0f} "
                f"loss={loss[-1, -1].item():.4f} "
                f"temp={temperature[-1, -1].item():.4f} "
                f"kl={kl[-1, -1].item():.4f} "
                f"SPS={sps}"
            )

            if args.track:
                import wandb
                lr = agent_state.opt_state[1].hyperparams["learning_rate"].item()
                wandb.log({
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length,
                    "charts/learning_rate": lr,
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy[-1, -1].item(),
                    "losses/kl_divergence": kl[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                    "reppo/temperature": temperature[-1, -1].item(),
                    "reppo/lagrangian": lagrangian[-1, -1].item(),
                }, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.track:
        import wandb
        wandb.finish()
