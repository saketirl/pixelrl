#!/usr/bin/env python
"""
PPO for Brax environments with state observations - CleanRL style adaptation.
Adapted from CleanRL's PPO implementation for continuous control with state observations.
This serves as a baseline to verify the algorithm works before testing pixel observations.
#To continue this session, run codex resume 019dac51-812f-7023-bfdd-553feaa73809
"""
import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

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
from optimizers import aurora, manifold_stiefel, manifold_stiefel_admm

# Import Brax from local source (pixelbrax/brax/brax)
import sys
sys.path.insert(0, "/users/apraka15/arjun/pixelrl/pixelbrax/brax")
from brax import envs as brax_envs
from pixelbrax.continual_dynamics import (
    apply_dynamics_task,
    continual_log_metrics,
    load_continual_dynamics_config,
    make_dynamics_task,
    resolve_schedule_seed,
    validate_continual_dynamics_config,
)
from spectrum.tracking import (
    add_activation_distributions,
    add_hessian_spectrum,
    add_weight_spectra,
    flatten_batch_tree,
    hessian_eigenvalues,
)

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
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Environment arguments
    env_name: str = "halfcheetah"
    """the name of the environment"""
    backend: str = "spring"
    """the physics backend (spring, generalized, positional)"""
    n_envs: int = 512
    """the number of parallel game environments"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    adam_eps: float = 1e-5
    """epsilon parameter for Adam"""
    heads_optimizer: str = "adam"
    """Optimizer for actor/critic matrix params: adam, stiefel, stiefel_admm, or aurora"""
    heads_stiefel_lr: float = 0.001
    """Learning rate for actor/critic matrix params when using Stiefel or Aurora"""
    stiefel_dual_lr: float = 0.01
    """Dual learning rate for manifold Stiefel"""
    stiefel_dual_steps: int = 5
    """Number of dual optimization steps for manifold Stiefel"""
    stiefel_msign_steps: int = 5
    """Number of matrix-sign iterations for manifold Stiefel"""
    actor_stiefel_max_grad_norm: float = 100.0
    """Maximum gradient norm for actor Stiefel params"""
    critic_stiefel_max_grad_norm: float = 1.0
    """Maximum gradient norm for critic Stiefel params"""
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
    actor_critic_activation: str = "swish"
    """Activation for all MLP hidden layers: swish or relu"""
    reward_normalize: bool = True
    """Normalize rewards using discounted-return RMS statistics"""

    # Action repeat
    action_repeat: int = 1
    """Number of times to repeat each action (frame skip)"""

    # Continual dynamics
    continual_dynamics_config: Optional[str] = None
    """Path to a YAML continual dynamics config"""
    spectrum_lanczos_order: int = 20
    """Lanczos order for task-boundary Hessian spectrum diagnostics"""
    spectrum_lanczos_draws: int = 1
    """Number of random Lanczos draws for task-boundary Hessian diagnostics"""
    spectrum_batch_size: int = 0
    """Batch size for spectrum diagnostics; 0 uses one PPO minibatch"""
    spectrum_hist_bins: int = 64
    """Number of histogram bins for spectrum and activation WandB payloads"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


class Network(nn.Module):
    """MLP encoder for state observations."""
    activation: str = "swish"
    
    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        # x: (B, obs_dim) - 1D state observations
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        x = activation(dense_1)
        if return_intermediates:
            return {
                "hidden": x,
                "Dense_0": hidden_0,
                "Dense_1": x,
                "Dense_0_pre": dense_0,
                "Dense_1_pre": dense_1,
            }
        return x


def actor_critic_activation_fn(name: str):
    if name == "swish":
        return nn.swish
    if name == "relu":
        return nn.relu
    raise ValueError(
        f"Unsupported actor_critic_activation={name!r}. Expected 'swish' or 'relu'."
    )


def resolve_heads_optimizer(heads_optimizer: str) -> str:
    optimizer = heads_optimizer.lower()
    if optimizer not in {"adam", "stiefel", "stiefel_admm", "aurora"}:
        raise ValueError(
            f"Unsupported heads_optimizer={heads_optimizer!r}. "
            "Expected one of: adam, stiefel, stiefel_admm, aurora."
        )
    return optimizer


def is_stiefel_matrix_param(param) -> bool:
    return param.ndim >= 2 and min(param.shape) > 1


def is_aurora_matrix_param(param) -> bool:
    return param.ndim == 2 and min(param.shape) > 1


def optimizer_label_for_param(path, param, heads_optimizer: str) -> str:
    top_level = path[0]
    if top_level == "network":
        return "network_adam"
    if top_level == "actor":
        if heads_optimizer == "stiefel" and is_stiefel_matrix_param(param):
            return "actor_stiefel"
        if heads_optimizer == "stiefel_admm" and is_stiefel_matrix_param(param):
            return "actor_stiefel_admm"
        if heads_optimizer == "aurora" and is_aurora_matrix_param(param):
            return "actor_aurora"
        return "heads_adam"
    if top_level == "critic":
        if heads_optimizer == "stiefel" and is_stiefel_matrix_param(param):
            return "critic_stiefel"
        if heads_optimizer == "stiefel_admm" and is_stiefel_matrix_param(param):
            return "critic_stiefel_admm"
        if heads_optimizer == "aurora" and is_aurora_matrix_param(param):
            return "critic_aurora"
        return "heads_adam"
    return "heads_adam"


def create_optimizer(
    learning_rate,
    *,
    adam_eps: float,
    max_grad_norm: float,
    heads_optimizer: str,
    heads_stiefel_lr: float,
    stiefel_dual_lr: float,
    stiefel_dual_steps: int,
    stiefel_msign_steps: int,
    actor_stiefel_max_grad_norm: float,
    critic_stiefel_max_grad_norm: float,
):
    heads_optimizer = resolve_heads_optimizer(heads_optimizer)

    def make_adam_tx():
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=learning_rate,
                eps=adam_eps,
            ),
        )

    actor_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        manifold_stiefel(
            learning_rate=heads_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            dual_steps=stiefel_dual_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    critic_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        manifold_stiefel(
            learning_rate=heads_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            dual_steps=stiefel_dual_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    actor_stiefel_admm_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        manifold_stiefel_admm(
            learning_rate=heads_stiefel_lr,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    critic_stiefel_admm_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        manifold_stiefel_admm(
            learning_rate=heads_stiefel_lr,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    actor_aurora_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        aurora(
            learning_rate=heads_stiefel_lr,
            adam_eps=adam_eps,
        ),
    )
    critic_aurora_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        aurora(
            learning_rate=heads_stiefel_lr,
            adam_eps=adam_eps,
        ),
    )

    transforms = {
        "network_adam": make_adam_tx(),
        "heads_adam": make_adam_tx(),
        "actor_stiefel": actor_stiefel_tx,
        "critic_stiefel": critic_stiefel_tx,
        "actor_stiefel_admm": actor_stiefel_admm_tx,
        "critic_stiefel_admm": critic_stiefel_admm_tx,
        "actor_aurora": actor_aurora_tx,
        "critic_aurora": critic_aurora_tx,
    }

    def label_fn(params):
        flat = flax.traverse_util.flatten_dict(params)
        labels = {
            path: optimizer_label_for_param(path, param, heads_optimizer)
            for path, param in flat.items()
        }
        return flax.core.freeze(flax.traverse_util.unflatten_dict(labels))

    return optax.multi_transform(transforms=transforms, param_labels=label_fn)


def count_optimizer_params(params, heads_optimizer: str) -> dict:
    flat = flax.traverse_util.flatten_dict(params)
    counts = {
        "network_adam": 0,
        "heads_adam": 0,
        "actor_stiefel": 0,
        "critic_stiefel": 0,
        "actor_stiefel_admm": 0,
        "critic_stiefel_admm": 0,
        "actor_aurora": 0,
        "critic_aurora": 0,
    }
    for path, param in flat.items():
        counts[optimizer_label_for_param(path, param, heads_optimizer)] += int(param.size)
    return counts


class Critic(nn.Module):
    """Value network with 2 hidden layers for sufficient capacity."""
    activation: str = "swish"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        hidden_1 = activation(dense_1)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(hidden_1)
        if return_intermediates:
            return {
                "value": value,
                "Dense_0": hidden_0,
                "Dense_1": hidden_1,
                "Dense_2": value,
            }
        return value


class Actor(nn.Module):
    """Continuous action actor with 2 hidden layers using Gaussian distribution."""
    action_dim: int
    activation: str = "swish"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        hidden_1 = activation(dense_1)
        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(hidden_1)
        actor_logstd = self.param(
            "log_std", 
            nn.initializers.zeros, 
            (self.action_dim,)
        )
        if return_intermediates:
            return {
                "actor_mean": actor_mean,
                "actor_logstd": actor_logstd,
                "Dense_0": hidden_0,
                "Dense_1": hidden_1,
                "Dense_2": actor_mean,
            }
        return actor_mean, actor_logstd


# Using FrozenDict for params instead of a dataclass for Flax compatibility
# Params structure: {'network': ..., 'actor': ..., 'critic': ...}


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
        # Update discounted returns: R_t = r_t + gamma * R_{t+1} * (1 - done)
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


def make_brax_envs(args):
    """Create Brax environments with state observations."""
    # Use brax.envs.create() which applies standard wrappers
    env = brax_envs.create(
        env_name=args.env_name,
        backend=args.backend,
        episode_length=1000,
        action_repeat=args.action_repeat,
        auto_reset=True,
        batch_size=args.n_envs,
    )
    
    action_dim = env.action_size
    obs_dim = env.observation_size
    
    return env, action_dim, obs_dim


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.actor_critic_activation not in {"swish", "relu"}:
        raise ValueError(
            "actor_critic_activation must be one of: swish, relu; "
            f"got {args.actor_critic_activation!r}"
        )
    args.heads_optimizer = resolve_heads_optimizer(args.heads_optimizer)
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
    print("JAX devices:", jax.devices())
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"adam_eps: {args.adam_eps}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")
    print(f"actor_critic_activation: {args.actor_critic_activation}")
    print(f"reward_normalize: {args.reward_normalize}")
    print(f"heads_optimizer: {args.heads_optimizer}")
    print(f"heads_stiefel_lr: {args.heads_stiefel_lr}")
    print(f"stiefel_dual_lr: {args.stiefel_dual_lr}")
    print(f"stiefel_dual_steps: {args.stiefel_dual_steps}")
    print(f"stiefel_msign_steps: {args.stiefel_msign_steps}")
    print(f"actor_stiefel_max_grad_norm: {args.actor_stiefel_max_grad_norm}")
    print(f"critic_stiefel_max_grad_norm: {args.critic_stiefel_max_grad_norm}")
    
    envs, action_dim, obs_dim = make_brax_envs(args)
    print(f"action_dim: {action_dim}")
    print(f"obs_dim: {obs_dim}")
    print(f"action_repeat: {args.action_repeat}")

    base_sys = envs.unwrapped.sys
    current_sys = base_sys
    continual_config = None
    current_task = None
    current_task_index = 0
    task_start_step = 0
    updates_per_task = 0
    schedule_seed = args.seed

    def env_with_sys(sys):
        envs.unwrapped.sys = sys
        return envs

    def reset_with_sys(sys, rng):
        return env_with_sys(sys).reset(rng)

    def step_with_sys(sys, state, action):
        return env_with_sys(sys).step(state, action)

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
    
    # Observation shape is 1D for state-based observations
    obs_shape = (obs_dim,)
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
    network = Network(activation=args.actor_critic_activation)
    actor = Actor(
        action_dim=action_dim,
        activation=args.actor_critic_activation,
    )
    critic = Critic(activation=args.actor_critic_activation)
    
    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)
    all_params = flax.core.freeze({
        "network": network_params,
        "actor": actor.init(actor_key, dummy_hidden),
        "critic": critic.init(critic_key, dummy_hidden),
    })
    optimizer_counts = count_optimizer_params(all_params, args.heads_optimizer)
    print("\nParameter breakdown by optimizer:")
    print(f"  network_adam: {optimizer_counts['network_adam']:,}")
    print(f"  heads_adam: {optimizer_counts['heads_adam']:,}")
    print(f"  actor_stiefel: {optimizer_counts['actor_stiefel']:,}")
    print(f"  critic_stiefel: {optimizer_counts['critic_stiefel']:,}")
    print(f"  actor_stiefel_admm: {optimizer_counts['actor_stiefel_admm']:,}")
    print(f"  critic_stiefel_admm: {optimizer_counts['critic_stiefel_admm']:,}")
    print(f"  actor_aurora: {optimizer_counts['actor_aurora']:,}")
    print(f"  critic_aurora: {optimizer_counts['critic_aurora']:,}")
    
    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=create_optimizer(
            linear_schedule if args.anneal_lr else args.learning_rate,
            adam_eps=args.adam_eps,
            max_grad_norm=args.max_grad_norm,
            heads_optimizer=args.heads_optimizer,
            heads_stiefel_lr=args.heads_stiefel_lr,
            stiefel_dual_lr=args.stiefel_dual_lr,
            stiefel_dual_steps=args.stiefel_dual_steps,
            stiefel_msign_steps=args.stiefel_msign_steps,
            actor_stiefel_max_grad_norm=args.actor_stiefel_max_grad_norm,
            critic_stiefel_max_grad_norm=args.critic_stiefel_max_grad_norm,
        ),
    )
    network_debug_apply = jax.jit(network.apply, static_argnames=("return_intermediates",))
    actor_debug_apply = jax.jit(actor.apply, static_argnames=("return_intermediates",))
    critic_debug_apply = jax.jit(critic.apply, static_argnames=("return_intermediates",))
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    spectrum_batch_size = args.spectrum_batch_size or args.minibatch_size

    def collect_spectrum_diagnostics(
        params,
        storage: Storage,
        key: jax.random.PRNGKey,
        prefix: str,
        task_index: int,
        global_step_value: int,
    ) -> dict:
        batch = flatten_batch_tree(storage, spectrum_batch_size)

        def spectrum_loss(loss_params, loss_batch):
            loss_value, _ = ppo_loss(
                loss_params,
                loss_batch.obs,
                loss_batch.actions,
                loss_batch.logprobs,
                loss_batch.advantages,
                loss_batch.returns,
                loss_batch.values,
            )
            return loss_value

        logs = {
            f"{prefix}/task_index": float(task_index),
            f"{prefix}/global_step": float(global_step_value),
            f"{prefix}/lanczos_order": float(args.spectrum_lanczos_order),
            f"{prefix}/lanczos_draws": float(args.spectrum_lanczos_draws),
        }
        eigvals = hessian_eigenvalues(
            spectrum_loss,
            params,
            batch,
            key,
            order=args.spectrum_lanczos_order,
            draws=args.spectrum_lanczos_draws,
        )
        add_hessian_spectrum(
            logs,
            f"{prefix}/hessian",
            eigvals,
            bins=args.spectrum_hist_bins,
        )
        add_weight_spectra(logs, params, prefix, bins=args.spectrum_hist_bins)

        net_debug = network_debug_apply(
            params["network"],
            batch.obs,
            return_intermediates=True,
        )
        actor_debug = actor_debug_apply(
            params["actor"],
            net_debug["hidden"],
            return_intermediates=True,
        )
        critic_debug = critic_debug_apply(
            params["critic"],
            net_debug["hidden"],
            return_intermediates=True,
        )
        activations = {
            **{f"network/{k}": v for k, v in net_debug.items() if k != "hidden"},
            **{f"actor/{k}": v for k, v in actor_debug.items() if k not in {"actor_logstd"}},
            **{f"critic/{k}": v for k, v in critic_debug.items()},
        }
        add_activation_distributions(
            logs,
            activations,
            prefix,
            bins=args.spectrum_hist_bins,
        )
        return logs

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
        value = critic.apply(agent_state.params['critic'], hidden)
        
        # Clip action
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

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values):
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
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                agent_state = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # Start the game
    global_step = 0
    start_time = time.time()

    # Reset environment
    key, reset_key = jax.random.split(key)
    env_state = reset_with_sys(current_sys, reset_key)
    
    # For state-based observations, directly use env_state.obs
    next_obs = env_state.obs
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    # Initialize reward normalizer (discounted return-based, like CleanRL)
    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    def step_once(current_sys, carry, step):
        agent_state, episode_stats, reward_norm, env_state, obs, done, key = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        # Step environment
        env_state = step_with_sys(current_sys, env_state, action)
        next_obs = env_state.obs  # State-based observation
        raw_reward = env_state.reward
        next_done = env_state.done.astype(jnp.bool_)  # Ensure bool type

        # Update reward statistics only when normalization is enabled.
        if args.reward_normalize:
            reward_norm = reward_norm.update(raw_reward, next_done.astype(jnp.float32))
            reward = reward_norm.normalize(raw_reward)
        else:
            reward = raw_reward

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
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,  # Use normalized reward for training
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (agent_state, episode_stats, reward_norm, env_state, next_obs, next_done, key), storage

    def rollout(
        current_sys,
        agent_state,
        episode_stats,
        reward_norm,
        env_state,
        next_obs,
        next_done,
        key,
        max_steps,
    ):
        (agent_state, episode_stats, reward_norm, env_state, next_obs, next_done, key), storage = jax.lax.scan(
            partial(step_once, current_sys),
            (agent_state, episode_stats, reward_norm, env_state, next_obs, next_done, key),
            jnp.arange(max_steps),
        )
        return agent_state, episode_stats, reward_norm, env_state, next_obs, next_done, storage, key

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("Starting training...")
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
            current_task_index = task_index
            current_sys = apply_dynamics_task(base_sys, current_task)
            task_start_step = global_step

            key, reset_key = jax.random.split(key)
            env_state = reset_with_sys(current_sys, reset_key)
            next_obs = env_state.obs
            next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)
            episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            )
            reward_normalizer = reward_normalizer.replace(
                discounted_return=jnp.zeros(args.n_envs),
            )

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
        agent_state, episode_stats, reward_normalizer, env_state, next_obs, next_done, storage, key = rollout(
            current_sys,
            agent_state,
            episode_stats,
            reward_normalizer,
            env_state,
            next_obs,
            next_done,
            key,
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        if (
            args.track
            and continual_config is not None
            and (iteration - 1) % updates_per_task == 0
        ):
            import wandb
            key, spectrum_key = jax.random.split(key)
            wandb.log(
                collect_spectrum_diagnostics(
                    agent_state.params,
                    storage,
                    spectrum_key,
                    "spectrum/task_start",
                    current_task_index,
                    global_step,
                ),
                step=global_step,
            )
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        if (
            args.track
            and continual_config is not None
            and (iteration % updates_per_task == 0 or iteration == args.num_updates)
        ):
            import wandb
            key, spectrum_key = jax.random.split(key)
            wandb.log(
                collect_spectrum_diagnostics(
                    agent_state.params,
                    storage,
                    spectrum_key,
                    "spectrum/task_end",
                    current_task_index,
                    global_step,
                ),
                step=global_step,
            )
        
        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))
            
            print(
                f"update={iteration} step={global_step} "
                f"ep_return={avg_episodic_return:.1f} "
                f"ep_len={avg_episodic_length * args.action_repeat:.0f} "  # Actual env steps
                f"loss={loss[-1, -1].item():.4f} "
                f"SPS={sps}"
            )
            
            if args.track:
                lr = float(linear_schedule(iteration * args.num_minibatches * args.update_epochs)) if args.anneal_lr else args.learning_rate
                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/learning_rate": lr,
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

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")
    
    if args.track:
        wandb.finish()
