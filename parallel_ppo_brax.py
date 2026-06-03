#!/usr/bin/env python
"""Parallel independent-agent PPO for Brax state observations.

This trainer keeps each agent equivalent to an independent ppo_brax.py run while
batching many agents with vmap so a GPU can see enough work. The agent axis is
separate from n_envs: n_envs remains the per-agent environment count.
"""

import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tyro
import sys
sys.path.insert(0, "/users/apraka15/arjun/pixelrl/pixelbrax/brax")
sys.path.insert(0, "/home/guests/arjun/pixelrl/pixelbrax/brax")
from brax import envs as brax_envs
from brax.envs.wrappers.training import EpisodeWrapper
from flax.training.train_state import TrainState

from configs.continual.slippery_ant_wrapper import (
    DEFAULT_SLIPPERY_FRICTIONS_CSV,
    AutoResetTimeStepWrapper,
    load_seeded_friction_schedule_table,
)
from ppo_brax import (
    Actor,
    Args,
    CRATEActor,
    CRATECritic,
    CRATENetwork,
    Critic,
    EpisodeStatistics,
    IdentityNetwork,
    LopActor,
    LopCritic,
    Network,
    ObsNormalizer,
    RewardNormalizer,
    SlipperyRolloutInfo,
    Storage,
    SUPPORTED_SLIPPERY_ENVS,
    count_optimizer_params,
    create_optimizer,
    latest_slippery_metrics,
    matrix_constraint_metrics,
    online_stiefel_constraint_metrics,
    resolve_base_optimizer,
    resolve_heads_optimizer,
    slippery_num_seeded_phases,
    slippery_phase_from_timestep,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("TF_XLA_FLAGS", "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


@dataclass
class ParallelArgs(Args):
    num_agents: int = 16
    """Number of independent PPO agents to batch with vmap."""
    agent_seed_stride: int = 1
    """Stride between per-agent training seeds."""
    slippery_schedule_seed_mode: Literal["agent", "fixed", "list"] = "agent"
    """How to assign slippery friction schedule rows across agents."""
    slippery_schedule_seeds: Optional[str] = None
    """Comma-separated schedule rows when slippery_schedule_seed_mode='list'."""
    max_logged_agents: int = 32
    """Maximum number of per-agent metric namespaces to log to W&B."""
    max_printed_agents: int = 32
    """Maximum number of per-agent values to print in console logs."""
    slippery_start_at_schedule: bool = False
    """If true, phase 0 uses the first CSV friction value instead of default Brax friction."""
    clip_actions: bool = True
    """If true, clip sampled continuous actions to [-max_action, max_action] before env step."""


class RegularSingleBraxEnv:
    def __init__(self, env_name: str, backend: str, action_repeat: int):
        self._env = brax_envs.create(
            env_name=env_name,
            backend=backend,
            episode_length=1000,
            action_repeat=action_repeat,
            auto_reset=True,
        )
        self.action_size = self._env.action_size
        self.observation_size = self._env.observation_size

    def reset(self, key):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, schedule_seed=None):
        del key, schedule_seed
        next_state = self._env.step(state, action)
        info = {
            "friction": jnp.zeros((3, 3), dtype=jnp.float32),
            "timestep": jnp.asarray(0, dtype=jnp.int32),
        }
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, info


class DynamicSlipperySingleBraxEnv:
    def __init__(
        self,
        env_name: str,
        backend: str,
        change_every: int,
        action_repeat: int,
        start_at_schedule: bool = False,
    ):
        self.change_every = int(change_every)
        self.action_repeat = int(action_repeat)
        self.start_at_schedule = bool(start_at_schedule)
        table = load_seeded_friction_schedule_table(str(DEFAULT_SLIPPERY_FRICTIONS_CSV.resolve()))
        self.friction_table = jnp.asarray(table, dtype=jnp.float32)
        self.num_schedule_rows = int(table.shape[0])
        self.num_phases = int(table.shape[1])

        env = brax_envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=action_repeat)
        env = AutoResetTimeStepWrapper(env, action_repeat=action_repeat)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = env.observation_size

    def env_fn(self, sys):
        env = self._env
        env.unwrapped.sys = sys
        return env

    def reset(self, key):
        state = self._env.reset(key)
        sys = self._env.unwrapped.sys
        default_geom_friction = jnp.array(sys.geom_friction)
        state.info["default_geom_friction"] = default_geom_friction
        state.info["sys_variation"] = {"geom_friction": default_geom_friction}
        return state.obs, state

    def step(self, key, state, action, schedule_seed):
        del key
        sys = self._env.unwrapped.sys
        default_geom_friction = state.info["default_geom_friction"]
        if self.start_at_schedule:
            phase = (state.info["timestep"] + self.action_repeat) // self.change_every
            schedule_phase = jnp.minimum(phase, self.num_phases - 1)
            friction = self.friction_table[schedule_seed, schedule_phase]
            new_geom_friction = default_geom_friction.at[:, 0].set(friction)
        else:
            phase = state.info["timestep"] // self.change_every
            schedule_phase = jnp.maximum(phase - 1, 0)
            schedule_phase = jnp.minimum(schedule_phase, self.num_phases - 1)
            friction = self.friction_table[schedule_seed, schedule_phase]
            scheduled_geom_friction = default_geom_friction.at[:, 0].set(friction)
            new_geom_friction = jax.lax.select(phase > 0, scheduled_geom_friction, default_geom_friction)
        variation = {"geom_friction": new_geom_friction}
        next_state = self.env_fn(sys.replace(**variation)).step(state, action)
        next_state.info["default_geom_friction"] = default_geom_friction
        next_state.info["sys_variation"] = variation
        info = {
            "friction": next_state.info["sys_variation"]["geom_friction"],
            "timestep": next_state.info["timestep"],
        }
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, info


def parse_schedule_seeds(args: ParallelArgs) -> np.ndarray:
    table = load_seeded_friction_schedule_table(str(DEFAULT_SLIPPERY_FRICTIONS_CSV))
    max_rows = table.shape[0]
    agent_seeds = args.seed + np.arange(args.num_agents, dtype=np.int32) * args.agent_seed_stride

    if args.slippery_schedule_seed_mode == "agent":
        schedule_seeds = agent_seeds.copy()
    elif args.slippery_schedule_seed_mode == "fixed":
        if args.slippery_schedule_seed is None:
            raise ValueError("--slippery-schedule-seed is required when mode='fixed'.")
        schedule_seeds = np.full(args.num_agents, args.slippery_schedule_seed, dtype=np.int32)
    elif args.slippery_schedule_seed_mode == "list":
        if not args.slippery_schedule_seeds:
            raise ValueError("--slippery-schedule-seeds is required when mode='list'.")
        schedule_seeds = np.asarray(
            [int(x.strip()) for x in args.slippery_schedule_seeds.split(",") if x.strip()],
            dtype=np.int32,
        )
        if schedule_seeds.shape[0] != args.num_agents:
            raise ValueError(
                f"Expected {args.num_agents} schedule seeds, got {schedule_seeds.shape[0]}."
            )
    else:
        raise ValueError(f"Unsupported slippery_schedule_seed_mode={args.slippery_schedule_seed_mode!r}.")

    bad = schedule_seeds[(schedule_seeds < 0) | (schedule_seeds >= max_rows)]
    if bad.size:
        raise ValueError(
            f"Slippery schedule seeds out of range [0, {max_rows - 1}]: {bad.tolist()}."
        )
    return schedule_seeds


def compute_gae_once(carry, inp, gamma, gae_lambda):
    advantages = carry
    nextdone, nextvalues, curvalues, reward = inp
    nextnonterminal = 1.0 - nextdone
    delta = reward + gamma * nextvalues * nextnonterminal - curvalues
    advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
    return advantages, advantages


def stack_replicas(tree, count: int):
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * count), tree)


def slice_tree(tree, index: int):
    return jax.tree_util.tree_map(lambda x: x[index], tree)


def get_agent_slippery_metrics(rollout_info: SlipperyRolloutInfo, args: ParallelArgs) -> list[dict[str, float]]:
    friction = np.asarray(jax.device_get(rollout_info.friction_slide))
    timestep = np.asarray(jax.device_get(rollout_info.timestep))
    num_seeded_phases = slippery_num_seeded_phases()
    metrics = []
    for agent_id in range(friction.shape[0]):
        first_timestep = int(timestep[agent_id, -1, 0])
        metrics.append({
            "slippery/friction_slide_mean": float(np.mean(friction[agent_id, -1])),
            "slippery/friction_slide_first_env": float(friction[agent_id, -1, 0]),
            "slippery/timestep_first_env": first_timestep,
            "slippery/phase_first_env": slippery_phase_from_timestep(
                timestep=first_timestep,
                change_every=args.slippery_change_every,
                num_seeded_phases=num_seeded_phases,
            ),
            "slippery/change_every": int(args.slippery_change_every),
        })
    return metrics


def main():
    args = tyro.cli(ParallelArgs)
    if args.num_agents <= 0:
        raise ValueError("--num-agents must be positive.")
    if args.actor_critic_activation not in {"swish", "relu"}:
        raise ValueError("actor_critic_activation must be one of: swish, relu.")
    if args.action_repeat <= 0:
        raise ValueError("--action-repeat must be positive.")
    if args.slippery_ant:
        args.slippery = True
        if args.env_name != "ant":
            print("--slippery-ant is deprecated; use --slippery for non-Ant slippery envs.")
    if args.slippery:
        if args.env_name not in SUPPORTED_SLIPPERY_ENVS:
            supported = ", ".join(sorted(SUPPORTED_SLIPPERY_ENVS))
            raise ValueError(f"--slippery supports only envs in {{{supported}}}; got {args.env_name!r}.")
        if args.slippery_change_every <= 0:
            raise ValueError("--slippery-change-every must be positive.")
        schedule_seeds_np = parse_schedule_seeds(args)
    else:
        schedule_seeds_np = np.zeros(args.num_agents, dtype=np.int32)

    args.base_optimizer = resolve_base_optimizer(args.base_optimizer)
    args.heads_optimizer = resolve_heads_optimizer(args.heads_optimizer)
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError("n_envs * num_steps must be divisible by num_minibatches.")
    args.num_updates = args.total_timesteps // args.batch_size
    agent_seeds_np = args.seed + np.arange(args.num_agents, dtype=np.int32) * args.agent_seed_stride
    run_name = f"{args.env_name}__{args.exp_name}__parallel{args.num_agents}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config={
                **vars(args),
                "agent_seeds": agent_seeds_np.tolist(),
                "agent_slippery_schedule_seeds": schedule_seeds_np.tolist(),
            },
            name=run_name,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    agent_keys = jax.random.split(key, args.num_agents)
    init_keys = jax.vmap(lambda k: jax.random.split(k, 4))(agent_keys)
    agent_keys = init_keys[:, 0]
    network_keys = init_keys[:, 1]
    actor_keys = init_keys[:, 2]
    critic_keys = init_keys[:, 3]
    schedule_seeds = jnp.asarray(schedule_seeds_np, dtype=jnp.int32)

    print("JAX devices:", jax.devices())
    print(f"env name: {args.env_name}")
    print(f"num_agents: {args.num_agents}")
    seed_preview = agent_seeds_np[: min(args.num_agents, max(args.max_printed_agents, 0))].tolist()
    seed_suffix = f" ... (+{args.num_agents - len(seed_preview)} more)" if len(seed_preview) < args.num_agents else ""
    print(f"agent_seeds: {seed_preview}{seed_suffix}")
    if args.slippery:
        schedule_preview = schedule_seeds_np[: min(args.num_agents, max(args.max_printed_agents, 0))].tolist()
        schedule_suffix = f" ... (+{args.num_agents - len(schedule_preview)} more)" if len(schedule_preview) < args.num_agents else ""
        print(f"agent_slippery_schedule_seeds: {schedule_preview}{schedule_suffix}")
        print(f"slippery_schedule_seed_mode: {args.slippery_schedule_seed_mode}")
        print(f"slippery_change_every: {args.slippery_change_every}")
        print(f"slippery_start_at_schedule: {args.slippery_start_at_schedule}")
        print(f"slippery_friction_csv: {DEFAULT_SLIPPERY_FRICTIONS_CSV}")
    print(f"total timesteps per agent: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs per agent: {args.n_envs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size per agent: {args.minibatch_size}")
    print(f"network_arch: {args.network_arch}")
    print(f"heads_optimizer: {args.heads_optimizer}")
    print(f"reward_normalize: {args.reward_normalize}")
    print(f"obs_normalize: {args.obs_normalize}")
    print(f"clip_actions: {args.clip_actions}")

    if args.slippery:
        env = DynamicSlipperySingleBraxEnv(
            args.env_name,
            args.backend,
            args.slippery_change_every,
            args.action_repeat,
            start_at_schedule=args.slippery_start_at_schedule,
        )
        action_dim = env.action_size
        obs_dim = env.observation_size
    else:
        env = RegularSingleBraxEnv(args.env_name, args.backend, args.action_repeat)
        action_dim = env.action_size
        obs_dim = env.observation_size
    obs_shape = (obs_dim,)
    print(f"action_dim: {action_dim}")
    print(f"obs_dim: {obs_dim}")
    print(f"obs_shape: {obs_shape}")

    if args.slippery_probe_only or args.slippery_probe_steps > 0:
        raise ValueError("parallel_ppo_brax.py does not support slippery probe mode yet.")

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    if args.network_arch == "lop":
        if args.use_crate_network or args.use_crate_head:
            raise ValueError("--network-arch=lop does not support CRATE trunk/head options.")
        network = IdentityNetwork()
        print("Using lop-jax style identity state trunk")
    elif args.network_arch == "ppo" and args.use_crate_network:
        network = CRATENetwork(
            activation=args.actor_critic_activation,
            crate_step_size=args.network_crate_step_size,
        )
        print(f"Using CRATE state trunk with step_size={args.network_crate_step_size}")
    elif args.network_arch == "ppo":
        network = Network(activation=args.actor_critic_activation)
    else:
        raise ValueError(f"Unsupported --network-arch={args.network_arch!r}; expected 'ppo' or 'lop'.")

    actor_kwargs = dict(
        action_dim=action_dim,
        activation=args.actor_critic_activation,
        logstd_min=args.actor_logstd_min,
        logstd_max=args.actor_logstd_max,
        clip_global_logstd=args.clip_global_logstd,
        bounded_global_logstd=args.bounded_global_logstd,
        actor_logstd_init=args.actor_logstd_init,
        actor_mean_tanh=args.actor_mean_tanh,
        actor_mean_scale=args.actor_mean_scale,
    )
    if args.network_arch == "lop":
        actor = LopActor(**actor_kwargs)
        critic = LopCritic(activation=args.actor_critic_activation)
        print("Using lop-jax style shallow actor/critic heads")
    elif args.use_crate_head:
        actor = CRATEActor(crate_step_size=args.crate_step_size, **actor_kwargs)
        critic = CRATECritic(crate_step_size=args.crate_step_size, activation=args.actor_critic_activation)
        print(f"Using CRATE heads with step_size={args.crate_step_size}")
    else:
        actor = Actor(**actor_kwargs)
        critic = Critic(activation=args.actor_critic_activation)

    dummy_obs = jnp.zeros((1,) + obs_shape)

    def init_agent(network_key, actor_key, critic_key):
        network_params = network.init(network_key, dummy_obs)
        dummy_hidden = network.apply(network_params, dummy_obs)
        all_params = flax.core.freeze({
            "network": network_params,
            "actor": actor.init(actor_key, dummy_hidden),
            "critic": critic.init(critic_key, dummy_hidden),
        })
        return TrainState.create(
            apply_fn=None,
            params=all_params,
            tx=create_optimizer(
                linear_schedule if args.anneal_lr else args.learning_rate,
                adam_eps=args.adam_eps,
                base_optimizer=args.base_optimizer,
                weight_decay=args.weight_decay,
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

    agent_state = jax.vmap(init_agent)(network_keys, actor_keys, critic_keys)
    optimizer_counts = count_optimizer_params(slice_tree(agent_state.params, 0), args.heads_optimizer)
    print("\nParameter breakdown by optimizer per agent:")
    for name, count in optimizer_counts.items():
        print(f"  {name}: {count:,}")

    network_apply = jax.jit(network.apply)
    actor_apply = jax.jit(actor.apply)
    critic_apply = jax.jit(critic.apply)

    def reset_one_agent(key):
        reset_keys = jax.random.split(key, args.n_envs)
        return jax.vmap(env.reset)(reset_keys)

    reset_all_agents = jax.jit(jax.vmap(reset_one_agent))

    def step_env_one_agent(key, state, action, schedule_seed):
        key, env_key = jax.random.split(key)
        step_keys = jax.random.split(env_key, args.n_envs)
        obs, next_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_keys, state, action, schedule_seed
        )
        rollout_info = SlipperyRolloutInfo(
            friction_slide=info["friction"][:, 0, 0],
            timestep=info["timestep"] if args.slippery else jnp.zeros(args.n_envs, dtype=jnp.int32),
        )
        return key, obs, next_state, reward, done.astype(jnp.bool_), rollout_info

    def get_action_and_value(single_agent_state, next_obs, key):
        hidden = network_apply(single_agent_state.params["network"], next_obs)
        actor_mean, actor_logstd = actor_apply(single_agent_state.params["actor"], hidden)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic_apply(single_agent_state.params["critic"], hidden)
        if args.clip_actions:
            action = jnp.clip(action, -args.max_action, args.max_action)
        return action, logprob, value.squeeze(-1), key

    def get_action_and_value2(params, x, action):
        hidden = network_apply(params["network"], x)
        actor_mean, actor_logstd = actor_apply(params["actor"], hidden)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic_apply(params["critic"], hidden).squeeze(-1)
        return logprob, entropy, value

    compute_gae_once_bound = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    def compute_gae_single(single_agent_state, next_obs, next_done, storage):
        next_value = critic_apply(
            single_agent_state.params["critic"],
            network_apply(single_agent_state.params["network"], next_obs),
        ).squeeze(-1)
        advantages = jnp.zeros((args.n_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once_bound,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        return storage.replace(advantages=advantages, returns=advantages + storage.values)

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values):
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
            v_clipped = mb_values + jnp.clip(newvalue - mb_values, -args.clip_eps, args.clip_eps)
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def update_ppo_single(single_agent_state, storage, key):
        def update_epoch(carry, unused_inp):
            single_agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x):
                x = jax.random.permutation(subkey, x)
                return jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])

            flat_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flat_storage)

            def update_minibatch(carry, minibatch):
                single_agent_state = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    single_agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                )
                single_agent_state = single_agent_state.apply_gradients(grads=grads)
                return single_agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            single_agent_state, metrics = jax.lax.scan(update_minibatch, single_agent_state, shuffled_storage)
            return (single_agent_state, key), metrics

        (single_agent_state, key), metrics = jax.lax.scan(
            update_epoch, (single_agent_state, key), (), length=args.update_epochs
        )
        loss, pg_loss, v_loss, entropy_loss, approx_kl = metrics
        return single_agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    def step_once(carry, _):
        single_agent_state, episode_stats, reward_norm, obs_norm, env_state, obs, done, key, schedule_seed = carry
        action, logprob, value, key = get_action_and_value(single_agent_state, obs, key)
        key, next_obs_raw, env_state, raw_reward, next_done, rollout_info = step_env_one_agent(
            key, env_state, action, schedule_seed
        )
        obs_norm = obs_norm.update(next_obs_raw)
        next_obs = obs_norm.normalize(next_obs_raw)
        if args.reward_normalize:
            reward_norm = reward_norm.update(raw_reward, next_done.astype(jnp.float32))
            reward = reward_norm.normalize(raw_reward)
        else:
            reward = raw_reward
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
        return (
            single_agent_state,
            episode_stats,
            reward_norm,
            obs_norm,
            env_state,
            next_obs,
            next_done,
            key,
            schedule_seed,
        ), (storage, rollout_info)

    def rollout_single(single_agent_state, episode_stats, reward_norm, obs_norm, env_state, next_obs, next_done, key, schedule_seed):
        carry = (single_agent_state, episode_stats, reward_norm, obs_norm, env_state, next_obs, next_done, key, schedule_seed)
        carry, (storage, rollout_info) = jax.lax.scan(step_once, carry, None, length=args.num_steps)
        single_agent_state, episode_stats, reward_norm, obs_norm, env_state, next_obs, next_done, key, _ = carry
        return single_agent_state, episode_stats, reward_norm, obs_norm, env_state, next_obs, next_done, storage, rollout_info, key

    rollout_all = jax.jit(jax.vmap(rollout_single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)))
    compute_gae_all = jax.jit(jax.vmap(compute_gae_single, in_axes=(0, 0, 0, 0)))
    update_ppo_all = jax.jit(jax.vmap(update_ppo_single, in_axes=(0, 0, 0)))

    reset_keys = jax.random.split(jax.random.fold_in(key, 12345), args.num_agents)
    next_obs, env_state = reset_all_agents(reset_keys)
    single_obs_norm = ObsNormalizer.create(obs_shape, enabled=args.obs_normalize, clip=args.obs_norm_clip)
    obs_normalizer = stack_replicas(single_obs_norm, args.num_agents)
    obs_normalizer = jax.vmap(lambda norm, obs: norm.update(obs))(obs_normalizer, next_obs)
    next_obs = jax.vmap(lambda norm, obs: norm.normalize(obs))(obs_normalizer, next_obs)
    next_done = jnp.zeros((args.num_agents, args.n_envs), dtype=jnp.bool_)
    reward_normalizer = stack_replicas(RewardNormalizer.create(args.n_envs, args.gamma), args.num_agents)
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros((args.num_agents, args.n_envs), dtype=jnp.float32),
        episode_lengths=jnp.zeros((args.num_agents, args.n_envs), dtype=jnp.int32),
        returned_episode_returns=jnp.zeros((args.num_agents, args.n_envs), dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros((args.num_agents, args.n_envs), dtype=jnp.int32),
    )

    global_step = 0
    start_time = time.time()
    print("Starting parallel training...")
    for iteration in range(1, args.num_updates + 1):
        iteration_time_start = time.time()
        (
            agent_state,
            episode_stats,
            reward_normalizer,
            obs_normalizer,
            env_state,
            next_obs,
            next_done,
            storage,
            rollout_info,
            agent_keys,
        ) = rollout_all(
            agent_state,
            episode_stats,
            reward_normalizer,
            obs_normalizer,
            env_state,
            next_obs,
            next_done,
            agent_keys,
            schedule_seeds,
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae_all(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, agent_keys = update_ppo_all(
            agent_state, storage, agent_keys
        )

        if iteration % args.log_interval == 0:
            returns = np.asarray(jax.device_get(episode_stats.returned_episode_returns)).mean(axis=1)
            lengths = np.asarray(jax.device_get(episode_stats.returned_episode_lengths)).mean(axis=1)
            loss_np = np.asarray(jax.device_get(loss[:, -1, -1]))
            pg_loss_np = np.asarray(jax.device_get(pg_loss[:, -1, -1]))
            v_loss_np = np.asarray(jax.device_get(v_loss[:, -1, -1]))
            entropy_np = np.asarray(jax.device_get(entropy_loss[:, -1, -1]))
            approx_kl_np = np.asarray(jax.device_get(approx_kl[:, -1, -1]))
            aggregate_step = global_step * args.num_agents
            sps = int(aggregate_step / (time.time() - start_time))
            sps_update = int(args.num_agents * args.n_envs * args.num_steps / (time.time() - iteration_time_start))
            printed_agents = min(args.num_agents, max(args.max_printed_agents, 0))
            returns_print = returns[:printed_agents]
            ret_text = np.array2string(returns_print, precision=1, separator=", ", max_line_width=160)
            if printed_agents < args.num_agents:
                ret_text = f"{ret_text} ... (+{args.num_agents - printed_agents} more)"
            base_msg = (
                f"update={iteration} step_per_agent={global_step} step_agg={aggregate_step} "
                f"return_mean={returns.mean():.1f} return_min={returns.min():.1f} "
                f"return_max={returns.max():.1f} loss_mean={loss_np.mean():.4f} SPS={sps} "
                f"returns={ret_text}"
            )
            if args.slippery:
                agent_slip = get_agent_slippery_metrics(rollout_info, args)
                phases = [m["slippery/phase_first_env"] for m in agent_slip[:printed_agents]]
                frictions = [m["slippery/friction_slide_first_env"] for m in agent_slip[:printed_agents]]
                base_msg += f" phases={phases} frictions={np.round(frictions, 4).tolist()}"
                if printed_agents < args.num_agents:
                    base_msg += f" printed_agents={printed_agents}/{args.num_agents}"
            else:
                agent_slip = []
            print(base_msg)

            if args.track:
                lr = float(linear_schedule(iteration * args.num_minibatches * args.update_epochs)) if args.anneal_lr else args.learning_rate
                log_dict = {
                    "global_step_per_agent": global_step,
                    "global_step_aggregate": aggregate_step,
                    "charts/avg_episodic_return_mean": float(returns.mean()),
                    "charts/avg_episodic_return_min": float(returns.min()),
                    "charts/avg_episodic_return_max": float(returns.max()),
                    "charts/avg_episodic_return_median": float(np.median(returns)),
                    "charts/failure_frac_return_lt_100": float(np.mean(returns < 100.0)),
                    "charts/avg_episodic_length_mean": float(lengths.mean() * args.action_repeat),
                    "charts/learning_rate": lr,
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/loss_mean": float(loss_np.mean()),
                    "losses/value_loss_mean": float(v_loss_np.mean()),
                    "losses/policy_loss_mean": float(pg_loss_np.mean()),
                    "losses/entropy_mean": float(entropy_np.mean()),
                    "losses/approx_kl_mean": float(approx_kl_np.mean()),
                }
                if args.slippery:
                    friction_mean_by_agent = np.asarray([m["slippery/friction_slide_mean"] for m in agent_slip], dtype=np.float32)
                    friction_first_by_agent = np.asarray([m["slippery/friction_slide_first_env"] for m in agent_slip], dtype=np.float32)
                    phase_by_agent = np.asarray([m["slippery/phase_first_env"] for m in agent_slip], dtype=np.float32)
                    timestep_by_agent = np.asarray([m["slippery/timestep_first_env"] for m in agent_slip], dtype=np.float32)
                    log_dict.update({
                        "slippery/friction_slide_mean": float(friction_mean_by_agent.mean()),
                        "slippery/friction_slide_min": float(friction_first_by_agent.min()),
                        "slippery/friction_slide_max": float(friction_first_by_agent.max()),
                        "slippery/friction_slide_std": float(friction_first_by_agent.std()),
                        "slippery/friction_slide_first_agent_first_env": float(friction_first_by_agent[0]),
                        "slippery/phase_first_env_mean": float(phase_by_agent.mean()),
                        "slippery/phase_first_env_min": float(phase_by_agent.min()),
                        "slippery/phase_first_env_max": float(phase_by_agent.max()),
                        "slippery/phase_first_agent_first_env": int(phase_by_agent[0]),
                        "slippery/timestep_first_env_mean": float(timestep_by_agent.mean()),
                        "slippery/timestep_first_agent_first_env": int(timestep_by_agent[0]),
                        "slippery/change_every": int(args.slippery_change_every),
                    })
                for agent_id in range(min(args.num_agents, args.max_logged_agents)):
                    prefix = f"agent/{agent_id}"
                    log_dict.update({
                        f"{prefix}/seed": int(agent_seeds_np[agent_id]),
                        f"{prefix}/slippery/schedule_seed": int(schedule_seeds_np[agent_id]),
                        f"{prefix}/charts/avg_episodic_return": float(returns[agent_id]),
                        f"{prefix}/charts/avg_episodic_length": float(lengths[agent_id] * args.action_repeat),
                        f"{prefix}/losses/loss": float(loss_np[agent_id]),
                        f"{prefix}/losses/value_loss": float(v_loss_np[agent_id]),
                        f"{prefix}/losses/policy_loss": float(pg_loss_np[agent_id]),
                        f"{prefix}/losses/entropy": float(entropy_np[agent_id]),
                        f"{prefix}/losses/approx_kl": float(approx_kl_np[agent_id]),
                    })
                    if args.slippery:
                        for k, v in agent_slip[agent_id].items():
                            log_dict[f"{prefix}/{k}"] = v
                    if iteration == args.log_interval:
                        per_agent_params = slice_tree(agent_state.params, agent_id)
                        log_dict.update({f"{prefix}/{k}": v for k, v in matrix_constraint_metrics(per_agent_params).items()})
                        if args.heads_optimizer == "stiefel_online":
                            per_agent_opt_state = slice_tree(agent_state.opt_state, agent_id)
                            log_dict.update({f"{prefix}/{k}": v for k, v in online_stiefel_constraint_metrics(per_agent_opt_state).items()})
                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average aggregate SPS: {args.total_timesteps * args.num_agents / elapsed:.0f}")
    print(f"Average per-agent SPS: {args.total_timesteps / elapsed:.0f}")
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
