from typing import Optional, Tuple, Union, Sequence
from functools import partial

from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from brax.base import System
from brax.envs.base import State
import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env
        if hasattr(env, '_unwrapped'):
            self._unwrapped = env._unwrapped
        else:
            self._unwrapped = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    discounted_episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_discounted_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnums=(0))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_discounted_episode_return = state.discounted_episode_returns + (self.gamma ** state.episode_lengths) * reward
        new_episode_length = state.episode_lengths + 1
        # TODO: add discounted_episode_returns here.
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
                                     + new_episode_return * done,
            returned_discounted_episode_returns=state.returned_discounted_episode_returns * (1 - done)
                                                + new_discounted_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                     + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_discounted_episode_returns"] = state.returned_discounted_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["reward"] = reward
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        self.max_steps_in_episode = 1000
        env = EpisodeWrapper(env, episode_length=self.max_steps_in_episode, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


@struct.dataclass
class SlipperyAntState:
    env_state: environment.EnvState
    step: chex.Array


class AutoResetTimeStepWrapper(AutoResetWrapper):
    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        state.info['timestep'] = 0
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        state.info['timestep'] += 1
        return state


class NonstationaryFrictionBraxWrapper(BraxGymnaxWrapper):
    
    def __init__(self, env_name, backend="positional",
                 change_every: int = int(1e5),
                 friction_schedule: Sequence[float] = (1e-2, 0.1, 0.5, 1.0)):
        self.change_every = change_every
        self.schedule = jnp.array(friction_schedule, dtype=jnp.float32)
        self.num_phases = len(self.schedule)

        env = envs.get_environment(env_name=env_name, backend=backend)
        self.max_steps_in_episode = 1000
        env = EpisodeWrapper(env, episode_length=self.max_steps_in_episode, action_repeat=1)
        env = AutoResetTimeStepWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def env_fn(self, sys: System):
        env = self._env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: chex.PRNGKey, params=None):
        state = self._env.reset(rng)
        sys = self._env.unwrapped.sys

        state.info['sys_variation'] = {'geom_friction': jnp.array(sys.geom_friction)}
        return state.obs, state

    def step(self, rng: chex.PRNGKey, state: State, action: jnp.ndarray, params=None):
        sys = self._env.unwrapped.sys
        new_timestep = state.info['timestep'] + 1
        phase = (new_timestep // self.change_every) % self.num_phases
        friction = self.schedule[phase]
        new_geom_friction = state.info['sys_variation']['geom_friction'].at[:, 0].set(friction)
        variation = {'geom_friction': new_geom_friction}
        new_sys = sys.replace(**variation)
        new_env = self.env_fn(new_sys)
        next_state = new_env.step(state, action)
        next_state.info['sys_variation'] = variation

        geom_fric = next_state.info['sys_variation']['geom_friction']
        info = {
            'friction': geom_fric,
        }
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, info

@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


from typing import Any, Optional, Tuple, Dict, Union, List

import chex
import jax
import gymnasium as gym
from gymnasium import Wrapper, core
from gymnasium.core import WrapperObsType, WrapperActType, SupportsFloat, Env
from gymnasium.wrappers import AddRenderObservation

from gymnax.environments import environment
from gymnax.environments import spaces


class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
    """Wrap Gymnax environment as OOP Gym environment."""

    def __init__(
            self,
            env: environment.Environment,
            params: Optional[environment.EnvParams] = None,
            seed: Optional[int] = None,
            num_envs: Optional[int] = None,
    ):
        """Wrap Gymnax environment as OOP Gym environment.


        Args:
            env: Gymnax Environment instance
            params: If provided, gymnax EnvParams for environment (otherwise uses
              default)
            seed: If provided, seed for JAX PRNG (otherwise picks 0)
        """
        super().__init__()
        self._env = env
        self.env_params = params if params is not None else env.default_params
        self.metadata.update(
            {
                "name": env.name,
                "render_modes": (
                    ["human", "rgb_array"] if hasattr(env, "render") else []
                ),
            }
        )
        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)
        self.num_envs = num_envs
        rng = self.rng
        if self.num_envs is not None:
            rng = jax.random.split(self.rng, self.num_envs)
        _, self.env_state = self._env.reset(rng, self.env_params)
        self.max_steps_in_episode = self.env_params.max_steps_in_episode

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)."""
        self.rng = jax.random.PRNGKey(seed or 0)

    def step(
            self, action: core.ActType
    ) -> Tuple[core.ObsType, float, bool, bool, Dict[Any, Any]]:
        """Step environment, follow new step API."""
        self.rng, step_key = jax.random.split(self.rng)
        step_keys = jax.random.split(step_key, self.num_envs)
        o, self.env_state, r, d, info = self._env.step(
            step_keys, self.env_state, action, self.env_params
        )
        return o, r, d, d, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[Any] = None,  # dict
    ) -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset environment, update parameters and seed if provided."""
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.rng, reset_key = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_key, self.num_envs)
        o, self.env_state = self._env.reset(reset_keys, self.env_params)
        return o, {}

    def render(
            self, mode="human"
    ) -> Optional[Union[core.RenderFrame, List[core.RenderFrame]]]:
        """use underlying environment rendering if it exists, otherwise return None."""
        return getattr(self._env, "render", lambda x, y: None)(
            self.env_state, self.env_params
        )

class PixelOnlyObservationWrapper(AddRenderObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['pixels']

    def observation(self, observation):
        dict_observations = super().observation(observation)
        return dict_observations['pixels']


class OnlineReturnsLogWrapper(Wrapper):
    def __init__(self, *args, gamma: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.episode_return = 0
        self.discounted_episode_return = 0
        self.episode_length = 0
        self.returned_episode_return = 0
        self.returned_discounted_episode_return = 0
        self.timestep = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.returned_episode_return = self.episode_return
        self.returned_discounted_episode_return = self.discounted_episode_return
        self.timestep = 0
        info = {
            'episode_return': self.episode_return,
            'returned_episode_return': self.returned_episode_return,
            'returned_discounted_episode_return': self.returned_discounted_episode_return,
            'episode_length': self.episode_length,
            'timestep': self.timestep,
            'returned_episode': False,
            # **info
        }

        self.episode_return = 0
        self.discounted_episode_return = 0
        self.episode_length = 0
        return obs, info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, truncation, info = self.env.step(action)

        new_episode_return = self.episode_return + reward
        new_discounted_episode_return = self.discounted_episode_return + (self.gamma ** self.episode_length) * reward
        new_episode_length = self.episode_length + 1

        not_done_or_not_trunc = (1 - done) * (1 - truncation)
        self.episode_return = new_episode_return * not_done_or_not_trunc
        self.discounted_episode_return = new_discounted_episode_return * not_done_or_not_trunc
        self.episode_length = new_episode_length * not_done_or_not_trunc

        self.returned_episode_return = self.returned_episode_return * not_done_or_not_trunc \
                                       + new_episode_return * (1 - not_done_or_not_trunc)
        self.returned_discounted_episode_return = self.returned_discounted_episode_return * not_done_or_not_trunc\
                                                  + new_discounted_episode_return * (1 - not_done_or_not_trunc)

        info = {
            'episode_return': self.episode_return,
            'returned_episode_return': self.returned_episode_return,
            'returned_discounted_episode_return': self.returned_discounted_episode_return,
            'episode_length': self.episode_length,
            'timestep': self.timestep + 1,
            'returned_episode': done
            # **info
        }
        return obs, reward, done, truncation, info