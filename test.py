import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import mujoco
from mujoco import mjx
print("mujoco:", mujoco.__version__)

import brax, inspect
print(inspect.getfile(brax))
