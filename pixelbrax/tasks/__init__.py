"""Runtime registration for PixelBrax-owned tasks."""

from brax import envs

from pixelbrax.tasks.maze import AntMaze
from pixelbrax.tasks.maze import HumanoidMaze


_REGISTERED = False


def register_pixelbrax_tasks() -> None:
    """Registers parent-repo tasks with Brax's env registry."""
    global _REGISTERED
    if _REGISTERED:
        return

    envs.register_environment("ant_u_maze", AntMaze)
    envs.register_environment("humanoid_u_maze", HumanoidMaze)
    _REGISTERED = True
