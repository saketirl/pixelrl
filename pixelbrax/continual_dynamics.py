"""Utilities for deterministic continual dynamics schedules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml


SUPPORTED_PARAMETERS = ("geom_friction_slide", "actuator_gear", "link_mass")
HALFCHEETAH_ACTUATOR_NAMES = (
    "bthigh",
    "bshin",
    "bfoot",
    "fthigh",
    "fshin",
    "ffoot",
)
WALKER2D_ACTUATOR_NAMES = (
    "thigh",
    "leg",
    "foot",
    "thigh_left",
    "leg_left",
    "foot_left",
)
SUPPORTED_ENV_BACKENDS = {
    "halfcheetah": "spring",
    "inverted_pendulum": "generalized",
    "walker2d": "spring",
}
ACTUATOR_NAMES_BY_ENV = {
    "halfcheetah": HALFCHEETAH_ACTUATOR_NAMES,
    "inverted_pendulum": ("slide",),
    "walker2d": WALKER2D_ACTUATOR_NAMES,
}
DEFAULT_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ContinualDynamicsConfig:
    path: str
    enabled: bool
    env_name: str
    backend: str
    switch_every_env_steps: int
    seed: Optional[int]
    first_task_uses_defaults: bool
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class DynamicsTask:
    task_index: int
    is_default: bool
    values: Dict[str, Any]


def load_continual_dynamics_config(path: str) -> ContinualDynamicsConfig:
    """Loads a continual dynamics YAML config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Continual dynamics config must be a mapping: {path}")

    return ContinualDynamicsConfig(
        path=str(config_path),
        enabled=bool(raw.get("enabled", True)),
        env_name=str(raw.get("env_name", "")),
        backend=str(raw.get("backend", "")),
        switch_every_env_steps=int(raw.get("switch_every_env_steps", 0)),
        seed=raw.get("seed", None),
        first_task_uses_defaults=bool(raw.get("first_task_uses_defaults", True)),
        parameters=dict(raw.get("parameters", {}) or {}),
    )


def validate_continual_dynamics_config(
    config: ContinualDynamicsConfig,
    *,
    env_name: str,
    backend: str,
    n_envs: int,
    num_steps: int,
    base_sys: Any,
) -> int:
    """Validates config and returns PPO updates per task."""
    if not config.enabled:
        return 0

    if env_name != config.env_name:
        raise ValueError(
            "Continual dynamics args env_name must match config env_name; "
            f"got args env={env_name!r}, config env={config.env_name!r}."
        )

    expected_backend = SUPPORTED_ENV_BACKENDS.get(env_name)
    if expected_backend is None:
        raise ValueError(
            "Continual dynamics only supports envs "
            f"{sorted(SUPPORTED_ENV_BACKENDS)}; got {env_name!r}."
        )
    if backend != config.backend:
        raise ValueError(
            "Continual dynamics args backend must match config backend; "
            f"got args backend={backend!r}, config backend={config.backend!r}."
        )
    if backend != expected_backend:
        raise ValueError(
            f"Continual dynamics for {env_name!r} requires backend "
            f"{expected_backend!r}; got {backend!r}."
        )
    if not config.first_task_uses_defaults:
        raise ValueError("first_task_uses_defaults must be true for v1.")
    if config.seed is not None and int(config.seed) < 0:
        raise ValueError("Continual dynamics seed must be null or non-negative.")
    if config.switch_every_env_steps <= 0:
        raise ValueError("switch_every_env_steps must be positive.")

    rollout_env_steps = int(n_envs) * int(num_steps)
    if rollout_env_steps <= 0:
        raise ValueError("n_envs * num_steps must be positive.")
    if config.switch_every_env_steps % rollout_env_steps != 0:
        raise ValueError(
            "switch_every_env_steps must be an exact multiple of "
            f"n_envs * num_steps ({rollout_env_steps}); got "
            f"{config.switch_every_env_steps}."
        )

    unknown = set(config.parameters) - set(SUPPORTED_PARAMETERS)
    if unknown:
        raise ValueError(f"Unsupported continual dynamics parameter(s): {sorted(unknown)}")

    enabled = tuple(_enabled_parameters(config))
    if not enabled:
        raise ValueError("At least one continual dynamics parameter must be enabled.")

    actuator_names = _actuator_names(env_name)
    for name in enabled:
        param_config = _parameter_config(config, name)
        if name == "geom_friction_slide":
            _validate_geom_friction_config(param_config, base_sys)
        elif name == "actuator_gear":
            _validate_named_vector_config(
                param_config,
                actuator_names,
                np.asarray(jax.device_get(base_sys.actuator.gear), dtype=np.float64),
                "actuator_gear",
            )
        elif name == "link_mass":
            _validate_named_vector_config(
                param_config,
                tuple(base_sys.link_names),
                np.asarray(jax.device_get(base_sys.link.inertia.mass), dtype=np.float64),
                "link_mass",
            )

    return config.switch_every_env_steps // rollout_env_steps


def resolve_schedule_seed(config: ContinualDynamicsConfig, training_seed: int) -> int:
    """Returns the seed used for deterministic task sampling."""
    return int(training_seed if config.seed is None else config.seed)


def make_dynamics_task(
    config: ContinualDynamicsConfig,
    base_sys: Any,
    task_index: int,
    schedule_seed: int,
) -> DynamicsTask:
    """Creates a deterministic task. Task 0 always uses defaults."""
    if task_index < 0:
        raise ValueError("task_index must be non-negative.")

    if task_index == 0:
        return DynamicsTask(
            task_index=0,
            is_default=True,
            values=default_task_values(config, base_sys),
        )

    key = jax.random.fold_in(jax.random.PRNGKey(int(schedule_seed)), int(task_index))
    values: Dict[str, Any] = {}
    actuator_names = _actuator_names(config.env_name)

    for name in _enabled_parameters(config):
        key, param_key = jax.random.split(key)
        param_config = _parameter_config(config, name)
        if name == "geom_friction_slide":
            values[name] = _sample_uniform(
                param_key,
                float(param_config["min"]),
                float(param_config["max"]),
            )
        elif name == "actuator_gear":
            values[name] = _sample_named_ranges(
                param_key,
                actuator_names,
                param_config["ranges"],
            )
        elif name == "link_mass":
            values[name] = _sample_named_ranges(
                param_key,
                tuple(base_sys.link_names),
                param_config["ranges"],
            )

    return DynamicsTask(task_index=task_index, is_default=False, values=values)


def default_task_values(
    config: ContinualDynamicsConfig,
    base_sys: Any,
) -> Dict[str, Any]:
    """Returns default values for the enabled config parameters."""
    values: Dict[str, Any] = {}
    actuator_names = _actuator_names(config.env_name)
    for name in _enabled_parameters(config):
        if name == "geom_friction_slide":
            friction = np.asarray(jax.device_get(base_sys.geom_friction), dtype=np.float64)
            values[name] = float(friction[0, 0])
        elif name == "actuator_gear":
            gear = np.asarray(jax.device_get(base_sys.actuator.gear), dtype=np.float64)
            values[name] = {
                actuator_name: float(gear[i])
                for i, actuator_name in enumerate(actuator_names)
            }
        elif name == "link_mass":
            mass = np.asarray(jax.device_get(base_sys.link.inertia.mass), dtype=np.float64)
            values[name] = {
                link_name: float(mass[i])
                for i, link_name in enumerate(base_sys.link_names)
            }
    return values


def apply_dynamics_task(base_sys: Any, task: DynamicsTask) -> Any:
    """Applies task dynamics to a fresh copy of base_sys."""
    if task.is_default:
        return base_sys

    sys = base_sys

    if "geom_friction_slide" in task.values:
        geom_friction = jnp.asarray(sys.geom_friction)
        geom_friction = geom_friction.at[:, 0].set(
            float(task.values["geom_friction_slide"])
        )
        sys = sys.replace(geom_friction=geom_friction)

    if "actuator_gear" in task.values:
        gear_values = _ordered_actuator_gear_values(task.values["actuator_gear"])
        gear = jnp.asarray(gear_values, dtype=jnp.asarray(sys.actuator.gear).dtype)
        sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    if "link_mass" in task.values:
        link_names = tuple(sys.link_names)
        base_mass = jnp.asarray(base_sys.link.inertia.mass)
        new_mass = jnp.asarray(
            [task.values["link_mass"][name] for name in link_names],
            dtype=base_mass.dtype,
        )
        mass_ratio = new_mass / base_mass
        inertia = sys.link.inertia.replace(
            mass=new_mass,
            i=jnp.asarray(base_sys.link.inertia.i) * mass_ratio[:, None, None],
        )
        sys = sys.replace(link=sys.link.replace(inertia=inertia))

    return sys


def flatten_task_values(task: DynamicsTask) -> Dict[str, float]:
    """Flattens nested task values into W&B-friendly scalar metrics."""
    flat: Dict[str, float] = {}
    for name, value in task.values.items():
        if isinstance(value, dict):
            for sub_name, sub_value in value.items():
                flat[f"{name}_{sub_name}"] = float(sub_value)
        else:
            flat[name] = float(value)
    return flat


def continual_log_metrics(
    task: DynamicsTask,
    *,
    switch_every_env_steps: int,
    task_start_step: int,
    global_step: int,
) -> Dict[str, float]:
    """Builds scalar metrics for the current continual dynamics task."""
    metrics: Dict[str, float] = {
        "continual/task_index": int(task.task_index),
        "continual/is_default_task": int(task.is_default),
        "continual/task_start_step": int(task_start_step),
        "continual/task_elapsed_steps": int(global_step - task_start_step),
        "continual/switch_every_env_steps": int(switch_every_env_steps),
    }
    for name, value in flatten_task_values(task).items():
        metrics[f"continual/{name}"] = float(value)
    return metrics


def _enabled_parameters(config: ContinualDynamicsConfig) -> Iterable[str]:
    for name in SUPPORTED_PARAMETERS:
        param_config = config.parameters.get(name)
        if isinstance(param_config, dict) and bool(param_config.get("enabled", False)):
            yield name


def _parameter_config(config: ContinualDynamicsConfig, name: str) -> Dict[str, Any]:
    param_config = config.parameters.get(name)
    if not isinstance(param_config, dict):
        raise ValueError(f"Parameter config for {name!r} must be a mapping.")
    return param_config


def _actuator_names(env_name: str) -> Tuple[str, ...]:
    try:
        return ACTUATOR_NAMES_BY_ENV[env_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported continual dynamics env: {env_name!r}") from exc


def _ordered_actuator_gear_values(values: Dict[str, Any]) -> Tuple[float, ...]:
    if not isinstance(values, dict):
        raise ValueError("actuator_gear task values must be a mapping.")

    value_keys = set(values)
    for names in ACTUATOR_NAMES_BY_ENV.values():
        if value_keys == set(names):
            return tuple(float(values[name]) for name in names)

    raise ValueError(
        "actuator_gear task values keys do not match any supported env; "
        f"got {sorted(value_keys)}."
    )


def _sample_uniform(key: jax.Array, minval: float, maxval: float) -> float:
    sample = jax.random.uniform(key, (), minval=minval, maxval=maxval)
    return float(jax.device_get(sample))


def _sample_named_ranges(
    key: jax.Array,
    names: Tuple[str, ...],
    ranges: Dict[str, Any],
) -> Dict[str, float]:
    keys = jax.random.split(key, len(names))
    return {
        name: _sample_uniform(keys[i], float(ranges[name][0]), float(ranges[name][1]))
        for i, name in enumerate(names)
    }


def _validate_geom_friction_config(param_config: Dict[str, Any], base_sys: Any) -> None:
    _require_scalar_range(param_config, "geom_friction_slide")
    friction = np.asarray(jax.device_get(base_sys.geom_friction), dtype=np.float64)
    slide = friction[:, 0]
    if not np.allclose(slide, slide[0], atol=DEFAULT_TOLERANCE, rtol=0.0):
        raise ValueError("Expected one default slide friction value across all geoms.")
    _assert_default_close(
        float(param_config.get("default")),
        float(slide[0]),
        "geom_friction_slide.default",
    )


def _validate_named_vector_config(
    param_config: Dict[str, Any],
    names: Tuple[str, ...],
    defaults: np.ndarray,
    label: str,
) -> None:
    config_defaults = param_config.get("defaults")
    ranges = param_config.get("ranges")
    if not isinstance(config_defaults, dict):
        raise ValueError(f"{label}.defaults must be a mapping.")
    if not isinstance(ranges, dict):
        raise ValueError(f"{label}.ranges must be a mapping.")
    if defaults.shape[0] != len(names):
        raise ValueError(
            f"{label} expected {len(names)} default values for {names}; "
            f"loaded Brax system has shape {defaults.shape}."
        )

    _assert_exact_keys(config_defaults, names, f"{label}.defaults")
    _assert_exact_keys(ranges, names, f"{label}.ranges")

    for i, name in enumerate(names):
        _assert_default_close(float(config_defaults[name]), float(defaults[i]), f"{label}.{name}")
        _require_range_pair(ranges[name], f"{label}.ranges.{name}")


def _require_scalar_range(param_config: Dict[str, Any], label: str) -> None:
    if "min" not in param_config or "max" not in param_config:
        raise ValueError(f"{label} must define min and max.")
    minval, maxval = float(param_config["min"]), float(param_config["max"])
    if minval > maxval:
        raise ValueError(f"{label} min must be <= max.")


def _require_range_pair(value: Any, label: str) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or float(value[0]) > float(value[1])
    ):
        raise ValueError(f"{label} must be [min, max] with min <= max.")


def _assert_exact_keys(mapping: Dict[str, Any], names: Tuple[str, ...], label: str) -> None:
    expected = set(names)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}."
        )


def _assert_default_close(config_value: float, actual_value: float, label: str) -> None:
    if not np.isclose(config_value, actual_value, atol=DEFAULT_TOLERANCE, rtol=0.0):
        raise ValueError(
            f"Stale default for {label}: config has {config_value}, "
            f"loaded Brax system has {actual_value}."
        )
