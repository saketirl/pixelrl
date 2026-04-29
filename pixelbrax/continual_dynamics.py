"""Utilities for deterministic continual dynamics schedules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml


SUPPORTED_PARAMETERS = (
    "geom_friction_slide",
    "actuator_gear",
    "link_mass",
    "gravity_z",
    "dof_damping",
    "pole_length",
    "limb_length",
)
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
HALFCHEETAH_LIMB_NAMES = HALFCHEETAH_ACTUATOR_NAMES
HALFCHEETAH_REAR_NAMES = ("bthigh", "bshin", "bfoot")
HALFCHEETAH_FRONT_NAMES = ("fthigh", "fshin", "ffoot")
HALFCHEETAH_THIGH_NAMES = ("bthigh", "fthigh")
HALFCHEETAH_SHIN_NAMES = ("bshin", "fshin")
HALFCHEETAH_FOOT_NAMES = ("bfoot", "ffoot")
HALFCHEETAH_STRUCTURED_MODES = (
    "rear_long_front_short",
    "front_long_rear_short",
    "rear_strong_front_weak",
    "front_strong_rear_weak",
    "long_thigh_short_shin",
    "short_thigh_long_shin",
    "heavy_torso_weak_legs",
    "light_torso_strong_legs",
    "low_friction_high_gravity",
    "high_friction_low_gravity",
)
SUPPORTED_RANGE_SAMPLERS = ("linear_uniform", "uniform", "log_uniform")
SUPPORTED_GEOM_FRICTION_APPLY_MODES = ("uniform_defaults", "shared_scalar")
SUPPORTED_ENV_BACKENDS = {
    "ant": ("spring", "generalized"),
    "halfcheetah": ("spring", "generalized"),
    "hopper": ("spring", "generalized"),
    "inverted_pendulum": ("spring", "generalized"),
    "walker2d": ("spring", "generalized"),
}
ACTUATOR_NAMES_BY_ENV = {
    "halfcheetah": HALFCHEETAH_ACTUATOR_NAMES,
    "inverted_pendulum": ("slide",),
    "walker2d": WALKER2D_ACTUATOR_NAMES,
}
DOF_NAMES_BY_ENV = {
    "inverted_pendulum": ("slider", "hinge"),
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
    task_sampler: Dict[str, Any]
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
        task_sampler=dict(raw.get("task_sampler", {}) or {}),
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

    expected_backends = SUPPORTED_ENV_BACKENDS.get(env_name)
    if expected_backends is None:
        raise ValueError(
            "Continual dynamics only supports envs "
            f"{sorted(SUPPORTED_ENV_BACKENDS)}; got {env_name!r}."
        )
    if backend != config.backend:
        raise ValueError(
            "Continual dynamics args backend must match config backend; "
            f"got args backend={backend!r}, config backend={config.backend!r}."
        )
    if backend not in expected_backends:
        raise ValueError(
            f"Continual dynamics for {env_name!r} requires one of backends "
            f"{expected_backends!r}; got {backend!r}."
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
    _validate_task_sampler_config(config)

    for name in enabled:
        param_config = _parameter_config(config, name)
        if name == "geom_friction_slide":
            _validate_geom_friction_config(param_config, base_sys)
        elif name == "actuator_gear":
            _validate_named_vector_config(
                param_config,
                _actuator_names(env_name),
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
        elif name == "gravity_z":
            _validate_gravity_z_config(param_config, base_sys)
        elif name == "dof_damping":
            _validate_named_vector_config(
                param_config,
                _dof_names(env_name),
                np.asarray(jax.device_get(base_sys.dof.damping), dtype=np.float64),
                "dof_damping",
            )
        elif name == "pole_length":
            _validate_pole_length_config(param_config, env_name, base_sys)
        elif name == "limb_length":
            _validate_limb_length_config(param_config, env_name, base_sys)

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
    if _task_sampler_type(config) == "halfcheetah_structured_asymmetric":
        values = _sample_halfcheetah_structured_task(
            key,
            config,
            base_sys,
            task_index=task_index,
        )
        return DynamicsTask(task_index=task_index, is_default=False, values=values)

    values: Dict[str, Any] = {}

    for name in _enabled_parameters(config):
        key, param_key = jax.random.split(key)
        param_config = _parameter_config(config, name)
        if name == "geom_friction_slide":
            values[name] = _sample_range(
                param_key,
                float(param_config["min"]),
                float(param_config["max"]),
                _range_sampler(param_config),
            )
        elif name == "actuator_gear":
            values[name] = _sample_named_ranges(
                param_key,
                _actuator_names(config.env_name),
                param_config["ranges"],
                _range_sampler(param_config),
            )
        elif name == "link_mass":
            values[name] = _sample_named_ranges(
                param_key,
                tuple(base_sys.link_names),
                param_config["ranges"],
                _range_sampler(param_config),
            )
        elif name == "gravity_z":
            values[name] = _sample_range(
                param_key,
                float(param_config["min"]),
                float(param_config["max"]),
                _range_sampler(param_config),
            )
        elif name == "dof_damping":
            values[name] = _sample_named_ranges(
                param_key,
                _dof_names(config.env_name),
                param_config["ranges"],
                _range_sampler(param_config),
            )
        elif name == "pole_length":
            values[name] = _sample_range(
                param_key,
                float(param_config["min"]),
                float(param_config["max"]),
                _range_sampler(param_config),
            )
        elif name == "limb_length":
            values[name] = _sample_named_ranges(
                param_key,
                HALFCHEETAH_LIMB_NAMES,
                param_config["ranges"],
                _range_sampler(param_config),
            )

    return DynamicsTask(task_index=task_index, is_default=False, values=values)


def default_task_values(
    config: ContinualDynamicsConfig,
    base_sys: Any,
) -> Dict[str, Any]:
    """Returns default values for the enabled config parameters."""
    values: Dict[str, Any] = {}
    for name in _enabled_parameters(config):
        param_config = _parameter_config(config, name)
        if name == "geom_friction_slide":
            if _geom_friction_apply_mode(param_config) == "shared_scalar":
                continue
            friction = np.asarray(jax.device_get(base_sys.geom_friction), dtype=np.float64)
            values[name] = float(friction[0, 0])
        elif name == "actuator_gear":
            gear = np.asarray(jax.device_get(base_sys.actuator.gear), dtype=np.float64)
            values[name] = {
                actuator_name: float(gear[i])
                for i, actuator_name in enumerate(_actuator_names(config.env_name))
            }
        elif name == "link_mass":
            mass = np.asarray(jax.device_get(base_sys.link.inertia.mass), dtype=np.float64)
            values[name] = {
                link_name: float(mass[i])
                for i, link_name in enumerate(base_sys.link_names)
            }
        elif name == "gravity_z":
            gravity = np.asarray(jax.device_get(base_sys.gravity), dtype=np.float64)
            values[name] = float(gravity[2])
        elif name == "dof_damping":
            damping = np.asarray(jax.device_get(base_sys.dof.damping), dtype=np.float64)
            values[name] = {
                dof_name: float(damping[i])
                for i, dof_name in enumerate(_dof_names(config.env_name))
            }
        elif name == "pole_length":
            values[name] = _default_pole_length(base_sys)
        elif name == "limb_length":
            values[name] = _default_halfcheetah_limb_lengths(base_sys)
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

    if "gravity_z" in task.values:
        gravity = jnp.asarray(sys.gravity).at[2].set(float(task.values["gravity_z"]))
        sys = sys.replace(gravity=gravity)

    if "dof_damping" in task.values:
        damping_values = _ordered_named_values(
            task.values["dof_damping"],
            _dof_names_for_task(task.values["dof_damping"]),
            "dof_damping",
        )
        damping = jnp.asarray(damping_values, dtype=jnp.asarray(sys.dof.damping).dtype)
        sys = sys.replace(dof=sys.dof.replace(damping=damping))

    if "pole_length" in task.values:
        sys = _apply_inverted_pendulum_pole_length(
            sys,
            base_sys,
            float(task.values["pole_length"]),
        )

    if "limb_length" in task.values:
        sys = _apply_halfcheetah_limb_lengths(
            sys,
            base_sys,
            task.values["limb_length"],
        )

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


def _task_sampler_type(config: ContinualDynamicsConfig) -> str:
    return str(config.task_sampler.get("type", "independent"))


def _validate_task_sampler_config(config: ContinualDynamicsConfig) -> None:
    sampler_type = _task_sampler_type(config)
    if sampler_type == "independent":
        return
    if sampler_type != "halfcheetah_structured_asymmetric":
        raise ValueError(
            "task_sampler.type must be 'independent' or "
            f"'halfcheetah_structured_asymmetric'; got {sampler_type!r}."
        )
    if config.env_name != "halfcheetah":
        raise ValueError(
            "task_sampler.type='halfcheetah_structured_asymmetric' is only "
            "configured for halfcheetah."
        )

    mode_cycle = config.task_sampler.get("mode_cycle", HALFCHEETAH_STRUCTURED_MODES)
    if not isinstance(mode_cycle, (list, tuple)) or not mode_cycle:
        raise ValueError("task_sampler.mode_cycle must be a non-empty list.")
    unknown_modes = set(mode_cycle) - set(HALFCHEETAH_STRUCTURED_MODES)
    if unknown_modes:
        raise ValueError(
            f"Unknown halfcheetah structured mode(s): {sorted(unknown_modes)}"
        )

    _require_range_pair(
        config.task_sampler.get("intensity_range", [0.75, 1.0]),
        "task_sampler.intensity_range",
    )
    intensity_min, intensity_max = (
        float(config.task_sampler.get("intensity_range", [0.75, 1.0])[0]),
        float(config.task_sampler.get("intensity_range", [0.75, 1.0])[1]),
    )
    if intensity_min < 0.0 or intensity_max > 1.0:
        raise ValueError("task_sampler.intensity_range must lie within [0, 1].")

    jitter_fraction = float(config.task_sampler.get("jitter_fraction", 0.08))
    if jitter_fraction < 0.0 or jitter_fraction > 1.0:
        raise ValueError("task_sampler.jitter_fraction must lie within [0, 1].")


def _sample_halfcheetah_structured_task(
    key: jax.Array,
    config: ContinualDynamicsConfig,
    base_sys: Any,
    *,
    task_index: int,
) -> Dict[str, Any]:
    mode_cycle = tuple(
        config.task_sampler.get("mode_cycle", HALFCHEETAH_STRUCTURED_MODES)
    )
    mode = mode_cycle[(task_index - 1) % len(mode_cycle)]
    intensity_range = config.task_sampler.get("intensity_range", [0.75, 1.0])
    jitter_fraction = float(config.task_sampler.get("jitter_fraction", 0.08))
    key, intensity_key = jax.random.split(key)
    intensity = _sample_range(
        intensity_key,
        float(intensity_range[0]),
        float(intensity_range[1]),
        "linear_uniform",
    )

    values: Dict[str, Any] = {}
    actuator_names = _actuator_names(config.env_name)
    link_names = tuple(base_sys.link_names)

    for name in _enabled_parameters(config):
        key, param_key = jax.random.split(key)
        param_config = _parameter_config(config, name)
        sampler = _range_sampler(param_config)
        if name == "geom_friction_slide":
            values[name] = _sample_directed_scalar(
                param_key,
                float(param_config["default"]),
                float(param_config["min"]),
                float(param_config["max"]),
                _halfcheetah_scalar_direction(mode, name),
                sampler,
                intensity,
                jitter_fraction,
            )
        elif name == "gravity_z":
            values[name] = _sample_directed_scalar(
                param_key,
                float(param_config["default"]),
                float(param_config["min"]),
                float(param_config["max"]),
                _halfcheetah_scalar_direction(mode, name),
                sampler,
                intensity,
                jitter_fraction,
            )
        elif name == "actuator_gear":
            values[name] = _sample_directed_named_ranges(
                param_key,
                actuator_names,
                param_config,
                _halfcheetah_named_directions(mode, name, actuator_names),
                intensity,
                jitter_fraction,
            )
        elif name == "link_mass":
            values[name] = _sample_directed_named_ranges(
                param_key,
                link_names,
                param_config,
                _halfcheetah_named_directions(mode, name, link_names),
                intensity,
                jitter_fraction,
            )
        elif name == "limb_length":
            values[name] = _sample_directed_named_ranges(
                param_key,
                HALFCHEETAH_LIMB_NAMES,
                param_config,
                _halfcheetah_named_directions(mode, name, HALFCHEETAH_LIMB_NAMES),
                intensity,
                jitter_fraction,
            )
    return values


def _sample_directed_named_ranges(
    key: jax.Array,
    names: Tuple[str, ...],
    param_config: Dict[str, Any],
    directions: Dict[str, int],
    intensity: float,
    jitter_fraction: float,
) -> Dict[str, float]:
    keys = jax.random.split(key, len(names))
    sampler = _range_sampler(param_config)
    defaults = param_config["defaults"]
    ranges = param_config["ranges"]
    return {
        name: _sample_directed_scalar(
            keys[i],
            float(defaults[name]),
            float(ranges[name][0]),
            float(ranges[name][1]),
            directions.get(name, 0),
            sampler,
            intensity,
            jitter_fraction,
        )
        for i, name in enumerate(names)
    }


def _sample_directed_scalar(
    key: jax.Array,
    default: float,
    minval: float,
    maxval: float,
    direction: int,
    sampler: str,
    intensity: float,
    jitter_fraction: float,
) -> float:
    if direction == 0:
        low = _blend_range_value(default, minval, jitter_fraction, sampler)
        high = _blend_range_value(default, maxval, jitter_fraction, sampler)
    elif direction > 0:
        low_intensity = max(0.0, intensity - jitter_fraction)
        high_intensity = min(1.0, intensity + jitter_fraction)
        low = _blend_range_value(default, maxval, low_intensity, sampler)
        high = _blend_range_value(default, maxval, high_intensity, sampler)
    else:
        low_intensity = max(0.0, intensity - jitter_fraction)
        high_intensity = min(1.0, intensity + jitter_fraction)
        low = _blend_range_value(default, minval, high_intensity, sampler)
        high = _blend_range_value(default, minval, low_intensity, sampler)

    min_sample, max_sample = sorted((low, high))
    return _sample_range(key, min_sample, max_sample, sampler)


def _blend_range_value(default: float, target: float, fraction: float, sampler: str) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if sampler == "log_uniform":
        return float(np.exp(np.log(default) + fraction * (np.log(target) - np.log(default))))
    return float(default + fraction * (target - default))


def _halfcheetah_scalar_direction(mode: str, parameter_name: str) -> int:
    if parameter_name == "geom_friction_slide":
        if mode == "low_friction_high_gravity":
            return -1
        if mode == "high_friction_low_gravity":
            return 1
    if parameter_name == "gravity_z":
        if mode in {"heavy_torso_weak_legs", "low_friction_high_gravity"}:
            return -1
        if mode in {"light_torso_strong_legs", "high_friction_low_gravity"}:
            return 1
    return 0


def _halfcheetah_named_directions(
    mode: str,
    parameter_name: str,
    names: Tuple[str, ...],
) -> Dict[str, int]:
    directions = {name: 0 for name in names}

    if mode == "rear_long_front_short":
        if parameter_name in {"actuator_gear", "link_mass", "limb_length"}:
            _set_directions(directions, HALFCHEETAH_REAR_NAMES, 1)
            _set_directions(directions, HALFCHEETAH_FRONT_NAMES, -1)
    elif mode == "front_long_rear_short":
        if parameter_name in {"actuator_gear", "link_mass", "limb_length"}:
            _set_directions(directions, HALFCHEETAH_REAR_NAMES, -1)
            _set_directions(directions, HALFCHEETAH_FRONT_NAMES, 1)
    elif mode == "rear_strong_front_weak":
        if parameter_name == "actuator_gear":
            _set_directions(directions, HALFCHEETAH_REAR_NAMES, 1)
            _set_directions(directions, HALFCHEETAH_FRONT_NAMES, -1)
        elif parameter_name == "link_mass":
            _set_directions(directions, HALFCHEETAH_REAR_NAMES, -1)
            _set_directions(directions, HALFCHEETAH_FRONT_NAMES, 1)
    elif mode == "front_strong_rear_weak":
        if parameter_name == "actuator_gear":
            _set_directions(directions, HALFCHEETAH_REAR_NAMES, -1)
            _set_directions(directions, HALFCHEETAH_FRONT_NAMES, 1)
        elif parameter_name == "link_mass":
            _set_directions(directions, HALFCHEETAH_REAR_NAMES, 1)
            _set_directions(directions, HALFCHEETAH_FRONT_NAMES, -1)
    elif mode == "long_thigh_short_shin":
        if parameter_name in {"link_mass", "limb_length"}:
            _set_directions(directions, HALFCHEETAH_THIGH_NAMES, 1)
            _set_directions(directions, HALFCHEETAH_SHIN_NAMES, -1)
            _set_directions(directions, HALFCHEETAH_FOOT_NAMES, 0)
    elif mode == "short_thigh_long_shin":
        if parameter_name in {"link_mass", "limb_length"}:
            _set_directions(directions, HALFCHEETAH_THIGH_NAMES, -1)
            _set_directions(directions, HALFCHEETAH_SHIN_NAMES, 1)
            _set_directions(directions, HALFCHEETAH_FOOT_NAMES, 0)
    elif mode == "heavy_torso_weak_legs":
        if parameter_name == "actuator_gear":
            _set_directions(directions, HALFCHEETAH_LIMB_NAMES, -1)
        elif parameter_name == "link_mass":
            _set_directions(directions, ("torso",), 1)
            _set_directions(directions, HALFCHEETAH_LIMB_NAMES, 1)
    elif mode == "light_torso_strong_legs":
        if parameter_name == "actuator_gear":
            _set_directions(directions, HALFCHEETAH_LIMB_NAMES, 1)
        elif parameter_name == "link_mass":
            _set_directions(directions, ("torso",), -1)
            _set_directions(directions, HALFCHEETAH_LIMB_NAMES, -1)

    return directions


def _set_directions(
    directions: Dict[str, int],
    names: Tuple[str, ...],
    direction: int,
) -> None:
    for name in names:
        if name in directions:
            directions[name] = direction


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


def _dof_names(env_name: str) -> Tuple[str, ...]:
    try:
        return DOF_NAMES_BY_ENV[env_name]
    except KeyError as exc:
        raise ValueError(
            f"Continual dynamics dof_damping is not configured for env {env_name!r}."
        ) from exc


def _ordered_actuator_gear_values(values: Dict[str, Any]) -> Tuple[float, ...]:
    if not isinstance(values, dict):
        raise ValueError("actuator_gear task values must be a mapping.")

    return _ordered_named_values(
        values,
        _actuator_names_for_task(values),
        "actuator_gear",
    )


def _ordered_named_values(
    values: Dict[str, Any],
    names: Tuple[str, ...],
    label: str,
) -> Tuple[float, ...]:
    if not isinstance(values, dict):
        raise ValueError(f"{label} task values must be a mapping.")

    value_keys = set(values)
    if value_keys == set(names):
        return tuple(float(values[name]) for name in names)

    raise ValueError(
        f"{label} task values keys do not match expected names; "
        f"expected {sorted(names)}, got {sorted(value_keys)}."
    )


def _actuator_names_for_task(values: Dict[str, Any]) -> Tuple[str, ...]:
    value_keys = set(values)
    for names in ACTUATOR_NAMES_BY_ENV.values():
        if value_keys == set(names):
            return names

    raise ValueError(
        "actuator_gear task values keys do not match any supported env; "
        f"got {sorted(value_keys)}."
    )


def _dof_names_for_task(values: Dict[str, Any]) -> Tuple[str, ...]:
    value_keys = set(values)
    for names in DOF_NAMES_BY_ENV.values():
        if value_keys == set(names):
            return names

    raise ValueError(
        "dof_damping task values keys do not match any supported env; "
        f"got {sorted(value_keys)}."
    )


def _sample_range(
    key: jax.Array,
    minval: float,
    maxval: float,
    sampler: str,
) -> float:
    if sampler == "log_uniform":
        sample = jnp.exp(
            jax.random.uniform(key, (), minval=jnp.log(minval), maxval=jnp.log(maxval))
        )
    else:
        sample = jax.random.uniform(key, (), minval=minval, maxval=maxval)
    return float(jax.device_get(sample))


def _sample_named_ranges(
    key: jax.Array,
    names: Tuple[str, ...],
    ranges: Dict[str, Any],
    sampler: str,
) -> Dict[str, float]:
    keys = jax.random.split(key, len(names))
    return {
        name: _sample_range(
            keys[i],
            float(ranges[name][0]),
            float(ranges[name][1]),
            sampler,
        )
        for i, name in enumerate(names)
    }


def _range_sampler(param_config: Dict[str, Any]) -> str:
    sampler = str(param_config.get("sampler", "linear_uniform"))
    if sampler == "uniform":
        return "linear_uniform"
    return sampler


def _geom_friction_apply_mode(param_config: Dict[str, Any]) -> str:
    mode = str(param_config.get("apply_mode", "uniform_defaults"))
    if mode not in SUPPORTED_GEOM_FRICTION_APPLY_MODES:
        raise ValueError(
            "geom_friction_slide.apply_mode must be one of "
            f"{SUPPORTED_GEOM_FRICTION_APPLY_MODES}; got {mode!r}."
        )
    return mode


def _validate_range_sampler(
    sampler: str,
    minval: float,
    maxval: float,
    label: str,
) -> None:
    if sampler not in SUPPORTED_RANGE_SAMPLERS:
        raise ValueError(
            f"{label}.sampler must be one of {SUPPORTED_RANGE_SAMPLERS}; got {sampler!r}."
        )
    if sampler == "log_uniform" and (minval <= 0.0 or maxval <= 0.0):
        raise ValueError(f"{label} log_uniform bounds must be positive.")


def _validate_geom_friction_config(param_config: Dict[str, Any], base_sys: Any) -> None:
    _require_scalar_range(param_config, "geom_friction_slide")
    friction = np.asarray(jax.device_get(base_sys.geom_friction), dtype=np.float64)
    slide = friction[:, 0]
    mode = _geom_friction_apply_mode(param_config)
    has_uniform_defaults = np.allclose(slide, slide[0], atol=DEFAULT_TOLERANCE, rtol=0.0)

    if mode == "uniform_defaults":
        if not has_uniform_defaults:
            raise ValueError("Expected one default slide friction value across all geoms.")
        if "default" not in param_config:
            raise ValueError("geom_friction_slide.default must be set.")
        _assert_default_close(
            float(param_config.get("default")),
            float(slide[0]),
            "geom_friction_slide.default",
        )
        return

    if "default" in param_config:
        if not has_uniform_defaults:
            raise ValueError(
                "geom_friction_slide.default must be omitted for shared_scalar "
                "when base geom frictions are nonuniform."
            )
        _assert_default_close(
            float(param_config.get("default")),
            float(slide[0]),
            "geom_friction_slide.default",
        )


def _validate_gravity_z_config(param_config: Dict[str, Any], base_sys: Any) -> None:
    _require_scalar_range(param_config, "gravity_z")
    gravity = np.asarray(jax.device_get(base_sys.gravity), dtype=np.float64)
    _assert_default_close(
        float(param_config.get("default")),
        float(gravity[2]),
        "gravity_z.default",
    )


def _validate_pole_length_config(
    param_config: Dict[str, Any],
    env_name: str,
    base_sys: Any,
) -> None:
    if env_name != "inverted_pendulum":
        raise ValueError("pole_length is only configured for inverted_pendulum.")
    _require_scalar_range(param_config, "pole_length")
    if float(param_config["min"]) <= 0.0:
        raise ValueError("pole_length min must be positive.")
    _assert_default_close(
        float(param_config.get("default")),
        _default_pole_length(base_sys),
        "pole_length.default",
    )


def _validate_limb_length_config(
    param_config: Dict[str, Any],
    env_name: str,
    base_sys: Any,
) -> None:
    if env_name != "halfcheetah":
        raise ValueError("limb_length is only configured for halfcheetah.")
    defaults = _default_halfcheetah_limb_lengths(base_sys)
    _validate_named_vector_config(
        param_config,
        HALFCHEETAH_LIMB_NAMES,
        np.asarray([defaults[name] for name in HALFCHEETAH_LIMB_NAMES], dtype=np.float64),
        "limb_length",
    )
    for name, range_pair in param_config["ranges"].items():
        if float(range_pair[0]) <= 0.0:
            raise ValueError(f"limb_length.ranges.{name} min must be positive.")


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
    sampler = _range_sampler(param_config)

    for i, name in enumerate(names):
        _assert_default_close(float(config_defaults[name]), float(defaults[i]), f"{label}.{name}")
        _require_range_pair(ranges[name], f"{label}.ranges.{name}")
        _validate_range_sampler(
            sampler,
            float(ranges[name][0]),
            float(ranges[name][1]),
            f"{label}.ranges.{name}",
        )


def _require_scalar_range(param_config: Dict[str, Any], label: str) -> None:
    if "min" not in param_config or "max" not in param_config:
        raise ValueError(f"{label} must define min and max.")
    minval, maxval = float(param_config["min"]), float(param_config["max"])
    if minval > maxval:
        raise ValueError(f"{label} min must be <= max.")
    _validate_range_sampler(_range_sampler(param_config), minval, maxval, label)


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


def _link_index(sys: Any, link_name: str) -> int:
    try:
        return tuple(sys.link_names).index(link_name)
    except ValueError as exc:
        raise ValueError(f"Expected link {link_name!r} in system.") from exc


def _pole_geom_index(sys: Any) -> int:
    pole_idx = _link_index(sys, "pole")
    geom_bodyid = np.asarray(sys.geom_bodyid)
    matches = np.flatnonzero(geom_bodyid == pole_idx + 1)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one pole geom for inverted_pendulum; "
            f"found {len(matches)}."
        )
    return int(matches[0])


def _single_link_geom_index(sys: Any, link_name: str) -> int:
    link_idx = _link_index(sys, link_name)
    geom_bodyid = np.asarray(sys.geom_bodyid)
    matches = np.flatnonzero(geom_bodyid == link_idx + 1)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one geom for link {link_name!r}; found {len(matches)}."
        )
    return int(matches[0])


def _default_pole_length(sys: Any) -> float:
    geom_idx = _pole_geom_index(sys)
    geom_size = np.asarray(jax.device_get(sys.geom_size), dtype=np.float64)
    return float(2.0 * geom_size[geom_idx, 1])


def _default_halfcheetah_limb_lengths(sys: Any) -> Dict[str, float]:
    geom_size = np.asarray(jax.device_get(sys.geom_size), dtype=np.float64)
    return {
        name: float(2.0 * geom_size[_single_link_geom_index(sys, name), 1])
        for name in HALFCHEETAH_LIMB_NAMES
    }


def _apply_inverted_pendulum_pole_length(
    sys: Any,
    base_sys: Any,
    pole_length: float,
) -> Any:
    if pole_length <= 0.0:
        raise ValueError("pole_length must be positive.")

    pole_idx = _link_index(base_sys, "pole")
    geom_idx = _pole_geom_index(base_sys)
    default_length = _default_pole_length(base_sys)
    length_ratio = pole_length / default_length
    inertia_length_scale = length_ratio * length_ratio

    inertia = sys.link.inertia
    inertia_pos = jnp.asarray(inertia.transform.pos)
    new_pole_com = (
        jnp.asarray(base_sys.link.inertia.transform.pos[pole_idx]) * length_ratio
    )
    inertia_transform = inertia.transform.replace(
        pos=inertia_pos.at[pole_idx].set(new_pole_com)
    )

    inertia_i = jnp.asarray(inertia.i)
    pole_i = inertia_i[pole_idx]
    pole_i = pole_i.at[0, 0].set(pole_i[0, 0] * inertia_length_scale)
    pole_i = pole_i.at[1, 1].set(pole_i[1, 1] * inertia_length_scale)
    inertia_i = inertia_i.at[pole_idx].set(pole_i)
    inertia = inertia.replace(transform=inertia_transform, i=inertia_i)

    geom_pos = jnp.asarray(sys.geom_pos)
    new_geom_pos = jnp.asarray(base_sys.geom_pos[geom_idx]) * length_ratio
    geom_pos = geom_pos.at[geom_idx].set(new_geom_pos)

    geom_size = jnp.asarray(sys.geom_size)
    base_geom_size = jnp.asarray(base_sys.geom_size[geom_idx])
    new_half_length = base_geom_size[1] * length_ratio
    geom_size = geom_size.at[geom_idx, 1].set(new_half_length)

    geom_rbound = jnp.asarray(sys.geom_rbound)
    new_rbound = base_geom_size[0] + new_half_length
    geom_rbound = geom_rbound.at[geom_idx].set(new_rbound)

    mj_model = None
    if sys.mj_model is not None:
        mj_model = copy.copy(sys.mj_model)
        mj_model.geom_pos[geom_idx] = np.asarray(jax.device_get(new_geom_pos))
        mj_model.geom_size[geom_idx] = np.asarray(jax.device_get(geom_size[geom_idx]))
        mj_model.geom_rbound[geom_idx] = float(jax.device_get(new_rbound))
        body_id = int(np.asarray(base_sys.geom_bodyid)[geom_idx])
        mj_model.body_ipos[body_id] = np.asarray(jax.device_get(new_pole_com))
        mj_model.body_inertia[body_id] = np.diag(np.asarray(jax.device_get(pole_i)))

    return sys.replace(
        link=sys.link.replace(inertia=inertia),
        geom_pos=geom_pos,
        geom_size=geom_size,
        geom_rbound=geom_rbound,
        mj_model=mj_model,
    )


def _apply_halfcheetah_limb_lengths(
    sys: Any,
    base_sys: Any,
    limb_lengths: Dict[str, Any],
) -> Any:
    if not isinstance(limb_lengths, dict):
        raise ValueError("limb_length task values must be a mapping.")
    _assert_exact_keys(limb_lengths, HALFCHEETAH_LIMB_NAMES, "limb_length")

    default_lengths = _default_halfcheetah_limb_lengths(base_sys)
    link_names = tuple(base_sys.link_names)
    link_parents = tuple(np.asarray(base_sys.link_parents))

    inertia = sys.link.inertia
    inertia_transform_pos = jnp.asarray(inertia.transform.pos)
    inertia_i = jnp.asarray(inertia.i)
    geom_pos = jnp.asarray(sys.geom_pos)
    geom_size = jnp.asarray(sys.geom_size)
    geom_rbound = jnp.asarray(sys.geom_rbound)
    link_transform_pos = jnp.asarray(sys.link.transform.pos)

    for limb_name in HALFCHEETAH_LIMB_NAMES:
        limb_length = float(limb_lengths[limb_name])
        if limb_length <= 0.0:
            raise ValueError(f"limb_length for {limb_name!r} must be positive.")

        ratio = limb_length / default_lengths[limb_name]
        link_idx = link_names.index(limb_name)
        geom_idx = _single_link_geom_index(base_sys, limb_name)

        inertia_transform_pos = inertia_transform_pos.at[link_idx].set(
            jnp.asarray(base_sys.link.inertia.transform.pos[link_idx]) * ratio
        )
        inertia_i = inertia_i.at[link_idx].set(inertia_i[link_idx] * (ratio * ratio))

        geom_pos = geom_pos.at[geom_idx].set(
            jnp.asarray(base_sys.geom_pos[geom_idx]) * ratio
        )
        geom_size = geom_size.at[geom_idx, 1].set(limb_length * 0.5)
        geom_rbound = geom_rbound.at[geom_idx].set(
            geom_size[geom_idx, 0] + limb_length * 0.5
        )

        for child_idx, parent_idx in enumerate(link_parents):
            if parent_idx == link_idx:
                link_transform_pos = link_transform_pos.at[child_idx].set(
                    jnp.asarray(base_sys.link.transform.pos[child_idx]) * ratio
                )

    inertia = inertia.replace(
        transform=inertia.transform.replace(pos=inertia_transform_pos),
        i=inertia_i,
    )
    link = sys.link.replace(
        transform=sys.link.transform.replace(pos=link_transform_pos),
        inertia=inertia,
    )

    mj_model = None
    if sys.mj_model is not None:
        mj_model = copy.copy(sys.mj_model)
        for limb_name in HALFCHEETAH_LIMB_NAMES:
            link_idx = link_names.index(limb_name)
            geom_idx = _single_link_geom_index(base_sys, limb_name)
            body_id = int(np.asarray(base_sys.geom_bodyid)[geom_idx])
            mj_model.geom_pos[geom_idx] = np.asarray(jax.device_get(geom_pos[geom_idx]))
            mj_model.geom_size[geom_idx] = np.asarray(jax.device_get(geom_size[geom_idx]))
            mj_model.geom_rbound[geom_idx] = float(jax.device_get(geom_rbound[geom_idx]))
            mj_model.body_ipos[body_id] = np.asarray(
                jax.device_get(inertia_transform_pos[link_idx])
            )
            mj_model.body_inertia[body_id] = np.diag(
                np.asarray(jax.device_get(inertia_i[link_idx]))
            )
            for child_idx, parent_idx in enumerate(link_parents):
                if parent_idx == link_idx:
                    mj_model.body_pos[child_idx + 1] = np.asarray(
                        jax.device_get(link_transform_pos[child_idx])
                    )

    return sys.replace(
        link=link,
        geom_pos=geom_pos,
        geom_size=geom_size,
        geom_rbound=geom_rbound,
        mj_model=mj_model,
    )
