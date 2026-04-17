import copy
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pixelbrax" / "brax"))

from brax import envs
from pixelbrax.continual_dynamics import (
    DynamicsTask,
    apply_dynamics_task,
    load_continual_dynamics_config,
    make_dynamics_task,
    validate_continual_dynamics_config,
)


def _halfcheetah_sys():
    return envs.create(
        env_name="halfcheetah",
        backend="spring",
        action_repeat=1,
    ).sys


def _inverted_pendulum_sys():
    return envs.create(
        env_name="inverted_pendulum",
        backend="generalized",
        action_repeat=1,
    ).sys


def _walker2d_sys():
    return envs.create(
        env_name="walker2d",
        backend="spring",
        action_repeat=1,
    ).sys


def _template_config():
    return load_continual_dynamics_config(
        str(REPO_ROOT / "configs" / "continual" / "halfcheetah_dynamics.yaml")
    )


def _inverted_pendulum_config():
    return load_continual_dynamics_config(
        str(REPO_ROOT / "configs" / "continual" / "inverted_pendulum_dynamics.yaml")
    )


def _walker2d_config():
    return load_continual_dynamics_config(
        str(REPO_ROOT / "configs" / "continual" / "walker2d_dynamics.yaml")
    )


def test_task_zero_uses_defaults_and_sampling_is_repeatable():
    base_sys = _halfcheetah_sys()
    config = _template_config()

    updates_per_task = validate_continual_dynamics_config(
        config,
        env_name="halfcheetah",
        backend="spring",
        n_envs=2,
        num_steps=2,
        base_sys=base_sys,
    )
    assert updates_per_task == config.switch_every_env_steps // 4

    task0 = make_dynamics_task(config, base_sys, task_index=0, schedule_seed=7)
    assert task0.is_default
    assert task0.values["geom_friction_slide"] == pytest.approx(0.4)
    assert task0.values["actuator_gear"]["bthigh"] == pytest.approx(120.0)
    assert task0.values["link_mass"]["torso"] == pytest.approx(6.2502093)
    assert apply_dynamics_task(base_sys, task0) is base_sys

    task1_a = make_dynamics_task(config, base_sys, task_index=1, schedule_seed=7)
    task1_b = make_dynamics_task(config, base_sys, task_index=1, schedule_seed=7)
    task2 = make_dynamics_task(config, base_sys, task_index=2, schedule_seed=7)

    assert task1_a == task1_b
    assert task1_a != task2
    assert not task1_a.is_default
    assert 0.2 <= task1_a.values["geom_friction_slide"] <= 1.0
    assert 90.0 <= task1_a.values["actuator_gear"]["bthigh"] <= 150.0
    assert 5.3127 <= task1_a.values["link_mass"]["torso"] <= 7.1877


def test_inverted_pendulum_config_uses_defaults_and_samples_tasks():
    base_sys = _inverted_pendulum_sys()
    base_mass = np.asarray(jax.device_get(base_sys.link.inertia.mass))
    base_inertia = np.asarray(jax.device_get(base_sys.link.inertia.i))
    base_pole_com = np.asarray(jax.device_get(base_sys.link.inertia.transform.pos))[1]
    base_geom_size = np.asarray(jax.device_get(base_sys.geom_size))
    base_pole_length = 2.0 * base_geom_size[2, 1]
    config = _inverted_pendulum_config()

    updates_per_task = validate_continual_dynamics_config(
        config,
        env_name="inverted_pendulum",
        backend="generalized",
        n_envs=2,
        num_steps=2,
        base_sys=base_sys,
    )
    assert updates_per_task == config.switch_every_env_steps // 4

    task0 = make_dynamics_task(config, base_sys, task_index=0, schedule_seed=7)
    assert task0.is_default
    assert "geom_friction_slide" not in task0.values
    assert task0.values["actuator_gear"]["slide"] == pytest.approx(100.0)
    assert task0.values["link_mass"]["cart"] == pytest.approx(10.4719753)
    assert task0.values["link_mass"]["pole"] == pytest.approx(5.0185914)
    assert task0.values["gravity_z"] == pytest.approx(-9.81)
    assert task0.values["dof_damping"]["slider"] == pytest.approx(1.0)
    assert task0.values["dof_damping"]["hinge"] == pytest.approx(1.0)
    assert task0.values["pole_length"] == pytest.approx(0.6, abs=1e-5)
    assert apply_dynamics_task(base_sys, task0) is base_sys

    task1_a = make_dynamics_task(config, base_sys, task_index=1, schedule_seed=7)
    task1_b = make_dynamics_task(config, base_sys, task_index=1, schedule_seed=7)
    assert task1_a == task1_b
    assert not task1_a.is_default
    assert "geom_friction_slide" not in task1_a.values
    assert 40.0 <= task1_a.values["actuator_gear"]["slide"] <= 140.0
    assert 6.0 <= task1_a.values["link_mass"]["cart"] <= 16.0
    assert 2.5 <= task1_a.values["link_mass"]["pole"] <= 8.0
    assert -18.0 <= task1_a.values["gravity_z"] <= -7.0
    assert 0.25 <= task1_a.values["dof_damping"]["slider"] <= 6.0
    assert 0.25 <= task1_a.values["dof_damping"]["hinge"] <= 4.0
    assert 0.35 <= task1_a.values["pole_length"] <= 0.9

    new_sys = apply_dynamics_task(base_sys, task1_a)
    assert new_sys.q_size() == base_sys.q_size()
    assert new_sys.qd_size() == base_sys.qd_size()
    assert new_sys.act_size() == base_sys.act_size()
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.geom_friction)),
        np.asarray(jax.device_get(base_sys.geom_friction)),
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.actuator.gear)),
        np.array([task1_a.values["actuator_gear"]["slide"]]),
    )
    expected_mass = np.array(
        [
            task1_a.values["link_mass"]["cart"],
            task1_a.values["link_mass"]["pole"],
        ]
    )
    length_ratio = task1_a.values["pole_length"] / base_pole_length
    expected_inertia = base_inertia * (expected_mass / base_mass)[:, None, None]
    expected_inertia[1, 0, 0] *= length_ratio * length_ratio
    expected_inertia[1, 1, 1] *= length_ratio * length_ratio
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.mass)),
        expected_mass,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.i)),
        expected_inertia,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.gravity)),
        np.array([0.0, 0.0, task1_a.values["gravity_z"]]),
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.dof.damping)),
        np.array(
            [
                task1_a.values["dof_damping"]["slider"],
                task1_a.values["dof_damping"]["hinge"],
            ]
        ),
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.transform.pos))[1],
        base_pole_com * length_ratio,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.geom_size))[2, 1],
        base_geom_size[2, 1] * length_ratio,
        rtol=1e-6,
        atol=1e-6,
    )
    assert new_sys.mj_model is not base_sys.mj_model
    assert new_sys.mj_model.geom_size[2, 1] == pytest.approx(
        base_geom_size[2, 1] * length_ratio
    )


def test_walker2d_config_uses_defaults_and_samples_tasks():
    base_sys = _walker2d_sys()
    base_mass = np.asarray(jax.device_get(base_sys.link.inertia.mass))
    base_inertia = np.asarray(jax.device_get(base_sys.link.inertia.i))
    config = _walker2d_config()

    updates_per_task = validate_continual_dynamics_config(
        config,
        env_name="walker2d",
        backend="spring",
        n_envs=2,
        num_steps=2,
        base_sys=base_sys,
    )
    assert updates_per_task == config.switch_every_env_steps // 4

    task0 = make_dynamics_task(config, base_sys, task_index=0, schedule_seed=7)
    assert task0.is_default
    assert "geom_friction_slide" not in task0.values
    assert task0.values["actuator_gear"]["thigh"] == pytest.approx(100.0)
    assert task0.values["actuator_gear"]["foot_left"] == pytest.approx(100.0)
    assert task0.values["link_mass"]["torso"] == pytest.approx(3.6651914)
    assert task0.values["link_mass"]["thigh"] == pytest.approx(4.0578904)
    assert task0.values["link_mass"]["foot_left"] == pytest.approx(3.1667254)
    assert apply_dynamics_task(base_sys, task0) is base_sys

    task1_a = make_dynamics_task(config, base_sys, task_index=1, schedule_seed=7)
    task1_b = make_dynamics_task(config, base_sys, task_index=1, schedule_seed=7)
    assert task1_a == task1_b
    assert not task1_a.is_default
    assert "geom_friction_slide" not in task1_a.values
    assert 50.0 <= task1_a.values["actuator_gear"]["thigh"] <= 150.0
    assert 50.0 <= task1_a.values["actuator_gear"]["foot_left"] <= 150.0
    assert 2.5656 <= task1_a.values["link_mass"]["torso"] <= 4.7647
    assert 2.8405 <= task1_a.values["link_mass"]["thigh"] <= 5.2753
    assert 2.2167 <= task1_a.values["link_mass"]["foot_left"] <= 4.1167

    new_sys = apply_dynamics_task(base_sys, task1_a)
    assert new_sys.q_size() == base_sys.q_size()
    assert new_sys.qd_size() == base_sys.qd_size()
    assert new_sys.act_size() == base_sys.act_size()
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.actuator.gear)),
        np.array(
            [
                task1_a.values["actuator_gear"]["thigh"],
                task1_a.values["actuator_gear"]["leg"],
                task1_a.values["actuator_gear"]["foot"],
                task1_a.values["actuator_gear"]["thigh_left"],
                task1_a.values["actuator_gear"]["leg_left"],
                task1_a.values["actuator_gear"]["foot_left"],
            ]
        ),
    )
    expected_mass = np.array(
        [
            task1_a.values["link_mass"]["torso"],
            task1_a.values["link_mass"]["thigh"],
            task1_a.values["link_mass"]["leg"],
            task1_a.values["link_mass"]["foot"],
            task1_a.values["link_mass"]["thigh_left"],
            task1_a.values["link_mass"]["leg_left"],
            task1_a.values["link_mass"]["foot_left"],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.mass)),
        expected_mass,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.i)),
        base_inertia * (expected_mass / base_mass)[:, None, None],
        rtol=1e-6,
        atol=1e-6,
    )


def test_apply_sampled_task_changes_only_dynamics_shapes():
    base_sys = _halfcheetah_sys()
    base_mass = np.asarray(jax.device_get(base_sys.link.inertia.mass))
    base_inertia = np.asarray(jax.device_get(base_sys.link.inertia.i))

    new_masses = {
        name: float(base_mass[i] * 1.1)
        for i, name in enumerate(base_sys.link_names)
    }
    task = DynamicsTask(
        task_index=1,
        is_default=False,
        values={
            "geom_friction_slide": 0.7,
            "actuator_gear": {
                "bthigh": 121.0,
                "bshin": 91.0,
                "bfoot": 61.0,
                "fthigh": 122.0,
                "fshin": 101.0,
                "ffoot": 102.0,
            },
            "link_mass": new_masses,
        },
    )

    new_sys = apply_dynamics_task(base_sys, task)

    assert new_sys.q_size() == base_sys.q_size()
    assert new_sys.qd_size() == base_sys.qd_size()
    assert new_sys.act_size() == base_sys.act_size()
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.geom_friction))[:, 0],
        np.full(base_sys.geom_friction.shape[0], 0.7),
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.actuator.gear)),
        np.array([121.0, 91.0, 61.0, 122.0, 101.0, 102.0]),
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.mass)),
        base_mass * 1.1,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(new_sys.link.inertia.i)),
        base_inertia * 1.1,
        rtol=1e-6,
        atol=1e-6,
    )


def test_validation_rejects_non_divisible_switch_steps():
    base_sys = _halfcheetah_sys()
    config = replace(_template_config(), switch_every_env_steps=5)

    with pytest.raises(ValueError, match="exact multiple"):
        validate_continual_dynamics_config(
            config,
            env_name="halfcheetah",
            backend="spring",
            n_envs=2,
            num_steps=2,
            base_sys=base_sys,
        )


def test_validation_rejects_stale_defaults():
    base_sys = _halfcheetah_sys()
    config = _template_config()
    params = copy.deepcopy(config.parameters)
    params["actuator_gear"]["defaults"]["bthigh"] = 1.0
    stale_config = replace(config, parameters=params)

    with pytest.raises(ValueError, match="Stale default"):
        validate_continual_dynamics_config(
            stale_config,
            env_name="halfcheetah",
            backend="spring",
            n_envs=2,
            num_steps=2,
            base_sys=base_sys,
        )
