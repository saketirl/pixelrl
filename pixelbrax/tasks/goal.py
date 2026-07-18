"""Goal-reaching locomotion tasks for PixelBrax.

These environments are kept outside the Brax submodule and registered at
runtime.  The robot definitions are loaded from the pinned Brax assets, then
augmented with a non-contact visual target.
"""

from __future__ import annotations

import math as _math
import os
import xml.etree.ElementTree as ET
from typing import Tuple

from brax import actuator
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco


HUMANOID_TARGET_Z = 1.25


def _brax_asset_path(filename: str) -> epath.Path:
    return epath.resource_path("brax") / "envs/assets" / filename


def _set_init_qpos_length(root: ET.Element, extra_qpos: int) -> None:
    custom = root.find("custom")
    if custom is None:
        return
    init_qpos = custom.find("./numeric[@name='init_qpos']")
    if init_qpos is not None:
        data = init_qpos.get("data", "")
        init_qpos.set("data", f"{data} {' '.join(['0.0'] * extra_qpos)}".strip())


def _add_target_body(worldbody: ET.Element, target_z: float) -> None:
    target = ET.SubElement(worldbody, "body", name="target", pos=f"0 0 {target_z}")
    ET.SubElement(
        target,
        "joint",
        name="target_x",
        type="slide",
        axis="1 0 0",
        limited="false",
        damping="0",
        armature="0",
    )
    ET.SubElement(
        target,
        "joint",
        name="target_y",
        type="slide",
        axis="0 1 0",
        limited="false",
        damping="0",
        armature="0",
    )
    ET.SubElement(
        target,
        "geom",
        name="target",
        type="sphere",
        size="0.4",
        contype="0",
        conaffinity="0",
        rgba="0.1 0.8 0.2 1.0",
    )


def _goal_xml(asset_filename: str, target_z: float) -> bytes:
    xml_path = _brax_asset_path(asset_filename)
    root = ET.parse(os.fspath(xml_path)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{asset_filename} has no worldbody")

    _add_target_body(worldbody, target_z)
    _set_init_qpos_length(root, extra_qpos=2)
    return ET.tostring(root)


class AntGoal(PipelineEnv):
    """Ant with a random non-contact visual target and dense goal reward."""

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        goal_radius=10.0,
        progress_reward_scale=10.0,
        success_reward=50.0,
        distance_reward_scale=0.0,
        **kwargs,
    ):
        sys = mjcf.loads(_goal_xml("ant.xml", target_z=0.01))

        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            sys = sys.replace(
                actuator=sys.actuator.replace(gear=200 * jp.ones_like(sys.actuator.gear))
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._goal_radius = goal_radius
        self._progress_reward_scale = progress_reward_scale
        self._success_reward = success_reward
        self._distance_reward_scale = distance_reward_scale

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        _, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0.0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "reward_goal": zero,
            "reward_progress": zero,
            "reward_success": zero,
            "reward_distance": zero,
            "progress": zero,
            "prev_dist": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        healthy_reward = (
            self._healthy_reward
            if self._terminate_when_unhealthy
            else self._healthy_reward * is_healthy
        )
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        target_pos = pipeline_state.x.pos[-1][:2]
        prev_dist = jp.linalg.norm(pipeline_state0.x.pos[0, :2] - target_pos)
        dist = jp.linalg.norm(pipeline_state.x.pos[0, :2] - target_pos)
        progress = prev_dist - dist
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        progress_reward = self._progress_reward_scale * progress
        success_reward = self._success_reward * success
        distance_reward = -self._distance_reward_scale * dist
        reward = (
            progress_reward
            + healthy_reward
            - ctrl_cost
            - contact_cost
            + success_reward
            + distance_reward
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_goal=progress_reward,
            reward_progress=progress_reward,
            reward_success=success_reward,
            reward_distance=distance_reward,
            progress=progress,
            prev_dist=prev_dist,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]
        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jp.concatenate([qpos, qvel, target_pos])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        rng, rng1 = jax.random.split(rng)
        angle = jp.pi * 2.0 * jax.random.uniform(rng1)
        target = self._goal_radius * jp.array([jp.cos(angle), jp.sin(angle)])
        return rng, target


class HumanoidGoal(PipelineEnv):
    """Humanoid with a random non-contact visual target and dense goal reward."""

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=False,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        min_goal_dist=5.0,
        max_goal_dist=10.0,
        max_goal_angle=jp.pi / 4,
        goal_pos=(5.0 * _math.cos(_math.pi / 4), 5.0 * _math.sin(_math.pi / 4)),
        progress_reward_scale=10.0,
        success_reward=25.0,
        distance_reward_scale=0.0,
        heading_reward_scale=0.0,
        upright_stability_scale=0.0,
        **kwargs,
    ):
        sys = mjcf.loads(_goal_xml("humanoid.xml", target_z=HUMANOID_TARGET_Z))

        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.0015})
            n_frames = 10
            gear = jp.array(
                [
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            )
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._min_goal_dist = min_goal_dist
        self._max_goal_dist = max_goal_dist
        self._max_goal_angle = max_goal_angle
        self._goal_pos = jp.array(goal_pos, dtype=float)
        self._heading_reward_scale = heading_reward_scale
        self._upright_stability_scale = upright_stability_scale
        self._progress_reward_scale = progress_reward_scale
        self._success_reward = success_reward
        self._distance_reward_scale = distance_reward_scale

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        target = self._goal_pos
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-2:].set(0.0)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "reward_goal": zero,
            "reward_progress": zero,
            "reward_success": zero,
            "reward_distance": zero,
            "reward_heading": zero,
            "reward_stability": zero,
            "progress": zero,
            "prev_dist": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        torso_z = pipeline_state.x.pos[0, 2]
        upright_reward = 0.1 * self._healthy_reward * jp.clip(torso_z / min_z, 0.0, 1.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state, action)
        prev_obs = state.obs
        prev_dist = jp.linalg.norm(prev_obs[:3] - prev_obs[-3:])
        dist = jp.linalg.norm(obs[:3] - obs[-3:])
        progress = prev_dist - dist
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        progress_reward = self._progress_reward_scale * progress
        success_reward = self._success_reward * success
        distance_reward = -self._distance_reward_scale * dist

        # Heading reward: dot product of facing direction with goal direction
        root_quat = pipeline_state.q[3:7]
        facing_world = math.rotate(jp.array([1.0, 0.0, 0.0]), root_quat)
        facing_dir = facing_world[:2] / (jp.linalg.norm(facing_world[:2]) + 1e-8)
        goal_vec = pipeline_state.x.pos[-1][:2] - com_after[:2]
        goal_dir = goal_vec / (jp.linalg.norm(goal_vec) + 1e-8)
        heading_reward = self._heading_reward_scale * jp.dot(facing_dir, goal_dir)

        # Upright stability: orientation alignment + angular velocity penalty
        up_world = math.rotate(jp.array([0.0, 0.0, 1.0]), root_quat)
        orientation_reward = self._upright_stability_scale * jp.clip(up_world[2], 0.0, 1.0)
        root_ang_vel = pipeline_state.qd[3:6]
        ang_vel_penalty = self._upright_stability_scale * 0.1 * jp.sum(jp.square(root_ang_vel))
        stability_reward = orientation_reward - ang_vel_penalty

        reward = (
            progress_reward
            + upright_reward
            - ctrl_cost
            + success_reward
            + distance_reward
            + heading_reward
            + stability_reward
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=upright_reward,
            reward_goal=progress_reward,
            reward_progress=progress_reward,
            reward_success=success_reward,
            reward_distance=distance_reward,
            reward_heading=heading_reward,
            reward_stability=stability_reward,
            progress=progress,
            prev_dist=prev_dist,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            dist=dist,
            success=success,
            success_easy=success_easy,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        target_pos = pipeline_state.x.pos[-1][:2]
        target = jp.concatenate([target_pos, jp.array([HUMANOID_TARGET_Z])])
        return jp.concatenate(
            [
                position,
                velocity,
                com_inertia.ravel(),
                com_velocity.ravel(),
                qfrc_actuator,
                target,
            ]
        )

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ["spring", "positional"]:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        return com, inertia, mass_sum, x_i

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = jax.random.uniform(
            rng1, minval=self._min_goal_dist, maxval=self._max_goal_dist
        )
        angle = jax.random.uniform(
            rng2, minval=-self._max_goal_angle, maxval=self._max_goal_angle
        )
        target = dist * jp.array([jp.cos(angle), jp.sin(angle)])
        return rng, target
