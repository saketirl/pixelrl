"""
Visualize upstairs and downstairs actor optimization in three dimensions.

This script specializes the matched upstairs/downstairs LQR comparison to
``ds=3`` and ``da=1``.  In this setting the downstairs Stiefel policy is a
point on the unit sphere.  The two policies are represented in the same
latent state-to-action coordinates:

    downstairs:  p_down = Wc_col.T
    upstairs:    p_up   = (W3 W2 W1) M
    optimum:     p_opt  = -K_opt

The upstairs point is deliberately not normalized.  Its full observation-
space policy has unit norm under Cayley updates, but its projection into the
three-dimensional latent subspace can lie inside the sphere.  Showing the raw
projection makes that part of the optimization dynamics visible.

Example:
    uv run python lqr_ac_compare/visualize_optimization_sphere.py \
        --iters 500 --save_plot optimization_sphere.png \
        --save_animation optimization_sphere.gif
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from compare_upstairs_downstairs import construct_compatible_lqr
from downstairs_learning_batch import batch_multistep_update
from downstairs_models import DownstairsActorShallowT2, DownstairsCriticShallowT2
from lqr_env import LQRParams, LQRUpDownEnv
from upstairs_learning_batch import upstairs_batch_multistep_update
from upstairs_models import ActorDLN_L3, CriticDLN_L3


DOWN_COLOR = "#20A4F3"
UP_COLOR = "#F18F01"
OPT_COLOR = "#FFD166"
ALIGN_COLOR = "#8E6CFF"
EPS = 1e-15


@dataclass
class OptimizationTrace:
    """Policy snapshots and per-update diagnostics for one learner."""

    policies: np.ndarray
    rewards: np.ndarray
    mse: np.ndarray
    actor_grad_norm: np.ndarray
    critic_grad_norm: np.ndarray
    actor_ortho_error: np.ndarray
    critic_ortho_error: np.ndarray


def _stat(stats: Dict[str, float], key: str) -> float:
    """Read an update diagnostic while tolerating a skipped update."""
    return float(stats.get(key, np.nan))


def _build_trace(
    policies: List[np.ndarray],
    rewards: List[float],
    mse: List[float],
    actor_grad_norm: List[float],
    critic_grad_norm: List[float],
    actor_ortho_error: List[float],
    critic_ortho_error: List[float],
) -> OptimizationTrace:
    return OptimizationTrace(
        policies=np.asarray(policies, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        mse=np.asarray(mse, dtype=float),
        actor_grad_norm=np.asarray(actor_grad_norm, dtype=float),
        critic_grad_norm=np.asarray(critic_grad_norm, dtype=float),
        actor_ortho_error=np.asarray(actor_ortho_error, dtype=float),
        critic_ortho_error=np.asarray(critic_ortho_error, dtype=float),
    )


def run_downstairs(
    env: LQRUpDownEnv,
    *,
    iters: int,
    n_steps: int,
    beta: float,
    eta_actor: float,
    eta_critic: float,
    exploration_std: float,
    init_seed: int,
    noise_seed: int,
    use_generator: bool,
    use_sgd: bool,
    log_every: int,
) -> Tuple[OptimizationTrace, Dict[str, np.ndarray]]:
    """Run downstairs learning and retain every effective policy vector."""
    actor = DownstairsActorShallowT2(ds=env.ds, da=1, seed=init_seed)
    critic = DownstairsCriticShallowT2(ds=env.ds, da=1, seed=init_seed + 100)
    critic.Zb = env.H.copy()
    env.rng = np.random.default_rng(noise_seed)

    init_params = {
        "Wc_col": actor.Wc_col.copy(),
        "Ub": critic.Ub.copy(),
        "Zb": critic.Zb.copy(),
    }

    policies = [actor.Wc.reshape(env.ds).copy()]
    rewards: List[float] = []
    mse: List[float] = []
    actor_grad_norm: List[float] = []
    critic_grad_norm: List[float] = []
    actor_ortho_error: List[float] = []
    critic_ortho_error: List[float] = []

    for iteration in range(iters):

        def policy(obs):
            return actor.act(obs["downstairs"])

        traj = env.rollout(
            policy,
            exploration_std=exploration_std,
            use_upstairs=False,
        )
        stats = batch_multistep_update(
            actor,
            critic,
            traj["s"],
            traj["a"],
            traj["r"],
            beta=beta,
            dt=env.dt,
            L=n_steps,
            eta_actor=eta_actor,
            eta_critic=eta_critic,
            use_generator=use_generator,
            use_sgd=use_sgd,
        )

        policies.append(actor.Wc.reshape(env.ds).copy())
        rewards.append(float(np.mean(traj["r"])))
        mse.append(_stat(stats, "mse"))
        actor_grad_norm.append(_stat(stats, "G_Wc_norm"))
        critic_grad_norm.append(_stat(stats, "G_Ub_norm"))
        actor_ortho_error.append(_stat(stats, "ortho_err_Wc"))
        critic_ortho_error.append(_stat(stats, "ortho_err_Ub"))

        if log_every > 0 and (
            (iteration + 1) % log_every == 0 or iteration + 1 == iters
        ):
            distance = np.linalg.norm(policies[-1] + env.K_opt.reshape(env.ds))
            print(
                f"  downstairs {iteration + 1:5d}/{iters}: "
                f"reward={rewards[-1]:+.4f}, distance={distance:.4f}"
            )

    return (
        _build_trace(
            policies,
            rewards,
            mse,
            actor_grad_norm,
            critic_grad_norm,
            actor_ortho_error,
            critic_ortho_error,
        ),
        init_params,
    )


def initialize_matched_upstairs(
    actor: ActorDLN_L3,
    critic: CriticDLN_L3,
    init_params: Dict[str, np.ndarray],
    M: np.ndarray,
) -> None:
    """Match the upstairs effective actor and critic to downstairs initialization."""
    d = M.shape[0]

    Wc = init_params["Wc_col"].T
    W3_target = Wc @ M.T
    actor.W3 = W3_target / (np.linalg.norm(W3_target, axis=1, keepdims=True) + 1e-12)
    actor.W1 = np.eye(d)
    actor.W2 = np.eye(d)

    Ub = init_params["Ub"]
    projection = M @ M.T
    Ueff_full = M @ Ub @ M.T + (np.eye(d) - projection)
    left, _, right_t = np.linalg.svd(Ueff_full)
    critic.U3 = left @ right_t
    critic.U1 = np.eye(d)
    critic.U2 = np.eye(d)

    Zb = init_params["Zb"]
    Z3_target = Zb.T @ M.T
    critic.Z3 = Z3_target / (np.linalg.norm(Z3_target, axis=1, keepdims=True) + 1e-12)
    critic.Z1 = np.eye(d)
    critic.Z2 = np.eye(d)


def run_upstairs(
    env: LQRUpDownEnv,
    init_params: Dict[str, np.ndarray],
    *,
    iters: int,
    n_steps: int,
    beta: float,
    eta_actor: float,
    eta_critic: float,
    exploration_std: float,
    init_seed: int,
    noise_seed: int,
    use_generator: bool,
    use_sgd: bool,
    log_every: int,
) -> OptimizationTrace:
    """Run upstairs learning and project every effective policy into latent space."""
    actor = ActorDLN_L3(d=env.d, da=1, seed=init_seed)
    critic = CriticDLN_L3(d=env.d, da=1, seed=init_seed + 100)
    initialize_matched_upstairs(actor, critic, init_params, env.M)
    env.rng = np.random.default_rng(noise_seed)

    def latent_policy_vector() -> np.ndarray:
        return (actor.effective_matrix() @ env.M).reshape(env.ds).copy()

    policies = [latent_policy_vector()]
    rewards: List[float] = []
    mse: List[float] = []
    actor_grad_norm: List[float] = []
    critic_grad_norm: List[float] = []
    actor_ortho_error: List[float] = []
    critic_ortho_error: List[float] = []

    for iteration in range(iters):

        def policy(obs):
            return actor.act_upstairs(obs["upstairs"])

        traj = env.rollout(
            policy,
            exploration_std=exploration_std,
            use_upstairs=True,
        )
        stats = upstairs_batch_multistep_update(
            actor,
            critic,
            traj["o"],
            traj["a"],
            traj["r"],
            beta=beta,
            dt=env.dt,
            L=n_steps,
            eta_actor=eta_actor,
            eta_critic=eta_critic,
            use_generator=use_generator,
            use_cayley=not use_sgd,
        )

        policies.append(latent_policy_vector())
        rewards.append(float(np.mean(traj["r"])))
        mse.append(_stat(stats, "mse"))
        actor_grad_norm.append(_stat(stats, "G_W1_norm"))
        critic_grad_norm.append(_stat(stats, "G_U1_norm"))
        actor_ortho_error.append(_stat(stats, "ortho_err_W1"))
        critic_ortho_error.append(_stat(stats, "ortho_err_U1"))

        if log_every > 0 and (
            (iteration + 1) % log_every == 0 or iteration + 1 == iters
        ):
            distance = np.linalg.norm(policies[-1] + env.K_opt.reshape(env.ds))
            print(
                f"  upstairs   {iteration + 1:5d}/{iters}: "
                f"reward={rewards[-1]:+.4f}, distance={distance:.4f}, "
                f"latent_norm={np.linalg.norm(policies[-1]):.4f}"
            )

    return _build_trace(
        policies,
        rewards,
        mse,
        actor_grad_norm,
        critic_grad_norm,
        actor_ortho_error,
        critic_ortho_error,
    )


def compute_diagnostics(
    down_trace: OptimizationTrace,
    up_trace: OptimizationTrace,
    optimum: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute shared, parameterization-independent policy diagnostics."""
    down_delta = np.diff(down_trace.policies, axis=0)
    up_delta = np.diff(up_trace.policies, axis=0)
    down_step = np.linalg.norm(down_delta, axis=1)
    up_step = np.linalg.norm(up_delta, axis=1)
    denominators = down_step * up_step
    alignment = np.full_like(denominators, np.nan, dtype=float)
    moving = denominators > EPS
    alignment[moving] = np.sum(down_delta[moving] * up_delta[moving], axis=1)
    alignment[moving] /= denominators[moving]
    alignment = np.clip(alignment, -1.0, 1.0)

    return {
        "down_distance": np.linalg.norm(down_trace.policies - optimum[None, :], axis=1),
        "up_distance": np.linalg.norm(up_trace.policies - optimum[None, :], axis=1),
        "down_norm": np.linalg.norm(down_trace.policies, axis=1),
        "up_norm": np.linalg.norm(up_trace.policies, axis=1),
        "down_step": down_step,
        "up_step": up_step,
        "alignment": alignment,
    }


def draw_unit_sphere(ax, limit: float, elev: float, azim: float) -> None:
    """Draw a translucent unit sphere and configure a comparable 3-D view."""
    longitude = np.linspace(0.0, 2.0 * np.pi, 64)
    latitude = np.linspace(0.0, np.pi, 32)
    x = np.outer(np.cos(longitude), np.sin(latitude))
    y = np.outer(np.sin(longitude), np.sin(latitude))
    z = np.outer(np.ones_like(longitude), np.cos(latitude))
    ax.plot_surface(
        x,
        y,
        z,
        color="#A9D6E5",
        alpha=0.075,
        linewidth=0,
        shade=False,
        zorder=0,
    )

    circle = np.linspace(0.0, 2.0 * np.pi, 240)
    for xs, ys, zs in (
        (np.cos(circle), np.sin(circle), np.zeros_like(circle)),
        (np.cos(circle), np.zeros_like(circle), np.sin(circle)),
        (np.zeros_like(circle), np.cos(circle), np.sin(circle)),
    ):
        ax.plot(xs, ys, zs, color="#7895A2", alpha=0.28, linewidth=0.7)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$s_1$ coefficient")
    ax.set_ylabel(r"$s_2$ coefficient")
    ax.set_zlabel(r"$s_3$ coefficient")
    ax.grid(False)


def _positive(values: np.ndarray) -> np.ndarray:
    """Make nonnegative diagnostics safe for a logarithmic axis."""
    return np.maximum(np.asarray(values, dtype=float), EPS)


def create_static_plot(
    down_trace: OptimizationTrace,
    up_trace: OptimizationTrace,
    optimum: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
    output_path: Path,
    *,
    dpi: int,
    elev: float,
    azim: float,
) -> None:
    """Save a sphere trajectory plot with synchronized diagnostics."""
    max_norm = max(
        1.0,
        float(np.max(diagnostics["down_norm"])),
        float(np.max(diagnostics["up_norm"])),
    )
    limit = 1.12 * max_norm

    fig = plt.figure(figsize=(19, 11))
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=(1.35, 1.35, 1.0, 1.0),
        hspace=0.32,
        wspace=0.28,
    )
    sphere_ax = fig.add_subplot(grid[:, :2], projection="3d")
    draw_unit_sphere(sphere_ax, limit, elev, azim)

    sphere_ax.plot(
        down_trace.policies[:, 0],
        down_trace.policies[:, 1],
        down_trace.policies[:, 2],
        color=DOWN_COLOR,
        linewidth=2.8,
        label="Downstairs",
        zorder=5,
    )
    sphere_ax.plot(
        up_trace.policies[:, 0],
        up_trace.policies[:, 1],
        up_trace.policies[:, 2],
        color=UP_COLOR,
        linewidth=2.8,
        label="Upstairs latent projection",
        zorder=6,
    )
    sphere_ax.scatter(
        *down_trace.policies[0],
        color="white",
        edgecolor="#252525",
        marker="D",
        s=75,
        linewidth=1.0,
        label="Matched initialization",
        depthshade=False,
        zorder=10,
    )
    sphere_ax.scatter(
        *optimum,
        color=OPT_COLOR,
        edgecolor="#3D3200",
        marker="*",
        s=260,
        linewidth=1.0,
        label="Optimal policy",
        depthshade=False,
        zorder=11,
    )
    sphere_ax.scatter(
        *down_trace.policies[-1],
        color=DOWN_COLOR,
        edgecolor="white",
        marker="o",
        s=85,
        linewidth=1.2,
        depthshade=False,
        zorder=12,
    )
    sphere_ax.scatter(
        *up_trace.policies[-1],
        color=UP_COLOR,
        edgecolor="white",
        marker="o",
        s=85,
        linewidth=1.2,
        depthshade=False,
        zorder=12,
    )
    sphere_ax.set_title(
        "Effective policy optimization in latent space\n"
        r"$p_{\rm down}=W_c^\top,\quad "
        r"p_{\rm up}=W_3W_2W_1M,\quad p_*=-K_*$",
        pad=20,
        fontsize=15,
    )
    sphere_ax.legend(loc="upper left", fontsize=9)

    update_axis = np.arange(len(down_trace.policies))
    transition_axis = np.arange(1, len(down_trace.policies))
    rollout_axis = np.arange(len(down_trace.rewards))

    distance_ax = fig.add_subplot(grid[0, 2])
    distance_ax.plot(
        update_axis, diagnostics["down_distance"], color=DOWN_COLOR, label="Down"
    )
    distance_ax.plot(
        update_axis, diagnostics["up_distance"], color=UP_COLOR, label="Up"
    )
    distance_ax.set_title("Distance to optimal policy")
    distance_ax.set_ylabel(r"$\|p_t-p_*\|_2$")
    distance_ax.legend(fontsize=8)

    reward_ax = fig.add_subplot(grid[0, 3])
    reward_ax.plot(rollout_axis, down_trace.rewards, color=DOWN_COLOR, label="Down")
    reward_ax.plot(rollout_axis, up_trace.rewards, color=UP_COLOR, label="Up")
    reward_ax.set_title("Rollout reward")
    reward_ax.set_ylabel("Mean reward")
    reward_ax.legend(fontsize=8)

    norm_ax = fig.add_subplot(grid[1, 2])
    norm_ax.plot(update_axis, diagnostics["down_norm"], color=DOWN_COLOR, label="Down")
    norm_ax.plot(update_axis, diagnostics["up_norm"], color=UP_COLOR, label="Up")
    norm_ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="Sphere")
    norm_ax.set_ylim(
        0.0,
        1.08
        * max(
            1.0,
            float(np.max(diagnostics["down_norm"])),
            float(np.max(diagnostics["up_norm"])),
        ),
    )
    norm_ax.set_title("Latent policy norm")
    norm_ax.set_ylabel(r"$\|p_t\|_2$")
    norm_ax.legend(fontsize=8)

    step_ax = fig.add_subplot(grid[1, 3])
    step_ax.semilogy(
        transition_axis,
        _positive(diagnostics["down_step"]),
        color=DOWN_COLOR,
        label="Down",
    )
    step_ax.semilogy(
        transition_axis,
        _positive(diagnostics["up_step"]),
        color=UP_COLOR,
        label="Up",
    )
    step_ax.set_title("Functional step size")
    step_ax.set_ylabel(r"$\|p_t-p_{t-1}\|_2$")
    step_ax.legend(fontsize=8)

    alignment_ax = fig.add_subplot(grid[2, 2])
    alignment_ax.plot(
        transition_axis,
        diagnostics["alignment"],
        color=ALIGN_COLOR,
        linewidth=1.4,
    )
    alignment_ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    alignment_ax.axhline(0.0, color="#AAAAAA", linewidth=0.8)
    alignment_ax.set_ylim(-1.05, 1.05)
    alignment_ax.set_title("Update-direction agreement")
    alignment_ax.set_ylabel(r"$\cos(\Delta p_{\rm down},\Delta p_{\rm up})$")

    gradient_ax = fig.add_subplot(grid[2, 3])
    gradient_ax.semilogy(
        transition_axis,
        _positive(down_trace.actor_grad_norm),
        color=DOWN_COLOR,
        label="Down",
    )
    gradient_ax.semilogy(
        transition_axis,
        _positive(up_trace.actor_grad_norm),
        color=UP_COLOR,
        label="Up",
    )
    gradient_ax.set_title("Raw actor gradient norm")
    gradient_ax.set_ylabel("Parameter-space norm")
    gradient_ax.legend(fontsize=8)

    for ax in (
        distance_ax,
        reward_ax,
        norm_ax,
        step_ax,
        alignment_ax,
        gradient_ax,
    ):
        ax.set_xlabel("Update")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Upstairs vs. downstairs optimization dynamics (ds=3, da=1)",
        fontsize=18,
        y=0.985,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved static visualization to {output_path}")


def create_animation(
    down_trace: OptimizationTrace,
    up_trace: OptimizationTrace,
    optimum: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
    output_path: Path,
    *,
    fps: int,
    max_frames: int,
    dpi: int,
    elev: float,
    azim: float,
) -> None:
    """Animate the two effective policy paths on the unit sphere."""
    n_updates = len(down_trace.policies) - 1
    n_frames = min(n_updates + 1, max_frames)
    frame_updates = np.unique(np.linspace(0, n_updates, num=n_frames, dtype=int))

    max_norm = max(
        1.0,
        float(np.max(diagnostics["down_norm"])),
        float(np.max(diagnostics["up_norm"])),
    )
    limit = 1.12 * max_norm

    fig = plt.figure(figsize=(13.5, 7.5))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1.55, 1.0),
        hspace=0.38,
        wspace=0.28,
    )
    sphere_ax = fig.add_subplot(grid[:, 0], projection="3d")
    draw_unit_sphere(sphere_ax, limit, elev, azim)
    sphere_ax.scatter(
        *optimum,
        color=OPT_COLOR,
        edgecolor="#3D3200",
        marker="*",
        s=230,
        linewidth=1.0,
        label="Optimal",
        depthshade=False,
    )
    sphere_ax.scatter(
        *down_trace.policies[0],
        color="white",
        edgecolor="#252525",
        marker="D",
        s=65,
        linewidth=1.0,
        label="Initialization",
        depthshade=False,
    )
    (down_line,) = sphere_ax.plot(
        [], [], [], color=DOWN_COLOR, linewidth=3.0, label="Downstairs"
    )
    (up_line,) = sphere_ax.plot(
        [], [], [], color=UP_COLOR, linewidth=3.0, label="Upstairs projection"
    )
    (down_point,) = sphere_ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=8,
        color=DOWN_COLOR,
        markeredgecolor="white",
        linestyle="",
    )
    (up_point,) = sphere_ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=8,
        color=UP_COLOR,
        markeredgecolor="white",
        linestyle="",
    )
    (down_radius,) = sphere_ax.plot(
        [], [], [], color=DOWN_COLOR, alpha=0.32, linestyle="--", linewidth=1
    )
    (up_radius,) = sphere_ax.plot(
        [], [], [], color=UP_COLOR, alpha=0.32, linestyle="--", linewidth=1
    )
    sphere_ax.legend(loc="upper left", fontsize=8)
    sphere_ax.set_title("Effective policy trajectory", fontsize=14)
    status = sphere_ax.text2D(
        0.03,
        0.02,
        "",
        transform=sphere_ax.transAxes,
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#BBBBBB"},
    )

    distance_ax = fig.add_subplot(grid[0, 1])
    distance_ax.set_xlim(0, max(1, n_updates))
    distance_max = max(
        float(np.max(diagnostics["down_distance"])),
        float(np.max(diagnostics["up_distance"])),
        EPS,
    )
    distance_ax.set_ylim(0.0, 1.08 * distance_max)
    distance_ax.set_title("Distance to optimum")
    distance_ax.set_ylabel(r"$\|p_t-p_*\|_2$")
    (distance_down,) = distance_ax.plot([], [], color=DOWN_COLOR, label="Down")
    (distance_up,) = distance_ax.plot([], [], color=UP_COLOR, label="Up")
    distance_ax.legend(fontsize=8)

    norm_ax = fig.add_subplot(grid[1, 1])
    norm_ax.set_xlim(0, max(1, n_updates))
    norm_max = max(
        1.0,
        float(np.max(diagnostics["down_norm"])),
        float(np.max(diagnostics["up_norm"])),
    )
    norm_ax.set_ylim(0.0, 1.08 * norm_max)
    norm_ax.set_title("Latent policy norm")
    norm_ax.set_ylabel(r"$\|p_t\|_2$")
    norm_ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    (norm_down,) = norm_ax.plot([], [], color=DOWN_COLOR)
    (norm_up,) = norm_ax.plot([], [], color=UP_COLOR)

    alignment_ax = fig.add_subplot(grid[2, 1])
    alignment_ax.set_xlim(0, max(1, n_updates))
    alignment_ax.set_ylim(-1.05, 1.05)
    alignment_ax.set_title("Update-direction agreement")
    alignment_ax.set_ylabel("Cosine")
    alignment_ax.set_xlabel("Update")
    alignment_ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    alignment_ax.axhline(0.0, color="#AAAAAA", linewidth=0.8)
    (alignment_line,) = alignment_ax.plot([], [], color=ALIGN_COLOR)

    for ax in (distance_ax, norm_ax, alignment_ax):
        ax.grid(True, alpha=0.25)
        if ax is not alignment_ax:
            ax.set_xlabel("Update")

    def update(frame_number: int):
        update_index = int(frame_updates[frame_number])
        visible = slice(0, update_index + 1)
        down_path = down_trace.policies[visible]
        up_path = up_trace.policies[visible]
        current_down = down_path[-1]
        current_up = up_path[-1]

        down_line.set_data(down_path[:, 0], down_path[:, 1])
        down_line.set_3d_properties(down_path[:, 2])
        up_line.set_data(up_path[:, 0], up_path[:, 1])
        up_line.set_3d_properties(up_path[:, 2])

        down_point.set_data([current_down[0]], [current_down[1]])
        down_point.set_3d_properties([current_down[2]])
        up_point.set_data([current_up[0]], [current_up[1]])
        up_point.set_3d_properties([current_up[2]])

        down_radius.set_data([0.0, current_down[0]], [0.0, current_down[1]])
        down_radius.set_3d_properties([0.0, current_down[2]])
        up_radius.set_data([0.0, current_up[0]], [0.0, current_up[1]])
        up_radius.set_3d_properties([0.0, current_up[2]])

        history_axis = np.arange(update_index + 1)
        distance_down.set_data(
            history_axis, diagnostics["down_distance"][: update_index + 1]
        )
        distance_up.set_data(
            history_axis, diagnostics["up_distance"][: update_index + 1]
        )
        norm_down.set_data(history_axis, diagnostics["down_norm"][: update_index + 1])
        norm_up.set_data(history_axis, diagnostics["up_norm"][: update_index + 1])
        if update_index > 0:
            alignment_axis = np.arange(1, update_index + 1)
            alignment_line.set_data(
                alignment_axis, diagnostics["alignment"][:update_index]
            )
        else:
            alignment_line.set_data([], [])

        status.set_text(
            f"update {update_index:4d}/{n_updates}\n"
            f"down distance {diagnostics['down_distance'][update_index]:.4f}\n"
            f"up   distance {diagnostics['up_distance'][update_index]:.4f}\n"
            f"up latent norm {diagnostics['up_norm'][update_index]:.4f}"
        )
        return (
            down_line,
            up_line,
            down_point,
            up_point,
            down_radius,
            up_radius,
            distance_down,
            distance_up,
            norm_down,
            norm_up,
            alignment_line,
            status,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_updates),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(
        f"Saved animation to {output_path} "
        f"({len(frame_updates)} frames for {n_updates} updates)"
    )


def save_trace_csv(
    down_trace: OptimizationTrace,
    up_trace: OptimizationTrace,
    optimum: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save the exact points and diagnostics used by the plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "update",
                "down_p1",
                "down_p2",
                "down_p3",
                "up_p1",
                "up_p2",
                "up_p3",
                "optimal_p1",
                "optimal_p2",
                "optimal_p3",
                "down_distance",
                "up_distance",
                "down_norm",
                "up_norm",
                "down_step",
                "up_step",
                "update_alignment",
                "down_pre_update_reward",
                "up_pre_update_reward",
                "down_actor_grad_norm",
                "up_actor_grad_norm",
                "down_actor_ortho_error",
                "up_actor_ortho_error",
            ]
        )
        for update in range(len(down_trace.policies)):
            transition = update - 1
            has_transition = transition >= 0
            writer.writerow(
                [
                    update,
                    *down_trace.policies[update],
                    *up_trace.policies[update],
                    *optimum,
                    diagnostics["down_distance"][update],
                    diagnostics["up_distance"][update],
                    diagnostics["down_norm"][update],
                    diagnostics["up_norm"][update],
                    diagnostics["down_step"][transition] if has_transition else "",
                    diagnostics["up_step"][transition] if has_transition else "",
                    diagnostics["alignment"][transition] if has_transition else "",
                    down_trace.rewards[transition] if has_transition else "",
                    up_trace.rewards[transition] if has_transition else "",
                    (down_trace.actor_grad_norm[transition] if has_transition else ""),
                    up_trace.actor_grad_norm[transition] if has_transition else "",
                    (
                        down_trace.actor_ortho_error[transition]
                        if has_transition
                        else ""
                    ),
                    up_trace.actor_ortho_error[transition] if has_transition else "",
                ]
            )
    print(f"Saved optimization trace to {output_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Animate matched upstairs/downstairs effective policies on the "
            "ds=3, da=1 Stiefel sphere."
        )
    )
    parser.add_argument("--d", type=int, default=64, help="Upstairs dimension")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--n_steps", "--n-steps", type=int, default=30)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--eta_actor", "--eta-actor", type=float, default=0.01)
    parser.add_argument("--eta_critic", "--eta-critic", type=float, default=0.05)
    parser.add_argument(
        "--exploration_std", "--exploration-std", type=float, default=0.1
    )
    parser.add_argument(
        "--transition_noise", "--transition-noise", type=float, default=0.1
    )
    parser.add_argument(
        "--observation_noise", "--observation-noise", type=float, default=0.05
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_seed", "--init-seed", type=int, default=None)
    parser.add_argument("--noise_seed", "--noise-seed", type=int, default=None)
    parser.add_argument("--use_generator", "--use-generator", action="store_true")
    parser.add_argument(
        "--use_sgd",
        "--use-sgd",
        action="store_true",
        help="Use unconstrained SGD instead of Cayley retraction",
    )
    parser.add_argument(
        "--save_plot",
        "--save-plot",
        type=Path,
        default=script_dir / "optimization_dynamics_sphere.png",
    )
    parser.add_argument(
        "--save_animation",
        "--save-animation",
        type=Path,
        default=script_dir / "optimization_dynamics_sphere.gif",
    )
    parser.add_argument(
        "--save_trace",
        "--save-trace",
        type=Path,
        default=script_dir / "optimization_dynamics_sphere.csv",
    )
    parser.add_argument(
        "--no_animation",
        "--no-animation",
        action="store_true",
        help="Only create the static plot and CSV",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--max_frames",
        "--max-frames",
        type=int,
        default=180,
        help="Maximum GIF frames; the full path is retained when subsampling",
    )
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument(
        "--log_every",
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N updates; use 0 to disable",
    )
    args = parser.parse_args()

    if args.d < 3:
        parser.error("--d must be at least 3")
    if args.iters < 1:
        parser.error("--iters must be positive")
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    init_seed = args.init_seed if args.init_seed is not None else args.seed + 1
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed + 2

    G, H, Q, R = construct_compatible_lqr(
        ds=3,
        da=1,
        alpha=args.alpha,
        seed=args.seed,
    )
    params = LQRParams(
        d=args.d,
        ds=3,
        da=1,
        dt=args.dt,
        T=args.T,
        G=G,
        H=H,
        Q=Q,
        R=R,
        sigma_explore=0.05,
        eps_obs=args.observation_noise,
        seed=args.seed,
        Sigma_base=args.transition_noise * np.eye(3),
    )
    env = LQRUpDownEnv(params)
    optimum = -env.K_opt.reshape(env.ds)

    print("=" * 72)
    print("3-D upstairs/downstairs optimization dynamics")
    print("=" * 72)
    print(
        f"ds=3, da=1, d={args.d}, iters={args.iters}, "
        f"rollout_steps={env.n_steps}, bootstrap_steps={args.n_steps}"
    )
    print(
        f"optimization={'SGD' if args.use_sgd else 'Cayley'}, "
        f"LQR/init/noise seeds={args.seed}/{init_seed}/{noise_seed}"
    )
    print(f"optimal policy vector: {np.array2string(optimum, precision=4)}")
    print("\nRunning downstairs...")

    down_trace, init_params = run_downstairs(
        env,
        iters=args.iters,
        n_steps=args.n_steps,
        beta=args.beta,
        eta_actor=args.eta_actor,
        eta_critic=args.eta_critic,
        exploration_std=args.exploration_std,
        init_seed=init_seed,
        noise_seed=noise_seed,
        use_generator=args.use_generator,
        use_sgd=args.use_sgd,
        log_every=args.log_every,
    )

    print("\nRunning upstairs from matched initialization...")
    up_trace = run_upstairs(
        env,
        init_params,
        iters=args.iters,
        n_steps=args.n_steps,
        beta=args.beta,
        eta_actor=args.eta_actor,
        eta_critic=args.eta_critic,
        exploration_std=args.exploration_std,
        init_seed=init_seed,
        noise_seed=noise_seed,
        use_generator=args.use_generator,
        use_sgd=args.use_sgd,
        log_every=args.log_every,
    )

    diagnostics = compute_diagnostics(down_trace, up_trace, optimum)
    matched_error = np.linalg.norm(down_trace.policies[0] - up_trace.policies[0])
    print("\nSummary")
    print(f"  matched-initialization error: {matched_error:.3e}")
    print(
        "  final distance to optimum: "
        f"down={diagnostics['down_distance'][-1]:.4f}, "
        f"up={diagnostics['up_distance'][-1]:.4f}"
    )
    print(
        "  final latent policy norm: "
        f"down={diagnostics['down_norm'][-1]:.4f}, "
        f"up={diagnostics['up_norm'][-1]:.4f}"
    )

    create_static_plot(
        down_trace,
        up_trace,
        optimum,
        diagnostics,
        args.save_plot,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
    )
    save_trace_csv(
        down_trace,
        up_trace,
        optimum,
        diagnostics,
        args.save_trace,
    )
    if not args.no_animation:
        create_animation(
            down_trace,
            up_trace,
            optimum,
            diagnostics,
            args.save_animation,
            fps=args.fps,
            max_frames=args.max_frames,
            dpi=args.dpi,
            elev=args.elev,
            azim=args.azim,
        )


if __name__ == "__main__":
    main()
