"""
Visualize actual two-dimensional upstairs/downstairs policy trajectories.

This experiment uses ``ds=2`` and ``da=1``. The effective policies therefore
have exactly two latent coefficients and can be plotted directly on the unit
circle without projection or dimensionality reduction:

    downstairs: p_down = Wc_col.T
    upstairs:   p_up   = (W3 W2 W1) M
    optimum:    p_opt  = -K_opt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from compare_upstairs_downstairs import construct_compatible_lqr
from lqr_env import LQRParams, LQRUpDownEnv
from visualize_optimization_sphere import run_downstairs, run_upstairs


DOWN_COLOR = "#20A4F3"
UP_COLOR = "#F18F01"
OPT_COLOR = "#FFD166"


def add_time_colored_path(
    ax,
    policies: np.ndarray,
    color: str,
    label: str,
    marker_every: int,
) -> None:
    """Draw an optimization path with time markers and increasing opacity."""
    points = policies.reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    base_rgb = np.asarray(plt.matplotlib.colors.to_rgb(color))
    alpha = np.linspace(0.28, 1.0, len(segments))
    colors = np.column_stack(
        (np.repeat(base_rgb[None, :], len(segments), axis=0), alpha)
    )
    ax.add_collection(
        LineCollection(
            segments,
            colors=colors,
            linewidths=3.3,
            capstyle="round",
            joinstyle="round",
            zorder=5,
        )
    )

    marker_indices = np.arange(marker_every, len(policies), marker_every)
    if len(marker_indices):
        ax.scatter(
            policies[marker_indices, 0],
            policies[marker_indices, 1],
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            zorder=6,
        )

    ax.scatter(
        *policies[-1],
        s=115,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=8,
    )
    ax.annotate(
        f"{label} final",
        xy=policies[-1],
        xytext=(9, -13 if label == "Downstairs" else 10),
        textcoords="offset points",
        color=color,
        fontsize=10,
        weight="bold",
    )


def create_circle_plot(
    down_policies: np.ndarray,
    up_policies: np.ndarray,
    optimum: np.ndarray,
    output_path: Path,
    *,
    marker_every: int,
    dpi: int,
) -> None:
    """Plot raw two-dimensional effective policy coefficients."""
    fig, ax = plt.subplots(figsize=(9, 9))

    angle = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.fill(
        np.cos(angle),
        np.sin(angle),
        color="#DCEEF5",
        alpha=0.28,
        zorder=0,
    )
    ax.plot(
        np.cos(angle),
        np.sin(angle),
        color="#7895A2",
        linewidth=1.4,
        alpha=0.8,
        zorder=1,
    )
    ax.axhline(0.0, color="#A7B2B8", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#A7B2B8", linewidth=0.8, zorder=1)

    add_time_colored_path(
        ax,
        down_policies,
        DOWN_COLOR,
        "Downstairs",
        marker_every,
    )
    add_time_colored_path(
        ax,
        up_policies,
        UP_COLOR,
        "Upstairs",
        marker_every,
    )

    ax.scatter(
        *down_policies[0],
        marker="D",
        s=110,
        facecolor="white",
        edgecolor="#252525",
        linewidth=1.4,
        zorder=9,
    )
    ax.annotate(
        "Matched initialization",
        xy=down_policies[0],
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
    )
    ax.scatter(
        *optimum,
        marker="*",
        s=390,
        facecolor=OPT_COLOR,
        edgecolor="#4A3D00",
        linewidth=1.3,
        zorder=9,
    )
    ax.annotate(
        "Optimal policy",
        xy=optimum,
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=11,
        weight="bold",
        color="#6B5700",
    )

    max_norm = max(
        1.0,
        float(np.max(np.linalg.norm(down_policies, axis=1))),
        float(np.max(np.linalg.norm(up_policies, axis=1))),
    )
    limit = 1.14 * max_norm
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$s_1$ policy coefficient", fontsize=12)
    ax.set_ylabel(r"$s_2$ policy coefficient", fontsize=12)
    ax.set_title(
        "Actual 2-D effective-policy optimization trajectories",
        fontsize=16,
        pad=15,
    )
    ax.text(
        0.98,
        0.02,
        f"Markers every {marker_every} updates  •  darker path is later",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=DOWN_COLOR,
                linewidth=3,
                label="Downstairs",
            ),
            Line2D(
                [0],
                [0],
                color=UP_COLOR,
                linewidth=3,
                label="Upstairs",
            ),
            Line2D(
                [0],
                [0],
                color="#7895A2",
                linewidth=1.4,
                label="Unit circle",
            ),
        ],
        loc="lower left",
        frameon=True,
    )
    ax.grid(True, alpha=0.17)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved actual 2-D trajectory plot to {output_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run a ds=2, da=1 matched experiment and plot the raw policy "
            "trajectory on the unit circle."
        )
    )
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--n_steps", "--n-steps", type=int, default=30)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--eta_actor", "--eta-actor", type=float, default=0.01)
    parser.add_argument("--eta_critic", "--eta-critic", type=float, default=0.05)
    parser.add_argument(
        "--exploration_std",
        "--exploration-std",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--transition_noise",
        "--transition-noise",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--observation_noise",
        "--observation-noise",
        type=float,
        default=0.05,
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_seed", "--init-seed", type=int, default=None)
    parser.add_argument("--noise_seed", "--noise-seed", type=int, default=None)
    parser.add_argument("--use_generator", "--use-generator", action="store_true")
    parser.add_argument("--use_sgd", "--use-sgd", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "optimization_trajectory_circle.png",
    )
    parser.add_argument("--marker-every", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.d < 2:
        parser.error("--d must be at least 2")
    if args.iters < 1:
        parser.error("--iters must be positive")
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    if args.marker_every < 1:
        parser.error("--marker-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    init_seed = args.init_seed if args.init_seed is not None else args.seed + 1
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed + 2

    G, H, Q, R = construct_compatible_lqr(
        ds=2,
        da=1,
        alpha=args.alpha,
        seed=args.seed,
    )
    params = LQRParams(
        d=args.d,
        ds=2,
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
        Sigma_base=args.transition_noise * np.eye(2),
    )
    env = LQRUpDownEnv(params)
    optimum = -env.K_opt.reshape(2)

    print("=" * 72)
    print("Actual 2-D upstairs/downstairs optimization trajectories")
    print("=" * 72)
    print(
        f"ds=2, da=1, d={args.d}, iters={args.iters}, "
        f"rollout_steps={env.n_steps}, bootstrap_steps={args.n_steps}"
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

    matched_error = np.linalg.norm(down_trace.policies[0] - up_trace.policies[0])
    down_distance = np.linalg.norm(down_trace.policies[-1] - optimum)
    up_distance = np.linalg.norm(up_trace.policies[-1] - optimum)
    print("\nSummary")
    print(f"  matched-initialization error: {matched_error:.3e}")
    print(
        f"  final distance to optimum: down={down_distance:.4f}, "
        f"up={up_distance:.4f}"
    )
    print(
        "  final policy norm: "
        f"down={np.linalg.norm(down_trace.policies[-1]):.6f}, "
        f"up={np.linalg.norm(up_trace.policies[-1]):.6f}"
    )
    create_circle_plot(
        down_trace.policies,
        up_trace.policies,
        optimum,
        args.output,
        marker_every=args.marker_every,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
