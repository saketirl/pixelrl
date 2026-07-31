"""
Plot upstairs/downstairs optimization paths in 2-D geodesic coordinates.

The input is the CSV written by ``visualize_optimization_sphere.py``. Policies
are mapped from the unit sphere to the tangent plane at their shared initial
policy. The horizontal axis points along the shortest great-circle route to
the optimum; the vertical axis measures deviation from that route.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


DOWN_COLOR = "#20A4F3"
UP_COLOR = "#F18F01"
OPT_COLOR = "#FFD166"


def load_trace(path: Path) -> Dict[str, np.ndarray]:
    """Load effective policy vectors from a sphere-visualizer CSV."""
    with path.open(newline="") as csv_file:
        rows: List[Dict[str, str]] = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")

    def vectors(prefix: str) -> np.ndarray:
        return np.asarray(
            [[float(row[f"{prefix}_p{axis}"]) for axis in range(1, 4)] for row in rows],
            dtype=float,
        )

    optimum = np.asarray(
        [float(rows[0][f"optimal_p{axis}"]) for axis in range(1, 4)],
        dtype=float,
    )
    return {
        "down": vectors("down"),
        "up": vectors("up"),
        "optimum": optimum,
    }


def tangent_basis(initial: np.ndarray, optimum: np.ndarray):
    """Construct tangent axes pointing toward and across from the optimum."""
    initial = initial / np.linalg.norm(initial)
    optimum = optimum / np.linalg.norm(optimum)
    toward = optimum - np.dot(optimum, initial) * initial
    toward_norm = np.linalg.norm(toward)
    if toward_norm < 1e-12:
        raise ValueError("Initial and optimal policies do not define a tangent axis")
    toward /= toward_norm
    across = np.cross(initial, toward)
    across /= np.linalg.norm(across)
    return initial, toward, across


def sphere_log_map(
    policies: np.ndarray,
    initial: np.ndarray,
    toward: np.ndarray,
    across: np.ndarray,
) -> np.ndarray:
    """Map sphere directions to tangent coordinates at the initial policy."""
    norms = np.linalg.norm(policies, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Cannot map a zero-length policy vector")
    directions = policies / norms

    cosine = np.clip(directions @ initial, -1.0, 1.0)
    angle = np.arccos(cosine)
    tangent = directions - cosine[:, None] * initial
    sine = np.sin(angle)
    moving = sine > 1e-12
    tangent[moving] /= sine[moving, None]
    tangent[~moving] = 0.0

    coordinates = np.column_stack(
        (
            angle * (tangent @ toward),
            angle * (tangent @ across),
        )
    )
    return np.degrees(coordinates)


def add_time_colored_path(
    ax,
    coordinates: np.ndarray,
    color: str,
    label: str,
    marker_every: int,
) -> None:
    """Draw a path whose opacity increases with optimization time."""
    points = coordinates.reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    base_rgb = np.asarray(plt.matplotlib.colors.to_rgb(color))
    progress = np.linspace(0.28, 1.0, len(segments))
    colors = np.column_stack(
        (
            np.repeat(base_rgb[None, :], len(segments), axis=0),
            progress,
        )
    )
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=3.2,
        capstyle="round",
        joinstyle="round",
        zorder=4,
    )
    ax.add_collection(collection)

    marker_indices = np.arange(marker_every, len(coordinates), marker_every)
    if len(marker_indices):
        ax.scatter(
            coordinates[marker_indices, 0],
            coordinates[marker_indices, 1],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
            zorder=5,
        )
    ax.scatter(
        *coordinates[-1],
        s=105,
        color=color,
        edgecolor="white",
        linewidth=1.4,
        zorder=7,
    )
    ax.annotate(
        f"{label} final",
        xy=coordinates[-1],
        xytext=(8, -2 if label == "Downstairs" else 9),
        textcoords="offset points",
        color=color,
        fontsize=10,
        weight="bold",
    )


def create_plot(
    down: np.ndarray,
    up: np.ndarray,
    optimum: np.ndarray,
    output_path: Path,
    *,
    dpi: int,
    marker_every: int,
) -> None:
    """Create the trajectory-only geodesic plot."""
    initial, toward, across = tangent_basis(down[0], optimum)
    down_xy = sphere_log_map(down, initial, toward, across)
    up_xy = sphere_log_map(up, initial, toward, across)
    optimum_xy = sphere_log_map(optimum.reshape(1, 3), initial, toward, across)[0]

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    ax.axhline(
        0.0,
        color="#6C757D",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Direct great-circle route",
        zorder=1,
    )
    add_time_colored_path(ax, down_xy, DOWN_COLOR, "Downstairs", marker_every)
    add_time_colored_path(ax, up_xy, UP_COLOR, "Upstairs", marker_every)

    ax.scatter(
        0.0,
        0.0,
        marker="D",
        s=105,
        facecolor="white",
        edgecolor="#252525",
        linewidth=1.3,
        zorder=8,
    )
    ax.annotate(
        "Matched initialization",
        xy=(0.0, 0.0),
        xytext=(9, 9),
        textcoords="offset points",
        fontsize=10,
    )
    ax.scatter(
        *optimum_xy,
        marker="*",
        s=360,
        facecolor=OPT_COLOR,
        edgecolor="#4A3D00",
        linewidth=1.2,
        zorder=8,
    )
    ax.annotate(
        "Optimal policy",
        xy=optimum_xy,
        xytext=(-8, 13),
        textcoords="offset points",
        ha="right",
        fontsize=11,
        weight="bold",
        color="#6B5700",
    )

    all_xy = np.vstack((down_xy, up_xy, optimum_xy[None, :]))
    x_margin = max(4.0, 0.055 * np.ptp(all_xy[:, 0]))
    y_margin = max(2.0, 0.12 * np.ptp(all_xy[:, 1]))
    ax.set_xlim(np.min(all_xy[:, 0]) - x_margin, np.max(all_xy[:, 0]) + x_margin)
    ax.set_ylim(np.min(all_xy[:, 1]) - y_margin, np.max(all_xy[:, 1]) + y_margin)

    ax.set_xlabel("Geodesic progress toward optimum (degrees)", fontsize=12)
    ax.set_ylabel("Deviation from direct great-circle route (degrees)", fontsize=12)
    ax.set_title(
        "Upstairs vs. downstairs effective-policy trajectories",
        fontsize=17,
        pad=15,
    )
    ax.text(
        0.99,
        0.02,
        "Markers: every " f"{marker_every} updates  •  darker path: later optimization",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    legend_handles = [
        Line2D([0], [0], color=DOWN_COLOR, linewidth=3, label="Downstairs"),
        Line2D([0], [0], color=UP_COLOR, linewidth=3, label="Upstairs"),
        Line2D(
            [0],
            [0],
            color="#6C757D",
            linestyle="--",
            linewidth=1.2,
            label="Direct great-circle route",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2-D trajectory plot to {output_path}")
    print(
        "Final geodesic coordinates: "
        f"down=({down_xy[-1, 0]:.2f}, {down_xy[-1, 1]:.2f}) deg, "
        f"up=({up_xy[-1, 0]:.2f}, {up_xy[-1, 1]:.2f}) deg, "
        f"optimal=({optimum_xy[0]:.2f}, {optimum_xy[1]:.2f}) deg"
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot a trajectory-only 2-D geodesic view of optimization."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "optimization_dynamics_sphere.csv",
        help="CSV generated by visualize_optimization_sphere.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "optimization_trajectory_2d.png",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--marker-every",
        type=int,
        default=50,
        help="Place a time marker every N updates",
    )
    args = parser.parse_args()
    if args.marker_every < 1:
        parser.error("--marker-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    trace = load_trace(args.input)
    create_plot(
        trace["down"],
        trace["up"],
        trace["optimum"],
        args.output,
        dpi=args.dpi,
        marker_every=args.marker_every,
    )


if __name__ == "__main__":
    main()
