#!/usr/bin/env python3
"""Plot cached CNN Adam/Stiefel vs staging plain-head Adam/Stiefel curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ENV_ORDER = [
    "halfcheetah",
    "walker2d",
    "ant",
    "humanoid",
    "reacher",
    "swimmer",
    "pusher",
    "hopper",
    "inverted_pendulum",
]

CURVES = [
    {
        "source": "cnn",
        "key": "cnn_adam",
        "label": "CNN Adam",
        "color": "#0072B2",
    },
    {
        "source": "cnn",
        "key": "cnn_muon",
        "label": "CNN Stiefel",
        "color": "#E69F00",
    },
    {
        "source": "staging",
        "key": "plainheads_adam",
        "label": "Plain Heads Adam",
        "color": "#CC79A7",
    },
    {
        "source": "staging",
        "key": "plainheads_stiefel",
        "label": "Plain Heads Stiefel",
        "color": "#000000",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CNN Adam/Stiefel vs staging plain-head Adam/Stiefel from cached CSVs."
    )
    parser.add_argument("--cnn-data-dir", type=Path, default=Path("plots/cnn_data"))
    parser.add_argument(
        "--staging-data-dir", type=Path, default=Path("plots/staging_data")
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots/cnn_vs_staging_plainheads"),
    )
    parser.add_argument(
        "--envs",
        default=None,
        help="Optional comma-separated environment subset.",
    )
    return parser.parse_args()


def env_title(env: str) -> str:
    names = {
        "halfcheetah": "HalfCheetah",
        "walker2d": "Walker2D",
        "ant": "Ant",
        "humanoid": "Humanoid",
        "reacher": "Reacher",
        "swimmer": "Swimmer",
        "pusher": "Pusher",
        "hopper": "Hopper",
        "inverted_pendulum": "Inverted Pendulum",
    }
    return names.get(env, env.replace("_", " ").title())


def load_curves(
    cnn_data_dir: Path, staging_data_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cnn = pd.read_csv(cnn_data_dir / "curve_summary.csv")
    staging = pd.read_csv(staging_data_dir / "canonical_curve_summary.csv")
    required = {"environment", "step", "count", "mean", "stderr"}
    missing_cnn = (required | {"condition"}) - set(cnn.columns)
    missing_staging = (required | {"canonical_group"}) - set(staging.columns)
    if missing_cnn:
        raise ValueError(f"Missing CNN columns: {sorted(missing_cnn)}")
    if missing_staging:
        raise ValueError(f"Missing staging columns: {sorted(missing_staging)}")
    return cnn, staging


def curve_frame(
    curve: dict, env: str, cnn: pd.DataFrame, staging: pd.DataFrame
) -> pd.DataFrame:
    if curve["source"] == "cnn":
        frame = cnn[(cnn["environment"] == env) & (cnn["condition"] == curve["key"])]
    else:
        frame = staging[
            (staging["environment"] == env)
            & (staging["canonical_group"] == curve["key"])
        ]
    return frame.sort_values("step")


def plot_env(
    env: str, cnn: pd.DataFrame, staging: pd.DataFrame, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for curve in CURVES:
        frame = curve_frame(curve, env, cnn, staging)
        if frame.empty:
            print(f"Warning: missing {env}/{curve['key']}")
            continue
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        count = int(frame["count"].max())
        label = f"{curve['label']} (n={count})"
        ax.plot(steps, mean, color=curve["color"], label=label, linewidth=2.2)
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=curve["color"],
            alpha=0.2,
            linewidth=0,
        )

    ax.set_title(env_title(env), fontsize=18)
    ax.set_xlabel("Environment steps", fontsize=14)
    ax.set_ylabel("Average Episodic Return", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    envs = ENV_ORDER
    if args.envs:
        envs = [env.strip().lower() for env in args.envs.split(",") if env.strip()]
        unknown = [env for env in envs if env not in ENV_ORDER]
        if unknown:
            raise SystemExit(f"Unknown envs: {', '.join(unknown)}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    cnn, staging = load_curves(args.cnn_data_dir, args.staging_data_dir)

    for env in envs:
        plot_env(env, cnn, staging, args.plots_dir / f"{env}.png")


if __name__ == "__main__":
    main()
