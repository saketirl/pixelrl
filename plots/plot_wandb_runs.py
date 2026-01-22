#!/usr/bin/env python3
"""
Plot average episodic return with standard error regions for muon vs baseline runs.

Usage:
    python plot_wandb_runs.py --prefix saketirl/benchmark --env-name walker2d
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Placeholder run IDs - replace with actual run IDs
MUON_RUN_IDS = [
    "o4ic6ulk",
    "a0t3llba",
    "ncs63o34",
    "yx2ho1le",
    "tk4qlt61",
    "7uqrbhem"
]

BASELINE_RUN_IDS = [
    "b7x6ffla",
    "r6hag3uz",
    "nbfpgjaw",
    "y2qvx8nu",
    "q8uduj8b"
    "gei2gbkf"
]


def fetch_run_data(run_path: str, metric_key: str = "charts/avg_episodic_return", samples: int = 500):
    """Fetch metric data from a wandb run using sampled history (much faster)."""
    api = wandb.Api(timeout=90)
    run = api.run(run_path)

    # Use history() with samples - much faster than scan_history
    history_df = run.history(keys=[metric_key, "_step"], samples=samples)

    # Filter out NaN values
    history_df = history_df.dropna(subset=[metric_key])

    steps = history_df["_step"].values
    values = history_df[metric_key].values

    # Sort by steps to ensure proper ordering
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]

    return np.array(steps), np.array(values)


def fetch_all_runs(prefix: str, run_ids: list, metric_key: str = "charts/avg_episodic_return"):
    """Fetch data from multiple runs and align them by steps."""
    all_steps = []
    all_values = []

    for run_id in run_ids:
        run_path = f"{prefix}/{run_id}"
        print(f"Fetching data from: {run_path}")
        try:
            steps, values = fetch_run_data(run_path, metric_key)
            if len(steps) > 0:
                all_steps.append(steps)
                all_values.append(values)
        except Exception as e:
            print(f"  Warning: Failed to fetch {run_path}: {e}")

    if not all_values:
        return None, None, None

    # Find common step range and interpolate
    min_step = max(s.min() for s in all_steps)
    max_step = min(s.max() for s in all_steps)

    # Create common x-axis
    num_points = 500
    common_steps = np.linspace(min_step, max_step, num_points)

    # Interpolate all runs to common steps
    interpolated_values = []
    for steps, values in zip(all_steps, all_values):
        interp_values = np.interp(common_steps, steps, values)
        interpolated_values.append(interp_values)

    interpolated_values = np.array(interpolated_values)

    # Compute mean and standard error
    mean = np.mean(interpolated_values, axis=0)
    std = np.std(interpolated_values, axis=0)
    stderr = std / np.sqrt(len(interpolated_values))

    return common_steps, mean, stderr


def plot_comparison(
    prefix: str,
    env_name: str,
    muon_ids: list,
    baseline_ids: list,
    metric_key: str = "charts/avg_episodic_return",
    output_path: str = None,
):
    """Plot muon vs baseline comparison with std error regions."""

    print(f"\nFetching Muon runs...")
    muon_steps, muon_mean, muon_stderr = fetch_all_runs(prefix, muon_ids, metric_key)

    print(f"\nFetching Baseline runs...")
    baseline_steps, baseline_mean, baseline_stderr = fetch_all_runs(prefix, baseline_ids, metric_key)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline
    if baseline_mean is not None:
        ax.plot(baseline_steps, baseline_mean, label="Baseline", color="tab:blue", linewidth=2)
        ax.fill_between(
            baseline_steps,
            baseline_mean - baseline_stderr,
            baseline_mean + baseline_stderr,
            alpha=0.3,
            color="tab:blue",
        )

    # Plot muon
    if muon_mean is not None:
        ax.plot(muon_steps, muon_mean, label="Manifold Muon", color="tab:orange", linewidth=2)
        ax.fill_between(
            muon_steps,
            muon_mean - muon_stderr,
            muon_mean + muon_stderr,
            alpha=0.3,
            color="tab:orange",
        )

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Average Episodic Return", fontsize=12)
    ax.set_title(f"{env_name} - Manifold Muon vs Baseline", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path is None:
        output_path = f"{env_name}_muon_vs_baseline.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot wandb runs comparison")
    parser.add_argument(
        "--prefix",
        type=str,
        default="saketirl/benchmark",
        help="Wandb project prefix (e.g., saketirl/benchmark)",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="walker2d",
        help="Environment name for the plot title",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="charts/avg_episodic_return",
        help="Metric key to plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: {env_name}_muon_vs_baseline.png)",
    )

    args = parser.parse_args()

    plot_comparison(
        prefix=args.prefix,
        env_name=args.env_name,
        muon_ids=MUON_RUN_IDS,
        baseline_ids=BASELINE_RUN_IDS,
        metric_key=args.metric,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
