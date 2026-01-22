#!/usr/bin/env python3
"""
Plot individual episodic return curves for each run in a list.

Usage:
    python plot_individual_runs.py --prefix saketirl/benchmark --env-name walker2d
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Placeholder run IDs - replace with actual run IDs
RUN_IDS = [
    "0wiec0ah",
    "wblu2aq5",
    "j0qha76k",
    "jrgdarke"
]


def fetch_run_data(run_path: str, metric_key: str = "charts/avg_episodic_return", samples: int = 500):
    """Fetch metric data from a wandb run using sampled history."""
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

    return np.array(steps), np.array(values), run.name


def plot_individual_runs(
    prefix: str,
    env_name: str,
    run_ids: list,
    metric_key: str = "charts/avg_episodic_return",
    output_path: str = None,
):
    """Plot each run's return curve individually."""

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))

    for i, run_id in enumerate(run_ids):
        run_path = f"{prefix}/{run_id}"
        print(f"Fetching data from: {run_path}")
        try:
            steps, values, run_name = fetch_run_data(run_path, metric_key)
            if len(steps) > 0:
                label = f"({run_id})"
                ax.plot(steps, values, label=label, color=colors[i], linewidth=1.5, alpha=0.8)
        except Exception as e:
            print(f"  Warning: Failed to fetch {run_path}: {e}")

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Average Episodic Return", fontsize=12)
    ax.set_title(f"{env_name} - Individual Runs", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path is None:
        output_path = f"{env_name}_individual_runs.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot individual wandb runs")
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
        help="Output file path (default: {env_name}_individual_runs.png)",
    )

    args = parser.parse_args()

    plot_individual_runs(
        prefix=args.prefix,
        env_name=args.env_name,
        run_ids=RUN_IDS,
        metric_key=args.metric,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
