#!/usr/bin/env python3
"""
Plot Muon vs Adam on actor/critic heads for best ViT config.
4 seeds each, conv-stem ViT (patch=21, hidden=128, heads=8, layers=4).

Usage:
    python plots/plot_vit_heads_muon.py
    python plots/plot_vit_heads_muon.py --output my_plot.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

PREFIX = "rl-power/benchmark"
ENV_NAME = "halfcheetah"

# Tasks 0-3: use_heads_muon=True (Muon on actor/critic matrices)
MUON_RUN_IDS = [
    "znpw82jt",  # seed 0
    "e1rk9676",  # seed 1
    "xrxugyiu",  # seed 2
    "7wwdfgnr",  # seed 3
]

# Tasks 4-7: use_heads_muon=False (Adam for all head params)
ADAM_RUN_IDS = [
    "d80orsck",  # seed 0
    "s8t6c2o9",  # seed 1
    "ygm3nj4t",  # seed 2
    "npo6dc2i",  # seed 3
]


def fetch_run_data(run_id: str, metric_key: str, samples: int = 500):
    api = wandb.Api(timeout=90)
    run = api.run(f"{PREFIX}/{run_id}")
    rows = run.history(keys=[metric_key, "_step"], samples=samples, pandas=False)
    steps, values = [], []
    for row in rows:
        if metric_key in row and row[metric_key] is not None:
            steps.append(row["_step"])
            values.append(row[metric_key])
    steps, values = np.array(steps), np.array(values)
    idx = np.argsort(steps)
    return steps[idx], values[idx]


def fetch_group(run_ids, metric_key, num_points=500):
    all_steps, all_values = [], []
    for run_id in run_ids:
        print(f"  Fetching {run_id}...")
        try:
            steps, values = fetch_run_data(run_id, metric_key)
            if len(steps) > 0:
                all_steps.append(steps)
                all_values.append(values)
        except Exception as e:
            print(f"  Warning: {run_id} failed: {e}")

    if not all_values:
        return None, None, None

    min_step = max(s.min() for s in all_steps)
    max_step = min(s.max() for s in all_steps)
    common_steps = np.linspace(min_step, max_step, num_points)

    interpolated = np.array([
        np.interp(common_steps, s, v)
        for s, v in zip(all_steps, all_values)
    ])

    mean = np.mean(interpolated, axis=0)
    stderr = np.std(interpolated, axis=0) / np.sqrt(len(interpolated))
    return common_steps, mean, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="charts/avg_episodic_return")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Fetching Muon (actor/critic) runs...")
    muon_steps, muon_mean, muon_stderr = fetch_group(MUON_RUN_IDS, args.metric)

    print("\nFetching Adam (actor/critic) runs...")
    adam_steps, adam_mean, adam_stderr = fetch_group(ADAM_RUN_IDS, args.metric)

    fig, ax = plt.subplots(figsize=(10, 6))

    if adam_mean is not None:
        ax.plot(adam_steps, adam_mean, label="Adam (actor/critic)", color="tab:blue", linewidth=2)
        ax.fill_between(adam_steps, adam_mean - adam_stderr, adam_mean + adam_stderr,
                        alpha=0.3, color="tab:blue")

    if muon_mean is not None:
        ax.plot(muon_steps, muon_mean, label="Manifold Muon (actor/critic)", color="tab:orange", linewidth=2)
        ax.fill_between(muon_steps, muon_mean - muon_stderr, muon_mean + muon_stderr,
                        alpha=0.3, color="tab:orange")

    ax.set_xlim(0, 5_000_000)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Average Episodic Return", fontsize=12)
    ax.set_title(f"{ENV_NAME} — ViT (conv-stem, 4 patches, 4L) · Muon vs Adam on Actor/Critic\n"
                 f"patch=21, hidden=128, heads=8 · 4 seeds ± stderr", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plots_dir = os.path.dirname(os.path.abspath(__file__))
    output = args.output or os.path.join(plots_dir, f"{ENV_NAME}_vit_heads_muon_vs_adam.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output}")
    plt.show()


if __name__ == "__main__":
    main()
