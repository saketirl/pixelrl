#!/usr/bin/env python3
"""Plot Muon vs baseline comparison for a finished vit_heads_muon_multienv SLURM array job.

Parses SLURM logs to recover env / heads_muon condition / wandb run IDs, then fetches
`charts/avg_episodic_return` from wandb and writes one comparison plot per environment.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


ENV_DISPLAY_NAMES = {
    "halfcheetah": "HalfCheetah",
    "walker2d": "Walker2D",
    "ant": "Ant",
    "humanoid": "Humanoid",
}

METRIC_KEY = "charts/avg_episodic_return"

ENV_RE = re.compile(r"Running TASK_ID=\d+/\d+ env=([a-zA-Z0-9_]+) seed=(\d+)")
MUON_RE = re.compile(r"Config: use_heads_muon=(true|false) seed=(\d+)")
RUN_RE = re.compile(r"wandb: 🚀 View run at https://wandb\.ai/[^/]+/[^/]+/runs/([a-z0-9]+)")


def parse_log_for_run_info(path: str) -> Tuple[str, bool, int, str]:
    env_name = None
    seed = None
    use_heads_muon = None
    run_id = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if env_name is None:
                m = ENV_RE.search(line)
                if m:
                    env_name = m.group(1)
                    seed = int(m.group(2))
                    continue
            if use_heads_muon is None:
                m = MUON_RE.search(line)
                if m:
                    use_heads_muon = (m.group(1) == "true")
                    continue
            if run_id is None:
                m = RUN_RE.search(line)
                if m:
                    run_id = m.group(1)
                    if env_name is not None and use_heads_muon is not None:
                        break

    if env_name is None or seed is None or use_heads_muon is None or run_id is None:
        raise ValueError(f"Failed to parse run metadata from log: {path}")
    return env_name, use_heads_muon, seed, run_id


def collect_runs_from_logs(log_glob: str) -> Dict[str, Dict[str, List[str]]]:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"muon": [], "baseline": []})

    for path in sorted(glob.glob(log_glob)):
        env_name, use_heads_muon, seed, run_id = parse_log_for_run_info(path)
        bucket = "muon" if use_heads_muon else "baseline"
        grouped[env_name][bucket].append(run_id)
        print(f"Parsed {os.path.basename(path)} -> env={env_name} seed={seed} {bucket} run={run_id}")

    return grouped


def fetch_run_data(api: wandb.Api, run_path: str, metric_key: str = METRIC_KEY, samples: int = 1000):
    run = api.run(run_path)
    history = run.history(keys=[metric_key, "_step"], samples=samples)

    # wandb returns a pandas DataFrame when pandas is available, otherwise a list[dict].
    if hasattr(history, "dropna"):
        history = history.dropna(subset=[metric_key])
        if history.empty:
            return None, None
        steps = history["_step"].to_numpy()
        values = history[metric_key].to_numpy()
    else:
        rows = []
        for row in history:
            if not isinstance(row, dict):
                continue
            if metric_key not in row or "_step" not in row:
                continue
            value = row.get(metric_key)
            step = row.get("_step")
            if value is None or step is None:
                continue
            try:
                rows.append((float(step), float(value)))
            except (TypeError, ValueError):
                continue
        if not rows:
            return None, None
        steps = np.array([r[0] for r in rows])
        values = np.array([r[1] for r in rows])

    sort_idx = np.argsort(steps)
    return steps[sort_idx], values[sort_idx]


def aggregate_runs(
    api: wandb.Api,
    prefix: str,
    run_ids: List[str],
    metric_key: str = METRIC_KEY,
    samples: int = 1000,
    num_points: int = 500,
):
    all_steps = []
    all_values = []

    for run_id in run_ids:
        run_path = f"{prefix}/{run_id}"
        print(f"Fetching {run_path}")
        try:
            steps, values = fetch_run_data(api, run_path, metric_key=metric_key, samples=samples)
        except Exception as e:
            print(f"  Warning: failed to fetch {run_path}: {e}")
            continue
        if steps is None or len(steps) == 0:
            print(f"  Warning: no metric data in {run_path}")
            continue
        all_steps.append(steps)
        all_values.append(values)

    if not all_values:
        return None, None, None

    min_step = max(s.min() for s in all_steps)
    max_step = min(s.max() for s in all_steps)
    if max_step <= min_step:
        return None, None, None

    common_steps = np.linspace(min_step, max_step, num_points)
    interp = np.array([np.interp(common_steps, s, v) for s, v in zip(all_steps, all_values)])
    mean = np.mean(interp, axis=0)
    stderr = np.std(interp, axis=0) / np.sqrt(len(interp))
    return common_steps, mean, stderr


def plot_env(
    env_name: str,
    muon: Tuple[np.ndarray, np.ndarray, np.ndarray],
    baseline: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: str,
):
    display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)
    fig, ax = plt.subplots(figsize=(10, 6))

    base_steps, base_mean, base_stderr = baseline
    if base_mean is not None:
        ax.plot(base_steps, base_mean, label="Baseline", color="tab:blue", linewidth=2)
        ax.fill_between(
            base_steps,
            base_mean - base_stderr,
            base_mean + base_stderr,
            alpha=0.3,
            color="tab:blue",
        )

    mu_steps, mu_mean, mu_stderr = muon
    if mu_mean is not None:
        ax.plot(mu_steps, mu_mean, label="Manifold Muon", color="tab:orange", linewidth=2)
        ax.fill_between(
            mu_steps,
            mu_mean - mu_stderr,
            mu_mean + mu_stderr,
            alpha=0.3,
            color="tab:orange",
        )

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Average Episodic Return", fontsize=12)
    ax.set_title(f"{display_name} - Manifold Muon vs Baseline", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="SLURM array job ID (e.g. 43428)")
    parser.add_argument(
        "--log-dir",
        default=".worktrees/vit-muon/slurm_logs",
        help="Directory containing vit_heads_muon_me_<jobid>_<task>.out logs",
    )
    parser.add_argument(
        "--prefix",
        default="rl-power/benchmark",
        help="Wandb project prefix <entity>/<project>",
    )
    parser.add_argument(
        "--plots-dir",
        default=".worktrees/vit-muon/plots",
        help="Where to write output plots",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of history samples per run to fetch from wandb",
    )
    args = parser.parse_args()

    log_glob = os.path.join(args.log_dir, f"vit_heads_muon_me_{args.job_id}_*.out")
    grouped = collect_runs_from_logs(log_glob)
    if not grouped:
        raise SystemExit(f"No logs matched {log_glob}")

    os.makedirs(args.plots_dir, exist_ok=True)
    api = wandb.Api(timeout=120)

    for env_name in sorted(grouped.keys()):
        muon_ids = sorted(grouped[env_name]["muon"])
        baseline_ids = sorted(grouped[env_name]["baseline"])
        print(
            f"\n{env_name}: {len(muon_ids)} muon runs, {len(baseline_ids)} baseline runs"
        )
        if not muon_ids or not baseline_ids:
            print(f"Skipping {env_name}: missing one condition")
            continue

        muon = aggregate_runs(api, args.prefix, muon_ids, samples=args.samples)
        baseline = aggregate_runs(api, args.prefix, baseline_ids, samples=args.samples)
        if muon[0] is None and baseline[0] is None:
            print(f"Skipping {env_name}: no data fetched")
            continue

        display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)
        output_path = os.path.join(args.plots_dir, f"{display_name}_muon_vs_baseline.png")
        plot_env(env_name, muon, baseline, output_path)


if __name__ == "__main__":
    main()
