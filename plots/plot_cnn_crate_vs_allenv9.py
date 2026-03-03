#!/usr/bin/env python3
"""Plot CRATE special-case comparisons against allenv9 CNN Adam baselines.

Produces two plots per environment (ant, humanoid):
  1) CNN Adam baseline vs CNN CRATE Manifold Muon
  2) CNN Adam baseline vs CNN CRATE Manifold Muon vs CNN CRATE Adam
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_METRIC_KEY = "charts/avg_episodic_return"
REQUIRED_TAGS = {"allenv9", "sweep6seeds", "cnn_vit_heads_muon_me"}
TARGET_ENVS = ["ant", "humanoid"]

LABELS = {
    "baseline_cnn_adam": "CNN Encoder + Adam",
    "crate_muon": "CNN CRATE Encoder + Manifold Muon (ours)",
    "crate_adam": "CNN CRATE Encoder + Adam",
}

COLORS = {
    "baseline_cnn_adam": "tab:blue",
    "crate_muon": "tab:orange",
    "crate_adam": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CRATE (csv) runs against allenv9 CNN Adam baselines."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Wandb project path <entity>/<project> (e.g. rl-power/benchmark).",
    )
    parser.add_argument(
        "--crate-data-dir",
        default=".worktrees/vit-muon/plots/crate_data",
        help="Directory containing *_crate_{muon,adam}.csv files.",
    )
    parser.add_argument(
        "--plots-dir",
        default=".worktrees/vit-muon/plots/crate",
        help="Directory where plot PNGs are written.",
    )
    parser.add_argument(
        "--metric-key",
        default=DEFAULT_METRIC_KEY,
        help=f"Wandb history metric to plot (default: {DEFAULT_METRIC_KEY}).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="History samples to fetch per wandb run.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=500,
        help="Number of points on common interpolation grid for wandb aggregation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Wandb API timeout (seconds).",
    )
    parser.add_argument(
        "--min-runs-per-condition",
        type=int,
        default=1,
        help="Minimum valid runs needed to aggregate a condition.",
    )
    return parser.parse_args()


def _coerce_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _extract_rows(history, metric_key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # wandb may return pandas DataFrame or list[dict].
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
            step = row.get("_step")
            value = row.get(metric_key)
            if step is None or value is None:
                continue
            try:
                rows.append((float(step), float(value)))
            except (TypeError, ValueError):
                continue
        if not rows:
            return None, None
        steps = np.array([r[0] for r in rows])
        values = np.array([r[1] for r in rows])

    order = np.argsort(steps)
    return steps[order], values[order]


def fetch_run_metric(
    run,
    metric_key: str,
    samples: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        history = run.history(keys=[metric_key, "_step"], samples=samples)
    except Exception as exc:
        print(f"  Warning: failed history fetch for {run.path}: {exc}")
        return None, None
    return _extract_rows(history, metric_key)


def aggregate_wandb_runs(
    runs: Iterable,
    metric_key: str,
    samples: int,
    num_points: int,
    min_runs_per_condition: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    all_steps = []
    all_values = []
    valid_count = 0

    for run in runs:
        steps, values = fetch_run_metric(run, metric_key=metric_key, samples=samples)
        if steps is None or values is None or len(steps) == 0:
            continue
        all_steps.append(steps)
        all_values.append(values)
        valid_count += 1

    if valid_count < min_runs_per_condition:
        return None, None, None, valid_count

    min_step = max(s.min() for s in all_steps)
    max_step = min(s.max() for s in all_steps)
    if max_step <= min_step:
        return None, None, None, valid_count

    common_steps = np.linspace(min_step, max_step, num_points)
    interp = np.array([np.interp(common_steps, s, v) for s, v in zip(all_steps, all_values)])
    mean = np.mean(interp, axis=0)
    stderr = np.std(interp, axis=0) / np.sqrt(len(interp))
    return common_steps, mean, stderr, valid_count


def parse_crate_csv(
    path: str,
    min_runs_per_condition: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CRATE CSV: {path}")

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None, None, None, 0

        step_idx = None
        for i, col in enumerate(header):
            if col.strip().strip('"') == "Step":
                step_idx = i
                break
        if step_idx is None:
            raise ValueError(f"Missing Step column in {path}")

        run_indices = []
        for i, col in enumerate(header):
            if i == step_idx:
                continue
            normalized = col.strip().strip('"')
            if normalized.endswith("__MIN") or normalized.endswith("__MAX"):
                continue
            run_indices.append(i)

        if not run_indices:
            return None, None, None, 0

        steps = []
        rows = []
        for row in reader:
            if step_idx >= len(row):
                continue
            try:
                step = float(row[step_idx])
            except (TypeError, ValueError):
                continue

            values = []
            for idx in run_indices:
                if idx >= len(row):
                    values.append(np.nan)
                    continue
                try:
                    values.append(float(row[idx]))
                except (TypeError, ValueError):
                    values.append(np.nan)

            steps.append(step)
            rows.append(values)

    if not steps:
        return None, None, None, 0

    step_arr = np.asarray(steps, dtype=float)
    val_arr = np.asarray(rows, dtype=float)
    order = np.argsort(step_arr)
    step_arr = step_arr[order]
    val_arr = val_arr[order]

    # number of runs with at least one valid point
    run_valid_mask = np.isfinite(val_arr).any(axis=0)
    valid_runs = int(run_valid_mask.sum())
    if valid_runs < min_runs_per_condition:
        return None, None, None, valid_runs

    agg_steps = []
    agg_means = []
    agg_stderr = []
    for i in range(len(step_arr)):
        row_vals = val_arr[i]
        row_vals = row_vals[np.isfinite(row_vals)]
        if row_vals.size == 0:
            continue
        agg_steps.append(step_arr[i])
        agg_means.append(float(np.mean(row_vals)))
        agg_stderr.append(float(np.std(row_vals) / np.sqrt(row_vals.size)))

    if not agg_steps:
        return None, None, None, valid_runs

    return (
        np.asarray(agg_steps, dtype=float),
        np.asarray(agg_means, dtype=float),
        np.asarray(agg_stderr, dtype=float),
        valid_runs,
    )


def collect_baseline_runs(api: wandb.Api, prefix: str) -> Dict[str, List]:
    grouped = defaultdict(list)
    seen_ids: Set[str] = set()

    runs = api.runs(path=prefix, filters={"tags": {"$in": ["allenv9"]}})
    fetched = 0
    matched = 0

    for run in runs:
        fetched += 1
        run_id = getattr(run, "id", None)
        if run_id is None or run_id in seen_ids:
            continue
        seen_ids.add(run_id)

        tags = set(getattr(run, "tags", []) or [])
        if not REQUIRED_TAGS.issubset(tags):
            continue

        config = getattr(run, "config", {}) or {}
        env_name = str(config.get("env_name", "")).strip().lower()
        if env_name not in TARGET_ENVS:
            continue

        encoder_type = str(config.get("encoder_type", "")).strip().lower()
        use_heads_muon = _coerce_bool(config.get("use_heads_muon"))
        if encoder_type != "cnn" or use_heads_muon is not False:
            continue

        grouped[env_name].append(run)
        matched += 1

    print(f"Fetched {fetched} runs with server-side allenv9 filter.")
    print(f"Matched {matched} allenv9 CNN Adam runs for target envs.")
    return grouped


def metric_to_ylabel(metric_key: str) -> str:
    if metric_key == DEFAULT_METRIC_KEY:
        return "Average Episodic Return"
    tail = metric_key.split("/")[-1]
    return tail.replace("_", " ").title()


def plot_curves(
    output_path: str,
    curves: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, steps, mean, stderr in curves:
        ax.plot(steps, mean, label=LABELS[key], color=COLORS[key], linewidth=2)
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=COLORS[key],
            alpha=0.25,
        )

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    api = wandb.Api(timeout=args.timeout)
    baseline_runs = collect_baseline_runs(api, args.prefix)

    y_label = metric_to_ylabel(args.metric_key)
    summary = []

    for env_name in TARGET_ENVS:
        print(f"\n=== {env_name} ===")
        env_summary = {"env": env_name, "written": [], "skipped": []}

        baseline = aggregate_wandb_runs(
            runs=baseline_runs.get(env_name, []),
            metric_key=args.metric_key,
            samples=args.samples,
            num_points=args.num_points,
            min_runs_per_condition=args.min_runs_per_condition,
        )
        if baseline[0] is None:
            env_summary["skipped"].append("baseline_cnn_adam")
            print(f"  baseline_cnn_adam: unavailable")
        else:
            print(f"  baseline_cnn_adam: aggregated {baseline[3]} runs")

        crate_muon_path = os.path.join(args.crate_data_dir, f"{env_name}_crate_muon.csv")
        crate_adam_path = os.path.join(args.crate_data_dir, f"{env_name}_crate_adam.csv")

        crate_muon = parse_crate_csv(crate_muon_path, args.min_runs_per_condition)
        if crate_muon[0] is None:
            env_summary["skipped"].append("crate_muon")
            print(f"  crate_muon: unavailable")
        else:
            print(f"  crate_muon: aggregated {crate_muon[3]} runs")

        crate_adam = parse_crate_csv(crate_adam_path, args.min_runs_per_condition)
        if crate_adam[0] is None:
            env_summary["skipped"].append("crate_adam")
            print(f"  crate_adam: unavailable")
        else:
            print(f"  crate_adam: aggregated {crate_adam[3]} runs")

        # Plot 1: baseline vs crate muon
        if baseline[0] is not None and crate_muon[0] is not None:
            output_1 = os.path.join(args.plots_dir, f"{env_name}_cnn_adam_vs_crate_muon.png")
            curves_1 = [
                ("crate_muon", crate_muon[0], crate_muon[1], crate_muon[2]),
                ("baseline_cnn_adam", baseline[0], baseline[1], baseline[2]),
            ]
            plot_curves(output_1, curves_1, y_label=y_label)
            env_summary["written"].append(os.path.basename(output_1))
        else:
            env_summary["skipped"].append("plot_baseline_vs_crate_muon")

        # Plot 2: baseline vs crate muon vs crate adam
        if baseline[0] is not None and crate_muon[0] is not None and crate_adam[0] is not None:
            output_2 = os.path.join(
                args.plots_dir, f"{env_name}_cnn_adam_vs_crate_muon_vs_crate_adam.png"
            )
            curves_2 = [
                ("crate_muon", crate_muon[0], crate_muon[1], crate_muon[2]),
                ("crate_adam", crate_adam[0], crate_adam[1], crate_adam[2]),
                ("baseline_cnn_adam", baseline[0], baseline[1], baseline[2]),
            ]
            plot_curves(output_2, curves_2, y_label=y_label)
            env_summary["written"].append(os.path.basename(output_2))
        else:
            env_summary["skipped"].append("plot_baseline_vs_crate_muon_vs_crate_adam")

        summary.append(env_summary)

    print("\nSummary:")
    for row in summary:
        print(f"  {row['env']}:")
        if row["written"]:
            print(f"    wrote: {', '.join(row['written'])}")
        if row["skipped"]:
            print(f"    skipped: {', '.join(row['skipped'])}")


if __name__ == "__main__":
    main()
