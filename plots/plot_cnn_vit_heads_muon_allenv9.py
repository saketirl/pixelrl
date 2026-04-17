#!/usr/bin/env python3
"""Plot allenv9 CNN/ViT Muon-vs-Adam comparisons from wandb runs.

This script fetches runs directly from wandb using tags, aggregates
`charts/avg_episodic_return` across runs with standard error, and writes
per-environment plots into:
  - combined/  (CNN+Adam, CNN+Muon, ViT+Adam, ViT+Muon)
  - cnn_only/  (CNN+Adam, CNN+Muon)
  - vit_only/  (ViT+Adam, ViT+Muon)
"""

from __future__ import annotations

import argparse
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

ENV_MAX_STEP = {
    "walker2d": 2_000_000.0,
}

CONDITION_META = {
    "cnn_adam": {"label": "CNN Encoder + Adam", "color": "tab:blue"},
    "cnn_muon": {"label": "CNN Encoder + Manifold Muon (ours)", "color": "tab:orange"},
    "vit_adam": {"label": "ViT Encoder + Adam", "color": "tab:green"},
    "vit_muon": {"label": "ViT Encoder + Manifold Muon (ours)", "color": "tab:red"},
}

FIGSIZE = (10, 6)  # default/wide aspect ratio
LINEWIDTH = 2.5
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CNN/ViT Muon-vs-Adam comparisons from allenv9 wandb runs."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Wandb project path <entity>/<project> (e.g. rl-power/benchmark).",
    )
    parser.add_argument(
        "--plots-root",
        default=".worktrees/vit-muon/plots/cnn_vit_heads_muon_allenv9",
        help="Output root directory for plots.",
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
        help="History samples to fetch per run.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=500,
        help="Number of points on common interpolation grid.",
    )
    parser.add_argument(
        "--min-runs-per-condition",
        type=int,
        default=1,
        help="Minimum valid runs required to aggregate a condition.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Wandb API timeout (seconds).",
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


def _condition_from_config(config: Dict) -> Optional[str]:
    encoder_type = str(config.get("encoder_type", "")).strip().lower()
    use_heads_muon = _coerce_bool(config.get("use_heads_muon"))

    if encoder_type not in {"cnn", "vit"} or use_heads_muon is None:
        return None

    if encoder_type == "cnn":
        return "cnn_muon" if use_heads_muon else "cnn_adam"
    return "vit_muon" if use_heads_muon else "vit_adam"


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

    sort_idx = np.argsort(steps)
    return steps[sort_idx], values[sort_idx]


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


def aggregate_condition(
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
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for condition_key, steps, mean, stderr in curves:
        meta = CONDITION_META[condition_key]
        ax.plot(steps, mean, label=meta["label"], color=meta["color"], linewidth=LINEWIDTH)
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=meta["color"],
            alpha=0.25,
        )

    ax.set_xlabel("Steps", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.xaxis.get_offset_text().set_size(TICK_LABEL_FONTSIZE)
    ax.yaxis.get_offset_text().set_size(TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def collect_runs(api: wandb.Api, prefix: str) -> Dict[str, Dict[str, List]]:
    grouped = defaultdict(lambda: defaultdict(list))
    seen_ids: Set[str] = set()

    # Pre-filter with allenv9 on server side, then enforce strict tag subset client-side.
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
        if env_name not in ENV_ORDER:
            continue

        condition = _condition_from_config(config)
        if condition is None:
            continue

        grouped[env_name][condition].append(run)
        matched += 1

    print(f"Fetched {fetched} runs with server-side allenv9 filter.")
    print(f"Matched {matched} runs after strict tag/config filtering.")
    return grouped


def ensure_output_dirs(plots_root: str) -> Dict[str, str]:
    dirs = {
        "combined": os.path.join(plots_root, "combined"),
        "cnn_only": os.path.join(plots_root, "cnn_only"),
        "vit_only": os.path.join(plots_root, "vit_only"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=args.timeout)

    grouped_runs = collect_runs(api, args.prefix)
    if not grouped_runs:
        raise SystemExit("No runs matched required filters and config fields.")

    print("\nRun counts by env/condition:")
    for env_name in ENV_ORDER:
        counts = grouped_runs.get(env_name, {})
        print(
            f"  {env_name:18s} "
            f"cnn_adam={len(counts.get('cnn_adam', [])):3d} "
            f"cnn_muon={len(counts.get('cnn_muon', [])):3d} "
            f"vit_adam={len(counts.get('vit_adam', [])):3d} "
            f"vit_muon={len(counts.get('vit_muon', [])):3d}"
        )

    aggregated = defaultdict(dict)
    print("\nAggregating conditions...")
    for env_name in ENV_ORDER:
        for condition in CONDITION_META:
            runs = grouped_runs.get(env_name, {}).get(condition, [])
            steps, mean, stderr, valid_count = aggregate_condition(
                runs=runs,
                metric_key=args.metric_key,
                samples=args.samples,
                num_points=args.num_points,
                min_runs_per_condition=args.min_runs_per_condition,
            )
            if steps is None:
                if runs:
                    print(
                        f"  {env_name}/{condition}: no aggregate (valid={valid_count}, total={len(runs)})"
                    )
                continue
            aggregated[env_name][condition] = (steps, mean, stderr)
            print(
                f"  {env_name}/{condition}: aggregated {valid_count} valid runs (total={len(runs)})"
            )

    output_dirs = ensure_output_dirs(args.plots_root)
    y_label = metric_to_ylabel(args.metric_key)

    folder_specs = {
        "combined": {
            "required_conditions": ["cnn_muon", "cnn_adam", "vit_muon", "vit_adam"],
            "filename_template": "{env}_combined_muon_vs_adam.png",
        },
        "cnn_only": {
            "required_conditions": ["cnn_muon", "cnn_adam"],
            "filename_template": "{env}_cnn_muon_vs_adam.png",
        },
        "vit_only": {
            "required_conditions": ["vit_muon", "vit_adam"],
            "filename_template": "{env}_vit_muon_vs_adam.png",
        },
    }

    print("\nWriting plots...")
    plotted_counts = {k: 0 for k in folder_specs}
    skipped = {k: [] for k in folder_specs}

    for folder_name, spec in folder_specs.items():
        required = spec["required_conditions"]
        for env_name in ENV_ORDER:
            env_data = aggregated.get(env_name, {})
            missing = [cond for cond in required if cond not in env_data]
            if missing:
                skipped[folder_name].append((env_name, missing))
                continue

            max_step = ENV_MAX_STEP.get(env_name)
            curves = []
            for cond in required:
                steps, mean, stderr = env_data[cond]
                if max_step is not None:
                    keep = steps <= max_step
                    if not np.any(keep):
                        continue
                    steps = steps[keep]
                    mean = mean[keep]
                    stderr = stderr[keep]
                curves.append((cond, steps, mean, stderr))

            if len(curves) != len(required):
                skipped[folder_name].append((env_name, ["step_cutoff_removed_condition"]))
                continue

            filename = spec["filename_template"].format(env=env_name)
            output_path = os.path.join(output_dirs[folder_name], filename)
            plot_curves(output_path=output_path, curves=curves, y_label=y_label)
            plotted_counts[folder_name] += 1

    print("\nPlot summary:")
    for folder_name in ["combined", "cnn_only", "vit_only"]:
        print(f"  {folder_name:9s}: {plotted_counts[folder_name]} plotted")
        for env_name, missing in skipped[folder_name]:
            print(f"    skipped {env_name}: missing {', '.join(missing)}")


if __name__ == "__main__":
    main()
