#!/usr/bin/env python3
"""Plot inverted_pendulum probe CSV metrics (returns and R2) with repo plot style."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (10, 6)  # default/wide aspect ratio
LINEWIDTH = 2.5
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18

CONDITIONS = ("cnn_adam", "cnn_muon")
LABELS = {
    "cnn_adam": "CNN Encoder + Adam",
    "cnn_muon": "CNN Encoder + Manifold Muon (ours)",
}
COLORS = {
    "cnn_adam": "tab:blue",
    "cnn_muon": "tab:orange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot probe CSV metrics for inverted_pendulum.")
    parser.add_argument(
        "--probe-data-dir",
        default=".worktrees/vit-muon/plots/probe_data",
        help="Directory containing split or combined probe CSV files.",
    )
    parser.add_argument(
        "--plots-dir",
        default=".worktrees/vit-muon/plots/probe",
        help="Directory where output PNGs will be written.",
    )
    return parser.parse_args()


def _condition_from_header_col(col_name: str) -> Optional[str]:
    lowered = col_name.lower()
    if "_cnn_adam_" in lowered or "cnn_adam" in lowered:
        return "cnn_adam"
    if "_cnn_muon_" in lowered or "cnn_muon" in lowered:
        return "cnn_muon"
    if "muon" in lowered:
        return "cnn_muon"
    if "adam" in lowered:
        return "cnn_adam"
    return None


def parse_probe_csv(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV has no rows: {path}")

        step_idx = None
        for i, col in enumerate(header):
            if col.strip().strip('"') == "Step":
                step_idx = i
                break
        if step_idx is None:
            raise ValueError(f"Missing Step column in {path}")

        run_indices_by_cond: Dict[str, List[int]] = {k: [] for k in CONDITIONS}
        for i, col in enumerate(header):
            if i == step_idx:
                continue
            normalized = col.strip().strip('"')
            if normalized.endswith("__MIN") or normalized.endswith("__MAX"):
                continue
            condition = _condition_from_header_col(normalized)
            if condition is not None:
                run_indices_by_cond[condition].append(i)

        if not any(run_indices_by_cond.values()):
            raise ValueError(f"No Adam/Muon run columns found in {path}")

        steps: List[float] = []
        values_by_cond: Dict[str, List[List[float]]] = {k: [] for k in CONDITIONS}
        for row in reader:
            if step_idx >= len(row):
                continue
            try:
                step = float(row[step_idx])
            except (TypeError, ValueError):
                continue

            row_values_by_cond: Dict[str, List[float]] = {k: [] for k in CONDITIONS}
            for condition, indices in run_indices_by_cond.items():
                for idx in indices:
                    if idx >= len(row):
                        row_values_by_cond[condition].append(np.nan)
                        continue
                    try:
                        row_values_by_cond[condition].append(float(row[idx]))
                    except (TypeError, ValueError):
                        row_values_by_cond[condition].append(np.nan)

            steps.append(step)
            for condition in CONDITIONS:
                values_by_cond[condition].append(row_values_by_cond[condition])

    if not steps:
        raise ValueError(f"No valid rows in {path}")

    step_arr = np.asarray(steps, dtype=float)
    order = np.argsort(step_arr)
    step_arr = step_arr[order]

    means: Dict[str, np.ndarray] = {}
    stderrs: Dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        raw = np.asarray(values_by_cond[condition], dtype=float)[order]
        if raw.shape[1] == 0:
            means[condition] = np.full(step_arr.shape, np.nan, dtype=float)
            stderrs[condition] = np.full(step_arr.shape, np.nan, dtype=float)
            continue

        mean_vals = np.zeros(step_arr.shape, dtype=float)
        stderr_vals = np.zeros(step_arr.shape, dtype=float)
        for i in range(raw.shape[0]):
            row_vals = raw[i]
            row_vals = row_vals[np.isfinite(row_vals)]
            if row_vals.size == 0:
                mean_vals[i] = np.nan
                stderr_vals[i] = np.nan
                continue
            mean_vals[i] = float(np.mean(row_vals))
            stderr_vals[i] = float(np.std(row_vals) / np.sqrt(row_vals.size))

        means[condition] = mean_vals
        stderrs[condition] = stderr_vals

    return step_arr, means, stderrs


def _load_single_condition_file(
    path: str,
    condition: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, means, stderrs = parse_probe_csv(path)
    mean = means[condition]
    stderr = stderrs[condition]
    if not np.isfinite(mean).any():
        raise ValueError(f"No finite {condition} values found in {path}")
    return steps, mean, stderr


def load_metric_data(
    probe_data_dir: str,
    base_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    split_adam = os.path.join(probe_data_dir, f"{base_name}_adam.csv")
    split_muon = os.path.join(probe_data_dir, f"{base_name}_muon.csv")
    combined = os.path.join(probe_data_dir, f"{base_name}.csv")

    steps_by_cond: Dict[str, np.ndarray] = {}
    means_by_cond: Dict[str, np.ndarray] = {}
    stderrs_by_cond: Dict[str, np.ndarray] = {}

    if os.path.exists(split_adam) and os.path.exists(split_muon):
        adam_steps, adam_mean, adam_stderr = _load_single_condition_file(split_adam, "cnn_adam")
        muon_steps, muon_mean, muon_stderr = _load_single_condition_file(split_muon, "cnn_muon")
        steps_by_cond["cnn_adam"] = adam_steps
        steps_by_cond["cnn_muon"] = muon_steps
        means_by_cond["cnn_adam"] = adam_mean
        means_by_cond["cnn_muon"] = muon_mean
        stderrs_by_cond["cnn_adam"] = adam_stderr
        stderrs_by_cond["cnn_muon"] = muon_stderr
        return steps_by_cond, means_by_cond, stderrs_by_cond

    if os.path.exists(combined):
        steps, means, stderrs = parse_probe_csv(combined)
        for condition in CONDITIONS:
            steps_by_cond[condition] = steps
            means_by_cond[condition] = means[condition]
            stderrs_by_cond[condition] = stderrs[condition]
        return steps_by_cond, means_by_cond, stderrs_by_cond

    raise FileNotFoundError(
        f"Missing metric files for {base_name}: "
        f"expected {os.path.basename(split_adam)} + {os.path.basename(split_muon)} "
        f"or {os.path.basename(combined)}"
    )


def plot_probe_metric(
    output_path: str,
    steps_by_cond: Dict[str, np.ndarray],
    means: Dict[str, np.ndarray],
    stderrs: Dict[str, np.ndarray],
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for condition in CONDITIONS:
        steps = steps_by_cond.get(condition)
        mean = means.get(condition)
        stderr = stderrs.get(condition)
        if steps is None or mean is None or stderr is None:
            continue
        valid = np.isfinite(mean) & np.isfinite(stderr)
        if not np.any(valid):
            continue
        plot_steps = steps[valid]
        plot_mean = mean[valid]
        plot_stderr = stderr[valid]
        ax.plot(
            plot_steps,
            plot_mean,
            color=COLORS[condition],
            linewidth=LINEWIDTH,
            label=LABELS[condition],
        )
        ax.fill_between(
            plot_steps,
            plot_mean - plot_stderr,
            plot_mean + plot_stderr,
            color=COLORS[condition],
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    returns_steps_by_cond, returns_means, returns_stderrs = load_metric_data(
        args.probe_data_dir, "ip_returns"
    )

    if (
        os.path.exists(os.path.join(args.probe_data_dir, "ip_r2_at_state_dim_adam.csv"))
        and os.path.exists(os.path.join(args.probe_data_dir, "ip_r2_at_state_dim_muon.csv"))
    ) or os.path.exists(os.path.join(args.probe_data_dir, "ip_r2_at_state_dim.csv")):
        r2_base = "ip_r2_at_state_dim"
        r2_out_name = "ip_probe_r2_at_state_dim.png"
    else:
        r2_base = "ip_r2_at_state_sim"
        r2_out_name = "ip_probe_r2_at_state_sim.png"

    r2_steps_by_cond, r2_means, r2_stderrs = load_metric_data(args.probe_data_dir, r2_base)

    plot_probe_metric(
        output_path=os.path.join(args.plots_dir, "ip_probe_returns.png"),
        steps_by_cond=returns_steps_by_cond,
        means=returns_means,
        stderrs=returns_stderrs,
        y_label="Average Episodic Return",
    )
    plot_probe_metric(
        output_path=os.path.join(args.plots_dir, r2_out_name),
        steps_by_cond=r2_steps_by_cond,
        means=r2_means,
        stderrs=r2_stderrs,
        y_label=r"$R^2$ with $d_s$ PCs",
    )


if __name__ == "__main__":
    main()
