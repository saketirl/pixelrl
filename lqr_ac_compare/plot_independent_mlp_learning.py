"""Aggregate independent-action MLP runs into a rebuttal-ready learning figure.

The input directory must contain one ``seed_*/metrics.csv`` and
``seed_*/summary.json`` pair per initialization/training-noise seed.  The
script deliberately computes standard errors across those seeds; the
``eval_return_*_se`` columns inside an individual run describe a different
source of uncertainty and are therefore not used for the plotted ribbons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

DOWN_COLOR = "#0072B2"
UP_COLOR = "#D55E00"
LQR_COLOR = "#009E73"
ZERO_COLOR = "0.45"


def mean_and_se(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the seed mean and its standard error.

    Math:
        mean(t) = (1/S) sum_s x_s(t),
        SE(t) = std_s(x_s(t), ddof=1) / sqrt(S).
    Code map:
        ``values`` has shape (S, T); both outputs have shape (T,).
    """
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Need at least two seed traces with shape (S, T)")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def read_run(seed_dir: Path) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Read one complete seed-level trajectory and its saved configuration.

    Math:
        A run supplies x_s(t) for t = 0,...,N-1 and one fixed experimental
        configuration. No aggregation or interpolation occurs here.
    """
    metrics_path = seed_dir / "metrics.csv"
    summary_path = seed_dir / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete run directory: {seed_dir}")
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with summary_path.open() as handle:
        summary = json.load(handle)
    return rows, summary


def validate_runs(
    runs: Sequence[Tuple[Path, List[Dict[str, str]], Dict[str, Any]]],
    expected_seeds: int,
) -> np.ndarray:
    """Verify a complete homogeneous epsilon-zero independent MLP ensemble.

    Math:
        Every seed must provide the same grid {0,...,N-1}; all plotted rows
        must have head=MLP, epsilon=0, and independent training trajectories.
    Code map:
        Return the common integer iteration grid after checking all runs.
    """
    if len(runs) != expected_seeds:
        raise ValueError(f"Expected {expected_seeds} seeds, found {len(runs)}")

    reference_iterations: Optional[np.ndarray] = None
    reference_eval_mask: Optional[np.ndarray] = None
    reference_config: Optional[Dict[str, Any]] = None
    seed_specific_keys = {"init_seed", "noise_seed", "output_dir"}
    for seed_dir, rows, summary in runs:
        config = summary.get("config", {})
        if config.get("head_types") != ["mlp"]:
            raise ValueError(f"{seed_dir}: expected head_types=['mlp']")
        if config.get("training_coupling") != "independent_actions":
            raise ValueError(f"{seed_dir}: expected independent_actions")
        if [float(x) for x in config.get("observation_noises", [])] != [0.0]:
            raise ValueError(f"{seed_dir}: expected epsilon=0 only")
        shared_config = {
            key: value for key, value in config.items() if key not in seed_specific_keys
        }
        if reference_config is None:
            reference_config = shared_config
        elif shared_config != reference_config:
            raise ValueError(f"{seed_dir}: non-seed configuration differs")
        expected_iterations = int(config["iters"])
        if len(rows) != expected_iterations:
            raise ValueError(
                f"{seed_dir}: expected {expected_iterations} rows, found {len(rows)}"
            )
        if any(
            row["head_type"] != "mlp"
            or row["training_coupling"] != "independent_actions"
            or float(row["epsilon"]) != 0.0
            for row in rows
        ):
            raise ValueError(f"{seed_dir}: metrics contain an unexpected condition")

        iterations = np.asarray([int(row["iteration"]) for row in rows])
        if not np.array_equal(iterations, np.arange(expected_iterations)):
            raise ValueError(f"{seed_dir}: incomplete or unordered iteration grid")
        eval_mask = np.asarray(
            [math.isfinite(float(row["eval_return_down"])) for row in rows]
        )
        if not np.array_equal(
            eval_mask,
            np.asarray([math.isfinite(float(row["eval_return_up"])) for row in rows]),
        ):
            raise ValueError(f"{seed_dir}: upstairs/downstairs checkpoints differ")
        if reference_iterations is None:
            reference_iterations = iterations
            reference_eval_mask = eval_mask
        elif not np.array_equal(iterations, reference_iterations):
            raise ValueError(f"{seed_dir}: iteration grid differs across seeds")
        elif not np.array_equal(eval_mask, reference_eval_mask):
            raise ValueError(f"{seed_dir}: evaluation checkpoints differ across seeds")

    assert reference_iterations is not None
    return reference_iterations


def stack_field(
    runs: Sequence[Tuple[Path, List[Dict[str, str]], Dict[str, Any]]],
    field: str,
) -> np.ndarray:
    """Stack a scalar metric as X[s,t] for across-seed statistics."""
    values = np.asarray(
        [[float(row[field]) for row in rows] for _, rows, _ in runs],
        dtype=float,
    )
    return values


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing mean independently within each seed.

    Math:
        x_tilde_s(t) = (1/W) sum_(j=0)^(W-1) x_s(t-j).
    Code map:
        Input shape is (S,T), output shape is (S,T-W+1). Smoothing before
        aggregation preserves the correct across-seed uncertainty.
    """
    if not 1 <= window <= values.shape[1]:
        raise ValueError(f"rolling window must be in [1,{values.shape[1]}]")
    kernel = np.ones(window, dtype=float) / window
    return np.stack([np.convolve(trace, kernel, mode="valid") for trace in values])


def make_aggregate(
    runs: Sequence[Tuple[Path, List[Dict[str, str]], Dict[str, Any]]],
    iterations: np.ndarray,
    rolling_window: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Construct plotted statistics and concise performance diagnostics.

    Math:
        Evaluation uses seed means J_s(t). Training uses rollout means
        rbar_s(t)=N^-1 sum_tau r_(s,t,tau), followed by a per-seed trailing
        mean. Upstairs/downstairs gaps are paired within each seed.
    """
    train_down = stack_field(runs, "train_mean_reward_down")
    train_up = stack_field(runs, "train_mean_reward_up")
    eval_down_full = stack_field(runs, "eval_return_down")
    eval_up_full = stack_field(runs, "eval_return_up")
    eval_mask = np.isfinite(eval_down_full).all(axis=0)
    if not np.array_equal(eval_mask, np.isfinite(eval_up_full).all(axis=0)):
        raise ValueError("Evaluation checkpoint masks differ")

    eval_down = eval_down_full[:, eval_mask]
    eval_up = eval_up_full[:, eval_mask]
    eval_iterations = iterations[eval_mask]
    smooth_down = trailing_mean(train_down, rolling_window)
    smooth_up = trailing_mean(train_up, rolling_window)
    smooth_iterations = iterations[rolling_window - 1 :]

    eval_down_mean, eval_down_se = mean_and_se(eval_down)
    eval_up_mean, eval_up_se = mean_and_se(eval_up)
    eval_gap_mean, eval_gap_se = mean_and_se(eval_up - eval_down)
    train_down_mean, train_down_se = mean_and_se(train_down)
    train_up_mean, train_up_se = mean_and_se(train_up)
    smooth_down_mean, smooth_down_se = mean_and_se(smooth_down)
    smooth_up_mean, smooth_up_se = mean_and_se(smooth_up)

    zero_per_seed = np.asarray(
        [
            next(
                float(row["eval_return_zero"])
                for row in rows
                if math.isfinite(float(row["eval_return_zero"]))
            )
            for _, rows, _ in runs
        ]
    )
    lqr_per_seed = np.asarray(
        [
            next(
                float(row["eval_return_lqr"])
                for row in rows
                if math.isfinite(float(row["eval_return_lqr"]))
            )
            for _, rows, _ in runs
        ]
    )
    zero_mean = float(zero_per_seed.mean())
    lqr_mean = float(lqr_per_seed.mean())
    final_down = eval_down[:, -1]
    final_up = eval_up[:, -1]
    initial_paired = 0.5 * (eval_down[:, 0] + eval_up[:, 0])
    final_paired = 0.5 * (final_down + final_up)
    eval_improvement = final_paired - initial_paired
    tail_eval_per_seed = 0.5 * (
        eval_down[:, -10:].mean(axis=1) + eval_up[:, -10:].mean(axis=1)
    )
    early_train_per_seed = 0.5 * (
        train_down[:, :100].mean(axis=1) + train_up[:, :100].mean(axis=1)
    )
    late_train_per_seed = 0.5 * (
        train_down[:, -100:].mean(axis=1) + train_up[:, -100:].mean(axis=1)
    )
    train_improvement = late_train_per_seed - early_train_per_seed
    denominator = lqr_per_seed - zero_per_seed
    normalized_final = (final_paired - zero_per_seed) / denominator

    aggregate = {
        "iterations": iterations,
        "eval_iterations": eval_iterations,
        "smooth_iterations": smooth_iterations,
        "eval_down_mean": eval_down_mean,
        "eval_down_se": eval_down_se,
        "eval_up_mean": eval_up_mean,
        "eval_up_se": eval_up_se,
        "eval_gap_mean": eval_gap_mean,
        "eval_gap_se": eval_gap_se,
        "train_down_mean": train_down_mean,
        "train_down_se": train_down_se,
        "train_up_mean": train_up_mean,
        "train_up_se": train_up_se,
        "smooth_down_mean": smooth_down_mean,
        "smooth_down_se": smooth_down_se,
        "smooth_up_mean": smooth_up_mean,
        "smooth_up_se": smooth_up_se,
    }
    summary = {
        "n_seeds": len(runs),
        "rolling_window": rolling_window,
        "zero_action_reference_mean": zero_mean,
        "zero_action_reference_se": float(
            zero_per_seed.std(ddof=1) / np.sqrt(len(runs))
        ),
        "discounted_lqr_reference_mean": lqr_mean,
        "discounted_lqr_reference_se": float(
            lqr_per_seed.std(ddof=1) / np.sqrt(len(runs))
        ),
        "initial_eval_return_mean": float(initial_paired.mean()),
        "final_eval_return_mean": float(final_paired.mean()),
        "final_eval_return_se": float(final_paired.std(ddof=1) / np.sqrt(len(runs))),
        "eval_return_improvement": float(eval_improvement.mean()),
        "eval_return_improvement_se": float(
            eval_improvement.std(ddof=1) / np.sqrt(len(runs))
        ),
        "tail_10_eval_return_mean": float(tail_eval_per_seed.mean()),
        "tail_10_eval_return_se": float(
            tail_eval_per_seed.std(ddof=1) / np.sqrt(len(runs))
        ),
        "normalized_zero_to_lqr_fraction_final_mean": float(normalized_final.mean()),
        "normalized_zero_to_lqr_fraction_final_se": float(
            normalized_final.std(ddof=1) / np.sqrt(len(runs))
        ),
        "seeds_improved": int(np.sum(final_paired > initial_paired)),
        "seeds_above_zero_action": int(np.sum(final_paired > zero_per_seed)),
        "seeds_at_or_above_lqr_reference": int(np.sum(final_paired >= lqr_per_seed)),
        "final_paired_up_minus_down_mean": float((final_up - final_down).mean()),
        "final_paired_up_minus_down_se": float(
            (final_up - final_down).std(ddof=1) / np.sqrt(len(runs))
        ),
        "max_abs_eval_gap_mean_curve": float(np.max(np.abs(eval_gap_mean))),
        "first_100_training_mean_reward": float(early_train_per_seed.mean()),
        "last_100_training_mean_reward": float(late_train_per_seed.mean()),
        "training_mean_reward_improvement": float(train_improvement.mean()),
        "training_mean_reward_improvement_se": float(
            train_improvement.std(ddof=1) / np.sqrt(len(runs))
        ),
    }
    return aggregate, summary


def write_aggregate_csv(aggregate: Mapping[str, np.ndarray], path: Path) -> None:
    """Write all full-grid aggregate statistics without interpolating evals."""
    iterations = aggregate["iterations"].astype(int)
    eval_lookup = {
        int(iteration): index
        for index, iteration in enumerate(aggregate["eval_iterations"])
    }
    smooth_lookup = {
        int(iteration): index
        for index, iteration in enumerate(aggregate["smooth_iterations"])
    }
    fields = [
        "iteration",
        "eval_down_mean",
        "eval_down_se",
        "eval_up_mean",
        "eval_up_se",
        "eval_gap_mean",
        "eval_gap_se",
        "train_down_mean",
        "train_down_se",
        "train_up_mean",
        "train_up_se",
        "train_rolling_down_mean",
        "train_rolling_down_se",
        "train_rolling_up_mean",
        "train_rolling_up_se",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, iteration in enumerate(iterations):
            row: Dict[str, Any] = {
                "iteration": int(iteration),
                "train_down_mean": aggregate["train_down_mean"][index],
                "train_down_se": aggregate["train_down_se"][index],
                "train_up_mean": aggregate["train_up_mean"][index],
                "train_up_se": aggregate["train_up_se"][index],
            }
            if iteration in eval_lookup:
                eval_index = eval_lookup[iteration]
                for field in (
                    "eval_down_mean",
                    "eval_down_se",
                    "eval_up_mean",
                    "eval_up_se",
                    "eval_gap_mean",
                    "eval_gap_se",
                ):
                    row[field] = aggregate[field][eval_index]
            if iteration in smooth_lookup:
                smooth_index = smooth_lookup[iteration]
                row.update(
                    {
                        "train_rolling_down_mean": aggregate["smooth_down_mean"][
                            smooth_index
                        ],
                        "train_rolling_down_se": aggregate["smooth_down_se"][
                            smooth_index
                        ],
                        "train_rolling_up_mean": aggregate["smooth_up_mean"][
                            smooth_index
                        ],
                        "train_rolling_up_se": aggregate["smooth_up_se"][smooth_index],
                    }
                )
            writer.writerow(row)


def plot_learning(
    aggregate: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot fixed-bank return and exploration-on reward with seed-level SE.

    Math:
        Panel (a) shows mean_s J_s(t) +/- SE_s[J_s(t)]. Panel (b) shows
        mean_s rbar_tilde_s(t) +/- SE_s[rbar_tilde_s(t)], where the rolling
        average is taken within each seed before aggregation.
    """
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    eval_axis, train_axis = axes

    eval_iterations = aggregate["eval_iterations"]
    mark_every = max(1, len(eval_iterations) // 12)
    up_band = eval_axis.fill_between(
        eval_iterations,
        aggregate["eval_up_mean"] - aggregate["eval_up_se"],
        aggregate["eval_up_mean"] + aggregate["eval_up_se"],
        facecolor=UP_COLOR,
        edgecolor=UP_COLOR,
        alpha=0.10,
        hatch="////",
        linewidth=0.3,
    )
    up_band.set_zorder(1)
    eval_axis.fill_between(
        eval_iterations,
        aggregate["eval_down_mean"] - aggregate["eval_down_se"],
        aggregate["eval_down_mean"] + aggregate["eval_down_se"],
        color=DOWN_COLOR,
        alpha=0.14,
        linewidth=0.0,
        zorder=2,
    )
    up_line = eval_axis.plot(
        eval_iterations,
        aggregate["eval_up_mean"],
        color=UP_COLOR,
        linestyle="--",
        marker="x",
        markevery=mark_every,
        linewidth=2.2,
        alpha=0.80,
        label="Upstairs",
        zorder=3,
    )[0]
    down_line = eval_axis.plot(
        eval_iterations,
        aggregate["eval_down_mean"],
        color=DOWN_COLOR,
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=3.2,
        markevery=mark_every,
        linewidth=1.5,
        label="Downstairs",
        zorder=4,
    )[0]
    zero_line = eval_axis.axhline(
        summary["zero_action_reference_mean"],
        color=ZERO_COLOR,
        linestyle=":",
        linewidth=1.2,
        label="Zero action",
    )
    lqr_line = eval_axis.axhline(
        summary["discounted_lqr_reference_mean"],
        color=LQR_COLOR,
        linestyle="-.",
        linewidth=1.2,
        label="Discounted LQR",
    )
    eval_axis.set_title("(a) Fixed-bank evaluation")
    eval_axis.set_ylabel("Discounted return ↑")
    eval_axis.set_xlabel("Training iteration")

    iterations = aggregate["iterations"]
    train_axis.plot(
        iterations,
        aggregate["train_up_mean"],
        color=UP_COLOR,
        alpha=0.12,
        linewidth=0.5,
    )
    train_axis.plot(
        iterations,
        aggregate["train_down_mean"],
        color=DOWN_COLOR,
        alpha=0.14,
        linewidth=0.5,
    )
    smooth_iterations = aggregate["smooth_iterations"]
    train_axis.fill_between(
        smooth_iterations,
        aggregate["smooth_up_mean"] - aggregate["smooth_up_se"],
        aggregate["smooth_up_mean"] + aggregate["smooth_up_se"],
        facecolor=UP_COLOR,
        edgecolor=UP_COLOR,
        alpha=0.10,
        hatch="////",
        linewidth=0.3,
    )
    train_axis.fill_between(
        smooth_iterations,
        aggregate["smooth_down_mean"] - aggregate["smooth_down_se"],
        aggregate["smooth_down_mean"] + aggregate["smooth_down_se"],
        color=DOWN_COLOR,
        alpha=0.14,
        linewidth=0.0,
    )
    train_axis.plot(
        smooth_iterations,
        aggregate["smooth_up_mean"],
        color=UP_COLOR,
        linestyle="--",
        linewidth=2.2,
        alpha=0.80,
    )
    train_axis.plot(
        smooth_iterations,
        aggregate["smooth_down_mean"],
        color=DOWN_COLOR,
        linestyle="-",
        linewidth=1.5,
    )
    train_axis.set_title(
        f"(b) Exploration-on training (trailing-{summary['rolling_window']})"
    )
    train_axis.set_ylabel("Training-rollout mean reward ↑")
    train_axis.set_xlabel("Training iteration")

    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.margins(x=0)
    fig.legend(
        [down_line, up_line, zero_line, lqr_line],
        ["Downstairs", "Upstairs", "Zero action", "Discounted LQR"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=2.0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Define paths, ensemble size, and the within-seed smoothing window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="mlp_independent_learning_20seeds")
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=25)
    return parser


def main() -> None:
    """Validate, aggregate, and export CSV, JSON, PNG, and PDF artifacts."""
    args = build_parser().parse_args()
    seed_dirs = sorted(path for path in args.input_root.glob("seed_*") if path.is_dir())
    runs = [(path, *read_run(path)) for path in seed_dirs]
    iterations = validate_runs(runs, args.expected_seeds)
    aggregate, summary = make_aggregate(runs, iterations, args.rolling_window)
    summary["input_root"] = str(args.input_root)
    summary["seed_directories"] = [str(path) for path in seed_dirs]
    first_config = runs[0][2]["config"]
    summary["common_config"] = {
        key: value
        for key, value in first_config.items()
        if key not in {"init_seed", "noise_seed", "output_dir"}
    }
    summary["seed_parameters"] = [
        {
            "directory": str(path),
            "init_seed": run_summary["config"]["init_seed"],
            "noise_seed": run_summary["config"]["noise_seed"],
        }
        for path, _, run_summary in runs
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.basename}.csv"
    json_path = args.output_dir / f"{args.basename}.json"
    png_path = args.output_dir / f"{args.basename}.png"
    pdf_path = args.output_dir / f"{args.basename}.pdf"
    write_aggregate_csv(aggregate, csv_path)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    plot_learning(aggregate, summary, png_path, pdf_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
