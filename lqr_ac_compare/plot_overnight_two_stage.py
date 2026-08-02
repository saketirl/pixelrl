"""Aggregate and plot the independent-trajectory two-stage MLP experiment.

Stage 1 must use the curvature-assisted advantage rate

    Psi_theta(z, a) = MLP_theta([z, c_a a]) - a^T R a,

whereas Stage 2 must use the pure MLP rate

    Psi_theta(z, a) = MLP_theta([z, c_a a]).

The action-input conditioning scale c_a is read and disclosed for each stage.

Both stages are required to contain exactly ``--expected-seeds`` complete
epsilon-zero, independent-actions runs. Standard errors in every export are
computed across runs, never from the within-run evaluation-episode SE columns.
The script does not drop failed, incomplete, or non-finite seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DOWN_COLOR = "#0072B2"
UP_COLOR = "#D55E00"
LQR_COLOR = "#009E73"
ZERO_COLOR = "0.45"

SEED_CONFIG_KEYS = {"init_seed", "noise_seed", "output_dir"}
FULL_FIELDS = (
    "train_mean_reward_down",
    "train_mean_reward_up",
    "down_policy_loss",
    "up_policy_loss",
    "down_value_loss",
    "up_value_loss",
    "state_gap_rms",
    "action_gap_rms",
    "policy_action_gap_rms",
    "rep_full",
    "rep_projected",
    "head_discrepancy",
    "orthogonality",
    "actual_update_full",
    "probe_update_full",
)
SMOOTH_FIELDS = (
    "train_mean_reward_down",
    "train_mean_reward_up",
    "down_policy_loss",
    "up_policy_loss",
    "down_value_loss",
    "up_value_loss",
)


@dataclass(frozen=True)
class StageSpec:
    """Identify one theoretical stage and its required curvature coefficient.

    Math:
        Psi(z,a) = MLP_theta([z,c_a a]) - c_R a^T R a,
        with c_R in {1,0}.
    """

    key: str
    label: str
    curvature_scale: float
    root: Path
    epsilon: float = 0.0


@dataclass(frozen=True)
class Run:
    """Store one seed trajectory x_s(t) and its immutable configuration."""

    path: Path
    rows: List[Dict[str, str]]
    summary: Dict[str, Any]

    @property
    def config(self) -> Mapping[str, Any]:
        """Return the trainer configuration saved with this seed."""
        return self.summary["config"]

    @property
    def seed_key(self) -> Tuple[int, int]:
        """Return (initialization seed, trajectory-noise seed)."""
        return int(self.config["init_seed"]), int(self.config["noise_seed"])


def mean_and_se(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean_s x_s and SE_s[x_s] along the seed axis.

    Math:
        mu(t) = S^-1 sum_s x_s(t),
        SE(t) = sqrt(sum_s (x_s(t)-mu(t))^2 / (S-1)) / sqrt(S).
    """
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("At least two seed values are required for an SE")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def scalar_mean_se(values: np.ndarray) -> Dict[str, float]:
    """Return a JSON-ready mean and across-seed standard error."""
    mean, se = mean_and_se(np.asarray(values, dtype=float))
    return {"mean": float(mean), "se": float(se)}


def read_run(seed_dir: Path) -> Run:
    """Read one seed's complete metrics.csv and summary.json pair."""
    metrics_path = seed_dir / "metrics.csv"
    summary_path = seed_dir / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete seed directory: {seed_dir}")
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with summary_path.open() as handle:
        summary = json.load(handle)
    if "config" not in summary:
        raise ValueError(f"{summary_path}: missing config")
    return Run(seed_dir, rows, summary)


def discover_runs(spec: StageSpec, expected_seeds: int) -> List[Run]:
    """Load exactly S seed directories; no run may be silently omitted."""
    seed_dirs = sorted(path for path in spec.root.glob("seed_*") if path.is_dir())
    if len(seed_dirs) != expected_seeds:
        raise ValueError(
            f"{spec.label}: expected {expected_seeds} seed_* directories, "
            f"found {len(seed_dirs)} under {spec.root}"
        )
    return [read_run(path) for path in seed_dirs]


def validate_stage(spec: StageSpec, runs: Sequence[Run]) -> np.ndarray:
    """Validate condition, c_R, configuration, grid, and finite observations.

    Math:
        All x_s(t) must live on one common grid and use the requested epsilon with
        c_R=1 (Stage 1) or c_R=0 (Stage 2). There is no interpolation and no
        finite-value filtering, so every requested seed contributes.
    """
    reference_iterations: Optional[np.ndarray] = None
    reference_eval_mask: Optional[np.ndarray] = None
    reference_config: Optional[Dict[str, Any]] = None
    seen_seed_keys = set()

    for run in runs:
        config = run.config
        if config.get("head_types") != ["mlp"]:
            raise ValueError(f"{run.path}: expected head_types=['mlp']")
        if config.get("training_coupling") != "independent_actions":
            raise ValueError(f"{run.path}: expected independent_actions")
        if [float(x) for x in config.get("observation_noises", [])] != [spec.epsilon]:
            raise ValueError(
                f"{run.path}: expected observation epsilon={spec.epsilon:g} only"
            )
        if "action_curvature_scale" not in config:
            raise ValueError(f"{run.path}: config does not disclose curvature scale")
        if not math.isclose(
            float(config["action_curvature_scale"]),
            spec.curvature_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{run.path}: curvature={config['action_curvature_scale']}, "
                f"expected {spec.curvature_scale}"
            )

        shared_config = {
            key: value for key, value in config.items() if key not in SEED_CONFIG_KEYS
        }
        if reference_config is None:
            reference_config = shared_config
        elif shared_config != reference_config:
            raise ValueError(f"{run.path}: non-seed configuration differs within stage")

        if run.seed_key in seen_seed_keys:
            raise ValueError(f"{run.path}: duplicate seed pair {run.seed_key}")
        seen_seed_keys.add(run.seed_key)

        expected_iterations = int(config["iters"])
        if len(run.rows) != expected_iterations:
            raise ValueError(
                f"{run.path}: expected {expected_iterations} rows, "
                f"found {len(run.rows)}"
            )
        required_fields = set(FULL_FIELDS) | {
            "iteration",
            "head_type",
            "training_coupling",
            "epsilon",
            "action_curvature_scale",
            "eval_return_down",
            "eval_return_up",
            "eval_return_zero",
            "eval_return_lqr",
        }
        missing = required_fields - set(run.rows[0]) if run.rows else required_fields
        if missing:
            raise ValueError(f"{run.path}: missing fields {sorted(missing)}")

        for row in run.rows:
            if (
                row["head_type"] != "mlp"
                or row["training_coupling"] != "independent_actions"
                or not math.isclose(
                    float(row["epsilon"]),
                    spec.epsilon,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["action_curvature_scale"]),
                    spec.curvature_scale,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"{run.path}: metrics contain another condition")

        iterations = np.asarray([int(row["iteration"]) for row in run.rows])
        if not np.array_equal(iterations, np.arange(expected_iterations)):
            raise ValueError(f"{run.path}: incomplete or unordered iteration grid")
        for field in FULL_FIELDS:
            values = np.asarray([float(row[field]) for row in run.rows])
            if not np.isfinite(values).all():
                raise ValueError(f"{run.path}: non-finite {field}; seed not omitted")

        down_eval = np.asarray([float(row["eval_return_down"]) for row in run.rows])
        up_eval = np.asarray([float(row["eval_return_up"]) for row in run.rows])
        eval_mask = np.isfinite(down_eval)
        if not np.array_equal(eval_mask, np.isfinite(up_eval)) or not eval_mask.any():
            raise ValueError(f"{run.path}: inconsistent or absent eval checkpoints")
        for field in ("eval_return_zero", "eval_return_lqr"):
            values = np.asarray([float(row[field]) for row in run.rows])[eval_mask]
            if not np.isfinite(values).all():
                raise ValueError(f"{run.path}: non-finite {field} at checkpoint")

        if reference_iterations is None:
            reference_iterations = iterations
            reference_eval_mask = eval_mask
        elif not np.array_equal(reference_iterations, iterations):
            raise ValueError(f"{run.path}: iteration grid differs across seeds")
        elif not np.array_equal(reference_eval_mask, eval_mask):
            raise ValueError(f"{run.path}: evaluation checkpoints differ across seeds")

    assert reference_iterations is not None
    return reference_iterations


def stack_field(runs: Sequence[Run], field: str) -> np.ndarray:
    """Stack one metric into X[s,t] without dropping or imputing values."""
    return np.asarray(
        [[float(row[field]) for row in run.rows] for run in runs], dtype=float
    )


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute xbar_s(t)=W^-1 sum_{j=0}^{W-1} x_s(t-j) per seed."""
    if not 1 <= window <= values.shape[1]:
        raise ValueError(f"rolling window must be in [1,{values.shape[1]}]")
    kernel = np.ones(window, dtype=float) / window
    return np.stack([np.convolve(trace, kernel, mode="valid") for trace in values])


def summarize_stage(
    spec: StageSpec,
    runs: Sequence[Run],
    iterations: np.ndarray,
    rolling_window: int,
    tail_eval_checkpoints: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Aggregate a stage and compute seed-level learning diagnostics.

    Math:
        Branch curves are mu_down/up(t) +/- SE_down/up(t). Performance
        summaries use J_pair,s=(J_down,s+J_up,s)/2, while correspondence is
        reported separately through state, action, and representation gaps.
    """
    arrays = {field: stack_field(runs, field) for field in FULL_FIELDS}
    aggregate: Dict[str, np.ndarray] = {"iterations": iterations}
    for field, values in arrays.items():
        aggregate[f"{field}_mean"], aggregate[f"{field}_se"] = mean_and_se(values)

    aggregate["smooth_iterations"] = iterations[rolling_window - 1 :]
    for field in SMOOTH_FIELDS:
        smoothed = trailing_mean(arrays[field], rolling_window)
        aggregate[f"{field}_smooth_mean"], aggregate[f"{field}_smooth_se"] = (
            mean_and_se(smoothed)
        )

    eval_down_full = stack_field(runs, "eval_return_down")
    eval_up_full = stack_field(runs, "eval_return_up")
    eval_mask = np.isfinite(eval_down_full).all(axis=0)
    if not np.array_equal(eval_mask, np.isfinite(eval_up_full).all(axis=0)):
        raise ValueError(f"{spec.label}: inconsistent aggregate eval mask")
    eval_down = eval_down_full[:, eval_mask]
    eval_up = eval_up_full[:, eval_mask]
    aggregate["eval_iterations"] = iterations[eval_mask]
    aggregate["eval_return_down_mean"], aggregate["eval_return_down_se"] = mean_and_se(
        eval_down
    )
    aggregate["eval_return_up_mean"], aggregate["eval_return_up_se"] = mean_and_se(
        eval_up
    )

    zero = np.asarray(
        [
            next(
                float(row["eval_return_zero"])
                for row in run.rows
                if math.isfinite(float(row["eval_return_zero"]))
            )
            for run in runs
        ]
    )
    lqr = np.asarray(
        [
            next(
                float(row["eval_return_lqr"])
                for row in run.rows
                if math.isfinite(float(row["eval_return_lqr"]))
            )
            for run in runs
        ]
    )
    paired_eval = 0.5 * (eval_down + eval_up)
    tail_count = min(tail_eval_checkpoints, paired_eval.shape[1])
    initial_paired = paired_eval[:, 0]
    final_paired = paired_eval[:, -1]
    tail_paired = paired_eval[:, -tail_count:].mean(axis=1)
    normalized_final = (final_paired - zero) / (lqr - zero)

    def tail_stats(field: str) -> Dict[str, float]:
        return scalar_mean_se(arrays[field][:, -rolling_window:].mean(axis=1))

    def final_stats(field: str) -> Dict[str, float]:
        return scalar_mean_se(arrays[field][:, -1])

    def max_stats(field: str) -> Dict[str, float]:
        return scalar_mean_se(arrays[field].max(axis=1))

    action_input_scale = float(
        runs[0].config.get("mlp_advantage_action_input_scale", 1.0)
    )
    if spec.curvature_scale == 1.0:
        curvature_disclosure = (
            "Psi_theta(z,a) = MLP_theta([z,c_a a]) - c_R a^T R a "
            f"(c_a={action_input_scale:g}, c_R={spec.curvature_scale:g})"
        )
    else:
        curvature_disclosure = (
            "Psi_theta(z,a) = MLP_theta([z,c_a a]) - c_R a^T R a "
            f"(c_a={action_input_scale:g}, c_R={spec.curvature_scale:g}; "
            "no explicit -a^T R a)"
        )

    summary: Dict[str, Any] = {
        "stage": spec.key,
        "label": spec.label,
        "curvature_scale": spec.curvature_scale,
        "epsilon": spec.epsilon,
        "advantage_action_input_scale": action_input_scale,
        "curvature_disclosure": curvature_disclosure,
        "n_seeds": len(runs),
        "se_definition": "sample standard deviation across seeds divided by sqrt(S)",
        "rolling_window": rolling_window,
        "tail_eval_checkpoints": tail_count,
        "zero_action_reference": scalar_mean_se(zero),
        "discounted_lqr_reference": scalar_mean_se(lqr),
        "initial_paired_eval_return": scalar_mean_se(initial_paired),
        "final_paired_eval_return": scalar_mean_se(final_paired),
        "tail_paired_eval_return": scalar_mean_se(tail_paired),
        "final_downstairs_eval_return": scalar_mean_se(eval_down[:, -1]),
        "final_upstairs_eval_return": scalar_mean_se(eval_up[:, -1]),
        "eval_return_improvement": scalar_mean_se(final_paired - initial_paired),
        "normalized_zero_to_lqr_fraction_final": scalar_mean_se(normalized_final),
        "seeds_improved": int(np.sum(final_paired > initial_paired)),
        "seeds_above_zero_action": int(np.sum(final_paired > zero)),
        "seeds_at_or_above_lqr": int(np.sum(final_paired >= lqr)),
        "last_window_training_mean_reward": {
            "downstairs": tail_stats("train_mean_reward_down"),
            "upstairs": tail_stats("train_mean_reward_up"),
        },
        "last_window_policy_loss": {
            "downstairs": tail_stats("down_policy_loss"),
            "upstairs": tail_stats("up_policy_loss"),
            "absolute_branch_gap": scalar_mean_se(
                np.abs(
                    arrays["up_policy_loss"][:, -rolling_window:].mean(axis=1)
                    - arrays["down_policy_loss"][:, -rolling_window:].mean(axis=1)
                )
            ),
        },
        "last_window_value_loss": {
            "downstairs": tail_stats("down_value_loss"),
            "upstairs": tail_stats("up_value_loss"),
            "absolute_branch_gap": scalar_mean_se(
                np.abs(
                    arrays["up_value_loss"][:, -rolling_window:].mean(axis=1)
                    - arrays["down_value_loss"][:, -rolling_window:].mean(axis=1)
                )
            ),
        },
        "final_path_and_representation_gaps": {
            "state_gap_rms": final_stats("state_gap_rms"),
            "executed_action_gap_rms": final_stats("action_gap_rms"),
            "policy_action_gap_rms": final_stats("policy_action_gap_rms"),
            "representation_gap_full": final_stats("rep_full"),
            "representation_gap_projected": final_stats("rep_projected"),
        },
        "per_seed_max_path_and_representation_gaps": {
            "state_gap_rms": max_stats("state_gap_rms"),
            "executed_action_gap_rms": max_stats("action_gap_rms"),
            "policy_action_gap_rms": max_stats("policy_action_gap_rms"),
            "representation_gap_full": max_stats("rep_full"),
            "representation_gap_projected": max_stats("rep_projected"),
        },
        "theory_diagnostics": {
            "final": {
                "head_discrepancy": final_stats("head_discrepancy"),
                "orthogonality": final_stats("orthogonality"),
                "actual_update_full": final_stats("actual_update_full"),
                "probe_update_full": final_stats("probe_update_full"),
            },
            "per_seed_max": {
                "head_discrepancy": max_stats("head_discrepancy"),
                "orthogonality": max_stats("orthogonality"),
                "actual_update_full": max_stats("actual_update_full"),
                "probe_update_full": max_stats("probe_update_full"),
            },
        },
        "final_paired_up_minus_down_return": scalar_mean_se(
            eval_up[:, -1] - eval_down[:, -1]
        ),
        "input_root": str(spec.root),
        "seed_directories": [str(run.path) for run in runs],
        "seed_parameters": [
            {
                "directory": str(run.path),
                "init_seed": run.seed_key[0],
                "noise_seed": run.seed_key[1],
            }
            for run in runs
        ],
        "common_config": {
            key: value
            for key, value in runs[0].config.items()
            if key not in SEED_CONFIG_KEYS
        },
    }
    return aggregate, summary


def write_aggregate_csv(
    stages: Sequence[Tuple[StageSpec, Mapping[str, np.ndarray]]], path: Path
) -> None:
    """Write both stage trajectories, with every mean paired with its SE."""
    statistic_names = sorted(
        {
            key[: -len("_mean")]
            for _, aggregate in stages
            for key in aggregate
            if key.endswith("_mean")
        }
    )
    fields = ["stage", "curvature_scale", "iteration"]
    for name in statistic_names:
        fields.extend((f"{name}_mean", f"{name}_se"))

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec, aggregate in stages:
            iterations = aggregate["iterations"].astype(int)
            eval_lookup = {
                int(value): index
                for index, value in enumerate(aggregate["eval_iterations"])
            }
            smooth_lookup = {
                int(value): index
                for index, value in enumerate(aggregate["smooth_iterations"])
            }
            for full_index, iteration in enumerate(iterations):
                row: Dict[str, Any] = {
                    "stage": spec.key,
                    "curvature_scale": spec.curvature_scale,
                    "iteration": int(iteration),
                }
                for name in statistic_names:
                    mean_key, se_key = f"{name}_mean", f"{name}_se"
                    if mean_key not in aggregate:
                        continue
                    if name.startswith("eval_return_"):
                        index = eval_lookup.get(int(iteration))
                    elif name.endswith("_smooth"):
                        index = smooth_lookup.get(int(iteration))
                    else:
                        index = full_index
                    if index is not None:
                        row[mean_key] = aggregate[mean_key][index]
                        row[se_key] = aggregate[se_key][index]
                writer.writerow(row)


def draw_branch(
    axis: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    se: np.ndarray,
    branch: str,
) -> None:
    """Draw one branch with an unambiguous color, line style, and SE ribbon."""
    is_up = branch == "upstairs"
    color = UP_COLOR if is_up else DOWN_COLOR
    axis.fill_between(
        x,
        mean - se,
        mean + se,
        facecolor=color,
        edgecolor=color if is_up else "none",
        alpha=0.10 if is_up else 0.15,
        hatch="////" if is_up else None,
        linewidth=0.3,
    )
    axis.plot(
        x,
        mean,
        color=color,
        linestyle="--" if is_up else "-",
        linewidth=1.8 if is_up else 1.5,
        label="Upstairs" if is_up else "Downstairs",
    )


def plot_stage_learning(
    spec: StageSpec,
    aggregate: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot return, exploration-on reward, policy loss, and value loss."""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2))
    return_axis, reward_axis, policy_axis, value_axis = axes.ravel()

    eval_x = aggregate["eval_iterations"]
    draw_branch(
        return_axis,
        eval_x,
        aggregate["eval_return_down_mean"],
        aggregate["eval_return_down_se"],
        "downstairs",
    )
    draw_branch(
        return_axis,
        eval_x,
        aggregate["eval_return_up_mean"],
        aggregate["eval_return_up_se"],
        "upstairs",
    )
    return_axis.axhline(
        summary["zero_action_reference"]["mean"],
        color=ZERO_COLOR,
        linestyle=":",
        linewidth=1.2,
        label="Zero action",
    )
    return_axis.axhline(
        summary["discounted_lqr_reference"]["mean"],
        color=LQR_COLOR,
        linestyle="-.",
        linewidth=1.2,
        label="Discounted LQR",
    )
    return_axis.set_title("Fixed-bank discounted return")
    return_axis.set_ylabel("Return ↑")

    smooth_x = aggregate["smooth_iterations"]
    for axis, down_field, up_field, title, ylabel in (
        (
            reward_axis,
            "train_mean_reward_down",
            "train_mean_reward_up",
            f"Training reward (trailing-{summary['rolling_window']})",
            "Mean reward ↑",
        ),
        (
            policy_axis,
            "down_policy_loss",
            "up_policy_loss",
            f"Policy loss (trailing-{summary['rolling_window']})",
            "Policy loss",
        ),
        (
            value_axis,
            "down_value_loss",
            "up_value_loss",
            f"Value loss (trailing-{summary['rolling_window']})",
            "Value loss",
        ),
    ):
        draw_branch(
            axis,
            smooth_x,
            aggregate[f"{down_field}_smooth_mean"],
            aggregate[f"{down_field}_smooth_se"],
            "downstairs",
        )
        draw_branch(
            axis,
            smooth_x,
            aggregate[f"{up_field}_smooth_mean"],
            aggregate[f"{up_field}_smooth_se"],
            "upstairs",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)

    for axis in axes.ravel():
        axis.set_xlabel("Training iteration")
        axis.grid(True, alpha=0.22)
        axis.margins(x=0)
    handles, labels = return_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"{spec.label}: {summary['curvature_disclosure']}", y=1.075)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_stage_gaps(
    spec: StageSpec,
    aggregate: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot across-seed mean +/- SE path and representation discrepancies."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.65))
    x = aggregate["iterations"]
    panels = (
        (axes[0], "state_gap_rms", "State-path gap", "RMS state gap"),
        (axes[1], "action_gap_rms", "Executed-action gap", "RMS action gap"),
        (axes[2], "rep_full", "Representation discrepancy", r"$E_{\mathrm{rep}}$"),
    )
    colors = ("#CC79A7", "#009E73", "#E69F00")
    for (axis, field, title, ylabel), color in zip(panels, colors):
        mean = aggregate[f"{field}_mean"]
        se = aggregate[f"{field}_se"]
        # Exact matched initialization gives literal zeros. Clipping those
        # zeros at ``float.tiny`` makes the log axis span roughly 300 decades,
        # obscuring the numerical-scale discrepancies that matter here.
        positive = 1.0e-16
        axis.fill_between(
            x,
            np.maximum(mean - se, positive),
            np.maximum(mean + se, positive),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(x, np.maximum(mean, positive), color=color, linewidth=1.5)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Training iteration")
        axis.grid(True, alpha=0.22)
        axis.margins(x=0)
    fig.suptitle(f"{spec.label}: {summary['curvature_disclosure']}", y=1.08)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def format_stat(stat: Mapping[str, float]) -> str:
    """Format one mean +/- SE pair for the Markdown audit report."""
    return f"{stat['mean']:.6g} ± {stat['se']:.3g}"


def compute_cross_stage_retention(
    stage_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute the deterministic ratio of stage-level improvement means.

    Math:
        retention = (mean(J_pair,stage2,final) - mean(J_0,stage2))
                    / (mean(J_pair,stage1,final) - mean(J_0,stage1)).

    This is deliberately a ratio of four aggregate means. It is not a mean
    of seed-level ratios, so this helper does not invent a seed-level standard
    error for it.
    """
    by_stage = {summary["stage"]: summary for summary in stage_summaries}
    missing = {"stage1", "stage2"} - set(by_stage)
    if missing:
        raise ValueError(f"Missing summaries for cross-stage retention: {missing}")

    stage1 = by_stage["stage1"]
    stage2 = by_stage["stage2"]
    stage1_final = float(stage1["final_paired_eval_return"]["mean"])
    stage1_zero = float(stage1["zero_action_reference"]["mean"])
    stage2_final = float(stage2["final_paired_eval_return"]["mean"])
    stage2_zero = float(stage2["zero_action_reference"]["mean"])
    numerator = stage2_final - stage2_zero
    denominator = stage1_final - stage1_zero
    if not all(
        math.isfinite(value)
        for value in (stage1_final, stage1_zero, stage2_final, stage2_zero)
    ):
        raise ValueError("Cross-stage retention inputs must be finite")
    if denominator == 0.0:
        raise ValueError("Stage-1 improvement over zero is zero; retention undefined")

    return {
        "definition": (
            "retention = (Stage 2 final paired return mean - Stage 2 "
            "zero-action return mean) / (Stage 1 final paired return mean - "
            "Stage 1 zero-action return mean)"
        ),
        "formula": "(mu_J2_final - mu_J2_zero) / (mu_J1_final - mu_J1_zero)",
        "estimate_type": "deterministic ratio-of-means point estimate",
        "standard_error": None,
        "standard_error_disclosure": (
            "No seed-level standard error is defined or reported for this "
            "ratio of stage-level means."
        ),
        "numerator": {
            "definition": (
                "Stage 2 final paired return mean - Stage 2 zero-action " "return mean"
            ),
            "final_paired_return_mean": stage2_final,
            "zero_action_return_mean": stage2_zero,
            "value": numerator,
        },
        "denominator": {
            "definition": (
                "Stage 1 final paired return mean - Stage 1 zero-action " "return mean"
            ),
            "final_paired_return_mean": stage1_final,
            "zero_action_return_mean": stage1_zero,
            "value": denominator,
        },
        "point_estimate": numerator / denominator,
    }


def validate_cross_stage_config_contract(
    stage_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require the complete saved stage configs to differ only in c_R."""
    by_stage = {summary["stage"]: summary for summary in stage_summaries}
    if set(by_stage) != {"stage1", "stage2"}:
        raise ValueError("Expected exactly Stage 1 and Stage 2 summaries")
    stage1 = dict(by_stage["stage1"]["common_config"])
    stage2 = dict(by_stage["stage2"]["common_config"])
    differing = sorted(
        key for key in set(stage1) | set(stage2) if stage1.get(key) != stage2.get(key)
    )
    if differing != ["action_curvature_scale"]:
        raise ValueError(
            "Cross-stage configurations must differ only in "
            f"action_curvature_scale; found {differing}"
        )
    if not math.isclose(
        float(stage1["action_curvature_scale"]), 1.0
    ) or not math.isclose(float(stage2["action_curvature_scale"]), 0.0):
        raise ValueError("Expected Stage 1 c_R=1 and Stage 2 c_R=0")
    return {
        "verified": True,
        "differing_keys": differing,
        "stage1_action_curvature_scale": 1.0,
        "stage2_action_curvature_scale": 0.0,
    }


def compute_paired_final_stage_difference(
    runs_by_stage: Sequence[Sequence[Run]],
) -> Dict[str, Any]:
    """Compute paired final J_pair,2-J_pair,1 mean and across-seed SE."""
    if len(runs_by_stage) != 2:
        raise ValueError("Expected exactly two stages for a paired difference")

    def final_paired_by_seed(runs: Sequence[Run]) -> Dict[Tuple[int, int], float]:
        values: Dict[Tuple[int, int], float] = {}
        for run in runs:
            final_eval = next(
                row
                for row in reversed(run.rows)
                if math.isfinite(float(row["eval_return_down"]))
                and math.isfinite(float(row["eval_return_up"]))
            )
            values[run.seed_key] = 0.5 * (
                float(final_eval["eval_return_down"])
                + float(final_eval["eval_return_up"])
            )
        return values

    stage1 = final_paired_by_seed(runs_by_stage[0])
    stage2 = final_paired_by_seed(runs_by_stage[1])
    if set(stage1) != set(stage2):
        raise ValueError("Paired final difference requires identical seed pairs")
    differences = np.asarray(
        [stage2[key] - stage1[key] for key in sorted(stage1)], dtype=float
    )
    stats = scalar_mean_se(differences)
    return {
        "definition": "final paired return Stage 2 minus Stage 1 by seed pair",
        **stats,
        "n_seed_pairs": int(differences.size),
        "stage2_wins": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "stage1_wins": int(np.sum(differences < 0.0)),
        "min": float(differences.min()),
        "max": float(differences.max()),
    }


def write_report(
    stage_summaries: Sequence[Mapping[str, Any]],
    cross_stage_retention: Mapping[str, Any],
    cross_stage_config_contract: Mapping[str, Any],
    paired_stage_difference: Mapping[str, Any],
    path: Path,
) -> None:
    """Write a compact report with Stage 1 and Stage 2 disclosed separately."""
    numerator = cross_stage_retention["numerator"]
    denominator = cross_stage_retention["denominator"]
    lines = [
        "# Independent upstairs/downstairs MLP: two-stage report",
        "",
        "All intervals are mean ± standard error across the complete seed "
        "ensemble. Within-run evaluation-episode SEs are not reused.",
        "",
        "## Cross-stage retention",
        "",
        f"- Exact definition: `{cross_stage_retention['definition']}`.",
        f"- Stage 2 numerator: {numerator['value']:.6g} = "
        f"{numerator['final_paired_return_mean']:.6g} - "
        f"{numerator['zero_action_return_mean']:.6g}.",
        f"- Stage 1 denominator: {denominator['value']:.6g} = "
        f"{denominator['final_paired_return_mean']:.6g} - "
        f"{denominator['zero_action_return_mean']:.6g}.",
        f"- Retention point estimate: "
        f"{cross_stage_retention['point_estimate']:.6g}.",
        f"- Uncertainty disclosure: "
        f"{cross_stage_retention['standard_error_disclosure']}",
        "",
        "## Cross-stage contract and paired difference",
        "",
        f"- Full saved-config validation: "
        f"`verified={cross_stage_config_contract['verified']}`; the only "
        f"differing key is "
        f"`{cross_stage_config_contract['differing_keys'][0]}` "
        "(`c_R: 1 -> 0`).",
        f"- Final paired return, Stage 2 minus Stage 1: "
        f"{format_stat(paired_stage_difference)}; "
        f"wins/ties/losses="
        f"{paired_stage_difference['stage2_wins']}/"
        f"{paired_stage_difference['ties']}/"
        f"{paired_stage_difference['stage1_wins']} across "
        f"{paired_stage_difference['n_seed_pairs']} seed pairs.",
        "- Policy-loss magnitudes are not comparable across stages because "
        "`c_R` changes the actor objective; compare upstairs against "
        "downstairs within a stage.",
        "",
    ]
    for summary in stage_summaries:
        gaps = summary["final_path_and_representation_gaps"]
        last_reward = summary["last_window_training_mean_reward"]
        last_policy_loss = summary["last_window_policy_loss"]
        last_value_loss = summary["last_window_value_loss"]
        max_theory = summary["theory_diagnostics"]["per_seed_max"]
        config = summary["common_config"]
        action_input_scale = float(config["mlp_advantage_action_input_scale"])
        curvature_scale = float(config["action_curvature_scale"])
        policy_bound = float(config["mlp_policy_action_limit"])
        policy_bound_text = (
            f"B={policy_bound:g}, pi(z)=B tanh(raw/B)"
            if policy_bound > 0.0
            else "B=0 (unbounded policy head)"
        )
        effective_actor_rate = float(config["eta_actor"]) * float(
            config["mlp_actor_lr_scale"]
        )
        actor_stop_iters = config["actor_stop_iters"]
        critic_stop_recorded = "critic_stop_iters" in config
        critic_stop_iters = config.get("critic_stop_iters")
        if not critic_stop_recorded:
            critic_stop_text = "not recorded (legacy summary)"
        elif critic_stop_iters is None:
            critic_stop_text = "none (updates through the full run)"
        else:
            critic_stop_text = f"{critic_stop_iters} (exclusive)"

        if critic_stop_iters is not None and actor_stop_iters == critic_stop_iters:
            stop_semantics = (
                f"- Joint-freeze semantics: `actor_stop_iters="
                f"critic_stop_iters={critic_stop_iters}` is exclusive; all "
                "trainable parameters are frozen from that iteration onward, "
                "while rollouts and loss diagnostics continue."
            )
        else:
            stop_semantics = (
                "- Stop semantics: stops are exclusive. A joint "
                "`actor_stop_iters == critic_stop_iters` freezes all trainable "
                "parameters from that iteration onward, while rollouts and "
                "loss diagnostics continue."
            )
        lines.extend(
            [
                f"## {summary['label']}",
                "",
                f"- Curvature disclosure: `{summary['curvature_disclosure']}`.",
                f"- Advantage gradient estimator: "
                f"`advantage_gradient={config['advantage_gradient']}`.",
                f"- Advantage-head constants: `c_a={action_input_scale:g}`, "
                f"`c_R={curvature_scale:g}` in "
                "`Psi_theta(z,a)=MLP_theta([z,c_a a])-c_R a^T R a`.",
                f"- Policy bound: `{policy_bound_text}`; separate environment "
                f"`action_clip={float(config['action_clip']):g}`.",
                f"- Optimizer timing: actor "
                f"`warmup={config['actor_warmup_iters']}`, "
                f"`stop={actor_stop_iters} (exclusive)`, "
                f"`cadence={config['mlp_actor_update_every']}`; critic "
                f"`stop={critic_stop_text}`; actor "
                f"`schedule={config['mlp_actor_lr_schedule']}`.",
                stop_semantics,
                f"- Actor rate: `eta_actor={float(config['eta_actor']):g}` × "
                f"`mlp_actor_lr_scale="
                f"{float(config['mlp_actor_lr_scale']):g}` = "
                f"`{effective_actor_rate:g}` before scheduling.",
                f"- Behavior exploration: "
                f"`exploration_std={float(config['exploration_std']):g}`.",
                f"- Complete seeds: {summary['n_seeds']}.",
                f"- Final downstairs return: "
                f"{format_stat(summary['final_downstairs_eval_return'])}.",
                f"- Final upstairs return: "
                f"{format_stat(summary['final_upstairs_eval_return'])}.",
                f"- Tail paired return: "
                f"{format_stat(summary['tail_paired_eval_return'])}.",
                f"- Zero-action reference: "
                f"{format_stat(summary['zero_action_reference'])}.",
                f"- Discounted-LQR reference: "
                f"{format_stat(summary['discounted_lqr_reference'])}.",
                f"- Final zero-to-LQR fraction: "
                f"{format_stat(summary['normalized_zero_to_lqr_fraction_final'])}.",
                f"- Last-window training reward (down/up): "
                f"{format_stat(last_reward['downstairs'])} / "
                f"{format_stat(last_reward['upstairs'])}.",
                f"- Last-window policy loss (down/up): "
                f"{format_stat(last_policy_loss['downstairs'])} / "
                f"{format_stat(last_policy_loss['upstairs'])}.",
                f"- Last-window value loss (down/up): "
                f"{format_stat(last_value_loss['downstairs'])} / "
                f"{format_stat(last_value_loss['upstairs'])}.",
                f"- Final state/action/representation gaps: "
                f"{format_stat(gaps['state_gap_rms'])} / "
                f"{format_stat(gaps['executed_action_gap_rms'])} / "
                f"{format_stat(gaps['representation_gap_full'])}.",
                f"- Mean across seeds of each seed's all-iteration maximum "
                f"(head/orthogonality/actual update/probe update): "
                f"{format_stat(max_theory['head_discrepancy'])} / "
                f"{format_stat(max_theory['orthogonality'])} / "
                f"{format_stat(max_theory['actual_update_full'])} / "
                f"{format_stat(max_theory['probe_update_full'])}.",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def write_seed_manifest(
    stages: Sequence[Tuple[StageSpec, Sequence[Run]]], path: Path
) -> None:
    """Export every included seed directory so omissions are auditable."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "curvature_scale",
                "seed_directory",
                "init_seed",
                "noise_seed",
            ),
        )
        writer.writeheader()
        for spec, runs in stages:
            for run in runs:
                writer.writerow(
                    {
                        "stage": spec.key,
                        "curvature_scale": spec.curvature_scale,
                        "seed_directory": run.path,
                        "init_seed": run.seed_key[0],
                        "noise_seed": run.seed_key[1],
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    """Define the two artifact roots and strict ensemble-size contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="overnight_two_stage")
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--tail-eval-checkpoints", type=int, default=10)
    parser.add_argument(
        "--allow-unpaired-seeds",
        action="store_true",
        help="Permit Stage 1 and Stage 2 to use different seed pairs.",
    )
    return parser


def main() -> None:
    """Validate all 40 runs, aggregate them, and export plots/tables/report."""
    args = build_parser().parse_args()
    if args.expected_seeds < 2:
        raise ValueError("expected-seeds must be at least 2")
    if args.rolling_window < 1 or args.tail_eval_checkpoints < 1:
        raise ValueError("window sizes must be positive")

    specs = (
        StageSpec("stage1", "Stage 1 (curvature-assisted)", 1.0, args.stage1_root),
        StageSpec("stage2", "Stage 2 (pure MLP)", 0.0, args.stage2_root),
    )
    runs_by_stage = [discover_runs(spec, args.expected_seeds) for spec in specs]
    grids = [validate_stage(spec, runs) for spec, runs in zip(specs, runs_by_stage)]
    if not args.allow_unpaired_seeds:
        first = {run.seed_key for run in runs_by_stage[0]}
        second = {run.seed_key for run in runs_by_stage[1]}
        if first != second:
            raise ValueError(
                "Stage seed pairs differ; pass --allow-unpaired-seeds only if "
                "this was intentional"
            )

    aggregates: List[Dict[str, np.ndarray]] = []
    summaries: List[Dict[str, Any]] = []
    for spec, runs, grid in zip(specs, runs_by_stage, grids):
        aggregate, summary = summarize_stage(
            spec,
            runs,
            grid,
            args.rolling_window,
            args.tail_eval_checkpoints,
        )
        aggregates.append(aggregate)
        summaries.append(summary)

    cross_stage_config_contract = validate_cross_stage_config_contract(summaries)
    if args.allow_unpaired_seeds:
        raise ValueError(
            "The definitive two-stage report requires paired seeds; "
            "--allow-unpaired-seeds is incompatible with paired statistics"
        )
    paired_stage_difference = compute_paired_final_stage_difference(runs_by_stage)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.basename
    write_aggregate_csv(list(zip(specs, aggregates)), prefix.with_suffix(".csv"))
    write_seed_manifest(
        list(zip(specs, runs_by_stage)),
        args.output_dir / f"{args.basename}_seed_manifest.csv",
    )
    combined_summary = {
        "experiment": "independent upstairs/downstairs MLP, epsilon=0",
        "expected_seeds_per_stage": args.expected_seeds,
        "paired_seed_ensembles": True,
        "cross_stage_config_contract": cross_stage_config_contract,
        "paired_final_stage2_minus_stage1_return": paired_stage_difference,
        "stages": {summary["stage"]: summary for summary in summaries},
    }
    cross_stage_retention = compute_cross_stage_retention(summaries)
    combined_summary["cross_stage_retention"] = cross_stage_retention
    with prefix.with_suffix(".json").open("w") as handle:
        json.dump(combined_summary, handle, indent=2)
    write_report(
        summaries,
        cross_stage_retention,
        cross_stage_config_contract,
        paired_stage_difference,
        args.output_dir / f"{args.basename}_report.md",
    )

    for spec, aggregate, summary in zip(specs, aggregates, summaries):
        plot_stage_learning(
            spec,
            aggregate,
            summary,
            args.output_dir / f"{args.basename}_{spec.key}_learning.png",
            args.output_dir / f"{args.basename}_{spec.key}_learning.pdf",
        )
        plot_stage_gaps(
            spec,
            aggregate,
            summary,
            args.output_dir / f"{args.basename}_{spec.key}_gaps.png",
            args.output_dir / f"{args.basename}_{spec.key}_gaps.pdf",
        )

    print(json.dumps(combined_summary, indent=2))
    print(f"Saved two-stage artifacts under {args.output_dir}")


if __name__ == "__main__":
    main()
