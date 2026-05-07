#!/usr/bin/env python3
"""Plot CNN baselines vs staging CRATE runs with tueplots NeurIPS styling."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles

try:
    import wandb
except ImportError:
    wandb = None


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

ENV_TITLES = {
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

MUON_ENV_FILES = {
    "halfcheetah": "half_cheetah_muon.csv",
    "walker2d": "walker2d_muon.csv",
    "ant": "ant_muon.csv",
    "humanoid": "humanoid_muon.csv",
    "reacher": "reacher_muon.csv",
    "swimmer": "swimmer_muon.csv",
    "pusher": "pusher_muon.csv",
    "hopper": "hopper_muon.csv",
    "inverted_pendulum": "pendulum_muon.csv",
}

ENV_MAX_STEPS = {
    "walker2d": 2_000_000,
    "hopper": 2_000_000,
    "swimmer": 2_000_000,
    "reacher": 2_000_000,
    "pusher": 2_000_000,
}

CONDITION_COLORS = {
    "cnn_adam": "#4C72B0",
    "cnn_muon": "#DD8452",
    "crate_adam": "#17BECF",
    "crate_stiefel": "#2F2F2F",
    "cnn_heads_adam": "#4C72B0",
    "cnn_heads_stiefel": "#DD8452",
    "cratecnn_crateheads_adam": "#17BECF",
    "cratecnn_crateheads_stiefel": "#2F2F2F",
    "muon": "#B79F00",
}

CURVES = [
    {
        "source": "cnn",
        "key": "cnn_adam",
        "label": "CNN Adam",
    },
    {
        "source": "cnn",
        "key": "cnn_muon",
        "label": "CNN manifold Muon",
    },
    {
        "source": "staging",
        "key": "crate_adam",
        "label": "ISTA Adam",
    },
    {
        "source": "staging",
        "key": "crate_stiefel",
        "label": "ISTA manifold Muon",
    },
]

GOAL_CONDITION_ORDER = [
    "cnn_heads_adam",
    "cnn_heads_stiefel",
    "cratecnn_crateheads_adam",
    "cratecnn_crateheads_stiefel",
]

GOAL_CONDITION_LABELS = {
    "cnn_heads_adam": "CNN Adam",
    "cnn_heads_stiefel": "CNN manifold Muon",
    "cratecnn_crateheads_adam": "ISTA Adam",
    "cratecnn_crateheads_stiefel": "ISTA manifold Muon",
}

ANT_GOAL_ALGORITHM_TO_CONDITION = {
    "Adam + CNN": "cnn_heads_adam",
    "Stiefel + CNN": "cnn_heads_stiefel",
    "Adam + CRATE": "cratecnn_crateheads_adam",
    "Stiefel + CRATE": "cratecnn_crateheads_stiefel",
}

ANT_GOAL_RUNS = {
    "cratecnn_crateheads_stiefel": [
        "nu3ozqzt",
        "m97k6s8s",
        "ieofyw8b",
        "5xtn3755",
        "ghczrm1f",
        "ou5ejwef",
    ],
    "cratecnn_crateheads_adam": [
        "th9g2hta",
        "lat9xwua",
        "0cvwelmj",
        "laxgsjc7",
        "1br0zuy0",
        "m56webl2",
    ],
    "cnn_heads_stiefel": [
        "sck8hbms",
        "u1o2tpu3",
        "kaffkok1",
        "c8142l7w",
        "qlpjht2q",
        "sjvab7y5",
    ],
    "cnn_heads_adam": [
        "7s62cwp2",
        "o7k2hprj",
        "66kyabiw",
        "g7art2wl",
        "pyt5j5e5",
        "ndav8rcj",
    ],
}

ANT_GOAL_METRIC_KEY = "charts/avg_episodic_return"
ANT_GOAL_Y_MIN = -175

LQR_CURVES = [
    {
        "key": "up_reward",
        "label": "Upstairs",
        "color": "#2F2F2F",
        "linestyle": "-",
    },
    {
        "key": "down_reward",
        "label": "Downstairs",
        "color": "#56B4E9",
        "linestyle": "-",
    },
]

LLM_TASKS = {
    "cot_math": "CoT Math",
    "spider": "Spider",
    "tooluse": "Tool Use",
}

LLM_ALGOS = [
    {
        "key": "adam",
        "label": "Adam",
        "color": CONDITION_COLORS["crate_adam"],
    },
    {
        "key": "stiefel",
        "label": "manifold Muon",
        "color": CONDITION_COLORS["crate_stiefel"],
    },
]
LLM_STEP_DIVISOR = 260

PROBE_CONDITIONS = [
    {
        "key": "adam",
        "label": "CNN Adam",
        "color": CONDITION_COLORS["cnn_adam"],
    },
    {
        "key": "stiefel",
        "label": "CNN manifold Muon",
        "color": CONDITION_COLORS["cnn_muon"],
    },
]

PROBE_METRICS = {
    "returns": {
        "base": "ip_returns",
        "ylabel": "Average episodic return",
        "filename": "probe_ip_returns.png",
    },
    "r2_at_state_dim": {
        "base": "ip_r2_at_state_dim",
        "ylabel": r"$R^2$ with $d_s$ PCs",
        "filename": "probe_ip_r2_at_state_dim.png",
    },
}

HUMANOID_GOAL_METRICS = {
    "return": {
        "key": "charts/avg_episodic_return",
        "ylabel": "Average episodic return",
        "filename": "humanoid_goal_return.png",
    },
    "success": {
        "key": "env/success_mean",
        "ylabel": "Success rate",
        "filename": "humanoid_goal_success.png",
    },
    "success_easy": {
        "key": "env/success_easy_mean",
        "ylabel": "Easy success rate",
        "filename": "humanoid_goal_success_easy.png",
    },
    "episode_length": {
        "key": "charts/avg_episodic_length_agent",
        "ylabel": "Average episode length",
        "filename": "humanoid_goal_episode_length.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tueplots NeurIPS plots from cached curve summaries."
    )
    parser.add_argument("--cnn-data-dir", type=Path, default=Path("plots/cnn_data"))
    parser.add_argument(
        "--staging-data-dir", type=Path, default=Path("plots/staging_data")
    )
    parser.add_argument("--muon-data-dir", type=Path, default=Path("plots/muon_data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots/neurips"))
    parser.add_argument(
        "--ant-goal-csv",
        type=Path,
        default=Path("plots/ant_goal/avg_episodic_returns.csv"),
    )
    parser.add_argument(
        "--ant-goal-data-dir",
        type=Path,
        default=Path("plots/ant_goal"),
    )
    parser.add_argument(
        "--ant-goal-prefix",
        default="saketirl/benchmark",
        help="W&B project path for refreshing ant-goal data.",
    )
    parser.add_argument(
        "--refresh-ant-goal",
        action="store_true",
        help="Fetch ant-goal histories from W&B and rewrite the local CSV cache.",
    )
    parser.add_argument(
        "--ant-goal-samples",
        type=int,
        default=2000,
        help="History samples per W&B run when refreshing ant-goal data.",
    )
    parser.add_argument(
        "--ant-goal-num-points",
        type=int,
        default=500,
        help="Interpolation points for ant-goal curve aggregation.",
    )
    parser.add_argument(
        "--ant-goal-max-step",
        type=float,
        default=10_000_000,
        help="Maximum x-axis step shown in the Ant goal plot.",
    )
    parser.add_argument(
        "--lqr-data-dir",
        type=Path,
        default=Path("plots/lqr"),
        help="Directory containing LQR results CSV files.",
    )
    parser.add_argument(
        "--lqr-glob",
        default="results_nsteps30_noise*.csv",
        help="Glob pattern for LQR CSV files relative to --lqr-data-dir.",
    )
    parser.add_argument(
        "--llm-data-dir",
        type=Path,
        default=Path("plots/llm"),
        help="Directory containing LLM history CSV files.",
    )
    parser.add_argument(
        "--probe-data-dir",
        type=Path,
        default=Path("probe_results"),
        help="Directory containing inverted-pendulum probe CSV exports.",
    )
    parser.add_argument(
        "--draft-plots-dir",
        type=Path,
        default=Path("plots/neurips/draft"),
        help="Directory for draft/alternative plot variants.",
    )
    parser.add_argument(
        "--make-draft-llm-plots",
        action="store_true",
        help="Generate cropped and inset LLM loss draft variants.",
    )
    parser.add_argument(
        "--llm-elbow-min-step",
        type=float,
        default=0,
        help="Minimum x-axis step for LLM draft elbow zooms.",
    )
    parser.add_argument(
        "--llm-elbow-max-step",
        type=float,
        default=80,
        help="Maximum x-axis step for LLM draft elbow zooms.",
    )
    parser.add_argument(
        "--llm-smoothing-window",
        type=int,
        default=10,
        help="Centered rolling window, in logged training steps, for LLM loss plots.",
    )
    parser.add_argument(
        "--llm-smoothing",
        choices=("rolling", "ema", "none"),
        default="rolling",
        help="Smoothing method for LLM loss plots.",
    )
    parser.add_argument(
        "--llm-ema-alpha",
        type=float,
        default=0.2,
        help="EMA alpha for LLM loss plots when --llm-smoothing=ema.",
    )
    parser.add_argument(
        "--humanoid-goal-data-dir",
        type=Path,
        default=Path("plots/humanoid_goal"),
    )
    parser.add_argument(
        "--humanoid-goal-prefix",
        default="rl-power/pixel-goal",
        help="W&B project path for refreshing humanoid-goal data.",
    )
    parser.add_argument(
        "--humanoid-goal-run-regex",
        default=r"humanoid_goal__ppo_humanoid_goal_final_exp_s125_t0p35.*10m",
        help="Regex matched against W&B display names when refreshing humanoid-goal data.",
    )
    parser.add_argument(
        "--refresh-humanoid-goal",
        action="store_true",
        help="Fetch humanoid-goal histories from W&B and rewrite the local CSV cache.",
    )
    parser.add_argument(
        "--humanoid-goal-samples",
        type=int,
        default=2000,
        help="History samples per W&B run when refreshing humanoid-goal data.",
    )
    parser.add_argument(
        "--humanoid-goal-num-points",
        type=int,
        default=500,
        help="Interpolation points for humanoid-goal curve aggregation.",
    )
    parser.add_argument(
        "--humanoid-goal-metrics",
        nargs="+",
        default=["return"],
        choices=sorted(HUMANOID_GOAL_METRICS),
        help="Humanoid-goal metrics to plot.",
    )
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=120,
        help="W&B API timeout in seconds.",
    )
    parser.add_argument(
        "--envs",
        default=None,
        help="Optional comma-separated environment subset.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--rel-width",
        type=float,
        default=1.0,
        help="Relative NeurIPS column width passed to tueplots.",
    )
    return parser.parse_args()


def standard_error(values: pd.Series) -> float:
    count = values.count()
    if count <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(count))


def load_curves(cnn_data_dir: Path, staging_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    curve: dict[str, str],
    env: str,
    cnn: pd.DataFrame,
    staging: pd.DataFrame,
) -> pd.DataFrame:
    if curve["source"] == "cnn":
        frame = cnn[(cnn["environment"] == env) & (cnn["condition"] == curve["key"])]
    else:
        frame = staging[
            (staging["environment"] == env)
            & (staging["canonical_group"] == curve["key"])
        ]
    return frame.sort_values("step")


def load_muon_curve(muon_data_dir: Path, env: str) -> pd.DataFrame:
    filename = MUON_ENV_FILES.get(env)
    if filename is None:
        return pd.DataFrame(columns=["step", "mean"])

    path = muon_data_dir / filename
    if not path.exists():
        print(f"Warning: missing Muon data for {env}: {path}")
        return pd.DataFrame(columns=["step", "mean"])

    frame = pd.read_csv(path)
    mean_col = "heads_optimizer: muon - charts/avg_episodic_return"
    lower_col = "heads_optimizer: muon - charts/avg_episodic_return__MIN"
    upper_col = "heads_optimizer: muon - charts/avg_episodic_return__MAX"
    required = {"Step", mean_col, lower_col, upper_col}
    if not required.issubset(frame.columns):
        print(f"Warning: Muon data has unexpected columns: {path}")
        return pd.DataFrame(columns=["step", "mean", "lower", "upper"])

    return (
        frame[["Step", mean_col, lower_col, upper_col]]
        .rename(
            columns={
                "Step": "step",
                mean_col: "mean",
                lower_col: "lower",
                upper_col: "upper",
            }
        )
        .dropna(subset=["step", "mean", "lower", "upper"])
        .astype({"step": float, "mean": float, "lower": float, "upper": float})
        .sort_values("step")
    )


def plot_env(env: str, cnn: pd.DataFrame, staging: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots()
    max_step = ENV_MAX_STEPS.get(env)

    for curve in CURVES:
        frame = curve_frame(curve, env, cnn, staging)
        if max_step is not None:
            frame = frame[frame["step"] <= max_step]
        if frame.empty:
            print(f"Warning: missing {env}/{curve['key']}")
            continue

        color = CONDITION_COLORS[curve["key"]]
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        count = int(frame["count"].max())

        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=color,
            linewidth=2.0,
            label=f"{curve['label']} (n={count})",
            errorbar=None,
        )
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Average episodic return")
    if max_step is not None:
        ax.set_xlim(left=0, right=max_step)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_additional_env(
    env: str,
    cnn: pd.DataFrame,
    staging: pd.DataFrame,
    muon_data_dir: Path,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots()
    max_step = ENV_MAX_STEPS.get(env)

    for curve in CURVES:
        frame = curve_frame(curve, env, cnn, staging)
        if max_step is not None:
            frame = frame[frame["step"] <= max_step]
        if frame.empty:
            print(f"Warning: missing {env}/{curve['key']}")
            continue

        color = CONDITION_COLORS[curve["key"]]
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        count = int(frame["count"].max())
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=color,
            linewidth=2.0,
            label=f"{curve['label']} (n={count})",
            errorbar=None,
        )
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    muon = load_muon_curve(muon_data_dir, env)
    if max_step is not None:
        muon = muon[muon["step"] <= max_step]
    if not muon.empty:
        steps = muon["step"].to_numpy(dtype=float)
        mean = muon["mean"].to_numpy(dtype=float)
        lower = muon["lower"].to_numpy(dtype=float)
        upper = muon["upper"].to_numpy(dtype=float)
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=CONDITION_COLORS["muon"],
            linewidth=2.2,
            label="Muon",
            errorbar=None,
        )
        ax.fill_between(
            steps,
            lower,
            upper,
            color=CONDITION_COLORS["muon"],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Average episodic return")
    if max_step is not None:
        ax.set_xlim(left=0, right=max_step)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ant_goal(csv_path: Path, output_path: Path, dpi: int) -> None:
    if not csv_path.exists():
        print(f"Warning: missing Ant goal CSV: {csv_path}")
        return

    data = pd.read_csv(csv_path).drop_duplicates("run_id")
    data["condition"] = data["algorithm"].map(ANT_GOAL_ALGORITHM_TO_CONDITION)
    data = data.dropna(subset=["condition", "avg_episodic_return"])
    if data.empty:
        print(f"Warning: no valid Ant goal rows in {csv_path}")
        return

    summary = (
        data.groupby("condition", dropna=False)["avg_episodic_return"]
        .agg(["count", "mean", "std", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr"})
    )
    summary["label"] = summary["condition"].map(GOAL_CONDITION_LABELS)
    summary["condition"] = pd.Categorical(
        summary["condition"],
        categories=GOAL_CONDITION_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("condition")

    fig, ax = plt.subplots()
    palette = sns.color_palette(n_colors=len(summary))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["stderr"].fillna(0.0).to_numpy(dtype=float),
        capsize=3,
        color=[CONDITION_COLORS[condition] for condition in summary["condition"]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_ylabel("Average episodic return")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row.label}\n(n={int(row.count)})" for row in summary.itertuples()],
        rotation=20,
        ha="right",
    )
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def fetch_ant_goal_runs(args: argparse.Namespace) -> dict[str, list]:
    if wandb is None:
        raise RuntimeError("wandb is required to refresh ant-goal data.")

    api = wandb.Api(timeout=args.wandb_timeout)
    grouped = {}
    for condition, run_ids in ANT_GOAL_RUNS.items():
        grouped[condition] = []
        for run_id in dict.fromkeys(run_ids):
            run_path = f"{args.ant_goal_prefix}/{run_id}"
            try:
                grouped[condition].append(api.run(run_path))
            except Exception as exc:
                print(f"  warning: failed to fetch {run_path}: {exc}")
        print(f"Fetched {len(grouped[condition])} ant-goal runs for {condition}")
    return grouped


def aggregate_ant_goal_metric(
    runs: list,
    samples: int,
    num_points: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    all_steps = []
    all_values = []
    for run in runs:
        try:
            history = run.history(keys=[ANT_GOAL_METRIC_KEY, "_step"], samples=samples)
        except Exception as exc:
            print(f"  warning: failed to fetch {ANT_GOAL_METRIC_KEY} for {run.path}: {exc}")
            continue
        steps, values = extract_history_arrays(history, ANT_GOAL_METRIC_KEY)
        if steps is None or values is None or len(steps) == 0:
            continue
        all_steps.append(steps)
        all_values.append(values)

    count = len(all_steps)
    if count == 0:
        return None, None, None, 0

    min_step = max(steps.min() for steps in all_steps)
    max_step = min(steps.max() for steps in all_steps)
    if max_step <= min_step:
        return None, None, None, count

    common_steps = np.linspace(min_step, max_step, num_points)
    interpolated = np.asarray(
        [
            np.interp(common_steps, steps, values)
            for steps, values in zip(all_steps, all_values)
        ]
    )
    mean = np.mean(interpolated, axis=0)
    stderr = np.std(interpolated, axis=0) / math.sqrt(count)
    return common_steps, mean, stderr, count


def refresh_ant_goal_cache(args: argparse.Namespace, cache_path: Path) -> pd.DataFrame:
    grouped_runs = fetch_ant_goal_runs(args)
    rows = []
    for condition in GOAL_CONDITION_ORDER:
        steps, mean, stderr, count = aggregate_ant_goal_metric(
            runs=grouped_runs.get(condition, []),
            samples=args.ant_goal_samples,
            num_points=args.ant_goal_num_points,
        )
        if steps is None:
            print(f"  {condition}: unavailable (n={count})")
            continue
        print(f"  {condition}: aggregated n={count}")
        for step, mean_value, stderr_value in zip(steps, mean, stderr):
            rows.append(
                {
                    "metric": "return",
                    "metric_key": ANT_GOAL_METRIC_KEY,
                    "condition": condition,
                    "label": GOAL_CONDITION_LABELS[condition],
                    "step": step,
                    "count": count,
                    "mean": mean_value,
                    "stderr": stderr_value,
                }
            )

    curves = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(cache_path, index=False)
    print(f"Wrote {cache_path}")
    return curves


def load_ant_goal_curves(args: argparse.Namespace) -> pd.DataFrame | None:
    cache_path = args.ant_goal_data_dir / "ant_goal_curves.csv"
    if args.refresh_ant_goal or not cache_path.exists():
        return refresh_ant_goal_cache(args, cache_path)
    return pd.read_csv(cache_path)


def plot_ant_goal_curves(
    curves: pd.DataFrame,
    output_path: Path,
    dpi: int,
    max_step: float | None,
) -> None:
    if curves.empty:
        print("Warning: no Ant goal curve data")
        return

    if max_step is not None:
        curves = curves[curves["step"] <= max_step]
        if curves.empty:
            print(f"Warning: no Ant goal curve data at or before step {max_step}")
            return

    fig, ax = plt.subplots()
    for condition in GOAL_CONDITION_ORDER:
        frame = curves[curves["condition"] == condition].sort_values("step")
        if frame.empty:
            continue
        color = CONDITION_COLORS[condition]
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        count = int(frame["count"].max())
        label = f"{GOAL_CONDITION_LABELS[condition]} (n={count})"
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=color,
            linewidth=2.0,
            label=label,
            errorbar=None,
        )
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Average episodic return")
    if max_step is not None:
        ax.set_xlim(left=0, right=max_step)
    ax.set_ylim(bottom=ANT_GOAL_Y_MIN)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def humanoid_goal_condition_from_name(name: str) -> str | None:
    for condition in GOAL_CONDITION_ORDER:
        if condition in name:
            return condition
    return None


def extract_history_arrays(history, metric_key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hasattr(history, "dropna"):
        history = history.dropna(subset=[metric_key])
        if history.empty:
            return None, None
        steps = history["_step"].to_numpy(dtype=float)
        values = history[metric_key].to_numpy(dtype=float)
    else:
        rows = []
        for row in history:
            if not isinstance(row, dict):
                continue
            step = row.get("_step")
            value = row.get(metric_key)
            if step is None or value is None:
                continue
            rows.append((float(step), float(value)))
        if not rows:
            return None, None
        steps = np.asarray([row[0] for row in rows], dtype=float)
        values = np.asarray([row[1] for row in rows], dtype=float)

    order = np.argsort(steps)
    return steps[order], values[order]


def fetch_humanoid_goal_runs(args: argparse.Namespace) -> dict[str, list]:
    if wandb is None:
        raise RuntimeError("wandb is required to refresh humanoid-goal data.")

    api = wandb.Api(timeout=args.wandb_timeout)
    compiled = re.compile(args.humanoid_goal_run_regex)
    grouped = defaultdict(list)
    runs = api.runs(
        path=args.humanoid_goal_prefix,
        filters={"display_name": {"$regex": args.humanoid_goal_run_regex}},
    )

    fetched = 0
    matched = 0
    for run in runs:
        fetched += 1
        if not compiled.search(run.name):
            continue
        condition = humanoid_goal_condition_from_name(run.name)
        if condition is None:
            continue
        grouped[condition].append(run)
        matched += 1

    print(f"Fetched {fetched} humanoid-goal runs from W&B.")
    print(f"Matched {matched} humanoid-goal runs across {len(grouped)} conditions.")
    return grouped


def aggregate_humanoid_goal_metric(
    runs: list,
    metric_key: str,
    samples: int,
    num_points: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    all_steps = []
    all_values = []
    for run in runs:
        try:
            history = run.history(keys=[metric_key, "_step"], samples=samples)
        except Exception as exc:
            print(f"  warning: failed to fetch {metric_key} for {run.path}: {exc}")
            continue
        steps, values = extract_history_arrays(history, metric_key)
        if steps is None or values is None or len(steps) == 0:
            continue
        all_steps.append(steps)
        all_values.append(values)

    count = len(all_steps)
    if count == 0:
        return None, None, None, 0

    min_step = max(steps.min() for steps in all_steps)
    max_step = min(steps.max() for steps in all_steps)
    if max_step <= min_step:
        return None, None, None, count

    common_steps = np.linspace(min_step, max_step, num_points)
    interpolated = np.asarray(
        [
            np.interp(common_steps, steps, values)
            for steps, values in zip(all_steps, all_values)
        ]
    )
    mean = np.mean(interpolated, axis=0)
    stderr = np.std(interpolated, axis=0) / math.sqrt(count)
    return common_steps, mean, stderr, count


def refresh_humanoid_goal_cache(args: argparse.Namespace, cache_path: Path) -> pd.DataFrame:
    grouped_runs = fetch_humanoid_goal_runs(args)
    rows = []

    for metric_name, spec in HUMANOID_GOAL_METRICS.items():
        print(f"Aggregating humanoid-goal {metric_name}: {spec['key']}")
        for condition in GOAL_CONDITION_ORDER:
            steps, mean, stderr, count = aggregate_humanoid_goal_metric(
                runs=grouped_runs.get(condition, []),
                metric_key=spec["key"],
                samples=args.humanoid_goal_samples,
                num_points=args.humanoid_goal_num_points,
            )
            if steps is None:
                print(f"  {condition}: unavailable (n={count})")
                continue
            print(f"  {condition}: n={count}")
            for step, mean_value, stderr_value in zip(steps, mean, stderr):
                rows.append(
                    {
                        "metric": metric_name,
                        "metric_key": spec["key"],
                        "condition": condition,
                        "label": GOAL_CONDITION_LABELS[condition],
                        "step": step,
                        "count": count,
                        "mean": mean_value,
                        "stderr": stderr_value,
                    }
                )

    curves = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(cache_path, index=False)
    print(f"Wrote {cache_path}")
    return curves


def load_humanoid_goal_curves(args: argparse.Namespace) -> pd.DataFrame | None:
    cache_path = args.humanoid_goal_data_dir / "humanoid_goal_final_archopt_curves.csv"
    if args.refresh_humanoid_goal or not cache_path.exists():
        return refresh_humanoid_goal_cache(args, cache_path)
    return pd.read_csv(cache_path)


def plot_humanoid_goal_metric(
    curves: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    metric_curves = curves[curves["metric"] == metric_name]
    if metric_curves.empty:
        print(f"Warning: no humanoid-goal data for {metric_name}")
        return

    spec = HUMANOID_GOAL_METRICS[metric_name]
    fig, ax = plt.subplots()

    for condition in GOAL_CONDITION_ORDER:
        frame = metric_curves[metric_curves["condition"] == condition].sort_values("step")
        if frame.empty:
            continue
        color = CONDITION_COLORS[condition]
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        count = int(frame["count"].max())
        label = f"{GOAL_CONDITION_LABELS[condition]} (n={count})"
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=color,
            linewidth=2.0,
            label=label,
            errorbar=None,
        )
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(spec["ylabel"])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def load_lqr_curves(data_dir: Path, pattern: str) -> pd.DataFrame | None:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        print(f"Warning: no LQR files matched {data_dir / pattern}")
        return None

    frames = []
    required = {"iter", "up_reward", "down_reward"}
    for path in paths:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            print(f"Warning: skipping {path}; missing columns {sorted(missing)}")
            continue
        frame = frame[["iter", "up_reward", "down_reward"]].copy()
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    long = raw.melt(
        id_vars=["source_file", "iter"],
        value_vars=["up_reward", "down_reward"],
        var_name="condition",
        value_name="reward",
    )
    return (
        long.groupby(["condition", "iter"], dropna=False)["reward"]
        .agg(["count", "mean", "std", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr", "iter": "step"})
    )


def plot_lqr(curves: pd.DataFrame, output_path: Path, dpi: int) -> None:
    if curves.empty:
        print("Warning: no LQR curve data")
        return

    fig, ax = plt.subplots()
    for curve in LQR_CURVES:
        frame = curves[curves["condition"] == curve["key"]].sort_values("step")
        if frame.empty:
            continue

        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=2.0,
            label=curve["label"],
            errorbar=None,
        )
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=curve["color"],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.legend(frameon=True)
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def load_probe_metric(probe_data_dir: Path, base_name: str) -> pd.DataFrame | None:
    frames = []
    for condition in PROBE_CONDITIONS:
        path = probe_data_dir / f"{base_name}_{condition['key']}.csv"
        if not path.exists():
            print(f"Warning: missing probe file: {path}")
            continue

        data = pd.read_csv(path)
        if "Step" not in data.columns:
            print(f"Warning: skipping {path}; missing Step column")
            continue

        value_columns = [
            column
            for column in data.columns
            if column != "Step"
            and not column.endswith("__MIN")
            and not column.endswith("__MAX")
        ]
        if not value_columns:
            print(f"Warning: skipping {path}; no probe value columns found")
            continue

        long = data.melt(
            id_vars="Step",
            value_vars=value_columns,
            var_name="run",
            value_name="value",
        )
        long["condition"] = condition["key"]
        long = long.rename(columns={"Step": "step"})
        frames.append(long[["condition", "step", "value"]])

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    raw["step"] = pd.to_numeric(raw["step"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["condition", "step", "value"])
    if raw.empty:
        return None

    return (
        raw.groupby(["condition", "step"], dropna=False)["value"]
        .agg(["count", "mean", "std", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr"})
    )


def plot_probe_metric(
    curves: pd.DataFrame,
    output_path: Path,
    dpi: int,
    ylabel: str,
) -> None:
    if curves.empty:
        print(f"Warning: no probe curve data for {output_path.name}")
        return

    fig, ax = plt.subplots()
    for condition in PROBE_CONDITIONS:
        frame = curves[curves["condition"] == condition["key"]].sort_values("step")
        if frame.empty:
            continue

        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        stderr = frame["stderr"].fillna(0.0).to_numpy(dtype=float)
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=condition["color"],
            linewidth=2.0,
            label=condition["label"],
            errorbar=None,
        )
        ax.fill_between(
            steps,
            mean - stderr,
            mean + stderr,
            color=condition["color"],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def load_llm_curves(data_dir: Path, task: str, algo: str) -> pd.DataFrame | None:
    path = data_dir / f"{task}_{algo}_history.csv"
    if not path.exists():
        print(f"Warning: missing LLM history file: {path}")
        return None

    required = ["_step", "train/loss"]
    try:
        frame = pd.read_csv(path, usecols=required)
    except ValueError:
        print(f"Warning: skipping {path}; expected columns {required}")
        return None

    frame = frame.dropna(subset=required).copy()
    if frame.empty:
        print(f"Warning: no train/loss rows in {path}")
        return None

    frame["condition"] = algo
    frame = frame.rename(columns={"_step": "step", "train/loss": "loss"})
    frame["step"] = (frame["step"] + 1) / LLM_STEP_DIVISOR
    return frame[["condition", "step", "loss"]]


def aggregate_llm_task(data_dir: Path, task: str) -> pd.DataFrame | None:
    frames = []
    for algo in [spec["key"] for spec in LLM_ALGOS]:
        frame = load_llm_curves(data_dir, task, algo)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    return (
        raw.groupby(["condition", "step"], dropna=False)["loss"]
        .agg(["count", "mean", "std", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr"})
    )


def smooth_llm_frame(
    frame: pd.DataFrame,
    method: str,
    window: int,
    ema_alpha: float,
) -> pd.DataFrame:
    frame = frame.sort_values("step").copy()
    frame["lower"] = frame["mean"] - frame["stderr"].fillna(0.0)
    frame["upper"] = frame["mean"] + frame["stderr"].fillna(0.0)
    if method == "none":
        return frame

    columns = ["mean", "lower", "upper"]
    if method == "rolling":
        if window <= 1:
            return frame
        frame[columns] = (
            frame[columns]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
    elif method == "ema":
        frame[columns] = frame[columns].ewm(alpha=ema_alpha, adjust=False).mean()
    return frame


def smooth_llm_curves(
    curves: pd.DataFrame,
    method: str,
    window: int,
    ema_alpha: float,
) -> pd.DataFrame:
    frames = [
        smooth_llm_frame(frame, method, window, ema_alpha)
        for _, frame in curves.groupby("condition", sort=False)
    ]
    return pd.concat(frames, ignore_index=True)


def plot_llm_task(
    curves: pd.DataFrame,
    task: str,
    output_path: Path,
    dpi: int,
    smoothing_method: str,
    smoothing_window: int,
    ema_alpha: float,
) -> None:
    if curves.empty:
        print(f"Warning: no LLM curve data for {task}")
        return

    curves = smooth_llm_curves(curves, smoothing_method, smoothing_window, ema_alpha)
    fig, ax = plt.subplots()
    for algo in LLM_ALGOS:
        frame = curves[curves["condition"] == algo["key"]].sort_values("step")
        if frame.empty:
            continue
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        lower = frame["lower"].to_numpy(dtype=float)
        upper = frame["upper"].to_numpy(dtype=float)
        count = int(frame["count"].max())
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=algo["color"],
            linewidth=2.0,
            label=f"{algo['label']} (n={count})",
            errorbar=None,
        )
        ax.fill_between(
            steps,
            lower,
            upper,
            color=algo["color"],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss")
    y_bottom, y_top = ax.get_ylim()
    source_padding = 0.05 * max(y_max - y_min, 1e-9)
    ax.set_ylim(
        min(y_bottom, y_min - source_padding),
        max(y_top, y_max + source_padding),
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def llm_elbow_limits(
    curves: pd.DataFrame,
    min_step: float,
    max_step: float,
) -> tuple[float, float, float, float]:
    x_min = min_step
    x_max = max_step
    elbow = curves[(curves["step"] >= x_min) & (curves["step"] <= x_max)]
    if elbow.empty:
        elbow = curves
    y_low = float(elbow["lower"].quantile(0.02))
    y_high = float(elbow["upper"].quantile(0.98))
    padding = 0.08 * max(y_high - y_low, 1e-9)
    return x_min, x_max, y_low - padding, y_high + padding


def plot_llm_task_cropped(
    curves: pd.DataFrame,
    task: str,
    output_path: Path,
    dpi: int,
    min_step: float,
    max_step: float,
    smoothing_method: str,
    smoothing_window: int,
    ema_alpha: float,
) -> None:
    if curves.empty:
        print(f"Warning: no LLM curve data for {task}")
        return

    curves = smooth_llm_curves(curves, smoothing_method, smoothing_window, ema_alpha)
    x_min, x_max, y_min, y_max = llm_elbow_limits(curves, min_step, max_step)
    fig, ax = plt.subplots()
    for algo in LLM_ALGOS:
        frame = curves[curves["condition"] == algo["key"]].sort_values("step")
        frame = frame[(frame["step"] >= x_min) & (frame["step"] <= x_max)]
        if frame.empty:
            continue
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        lower = frame["lower"].to_numpy(dtype=float)
        upper = frame["upper"].to_numpy(dtype=float)
        count = int(frame["count"].max())
        sns.lineplot(
            x=steps,
            y=mean,
            ax=ax,
            color=algo["color"],
            linewidth=2.0,
            label=f"{algo['label']} (n={count})",
            errorbar=None,
        )
        ax.fill_between(
            steps,
            lower,
            upper,
            color=algo["color"],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss")
    y_bottom, y_top = ax.get_ylim()
    source_padding = 0.08 * max(y_max - y_min, 1e-9)
    ax.set_ylim(
        min(y_bottom, y_min - source_padding),
        max(y_top, y_max + source_padding),
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(fig=fig, ax=ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_llm_task_with_inset(
    curves: pd.DataFrame,
    task: str,
    output_path: Path,
    dpi: int,
    min_step: float,
    max_step: float,
    smoothing_method: str,
    smoothing_window: int,
    ema_alpha: float,
) -> None:
    if curves.empty:
        print(f"Warning: no LLM curve data for {task}")
        return

    curves = smooth_llm_curves(curves, smoothing_method, smoothing_window, ema_alpha)
    x_min, x_max, y_min, y_max = llm_elbow_limits(curves, min_step, max_step)
    fig, ax = plt.subplots()
    inset = ax.inset_axes([0.50, 0.42, 0.45, 0.45])

    for algo in LLM_ALGOS:
        frame = curves[curves["condition"] == algo["key"]].sort_values("step")
        if frame.empty:
            continue
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean"].to_numpy(dtype=float)
        lower = frame["lower"].to_numpy(dtype=float)
        upper = frame["upper"].to_numpy(dtype=float)
        count = int(frame["count"].max())
        label = f"{algo['label']} (n={count})"
        for axis, alpha in ((ax, 0.18), (inset, 0.14)):
            sns.lineplot(
                x=steps,
                y=mean,
                ax=axis,
                color=algo["color"],
                linewidth=2.0,
                label=label if axis is ax else None,
                errorbar=None,
            )
            axis.fill_between(
                steps,
                lower,
                upper,
                color=algo["color"],
                alpha=alpha,
                linewidth=0,
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss")
    y_bottom, y_top = ax.get_ylim()
    source_padding = 0.08 * max(y_max - y_min, 1e-9)
    ax.set_ylim(
        min(y_bottom, y_min - source_padding),
        max(y_top, y_max + source_padding),
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    inset.set_xlim(x_min, x_max)
    inset.set_ylim(y_min, y_max)
    inset.tick_params(labelsize=7)
    inset.set_xlabel("")
    inset.set_ylabel("")
    indicator = ax.indicate_inset_zoom(inset, edgecolor="0.0", linewidth=1.4)
    try:
        indicator.rectangle.set_zorder(5)
        indicator.rectangle.set_clip_on(False)
        for connector in indicator.connectors:
            connector.set_visible(False)
            connector.set_zorder(5)
            connector.set_clip_on(False)
    except AttributeError:
        rectangle, connectors = indicator
        rectangle.set_zorder(5)
        rectangle.set_clip_on(False)
        for connector in connectors:
            connector.set_visible(False)
            connector.set_zorder(5)
            connector.set_clip_on(False)
    top_connector = ConnectionPatch(
        xyA=(0, 1),
        coordsA=inset.transAxes,
        xyB=(x_max, y_max),
        coordsB=ax.transData,
        arrowstyle="-",
        color="0.0",
        linewidth=1.4,
        clip_on=False,
        zorder=5,
    )
    ax.add_artist(top_connector)
    sns.despine(fig=fig, ax=ax)
    sns.despine(fig=fig, ax=inset)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_llm_drafts(
    llm_curves_by_task: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
    min_step: float,
    max_step: float,
    smoothing_method: str,
    smoothing_window: int,
    ema_alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, curves in llm_curves_by_task.items():
        plot_llm_task_cropped(
            curves,
            task,
            output_dir / f"llm_{task}_loss_cropped.png",
            dpi,
            min_step,
            max_step,
            smoothing_method,
            smoothing_window,
            ema_alpha,
        )
        plot_llm_task_with_inset(
            curves,
            task,
            output_dir / f"llm_{task}_loss_inset.png",
            dpi,
            min_step,
            max_step,
            smoothing_method,
            smoothing_window,
            ema_alpha,
        )


def save_legend(
    handles: list[Line2D],
    output_path: Path,
    dpi: int,
    ncol: int,
) -> None:
    fig, ax = plt.subplots(figsize=(max(1.5 * ncol, 3.0), 0.45))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.3,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {output_path}")


def save_shared_legends(plots_dir: Path, dpi: int) -> None:
    rl_handles = [
        Line2D([0], [0], color=CONDITION_COLORS["cnn_adam"], lw=2.0, label="CNN Adam"),
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS["cnn_muon"],
            lw=2.0,
            label="CNN manifold Muon",
        ),
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS["crate_adam"],
            lw=2.0,
            label="ISTA Adam",
        ),
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS["crate_stiefel"],
            lw=2.0,
            label="ISTA manifold Muon",
        ),
    ]
    save_legend(rl_handles, plots_dir / "rl_legend.png", dpi, ncol=4)

    rl_muon_handles = [
        rl_handles[0],
        Line2D([0], [0], color=CONDITION_COLORS["muon"], lw=2.2, label="CNN Muon"),
        *rl_handles[1:],
    ]
    save_legend(rl_muon_handles, plots_dir / "rl_muon_legend.png", dpi, ncol=5)

    llm_handles = [
        Line2D([0], [0], color=CONDITION_COLORS["crate_adam"], lw=2.0, label="Adam"),
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS["crate_stiefel"],
            lw=2.0,
            label="manifold Muon",
        ),
    ]
    save_legend(llm_handles, plots_dir / "llm_legend.png", dpi, ncol=2)

    probe_handles = [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS["cnn_adam"],
            lw=2.0,
            label="CNN Adam",
        ),
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS["cnn_muon"],
            lw=2.0,
            label="CNN manifold Muon",
        ),
    ]
    save_legend(probe_handles, plots_dir / "probe_legend.png", dpi, ncol=2)


def main() -> None:
    args = parse_args()
    envs = ENV_ORDER
    if args.envs:
        envs = [env.strip().lower() for env in args.envs.split(",") if env.strip()]
        unknown = [env for env in envs if env not in ENV_ORDER]
        if unknown:
            raise SystemExit(f"Unknown envs: {', '.join(unknown)}")

    plt.rcParams.update(
        bundles.neurips2024(usetex=False, rel_width=args.rel_width, family="sans-serif")
    )
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="deep",
        rc=dict(plt.rcParams),
    )
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.28,
            "legend.framealpha": 0.92,
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica",
                "Helvetica Neue",
                "TeX Gyre Heros",
                "Nimbus Sans",
                "Nimbus Sans L",
            ],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    cnn, staging = load_curves(args.cnn_data_dir, args.staging_data_dir)

    for env in envs:
        plot_env(env, cnn, staging, args.plots_dir / f"{env}.png", args.dpi)
        plot_additional_env(
            env,
            cnn,
            staging,
            args.muon_data_dir,
            args.plots_dir / f"additional_{env}.png",
            args.dpi,
        )

    ant_goal_curves = load_ant_goal_curves(args)
    if ant_goal_curves is not None and not ant_goal_curves.empty:
        plot_ant_goal_curves(
            ant_goal_curves,
            args.plots_dir / "ant_goal_return.png",
            args.dpi,
            args.ant_goal_max_step,
        )
    else:
        plot_ant_goal(
            args.ant_goal_csv,
            args.plots_dir / "ant_goal_return.png",
            args.dpi,
        )

    humanoid_goal_curves = load_humanoid_goal_curves(args)
    if humanoid_goal_curves is not None and not humanoid_goal_curves.empty:
        for metric_name in args.humanoid_goal_metrics:
            spec = HUMANOID_GOAL_METRICS[metric_name]
            plot_humanoid_goal_metric(
                humanoid_goal_curves,
                metric_name,
                args.plots_dir / spec["filename"],
                args.dpi,
            )

    lqr_curves = load_lqr_curves(args.lqr_data_dir, args.lqr_glob)
    if lqr_curves is not None and not lqr_curves.empty:
        plot_lqr(lqr_curves, args.plots_dir / "lqr_reward.png", args.dpi)

    for spec in PROBE_METRICS.values():
        probe_curves = load_probe_metric(args.probe_data_dir, spec["base"])
        if probe_curves is not None and not probe_curves.empty:
            plot_probe_metric(
                probe_curves,
                args.plots_dir / spec["filename"],
                args.dpi,
                spec["ylabel"],
            )

    llm_curves_by_task = {}
    for task in LLM_TASKS:
        llm_curves = aggregate_llm_task(args.llm_data_dir, task)
        if llm_curves is not None and not llm_curves.empty:
            llm_curves_by_task[task] = llm_curves
            plot_llm_task_with_inset(
                llm_curves,
                task,
                args.plots_dir / f"llm_{task}_loss.png",
                args.dpi,
                args.llm_elbow_min_step,
                args.llm_elbow_max_step,
                args.llm_smoothing,
                args.llm_smoothing_window,
                args.llm_ema_alpha,
            )

    save_shared_legends(args.plots_dir, args.dpi)

    if args.make_draft_llm_plots and llm_curves_by_task:
        plot_llm_drafts(
            llm_curves_by_task,
            args.draft_plots_dir,
            args.dpi,
            args.llm_elbow_min_step,
            args.llm_elbow_max_step,
            args.llm_smoothing,
            args.llm_smoothing_window,
            args.llm_ema_alpha,
        )


if __name__ == "__main__":
    main()
