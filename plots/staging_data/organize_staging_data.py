#!/usr/bin/env python3
"""Normalize staging W&B data into canonical groups and plot per environment.

The original staging runs used ``muon`` where the later scripts use
``stiefel``.  This script maps both names to the canonical optimizer
``stiefel`` so every environment has the intended four groups:

  - plainheads_adam
  - plainheads_stiefel
  - crate_adam
  - crate_stiefel
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CANONICAL_GROUPS = [
    "plainheads_adam",
    "plainheads_stiefel",
    "crate_adam",
    "crate_stiefel",
]
EXPECTED_SEEDS = tuple(range(6))
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
GROUP_STYLE = {
    "plainheads_adam": {"label": "Plain heads Adam", "color": "#4C78A8"},
    "plainheads_stiefel": {"label": "Plain heads Stiefel", "color": "#F58518"},
    "crate_adam": {"label": "CRATE heads Adam", "color": "#54A24B"},
    "crate_stiefel": {"label": "CRATE heads Stiefel", "color": "#E45756"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize staging/staging2 W&B data and make per-env plots."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing runs.csv and history.csv.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "staging",
        help="Directory for per-environment plots.",
    )
    return parser.parse_args()


def canonical_headarch(row: pd.Series) -> str:
    headarch = str(row.get("headarch", "")).strip().lower()
    architecture = str(row.get("architecture", "")).strip().lower()
    name = str(row.get("name", "")).strip().lower()
    group = str(row.get("group", "")).strip().lower()

    if headarch in {"crate", "plain"}:
        return headarch
    if "plain_heads" in architecture or "plainheads" in name or "plainheads" in group:
        return "plain"
    if "crate_heads" in architecture or "crate_stiefel" in name or "crate_adam" in name:
        return "crate"
    raise ValueError(f"Could not infer head architecture for run {row.get('run_id')}")


def canonical_optimizer(row: pd.Series) -> str:
    optimizer = str(row.get("optimizer", "")).strip().lower()
    name = str(row.get("name", "")).strip().lower()
    group = str(row.get("group", "")).strip().lower()

    if optimizer == "adam" or "_adam_" in name or "_adam_" in group:
        return "adam"
    if optimizer in {"muon", "stiefel"} or "_muon_" in name or "_stiefel_" in name:
        return "stiefel"
    raise ValueError(f"Could not infer optimizer for run {row.get('run_id')}")


def add_canonical_columns(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    runs["canonical_headarch"] = runs.apply(canonical_headarch, axis=1)
    runs["canonical_optimizer"] = runs.apply(canonical_optimizer, axis=1)
    runs["canonical_group"] = (
        runs["canonical_headarch"].map({"plain": "plainheads", "crate": "crate"})
        + "_"
        + runs["canonical_optimizer"]
    )
    unknown = sorted(set(runs["canonical_group"]) - set(CANONICAL_GROUPS))
    if unknown:
        raise ValueError(f"Unexpected canonical groups: {unknown}")
    return runs


def standard_error(values: pd.Series) -> float:
    count = values.count()
    if count <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(count))


def seed_coverage(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = runs.groupby(["environment", "canonical_group"], dropna=False)
    for env in ENV_ORDER:
        for group in CANONICAL_GROUPS:
            subset = grouped.get_group((env, group)) if (env, group) in grouped.groups else runs.iloc[0:0]
            seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
            missing = [seed for seed in EXPECTED_SEEDS if seed not in seeds]
            duplicates = (
                subset.groupby("seed")["run_id"].count().loc[lambda s: s > 1].to_dict()
                if not subset.empty
                else {}
            )
            source_sets = sorted(str(source) for source in subset["staging_set"].dropna().unique())
            seed_sources = []
            if not subset.empty:
                for seed, seed_subset in subset.groupby("seed"):
                    sources = sorted(str(source) for source in seed_subset["staging_set"].dropna().unique())
                    seed_sources.append(f"{int(seed)}:{'+'.join(sources)}")
            rows.append(
                {
                    "environment": env,
                    "canonical_group": group,
                    "run_count": int(len(subset)),
                    "unique_seed_count": int(len(seeds)),
                    "present_seeds": " ".join(str(seed) for seed in seeds),
                    "missing_seeds": " ".join(str(seed) for seed in missing),
                    "source_sets": " ".join(source_sets),
                    "seed_sources": " ".join(seed_sources),
                    "duplicate_seed_counts": " ".join(
                        f"{int(seed)}:{int(count)}" for seed, count in sorted(duplicates.items())
                    ),
                    "complete": len(missing) == 0 and not duplicates,
                }
            )
    return pd.DataFrame(rows)


def final_summary(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["environment", "canonical_group"], dropna=False)[
            "final_avg_episodic_return"
        ]
        .agg(["count", "mean", "std", "min", "max", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr"})
    )


def curve_summary(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.groupby(["environment", "canonical_group", "step"], dropna=False)[
            "avg_episodic_return"
        ]
        .agg(["count", "mean", "std", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr"})
    )


def plot_environment(env: str, curves: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    env_curves = curves[curves["environment"] == env]
    env_coverage = coverage[coverage["environment"] == env].set_index("canonical_group")

    for group in CANONICAL_GROUPS:
        group_curves = env_curves[env_curves["canonical_group"] == group]
        if group_curves.empty:
            continue
        style = GROUP_STYLE[group]
        steps = group_curves["step"].to_numpy()
        mean = group_curves["mean"].to_numpy()
        stderr = group_curves["stderr"].fillna(0.0).to_numpy()
        seed_count = int(env_coverage.loc[group, "unique_seed_count"])
        missing = str(env_coverage.loc[group, "missing_seeds"])
        label = f"{style['label']} (n={seed_count})"
        if missing and missing != "nan":
            label += f", missing {missing}"
        ax.plot(steps, mean, label=label, color=style["color"], linewidth=1.8)
        ax.fill_between(steps, mean - stderr, mean + stderr, color=style["color"], alpha=0.18)

    ax.set_title(env.replace("_", " ").title())
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("avg_episodic_return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.data_dir / "runs.csv")
    history = pd.read_csv(args.data_dir / "history.csv")
    runs = add_canonical_columns(runs)

    canonical_cols = [
        "run_id",
        "run_path",
        "name",
        "group",
        "staging_set",
        "environment",
        "canonical_group",
        "canonical_headarch",
        "canonical_optimizer",
        "architecture",
        "optimizer",
        "headarch",
        "seed",
        "history_points",
        "final_step",
        "final_avg_episodic_return",
        "tags",
    ]
    canonical_runs = runs[canonical_cols].sort_values(
        ["environment", "canonical_group", "seed", "staging_set", "run_id"]
    )

    history = history.merge(
        canonical_runs[
            [
                "run_id",
                "canonical_group",
                "canonical_headarch",
                "canonical_optimizer",
            ]
        ],
        on="run_id",
        how="inner",
    )
    canonical_history = history[
        [
            "run_id",
            "staging_set",
            "environment",
            "canonical_group",
            "canonical_headarch",
            "canonical_optimizer",
            "architecture",
            "optimizer",
            "seed",
            "step",
            "avg_episodic_return",
        ]
    ].sort_values(["environment", "canonical_group", "seed", "step"])

    coverage = seed_coverage(canonical_runs)
    final = final_summary(canonical_runs)
    curves = curve_summary(canonical_history)

    canonical_runs.to_csv(args.data_dir / "canonical_runs.csv", index=False)
    canonical_history.to_csv(args.data_dir / "canonical_history.csv", index=False)
    coverage.to_csv(args.data_dir / "seed_coverage.csv", index=False)
    final.to_csv(args.data_dir / "canonical_final_summary.csv", index=False)
    curves.to_csv(args.data_dir / "canonical_curve_summary.csv", index=False)

    for env in ENV_ORDER:
        plot_environment(env, curves, coverage, args.plots_dir / f"{env}.png")

    print(f"Wrote {args.data_dir / 'canonical_runs.csv'}")
    print(f"Wrote {args.data_dir / 'canonical_history.csv'}")
    print(f"Wrote {args.data_dir / 'seed_coverage.csv'}")
    print(f"Wrote {args.data_dir / 'canonical_final_summary.csv'}")
    print(f"Wrote {args.data_dir / 'canonical_curve_summary.csv'}")
    print(f"Wrote plots to {args.plots_dir}")
    print("\nMissing seed report:")
    missing = coverage[~coverage["complete"]]
    if missing.empty:
        print("  All environment/group combinations have exactly seeds 0..5.")
    else:
        for _, row in missing.iterrows():
            print(
                f"  {row['environment']:18s} {row['canonical_group']:18s} "
                f"present=[{row['present_seeds']}] missing=[{row['missing_seeds']}] "
                f"duplicates=[{row['duplicate_seed_counts']}]"
            )


if __name__ == "__main__":
    main()
