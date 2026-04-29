#!/usr/bin/env python3
"""Pull staging avg_episodic_return data from W&B.

Writes raw run metadata, raw metric histories, and grouped summaries to
``plots/staging_data`` by default.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from typing import Any, Iterable

import pandas as pd
import wandb


DEFAULT_METRIC_KEY = "charts/avg_episodic_return"
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
ENV_ALIASES = {
    "half_cheetah": "halfcheetah",
    "invertedpendulum": "inverted_pendulum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull finished staging/staging2/staging3 W&B runs into local CSV files."
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("WANDB_PREFIX", "rl-power/encoder"),
        help="W&B entity/project path, e.g. rl-power/encoder.",
    )
    parser.add_argument(
        "--metric-key",
        default=DEFAULT_METRIC_KEY,
        help="Metric key to fetch from run histories.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/staging_data",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Use sampled run.history instead of full scan_history when set.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="W&B API timeout in seconds.",
    )
    return parser.parse_args()


def normalize_env(value: Any) -> str | None:
    if value is None:
        return None
    env = str(value).strip().lower().replace("-", "_")
    env = ENV_ALIASES.get(env, env)
    return env if env in ENV_ORDER else None


def tag_value(tags: Iterable[str], prefix: str) -> str | None:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def infer_env(run: Any, tags: set[str], config: dict[str, Any]) -> str | None:
    env = normalize_env(tag_value(tags, "env_"))
    if env is not None:
        return env

    for key in ("env_name", "env", "environment"):
        env = normalize_env(config.get(key))
        if env is not None:
            return env

    haystack = " ".join(
        str(part)
        for part in (
            getattr(run, "name", ""),
            getattr(run, "group", ""),
            getattr(run, "job_type", ""),
        )
        if part
    ).lower()
    for env_name in ENV_ORDER:
        if re.search(rf"(^|[_\-\s]){re.escape(env_name)}($|[_\-\s])", haystack):
            return env_name
    return None


def infer_seed(tags: set[str], config: dict[str, Any]) -> int | None:
    seed = tag_value(tags, "seed_")
    if seed is None:
        seed = config.get("seed")
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def infer_architecture(tags: set[str], config: dict[str, Any]) -> str:
    encoder = tag_value(tags, "encoder_") or config.get("encoder_type") or "unknown"
    headarch = tag_value(tags, "headarch_")
    if headarch is None:
        headarch = "crate" if config.get("use_crate_head") else "plain"
    return f"{encoder}+{headarch}_heads"


def infer_optimizer(tags: set[str], config: dict[str, Any]) -> str:
    return (
        tag_value(tags, "opt_")
        or config.get("heads_optimizer")
        or config.get("optimizer")
        or "unknown"
    )


def infer_staging_set(tags: set[str]) -> str:
    if "staging3" in tags:
        return "staging3"
    if "staging2" in tags:
        return "staging2"
    return "staging"


def fetch_history(run: Any, metric_key: str, samples: int | None) -> pd.DataFrame:
    if samples is not None:
        history = run.history(keys=[metric_key], samples=samples, pandas=True)
        if history is None or len(history) == 0 or metric_key not in history:
            return pd.DataFrame(columns=["step", "avg_episodic_return"])
        step_col = "_step" if "_step" in history else history.index.name
        out = pd.DataFrame(
            {
                "step": history[step_col].to_numpy()
                if step_col in history
                else history.index.to_numpy(),
                "avg_episodic_return": history[metric_key].to_numpy(),
            }
        )
        return out.dropna(subset=["avg_episodic_return"])

    rows = []
    for row in run.scan_history(keys=["_step", metric_key], page_size=1000):
        if metric_key not in row or row[metric_key] is None:
            continue
        rows.append(
            {
                "step": row.get("_step"),
                "avg_episodic_return": row[metric_key],
            }
        )
    return pd.DataFrame(rows, columns=["step", "avg_episodic_return"])


def standard_error(values: pd.Series) -> float:
    count = values.count()
    if count <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(count))


def summarize_final(run_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if run_rows.empty:
        return pd.DataFrame()
    return (
        run_rows.groupby(group_cols, dropna=False)["final_avg_episodic_return"]
        .agg(["count", "mean", "std", "min", "max", standard_error])
        .reset_index()
        .rename(columns={"standard_error": "stderr"})
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    api = wandb.Api(timeout=args.timeout)
    runs = api.runs(path=args.prefix, filters={"tags": {"$in": ["staging"]}, "state": "finished"})

    run_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    fetched = 0
    matched = 0

    for run in runs:
        fetched += 1
        tags = set(getattr(run, "tags", []) or [])
        if "staging" not in tags:
            continue

        config = dict(getattr(run, "config", {}) or {})
        env_name = infer_env(run, tags, config)
        if env_name is None:
            continue

        matched += 1
        staging_set = infer_staging_set(tags)
        print(
            f"[{matched:04d}] fetching {staging_set} env={env_name} "
            f"run={run.id} name={getattr(run, 'name', '')}",
            flush=True,
        )
        history = fetch_history(run, args.metric_key, args.samples)
        print(
            f"[{matched:04d}] fetched {len(history)} points for run={run.id}",
            flush=True,
        )
        final_return = (
            float(history["avg_episodic_return"].iloc[-1]) if not history.empty else float("nan")
        )

        row = {
            "run_id": run.id,
            "run_path": f"{args.prefix}/{run.id}",
            "name": getattr(run, "name", ""),
            "group": getattr(run, "group", ""),
            "state": getattr(run, "state", ""),
            "staging_set": staging_set,
            "environment": env_name,
            "architecture": infer_architecture(tags, config),
            "optimizer": infer_optimizer(tags, config),
            "headarch": tag_value(tags, "headarch_") or "",
            "seed": infer_seed(tags, config),
            "history_points": len(history),
            "final_step": int(history["step"].iloc[-1]) if not history.empty else None,
            "final_avg_episodic_return": final_return,
            "tags": ",".join(sorted(tags)),
        }
        run_rows.append(row)

        if not history.empty:
            history.insert(0, "run_id", run.id)
            history.insert(1, "staging_set", staging_set)
            history.insert(2, "environment", env_name)
            history.insert(3, "architecture", row["architecture"])
            history.insert(4, "optimizer", row["optimizer"])
            history.insert(5, "seed", row["seed"])
            history_frames.append(history)

    runs_df = pd.DataFrame(run_rows).sort_values(
        ["staging_set", "environment", "architecture", "optimizer", "seed", "run_id"]
    )
    history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame(
            columns=[
                "run_id",
                "staging_set",
                "environment",
                "architecture",
                "optimizer",
                "seed",
                "step",
                "avg_episodic_return",
            ]
        )
    )

    env_summary = summarize_final(runs_df, ["staging_set", "environment"])
    grouped_summary = summarize_final(
        runs_df, ["staging_set", "environment", "architecture", "optimizer"]
    )

    runs_path = os.path.join(args.output_dir, "runs.csv")
    history_path = os.path.join(args.output_dir, "history.csv")
    env_summary_path = os.path.join(args.output_dir, "env_summary.csv")
    grouped_summary_path = os.path.join(args.output_dir, "grouped_summary.csv")

    runs_df.to_csv(runs_path, index=False)
    history_df.to_csv(history_path, index=False)
    env_summary.to_csv(env_summary_path, index=False)
    grouped_summary.to_csv(grouped_summary_path, index=False)

    print(f"Fetched {fetched} finished runs with server-side staging filter.")
    print(f"Matched {matched} runs with staging tag and recognized environment.")
    print(f"Wrote {runs_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {env_summary_path}")
    print(f"Wrote {grouped_summary_path}")
    if not env_summary.empty:
        print("\nRun counts by staging set and environment:")
        for _, row in env_summary.iterrows():
            print(
                f"  {row['staging_set']:8s} {row['environment']:18s} "
                f"n={int(row['count']):3d} mean_final={row['mean']:.3f}"
            )


if __name__ == "__main__":
    main()
