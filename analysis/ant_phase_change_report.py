#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


wandb._assert_is_internal_process = True

RETURN_METRIC = "charts/avg_episodic_return"
DEFAULT_THRESHOLDS = [0.0, 100.0, 200.0, 400.0, 600.0, 800.0]
DEFAULT_CHECKPOINT_STEPS = [128000, 256000, 512000, 768000, 1000000]
KEY_METRICS = [
    "repr/unit_std_avg",
    "repr/dead_units_frac",
    "cnn_dense/tanh_sat_frac_0.95",
    "policy/obs_to_noise_ratio",
    "policy/mean_action_obs_std_avg",
    "value/obs_std",
    "losses/entropy",
    "losses/innovation_coef",
    "losses/innovation_total",
    "innovation_debug/residual_r2",
    "innovation_debug/residual_to_target_ratio",
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local ant W&B runs and check whether the phase change happens before innovation turns on."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Paths to local wandb run directories (for example wandb/run-20260319_150108-3n6lp9yn).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Return thresholds to report first-hit steps for.",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=DEFAULT_CHECKPOINT_STEPS,
        help="Absolute env-step checkpoints to summarize.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "files" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
    return config


def load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def find_run_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("run-*.wandb"))
    if not matches:
        raise FileNotFoundError(f"Could not find run-*.wandb in {run_dir}")
    return matches[0]


def load_history(run_dir: Path) -> List[Dict]:
    ds = DataStore()
    ds.open_for_scan(str(find_run_file(run_dir)))
    rows: List[Dict] = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if not rec.HasField("history"):
            continue
        row = {}
        for item in rec.history.item:
            try:
                row[item.key] = json.loads(item.value_json)
            except Exception:
                row[item.key] = item.value_json
        rows.append(row)
    return rows


def metric_value(row: Dict, key: str) -> float:
    value = row.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def nearest_row(rows: List[Dict], step: int) -> Dict:
    return min(rows, key=lambda row: abs(int(metric_value(row, "global_step")) - step))


def first_threshold_hit(rows: List[Dict], threshold: float) -> Optional[Dict]:
    for row in rows:
        if metric_value(row, RETURN_METRIC) >= threshold:
            return row
    return None


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


def run_label(run_dir: Path, config: Dict) -> str:
    exp_name = str(config.get("exp_name", run_dir.name))
    seed = config.get("seed", "?")
    return f"{exp_name} (seed={seed})"


def innovation_start_step(config: Dict) -> int:
    warmup_updates = int(config.get("innovation_warmup_updates", 0) or 0)
    if warmup_updates <= 0:
        return 0
    batch_size = config.get("batch_size")
    if batch_size is None:
        n_envs = int(config.get("n_envs", 0) or 0)
        num_steps = int(config.get("num_steps", 0) or 0)
        batch_size = n_envs * num_steps
    return int(batch_size) * warmup_updates


def print_run_report(run_dir: Path, thresholds: List[float], steps: List[int]) -> None:
    config = load_config(run_dir)
    summary = load_summary(run_dir)
    rows = load_history(run_dir)
    label = run_label(run_dir, config)
    innov_start = innovation_start_step(config)

    print(f"\n## {label}")
    print(f"- Local run: `{run_dir}`")
    print(f"- Final return: `{fmt(float(summary.get(RETURN_METRIC, math.nan)), 4)}`")
    print(
        f"- Innovation warmup start: `{innov_start}` env steps "
        f"(warmup updates={config.get('innovation_warmup_updates', 0)})"
    )

    print("- Threshold crossings:")
    for threshold in thresholds:
        hit = first_threshold_hit(rows, threshold)
        if hit is None:
            print(f"  - return >= {threshold:.0f}: never")
            continue
        step = int(metric_value(hit, "global_step"))
        before_innov = "before" if step < innov_start else "after"
        print(
            "  - "
            f"return >= {threshold:.0f}: step `{step}` ({before_innov} innovation warmup), "
            f"repr std `{fmt(metric_value(hit, 'repr/unit_std_avg'))}`, "
            f"dead frac `{fmt(metric_value(hit, 'repr/dead_units_frac'))}`, "
            f"entropy `{fmt(metric_value(hit, 'losses/entropy'))}`, "
            f"innovation coef `{fmt(metric_value(hit, 'losses/innovation_coef'))}`"
        )

    print("- Fixed checkpoints:")
    header = ["step", "return"] + KEY_METRICS
    print("  " + " | ".join(header))
    for step in steps:
        row = nearest_row(rows, step)
        values = [str(int(metric_value(row, "global_step"))), fmt(metric_value(row, RETURN_METRIC), 4)]
        values.extend(fmt(metric_value(row, key)) for key in KEY_METRICS)
        print("  " + " | ".join(values))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    for run in args.runs:
        print_run_report(Path(run), args.thresholds, args.steps)


if __name__ == "__main__":
    main()
