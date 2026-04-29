# Staging Data Organization

This folder contains raw and organized W&B data for the `encoder` project runs
tagged `staging` with state `finished`.  The raw pull keeps `staging`,
`staging2`, and `staging3` as separate source sets in the `staging_set` column.

## Raw Inputs

- `runs.csv`: one row per finished W&B run.
- `history.csv`: per-step `charts/avg_episodic_return` for each run.
- `env_summary.csv`: original environment-level final-return summary.
- `grouped_summary.csv`: original summary using the raw W&B optimizer names.

## Canonical Grouping

Run groups are normalized to match the scripts in `scripts/staging`:

- `plainheads_adam`
- `plainheads_stiefel`
- `crate_adam`
- `crate_stiefel`

The confusing convention is handled as follows:

- Original `staging` runs used `muon` in tags and run names for what should be
  compared as the Stiefel optimizer condition.
- `staging2` extra seed runs use `stiefel`.
- `staging3` gap-fill runs also use `stiefel`.
- Therefore raw optimizers `muon` and `stiefel` are both mapped to canonical
  optimizer `stiefel`.
- Raw optimizer `adam` remains canonical optimizer `adam`.
- Head architecture comes from `headarch_plain` / `headarch_crate` where
  available, with run names and architecture labels used as fallbacks.

Each environment/group is expected to have seeds `0 1 2 3 4 5`.
Seeds `0..3` generally come from `staging`, and seeds `4..5` generally come
from `staging2`.  Any holes filled by the gap launchers come from `staging3`.
Missing seeds are reported explicitly in `seed_coverage.csv`, along with
`source_sets` and `seed_sources` columns that show which source set contributed
each seed.

## Refreshing After Staging3 Gap Runs

After `staging3` runs complete, refresh raw W&B data first, then re-run the
organizer:

```bash
uv run python plots/pull_staging_wandb_data.py --prefix rl-power/encoder --output-dir plots/staging_data --samples 100000
uv run python plots/staging_data/organize_staging_data.py
```

The pull script queries finished runs tagged `staging`; because the gap runs
also have `staging3`, they are marked as `staging_set=staging3` instead of being
folded into the original `staging` source set.

## Generated Outputs

Run:

```bash
uv run python plots/staging_data/organize_staging_data.py
```

This writes:

- `canonical_runs.csv`: run metadata with `canonical_group`,
  `canonical_headarch`, and `canonical_optimizer`.
- `canonical_history.csv`: per-step histories with canonical grouping columns.
- `seed_coverage.csv`: present/missing seeds for every environment/group.
- `canonical_final_summary.csv`: final-return mean/std/stderr per
  environment/group.
- `canonical_curve_summary.csv`: per-step mean/std/stderr per environment/group.
- `../staging/*.png`: one plot per environment.
