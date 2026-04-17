# Repository Guidelines

## Project Structure & Module Organization
- `pixelbrax/` contains the core package (environment utilities and renderer code).
- `pixelbrax/brax/` is a vendored Brax codebase; prefer minimal, targeted edits here.
- Top-level trainer entry points include `ppo_pixelbrax_jax2.py`, `ppo_jepa.py`, and `ddpg_pixelbrax_jax.py`.
- `scripts/` contains launch scripts by algorithm family (for example, `scripts/ppo_jax_pixel/` and `scripts/ddpg_jax_pixel/`).
- `datasets/` stores external assets (DAVIS backgrounds), while `plots/`, `wandb/`, and `slurm_logs/` hold experiment outputs.

## Build, Test, and Development Commands
- `uv sync`: install pinned dependencies from `pyproject.toml` and `uv.lock`.
- `pip install -e .`: install this repo in editable mode if you are not using `uv`.
- `uv run python test.py`: smoke-check JAX, MuJoCo, and Brax wiring.
- `uv run ppo_pixelbrax_jax2.py --env-name halfcheetah --n-envs 128 --total-timesteps 100000`: quick local PPO sanity run.
- `uv run ddpg_pixelbrax_jax.py --env-name walker2d --n-envs 16 --total-timesteps 100000`: quick local DDPG sanity run.
- `sbatch scripts/ppo_jax_pixel/ppo.sh`: submit PPO jobs on SLURM.

## Coding Style & Naming Conventions
- Use Python 3.9, 4-space indentation, and standard PEP 8 naming (`snake_case` functions/variables, `CamelCase` classes).
- Keep configs explicit and typed (dataclasses are the established pattern in trainer files).
- Name new trainer variants consistently (for example, `ppo_<variant>.py`) and keep matching launch scripts in `scripts/<algo>_jax_pixel/`.
- Format and lint before opening a PR: `uv run black .`, `uv run isort .`, `uv run flake8 .`.

## Testing Guidelines
- There is no enforced repository-wide coverage threshold yet.
- Add targeted tests for new utility modules and environment wrappers using `pytest` with `*_test.py` filenames.
- Run focused tests with `uv run pytest pixelbrax/brax/brax -k <module_or_feature>`.
- For training-path changes, include at least one reduced-timestep smoke run command in your PR.

## Commit & Pull Request Guidelines
- Commit messages in history are short and imperative (for example, `split encoders`, `working config`).
- Prefer `<scope>: <imperative summary>` going forward (example: `ppo: add frame-stack ablation`).
- PRs should include: what changed, why, exact reproduce commands, and expected metric/plot impact.
- Link related issues and include relevant plots/screenshots when behavior or performance changes.

## Security & Configuration Tips
- Keep secrets and machine-specific settings out of committed code.
- Pass `WANDB_API_KEY`, `PYTHONPATH`, and cluster-specific paths via environment variables or local shell config.
