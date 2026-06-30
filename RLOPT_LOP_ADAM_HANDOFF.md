# rlopt_lop-Aligned PixelRL Adam/Stiefel Runs

This handoff documents PixelRL launchers that match the
`lop-jax/rlopt_lop/scripts/slippery_ant_bp_l40s.sh` run shape, network, and
policy parameterization as closely as possible while retaining PixelRL's PPO
loss implementation.

## Added Files

- `ppo_brax_rlopt_lop_adam.py`
  - Copy of the current `ppo_brax.py`.
  - Keeps the PixelRL/CleanRL-style PPO objective in `ppo_loss`.
  - Changes defaults to the `rlopt_lop` BP baseline:
    - `env_name=ant`, `backend=positional`, `n_envs=1`
    - `total_timesteps=20_000_000`
    - `num_steps=2048`, `num_minibatches=16`, `update_epochs=10`
    - `learning_rate=1e-4`, `adam_eps=1e-8`, `max_grad_norm=1e9`
    - `ent_coef=0.0`, `vf_coef=1.0`
    - `actor_critic_activation=relu`
    - `network_arch=lop_reference`
    - `reward_normalize=False`, `obs_normalize=False`
    - `slippery=True`, `slippery_change_every=2_000_000`
  - Adds `clip_actions: bool = False` and only clips sampled actions when that
    flag is enabled. This matches the `rlopt_lop` policy/action path more
    closely than the original `ppo_brax.py`, which always clips sampled actions
    before stepping the environment.

- `scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seed0.sh`
  - Single-job sbatch launcher, no Slurm array.
  - Uses direct `SEED=0`.
  - Defaults to the action-clipping comparison condition:
    - `CONDITION=rlopt_lop_plain_adam_clip_actions`
    - `CLIP_ACTIONS=1`
  - The earlier no-clip seed-0 behavior can still be recovered with
    `CLIP_ACTIONS=0 CONDITION=rlopt_lop_plain_adam`.
  - Runs `ppo_brax_rlopt_lop_adam.py` with plain Adam:
    - `--base-optimizer adam`
    - `--heads-optimizer adam`
  - Uses `--network-arch lop_reference` for two 256-wide hidden layers in both
    actor and critic.
  - Does not pass `--actor-mean-tanh` or `--bounded-global-logstd`, so the
    policy uses an unconstrained mean and global `log_std` initialized at
    `0.0`. The seed-0 script now passes `--clip-actions` by default for the
    comparison run.
  - Provides `DRY_RUN=1` output with repo root, condition, seed, W&B metadata,
    rollout/minibatch sizing, and the final quoted command.
  - Aligns `TOTAL_TIMESTEPS` down to a whole rollout batch, matching the
    existing PixelRL launcher behavior. With defaults, `20_000_000` becomes
    `19_998_720` because `2048 * 9765 = 19_998_720`.

- `scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seeds1_5_serial.sh`
  - Serial completion launcher for seeds `1..5` after seed `0`.
  - Uses `#SBATCH --array=1-5%1`, so only one GPU array task can run at a
    time for this experiment.
  - Uses the same W&B project and group as seed `0`:
    - project: `continual_brax`
    - group: `slippery-ant-rlopt-lop-plain-adam`
  - Maps `SLURM_ARRAY_TASK_ID` directly to `--seed`.

- `scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_stiefel_seeds0_2_serial.sh`
  - Stiefel comparison launcher for seeds `0..2`.
  - Uses `#SBATCH --array=0-2`, so all three seeds are eligible to run
    concurrently when Oscar has GPU capacity.
  - Keeps the completed plain-Adam run configuration unchanged where it is
    shared:
    - no action clipping
    - `--network-arch lop_reference`
    - `--ent-coef 0.0`, `--vf-coef 1.0`
    - `--num-steps 2048`, `--num-minibatches 16`, `--update-epochs 10`
    - no reward normalization and no observation normalization
    - unconstrained actor mean and global log std initialized at `0.0`
  - Changes the head optimizer condition to regular Stiefel:
    - `--base-optimizer adam`
    - `--heads-optimizer stiefel`
    - `--heads-stiefel-lr 0.001`
    - `--stiefel-dual-lr 0.01`
    - `--stiefel-dual-steps 5`
    - `--stiefel-msign-steps 5`
    - `--actor-stiefel-max-grad-norm 100`
    - `--critic-stiefel-max-grad-norm 1`
  - Uses W&B project `continual_brax` and group
    `slippery-ant-rlopt-lop-stiefel`.

- `scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seed0_adam_vs_stiefel.sh`
  - Seed-`0` Adam-vs-Stiefel launcher that runs the copied
    `ppo_brax_rlopt_lop_adam.py` entrypoint with the original `ppo_brax`
    Slippery Ant settings from
    `scripts/CL/brax/humanoid_sweep/ant_lop_final_adam_vs_stiefel_6seeds.sh`.
  - Uses `#SBATCH --array=0-1`:
    - task `0`: Adam heads, seed `0`, learning rate `1e-4`
    - task `1`: Stiefel heads, seed `0`, base learning rate `3e-5`
  - Keeps the original PPO-run policy and model shape:
    - `--network-arch lop`, which is the single 256-wide hidden layer actor
      and critic head configuration in this entrypoint
    - `--num-minibatches 128`
    - `--ent-coef 0.01`, `--vf-coef 1.0`
    - `--actor-mean-tanh`
    - `--bounded-global-logstd`
    - `--actor-logstd-init -2.0`
    - `--actor-logstd-max -1.4`
    - `--clip-actions`
  - Uses W&B project `continual_brax` and group
    `slippery-ant-rlopt-lop-original-ppo-config-seed0-adam-vs-stiefel`.

- `scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seeds3_5_adam_vs_stiefel.sh`
  - Extension launcher for the same original-`ppo_brax` config experiment,
    using the copied `ppo_brax_rlopt_lop_adam.py` entrypoint.
  - Uses `#SBATCH --array=0-5`:
    - tasks `0..2`: Adam heads, seeds `3..5`, learning rate `1e-4`
    - tasks `3..5`: Stiefel heads, seeds `3..5`, base learning rate `3e-5`
  - Keeps the same original PPO-run policy and model shape as the seed-`0`
    script:
    - `--network-arch lop`
    - `--num-minibatches 128`
    - `--ent-coef 0.01`, `--vf-coef 1.0`
    - `--actor-mean-tanh`
    - `--bounded-global-logstd`
    - `--actor-logstd-init -2.0`
    - `--actor-logstd-max -1.4`
    - `--clip-actions`
  - Uses the same W&B project and group as the seed-`0` original-config
    comparison, even though the group name contains `seed0` from the first
    launcher.

## Intentional Objective Choice

The new copied entrypoint keeps PixelRL's PPO objective code:

- minibatch-local advantage normalization when `norm_adv=True`
- clipped value loss when `clip_vloss=True`
- entropy term in the formula
- PixelRL's `0.5` factor inside value loss

The run hyperparameters follow `rlopt_lop`, so `ent_coef=0.0` even though the
entropy term remains present in the implementation. This isolates the PPO
implementation from the `rlopt_lop` network/policy/hyperparameter bundle.

## Important Differences That Remain

- The Slippery Ant friction wrapper still comes from PixelRL
  (`configs/continual/slippery_ant_wrapper.py`), not `rlopt_lop`. It keeps the
  default Brax friction in phase 0 and then follows the CSV schedule.
- The optimizer still uses PixelRL's multi-transform plumbing, although all
  active labels resolve to Adam in this run.
- The copied entrypoint does not save an Orbax result tree like `rlopt_lop`.
  It logs the PixelRL W&B scalar metrics from the original training loop.

## Validation Before Submitting

Run these from the PixelRL repo root:

```bash
bash -n scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seed0.sh
DRY_RUN=1 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seed0.sh
sbatch --test-only scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seed0.sh
bash -n scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seeds1_5_serial.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seeds1_5_serial.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=5 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seeds1_5_serial.sh
sbatch --test-only scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_plain_adam_seeds1_5_serial.sh
bash -n scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_stiefel_seeds0_2_serial.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_stiefel_seeds0_2_serial.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=2 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_stiefel_seeds0_2_serial.sh
sbatch --test-only scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_stiefel_seeds0_2_serial.sh
bash -n scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seed0_adam_vs_stiefel.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seed0_adam_vs_stiefel.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seed0_adam_vs_stiefel.sh
sbatch --test-only scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seed0_adam_vs_stiefel.sh
bash -n scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seeds3_5_adam_vs_stiefel.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seeds3_5_adam_vs_stiefel.sh
DRY_RUN=1 SLURM_ARRAY_TASK_ID=5 bash scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seeds3_5_adam_vs_stiefel.sh
sbatch --test-only scripts/CL/brax/humanoid_sweep/ant_rlopt_lop_original_config_seeds3_5_adam_vs_stiefel.sh
```

For a cheap Python-level check, run:

```bash
UV_CACHE_DIR=/tmp/pixelrl-uv-cache \
MPLCONFIGDIR=/tmp/pixelrl-mpl \
PYTHONPATH="$PWD/pixelbrax/brax:${PYTHONPATH:-}" \
uv run python ppo_brax_rlopt_lop_adam.py --help
```

The expected seed-0 dry-run command should include:

- `--seed 0`
- `--num-minibatches 16`
- `--network-arch lop_reference`
- `--ent-coef 0.0`
- `--vf-coef 1.0`
- no `--actor-mean-tanh`
- no `--bounded-global-logstd`
- `--clip-actions`
- condition `rlopt_lop_plain_adam_clip_actions`

The expected Stiefel dry-run command should include:

- `--seed 0` for task `0` and `--seed 2` for task `2`
- `--heads-optimizer stiefel`
- `--heads-stiefel-lr 0.001`
- `--num-minibatches 16`
- `--network-arch lop_reference`
- `--ent-coef 0.0`
- no `--actor-mean-tanh`
- no `--bounded-global-logstd`
- no `--clip-actions`
- condition `rlopt_lop_stiefel`

The expected original-`ppo_brax` seed-0 Adam-vs-Stiefel dry-run commands should
include:

- task `0`: `--heads-optimizer adam`, `--learning-rate 1e-4`, `--seed 0`
- task `1`: `--heads-optimizer stiefel`, `--learning-rate 3e-5`,
  `--heads-stiefel-lr 0.001`, `--seed 0`
- both tasks: `--network-arch lop`, `--num-minibatches 128`,
  `--ent-coef 0.01`, `--actor-mean-tanh`, `--bounded-global-logstd`,
  `--clip-actions`
- neither task should use `--network-arch lop_reference`

The expected original-`ppo_brax` seeds-`3..5` Adam-vs-Stiefel dry-run commands
should include:

- task `0`: `--heads-optimizer adam`, `--learning-rate 1e-4`, `--seed 3`
- task `5`: `--heads-optimizer stiefel`, `--learning-rate 3e-5`,
  `--heads-stiefel-lr 0.001`, `--seed 5`
- both tasks: `--network-arch lop`, `--num-minibatches 128`,
  `--ent-coef 0.01`, `--actor-mean-tanh`, `--bounded-global-logstd`,
  `--clip-actions`

## Experiment Progress

- Plain Adam no-action-clipping runs for seeds `0`, `1`, and `2` are complete
  according to the current experiment state.
- Stiefel comparison launcher added for seeds `0..2`.
- Initial Stiefel array job `3430821` was cancelled manually before running.
- Stiefel resubmission status: submitted as Slurm array job `3430845` with no
  dependency and no array concurrency throttle. The submitted array is `0-2`,
  so all three seeds are eligible to run concurrently when scheduled.
- Original-`ppo_brax` config seed-0 Adam-vs-Stiefel launcher added for the
  copied `ppo_brax_rlopt_lop_adam.py` entrypoint.
- Original-`ppo_brax` config seed-0 Adam-vs-Stiefel submission status:
  submitted as Slurm array job `3444203` after `bash -n`, Adam dry-run,
  Stiefel dry-run, and `sbatch --test-only` succeeded.
- Queue check immediately after submission showed task `3444203_0` running and
  task `3444203_1` pending on `QOSMaxGRESPerUser`.
- Original-`ppo_brax` config seeds-`3..5` Adam-vs-Stiefel launcher added for
  the copied `ppo_brax_rlopt_lop_adam.py` entrypoint.
- Original-`ppo_brax` config seeds-`3..5` Adam-vs-Stiefel submission status:
  submitted as Slurm array job `3448656` after `bash -n`, Adam seed-`3`
  dry-run, Stiefel seed-`5` dry-run, and `sbatch --test-only` succeeded.

## Suggested Next Steps

1. Run the dry-run and `sbatch --test-only` checks above on Oscar.
2. Submit the single job if the command matches the expected settings.
3. Compare against the existing PixelRL Adam branch and `rlopt_lop` BP baseline
   using return collapse timing, entropy, value loss, and friction phase.
4. If this run still differs materially from `rlopt_lop`, the next likely
   source is the environment/friction wrapper rather than network or policy
   parameterization.
