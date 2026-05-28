# Humanoid Sweep Handoff

Last updated: 2026-05-26

## Scope

This directory contains the non-pixel Brax continual-learning sweeps for Slippery Humanoid and Slippery Ant.

Allowed edit scope during this work:

- `ppo_brax.py`
- `scripts/CL/brax/humanoid_sweep/`

## Current Best Ant Plasticity Setup

The cleanest Ant loss-of-plasticity setup is now the lop-cadence Slippery Ant protocol with the stable bounded policy profile.

Launcher:

```bash
scripts/CL/brax/humanoid_sweep/ant_lop_online_stiefel_4seeds.sh
```

Key config:

- `env=ant`
- `backend=positional`
- `NETWORK_ARCH=lop`
- stable bounded policy:
  - `--actor-mean-tanh`
  - `--actor-mean-scale 1.0`
  - `--bounded-global-logstd`
  - `--actor-logstd-init -2.0`
  - `--actor-logstd-min -5.0`
  - `--actor-logstd-max -1.4`
- raw observations, no obs normalization
- `learning_rate=1e-4`
- `num_envs=1`
- `num_steps=2048`
- `num_minibatches=128`
- `update_epochs=10`
- `clip_eps=0.2`
- `ent_coef=0.01`
- `vf_coef=1.0`
- `change_every=2_000_000`
- `TOTAL_TIMESTEPS=20_000_000`, aligned by the launcher to `19_998_720`
- final phase at 20M: `phase=9`, `friction=1.103`

Example Adam relaunch:

```bash
sbatch --parsable --gres=none --mem=16GB --time=02:30:00 --array=0-3%4 \
  --export=ALL,JAX_PLATFORMS=cpu,TOTAL_TIMESTEPS=20000000 \
  scripts/CL/brax/humanoid_sweep/ant_lop_online_stiefel_4seeds.sh continual_brax rl-power
```

Example online-Stiefel relaunch:

```bash
sbatch --parsable --gres=none --mem=16GB --time=04:00:00 --array=4-7%4 \
  --export=ALL,JAX_PLATFORMS=cpu,TOTAL_TIMESTEPS=20000000 \
  scripts/CL/brax/humanoid_sweep/ant_lop_online_stiefel_4seeds.sh continual_brax rl-power
```

Runtime on CPU:

- Adam 20M: about `1h20m`
- online Stiefel 20M: about `2h41m`

## Current Best Ant Result

Adam, 20M steps:

- Job: `84531`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_lop_online_ablation_positional_84531`

| Seed | W&B | Final return | Runtime | Notes |
| --- | --- | --- | --- | --- |
| 0 | `https://wandb.ai/rl-power/continual_brax/runs/qjdur0qj` | `-2241.4` | `1:20:05` | Catastrophic late collapse. |
| 1 | `https://wandb.ai/rl-power/continual_brax/runs/zl2u8w5d` | `4532.9` | `1:19:56` | Strong non-collapsed seed. |
| 2 | `https://wandb.ai/rl-power/continual_brax/runs/82mc2oi5` | `-130.4` | `1:19:57` | Late collapse/near failure. |
| 3 | `https://wandb.ai/rl-power/continual_brax/runs/8q8tu2il` | `-2337.4` | `1:19:51` | Catastrophic collapse; this seed was already weak early. |

Online Stiefel, 20M steps:

- Job: `84542`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_lop_online_ablation_positional_84542`

| Seed | W&B | Final return | Runtime | Notes |
| --- | --- | --- | --- | --- |
| 0 | `https://wandb.ai/rl-power/continual_brax/runs/4idk81ms` | `400.1` | `2:41:49` | Low final return, but not catastrophic negative collapse. |
| 1 | `https://wandb.ai/rl-power/continual_brax/runs/pkb81nol` | `1421.0` | `2:41:50` | Survived late phases. |
| 2 | `https://wandb.ai/rl-power/continual_brax/runs/hn5plau6` | `1748.2` | `2:42:06` | Best online-Stiefel seed. |
| 3 | `https://wandb.ai/rl-power/continual_brax/runs/fmd961eh` | `1272.6` | `2:40:54` | Survived despite early instability. |

Interpretation:

- This is the cleanest Ant plasticity setup so far.
- Adam collapses badly in 3/4 seeds by phase 9.
- Online Stiefel is slower and lower-return than Adam's best seed, but it avoids catastrophic negative-return collapse in all 4 seeds.

Regular Stiefel, 20M steps:

- Job: `84624`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_lop_stiefel_ablation_positional_84624`

| Seed | W&B | Final return | Runtime | Notes |
| --- | --- | --- | --- | --- |
| 0 | `https://wandb.ai/rl-power/continual_brax/runs/27yvs20k` | `2571.1` | `4:59:13` | Survived phase 9 with strong final return. |
| 1 | `https://wandb.ai/rl-power/continual_brax/runs/n3b19twz` | `1829.6` | `5:01:16` | Survived phase 9; some late dips but no collapse. |
| 2 | `https://wandb.ai/rl-power/continual_brax/runs/hxscsi28` | `2495.3` | `5:00:15` | Survived phase 9 with strong final return. |
| 3 | `https://wandb.ai/rl-power/continual_brax/runs/m2g11ag2` | `2604.8` | `5:00:28` | Survived phase 9 with strong final return. |

Updated interpretation:

- Regular Stiefel is the strongest mitigation tested on this Ant setup.
- It is much slower on CPU than Adam and online Stiefel, but it avoids Adam's catastrophic collapse and beats online Stiefel final returns on all four matched seeds.

Regular Stiefel LR sweep:

- Job: `84670`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_lop_stiefel_lr_sweep_positional_84670`
- Seeds: `0`, `2`
- Grid:
  - base/general LR: `3e-5`, `1e-4`, `3e-4`
  - Stiefel head LR: `3e-4`, `1e-3`, `3e-3`

| Base LR | Stiefel head LR | Seed 0 | Seed 2 | Mean |
| --- | --- | ---: | ---: | ---: |
| `3e-5` | `3e-4` | `1566.8` | `3887.2` | `2727.0` |
| `3e-5` | `1e-3` | `2982.3` | `3721.1` | `3351.7` |
| `3e-5` | `3e-3` | `2387.1` | `2356.5` | `2371.8` |
| `1e-4` | `3e-4` | `2044.5` | `3512.0` | `2778.2` |
| `1e-4` | `1e-3` | `2571.1` | `2495.3` | `2533.2` |
| `1e-4` | `3e-3` | `929.0` | `624.7` | `776.8` |
| `3e-4` | `3e-4` | `2834.7` | `3473.9` | `3154.3` |
| `3e-4` | `1e-3` | `1275.3` | `1035.3` | `1155.3` |
| `3e-4` | `3e-3` | `1086.1` | `48.9` | `567.5` |

Sweep interpretation:

- Best two-seed mean: base LR `3e-5`, Stiefel head LR `1e-3`, mean `3351.7`.
- This improves substantially over the previous regular-Stiefel matched seed-0/seed-2 mean at base LR `1e-4`, head LR `1e-3` (`2533.2`).
- It still does not beat Adam's best individual seed (`4532.9`), but it is much more reliable than Adam on collapse-prone seeds.
- High Stiefel head LR `3e-3` is consistently worse, especially with larger base LRs.
- Next sensible confirmation is 4 seeds with base LR `3e-5`, Stiefel head LR `1e-3`.

Final 6-seed Adam vs regular-Stiefel launcher:

```bash
scripts/CL/brax/humanoid_sweep/ant_lop_final_adam_vs_stiefel_6seeds.sh
```

Mapping:

- tasks `0..5`: Adam baseline, seeds `0..5`, base LR `1e-4`
- tasks `6..11`: tuned regular Stiefel, seeds `0..5`, base LR `3e-5`, Stiefel head LR `1e-3`

Launch command:

```bash
sbatch --parsable --gres=none --mem=16GB --time=06:00:00 --array=0-11%6 \
  --export=ALL,JAX_PLATFORMS=cpu \
  scripts/CL/brax/humanoid_sweep/ant_lop_final_adam_vs_stiefel_6seeds.sh continual_brax rl-power
```

## Trainer Changes In `ppo_brax.py`

The successful non-pixel trainer required mirroring the working pixelbrax behavior more closely.

Added policy controls:

- `--actor-logstd-min`
- `--actor-logstd-max`
- `--clip-global-logstd`
- `--bounded-global-logstd`
- `--actor-logstd-init`
- `--actor-mean-tanh`
- `--actor-mean-scale`

Added network/head controls:

- `--use-crate-network`
- `--network-crate-step-size`
- `--use-crate-head`
- `--crate-step-size`
- `--network-arch {ppo,lop}`

Added normalization:

- `--obs-normalize`
- `--obs-norm-clip`

Added optimizer controls:

- `--base-optimizer {adam,adamw}`
- `--weight-decay`
- `--heads-optimizer adam`
- `--heads-optimizer stiefel`
- `--heads-optimizer stiefel_admm`
- `--heads-optimizer stiefel_online`
- `--heads-optimizer aurora`

`--network-arch lop` uses an identity state trunk and lop-jax style shallow actor/critic heads:

- actor: one hidden layer, width 256, ReLU, direct observation input
- critic: one hidden layer, width 256, ReLU, direct observation input

## Slippery Humanoid Findings

Initial generalized Slippery Humanoid did not learn under the first non-pixel wrapper settings. The successful recipe required:

- bounded global actor log std
- tanh-bounded actor means
- running observation normalization
- CRATE state trunk
- CRATE actor and critic heads
- low learning rate, best tested `3e-5`

Saved successful Adam script:

```bash
scripts/CL/brax/humanoid_sweep/successful_slippery_humanoid_adam_crate_obsnorm.sh
```

Saved Stiefel variant:

```bash
scripts/CL/brax/humanoid_sweep/successful_slippery_humanoid_stiefel_crate_obsnorm.sh
```

Humanoid Adam fast plasticity validation:

- Job: `82883`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_humanoid_envonly_validate_82883`
- `LEARNING_RATE=3e-5`
- `VALIDATION_PHASE_ENV_STEPS=999936`
- `VALIDATION_PHASE_COUNTS=1,5,10,20`
- `BEST_SCHEDULE_SEED=14`

Final or late returns:

| Phase count | Seed 0 | Seed 1 |
| --- | --- | --- |
| 1 | `732` | `742` |
| 5 | `4482` | `2513` |
| 10 | `1134` | `925` |
| 20 | around `92` at phase 14, cancelled | around `153` at phase 14, cancelled |

Interpretation: Adam learns well with moderate task changes, then collapses as phase count increases.

Humanoid Stiefel + Adam + ReLU confirmation:

- Job: `82988`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_humanoid_envonly_confirm_82988`
- Seeds 2-9 final returns: `1969.1`, `3639.8`, `2994.3`, `2904.5`, `1841.7`, `2422.9`, `1253.2`, `3071.1`
- Combined with ablation seeds 0-1, mean final return was about `2805.0`, range `1253.2-4276.6`

Interpretation: Stiefel + Adam + ReLU avoids the near-collapse seen in the prior Adam-only 20-phase Humanoid baseline and was the best tested Humanoid config.

## Humanoid Optimizer/Activation Ablation

Job: `82974`

W&B group:
`https://wandb.ai/rl-power/continual_brax/groups/slippery_humanoid_envonly_optimizer_ablation_82974`

Settings:

```bash
WAVE=optimizer_ablation LEARNING_RATE=3e-5 BEST_SCHEDULE_SEED=14 ABLATION_PHASE_ENV_STEPS=999936 ABLATION_NUM_PHASES=20 \
  sbatch --parsable --array=0-7%8 scripts/CL/brax/humanoid_sweep/start.sh continual_brax rl-power
```

| Config | Seeds | Final returns | Mean |
| --- | --- | --- | --- |
| Stiefel + Adam + ReLU | 0, 1 | `3676.9`, `4276.6` | `3976.8` |
| Stiefel + Adam + Swish | 0, 1 | `1983.1`, `2556.5` | `2269.8` |
| Stiefel + AdamW + ReLU | 0, 1 | `3167.2`, `1297.8` | `2232.5` |
| Stiefel + AdamW + Swish | 0, 1 | `2955.2`, `3594.3` | `3274.8` |

Outcome: AdamW + Swish helped relative to the weaker Swish-only and AdamW-only variants, but it did not beat Stiefel + Adam + ReLU.

## Earlier Ant Transfer And Controls

Stiefel + Adam + ReLU transfer to Ant:

- Job: `83030`
- W&B group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_envonly_confirm_83030`
- Final returns for seeds 0-7: `2472.4`, `2540.2`, `2371.5`, `2976.0`, `2692.9`, `1863.4`, `1912.8`, `2801.7`
- Mean final return: `2453.9`

Outcome: the Humanoid Stiefel + Adam + ReLU config transfers to Ant and remains functional through 20 task phases. It did not show a late-run collapse in that Ant run.

Large-batch Ant controls:

- Script: `scripts/CL/brax/humanoid_sweep/ant_adam_batch_sweep.sh`
- Adam standard batch group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_adam_batch_lr1e_4_spring_83501`
- Adam extreme batch group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_adam_batch_extreme_lr1e_4_spring_83511`
- Matched Stiefel extreme-batch group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_stiefel_batch_extreme_lr1e_4_spring_83521`

Interpretation: larger batches break performance, but Stiefel does not mitigate. A fixed-budget phase-count control at batch 81920 showed the same failure with 1, 5, 10, and 20 phases, so this was under-updating rather than task-change plasticity.

Small-batch fixed-budget phase-count control:

- Script: `scripts/CL/brax/humanoid_sweep/ant_adam_phasecount_batch1280.sh`
- Adam group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_adam_phasecount_batch1280_lr1e_4_spring_83538`
- Stiefel group: `https://wandb.ai/rl-power/continual_brax/groups/slippery_ant_stiefel_phasecount_batch1280_lr1e_4_spring_83548`

Adam final-return means:

| Phase count | Mean |
| --- | --- |
| 1 | `4172.9` |
| 5 | `3676.1` |
| 10 | `3598.1` |
| 20 | `2899.7` |

Stiefel final-return means:

| Phase count | Mean |
| --- | --- |
| 1 | `2712.4` |
| 5 | `2068.7` |
| 10 | `2420.5` |
| 20 | `1862.5` |

Interpretation: batch 1280 at LR `1e-4` gave an Ant task-count effect, but Stiefel heads did not rescue it and were lower-return across phase counts.

## lop-jax Reproduction Notes

Reference repo:
`https://github.com/KevinGuo27/lop-jax/tree/main/rlopt`

Reference Slippery Ant protocol:

- `env=slippery_ant`
- backend effectively `positional`
- `num_envs=1`
- `total_steps=10_000_000`
- `change_every=2_000_000`
- `num_steps=2048`
- `num_minibatches=128`
- `update_epochs=10`
- `vf_coeff=1.0`
- `entropy_coeff=0.01`
- `clip_eps=0.2`
- Adam LR sweep includes `1e-4`
- actor/critic are independent one-hidden-layer width-256 ReLU MLPs directly on observations

Launcher:

```bash
scripts/CL/brax/humanoid_sweep/ant_lop_protocol.sh
```

Outcome:

- Exact lop-jax style high-std policy (`POLICY_PROFILE=lop`, global logstd init 0) did not learn in `ppo_brax.py`.
- LR `1e-4` failed before first switch.
- LR `1e-3` reached NaNs around 1.8M steps.
- LR `1e-2` reached NaNs around 0.39M steps.
- Observation normalization did not fix the exact high-std policy.
- The stable bounded policy with `NETWORK_ARCH=lop`, backend `positional`, raw observations, LR `1e-4`, and the lop cadence did learn.

Stable Adam 10M raw-observation runs:

| Job | Seed | W&B | Final return | Notes |
| --- | --- | --- | --- | --- |
| `84040_0` | 0 | `https://wandb.ai/rl-power/continual_brax/runs/pe5p57ie` | `2429.1` | Learned, then degraded sharply in later phases and partially recovered. |
| `84035_1` | 1 | `https://wandb.ai/rl-power/continual_brax/runs/srvgev34` | `4277.1` | Learned strongly through the full 5-phase schedule. |

Matched old Stiefel-head comparison on seed 0:

| Job | Seed | W&B | Final return | Notes |
| --- | --- | --- | --- | --- |
| `84050_0` | 0 | `https://wandb.ai/rl-power/continual_brax/runs/8ohk7udh` | `1766.6` | Learned phase 0 but did not rescue late degradation; slower on CPU than Adam. |

Interpretation: the lop cadence/backend/friction schedule can learn in this repo, but only with the stable bounded policy parameterization. The exact high-std lop policy appears incompatible with this trainer's action/logprob dynamics.

## Useful Commands

Syntax checks:

```bash
bash -n scripts/CL/brax/humanoid_sweep/sweep.sh
bash -n scripts/CL/brax/humanoid_sweep/ant_lop_online_stiefel_4seeds.sh
bash -n scripts/CL/brax/humanoid_sweep/ant_lop_protocol.sh
```

CPU smoke test for lop architecture:

```bash
PYTHONPATH=/home/guests/arjun/pixelrl/pixelbrax/brax JAX_PLATFORMS=cpu uv run python ppo_brax.py \
  --env-name ant --backend positional --n-envs 1 --total-timesteps 64 --num-steps 32 \
  --num-minibatches 1 --update-epochs 1 --network-arch lop --actor-critic-activation relu \
  --heads-optimizer adam --slippery-ant --slippery-change-every 2000000 \
  --slippery-schedule-seed 0 --actor-mean-tanh --bounded-global-logstd \
  --actor-logstd-init -2 --actor-logstd-min -5 --actor-logstd-max -1.4 --log-interval 1
```

Check recent arrays:

```bash
sacct -j 84531,84542 --format=JobID,JobName%30,State,ExitCode,Elapsed -n -P
```

Extract final summaries:

```bash
for f in slurm_logs/slippery_ant_lop_online_84531_{0..3}.out slurm_logs/slippery_ant_lop_online_84542_{4..7}.out; do
  echo "$f"
  tail -n 50 "$f" | rg "Run summary|avg_episodic_return|global_step|phase_first_env|timestep_first_env|Training finished|View run|update="
done
```

## Caveats

- Compare optimizers by environment steps, not wall time. Stiefel variants are much slower.
- Exact lop-jax high-std policy did not transfer directly; the working setup uses bounded std and tanh means.
- Adam seed 3 in the 20M Ant run was weak early, so seed 0 and seed 2 are the cleaner late-collapse evidence.
- Online Stiefel avoids catastrophic collapse in the 20M Ant run, but its returns are lower than Adam's best surviving seed.
