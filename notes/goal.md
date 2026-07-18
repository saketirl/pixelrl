# Pixel Goal Notes

Date: 2026-05-01

## Environments

Added PixelBrax-local goal reaching tasks:

- `ant_goal`
- `humanoid_goal`

These live outside the Brax submodule in `pixelbrax/tasks/goal.py` and are registered from `pixelbrax/tasks/__init__.py`.

Both tasks add a non-contact green target body to the Brax XML at runtime. The goal is visual and appears in the rendered pixels.

## Camera And Rendering

The initial top-down camera was easy to debug but not the desired training view.

Current goal camera is a goal-aware tracking view:

- camera follows the agent
- camera sits behind the agent relative to the goal direction
- camera looks toward a point between the agent and the goal
- floor uses a muted repeated texture for `ant_goal` / `humanoid_goal`
- U-maze envs keep the plain floor

This keeps the task visually observable while avoiding pure bird's-eye view.

## Original Reward

Initial `ant_goal` reward followed the upstream-style dense distance reward:

```text
reward = -dist + healthy_reward - ctrl_cost - contact_cost
```

With ant target radius `10`, this produces about:

```text
stationary reward ~= -10 + 1 = -9 per physics step
episode return ~= -9000 over 1000 physics steps
```

This matched the observed runs. Ant stayed near `dist ~= 10`, with `success_easy = 0`.

Humanoid looked much better in raw return, but that was mostly reward-scale/task-scale:

- humanoid goal distance is sampled from `1..5`
- humanoid has `healthy_reward = 5`
- humanoid often terminates around `110-120` physics steps

Conclusion: raw episodic return is not comparable across `ant_goal` and `humanoid_goal`. Prefer:

- `env/dist_mean`
- `env/success_easy_mean`
- `env/success_mean`
- `env/distance_from_origin_mean`
- rollout GIFs

## Reward Shaping Tried

Implemented shaped reward:

```text
reward = progress_scale * (prev_dist - dist)
       - distance_scale * dist
       + healthy_reward
       - ctrl_cost
       - contact_cost
       + success_reward * success
```

where:

```text
success = 1 if dist < 0.5 else 0
```

Logged reward components:

- `env/progress_mean`
- `env/prev_dist_mean`
- `env/reward_progress_mean`
- `env/reward_distance_mean`
- `env/reward_success_mean`

## 8-Config Ant Sweep

Launched array job `73818`.

Each reward config ran:

- Adam heads, seeds `0..3`
- Stiefel heads, seeds `0..3`
- `action_repeat = 4`
- CRATE-CNN encoder + CRATE heads
- GIF saved around `1,000,960` steps

GIF directory:

```text
outputs/goal_reward_sweep/
```

Configs:

| Config | Reward extras |
|---|---|
| `original_dist1` | `-1.0 * dist` |
| `progress10` | `10 * progress` |
| `progress20` | `20 * progress` |
| `progress50` | `50 * progress` |
| `progress10_dist0p1` | `10 * progress - 0.1 * dist` |
| `progress20_dist0p1` | `20 * progress - 0.1 * dist` |
| `progress20_success50` | `20 * progress + 50 * success` |
| `progress20_dist0p1_success50` | `20 * progress - 0.1 * dist + 50 * success` |

All array tasks hit the 4h limit, but summaries were usable.

Best sweep config:

```text
progress20_dist0p1_success50
```

Aggregate:

```text
mean dist ~= 9.37
mean origin distance ~= 1.00
success_easy = 0
success = 0
```

Stiefel was better than Adam on the best config:

```text
progress20_dist0p1_success50 stiefel:
  mean dist ~= 9.14
  mean origin distance ~= 1.25

progress20_dist0p1_success50 adam:
  mean dist ~= 9.60
  mean origin distance ~= 0.75
```

Interpretation:

- pure progress mostly learned standing/small motion
- adding `-0.1 * dist` made the goal signal materially better
- success bonus did not fire yet, but the config with it looked best
- Stiefel produced more consistent locomotion than Adam

## Current Best Reward

Current best candidate for `ant_goal`:

```text
reward = 20 * (prev_dist - dist)
       - 0.1 * dist
       + 1.0
       - 0.5 * sum(action^2)
       + 50 * success
```

with:

```text
success = 1 if dist < 0.5 else 0
contact_cost = 0.0
```

Equivalent CLI flags:

```text
--goal-progress-reward-scale 20
--goal-distance-reward-scale 0.1
--goal-success-reward 50
```

## Focused Follow-Up

Launched focused job `73980`:

- `73980_0`: Adam, seeds `0..3`, one GPU
- `73980_1`: Stiefel, seeds `0..3`, one GPU

Launcher:

```text
scripts/goal/ant_goal_progress20_dist0p1_success50_4seeds_byopt.sh
```

Renamed clearer launcher:

```text
scripts/goal/ant_goal_cratecnn_crateheads_bestreward_4seeds_byopt.sh
```

At around 3.2M-5.1M steps, Stiefel looked better:

```text
Adam:
  mean dist ~= 9.79
  mean origin distance ~= 0.66
  1/4 seeds moving meaningfully

Stiefel:
  mean dist ~= 8.83
  mean origin distance ~= 1.65
  3/4 seeds moving meaningfully
```

Best current Stiefel seed:

```text
seed 1:
  dist ~= 7.85
  origin distance ~= 2.84
```

No run has reached `success_easy > 0` yet, but the shaped Stiefel runs are clearly making more goal-directed progress than the original reward.

## Next Ideas

If `progress20_dist0p1_success50` plateaus before `success_easy`, try:

- `progress50_dist0p1_success50`
- `progress20_dist0p2_success50`
- `progress50_dist0p2_success50`
- lower `healthy_reward` for ant to reduce standing incentive
- curriculum on `goal_radius`, e.g. start at `3-5` before radius `10`

The network and PPO hyperparameters appear serviceable; the main issue still looks like reward geometry and exploration distance.

## Main Launchers

All launchers below use the current best reward flags:

```text
--goal-progress-reward-scale 20
--goal-distance-reward-scale 0.1
--goal-success-reward 50
```

They also use:

```text
--backend spring
--action-repeat 4
--frame-stack 4
--save-rollout-gif
--rollout-gif-step 1000000
```

Each launcher is a two-task Slurm array:

- array task `0`: Adam heads, seeds `0..3`
- array task `1`: Stiefel heads, seeds `0..3`

### CRATE-CNN Encoder + CRATE Heads

Ant:

```text
scripts/goal/ant_goal_cratecnn_crateheads_bestreward_4seeds_byopt.sh
```

Humanoid:

```text
scripts/goal/humanoid_goal_cratecnn_crateheads_bestreward_4seeds_byopt.sh
```

Architecture:

```text
--encoder-type crate_cnn
--encoder-crate-step-size 0.1
--use-crate-head
--crate-step-size 0.1
--sigreg-mode off
```

Common W&B tags:

```text
goal,pixel_goal,reward_focus,progress20_dist0p1_success50,encoder_crate_cnn,headarch_crate,action_repeat_4,gif_every_1m
```

### Baseline CNN Encoder + Standard Heads

Ant:

```text
scripts/goal/ant_goal_cnn_heads_bestreward_4seeds_byopt.sh
```

Humanoid:

```text
scripts/goal/humanoid_goal_cnn_heads_bestreward_4seeds_byopt.sh
```

Architecture, matching `scripts/full_runs/full_run.sh`:

```text
--encoder-type cnn
--encoder-tanh-scale 0.5
```

No CRATE head flags are used.

Common W&B tags:

```text
goal,pixel_goal,reward_focus,progress20_dist0p1_success50,encoder_cnn,encoder_scale_0.5,headarch_mlp,action_repeat_4,gif_every_1m
```

The clean W&B filter for the two baseline launchers is:

```text
encoder_cnn AND headarch_mlp AND progress20_dist0p1_success50
```

The clean W&B filter for the two CRATE-CNN launchers is:

```text
encoder_crate_cnn AND headarch_crate AND progress20_dist0p1_success50
```

## Configuration Update (2026-05-01)

### Explicit Episode Length for Ant

`ant_goal` now has `episode_length=1000` set explicitly in `env_utils.py` (previously relied on the Brax default). With `action_repeat=4` the `EpisodeWrapper` increments the step counter by 4 per call, so the episode ends after 250 `step()` calls = 1000 physics steps.

This made the W&B graphs much cleaner and more comparable across runs — previously episodes could end at irregular lengths depending on Brax version defaults; now every ant episode is guaranteed to terminate at exactly 1000 steps.

### All-Combos Sweep Launcher

Added `scripts/goal/ant_goal_all_combos_bestreward_4seeds.sh`, a 4-task Slurm array covering all encoder/head/optimizer combinations in one submission:

| Task | Encoder | Heads | Optimizer |
|------|---------|-------|-----------|
| 0 | crate_cnn | crate | adam |
| 1 | crate_cnn | crate | stiefel |
| 2 | cnn | mlp | adam |
| 3 | cnn | mlp | stiefel |

Each task gets one GPU and runs seeds 0–3 concurrently (`JAX_MEM_FRACTION=0.22` per seed).

## Humanoid Goal Findings (2026-05-03)

The humanoid goal is fixed at distance `2.5` and angle `45` degrees. With
`action_repeat=4`, Brax's default `episode_length=1000` corresponds to:

```text
250 agent action steps = 1000 simulator/physics steps
```

The PPO rollout fragment length is separate. Current sweep launchers use:

```text
--num-steps 10
```

This means PPO updates happen every 10 agent steps per env, but episodes continue
across update boundaries until 250 agent steps, unless unhealthy termination fires.

### Failed / Weak Humanoid Setup

The first 64-run humanoid reward sweep did not use unhealthy termination. Many
policies exploited bad states near or below the floor. Metrics such as negative
`env/torso_z_mean` made those runs unusable for selecting a reward.

### Termination Sweep

Follow-up job:

```text
74540
```

Launcher:

```text
scripts/goal/humanoid_goal_cratecnn_reward_sweep_32_term_seed0_stiefel.sh
```

Fixed settings:

```text
--goal-progress-reward-scale 20
--goal-healthy-reward 5
--goal-success-easy-reward 10
--goal-success-reward 50
--goal-gate-progress-by-standing
--goal-terminate-when-unhealthy
--action-repeat 4
--backend spring
--encoder-type crate_cnn
--use-crate-head
--heads-optimizer stiefel
```

Sweep axes:

```text
standing_scale in {0, 2, 4, 8}
heading_scale  in {0.5, 1}
distance_scale in {0.1, 0.25, 0.5, 0.75}
```

Finding:

- `--goal-terminate-when-unhealthy` fixed the bad floor/fall-through failure mode.
- Better runs stayed upright, with `torso_z_mean ~= 1.2` and
  `stand_gate_mean ~= 0.8-0.9`.
- Goal reaching remained weak. At about 2.1M agent steps, the best runs still had
  `dist_mean ~= 2.43-2.44` for a goal initialized at distance `2.5`.
- Strict success remained `0.0`; easy success was only about `0.03-0.06`.
- The strongest configs often had `standing_scale=0`, suggesting explicit
  standing reward can overpay standing still once unhealthy termination and
  heading shaping are present.

Representative best run before cancelling:

```text
humanoid_goal__ppo_humanoid_goal_rs32term_stand0_heading0p5_dist0p25_cratecnn_stiefel_s0_goal2p5__0__1777772227
https://wandb.ai/rl-power/pixel-goal/runs/hdnofmyt

step ~= 2.17M
dist_mean ~= 2.442
success_easy_mean ~= 0.048
success_mean = 0.000
stand_gate_mean ~= 0.842
torso_z_mean ~= 1.210
torso_up_z_mean ~= 0.866
heading_mean ~= 0.847
```

Conclusion: termination plus posture/heading shaping stabilizes humanoid, but
the policy mostly learns to stand and face the target. The reward does not yet
pay strongly enough for goal-directed locomotion.

## Current Humanoid Sweep

Cancelled old job:

```text
74540
```

Launched new job:

```text
74630
```

Launcher:

```text
scripts/goal/humanoid_goal_cratecnn_reward_sweep_64_seed0_stiefel.sh
```

This 64-run sweep is based on the termination-sweep finding above. It keeps the
stable settings and increases goal-directed reward pressure.

Fixed settings:

```text
--goal-healthy-reward 2
--goal-success-reward 50
--goal-gate-progress-by-standing
--goal-terminate-when-unhealthy
--action-repeat 4
--backend spring
--encoder-type crate_cnn
--use-crate-head
--heads-optimizer stiefel
--save-rollout-gif
--rollout-gif-step 1000000
--rollout-gif-steps 250
```

Sweep axes:

```text
progress_scale      in {50, 100, 200, 400}
success_easy_reward in {10, 25}
distance_scale      in {0.25, 0.5}
heading_scale       in {0.5, 1}
standing_scale      in {0, 2}
```

Resource layout:

```text
8 Slurm array tasks
8 concurrent runs per GPU
64 configs total
JAX_MEM_FRACTION=0.11
```

Selection criteria:

```text
Hard posture filters:
  stand_gate_mean > 0.75
  torso_z_mean > 1.0
  torso_up_z_mean > 0.75

Then rank by:
  lower env/dist_mean
  higher env/success_easy_mean
  nonzero env/success_mean if it appears

Reject by GIF if the policy only stands, rotates, or exploits physics.
```

### 64-Run Progress/Easy-Success Sweep Result

Initial submission:

```text
74630
```

`74630` showed promising early metrics but hit Slurm node/launch failures and was
cancelled/replaced. It reached only about `0.36M-0.47M` steps, with all W&B runs
marked crashed. Logs did not show Python, CUDA, or OOM errors; `sacct` showed
repeated `NODE_FAIL` / launch-failure requeues.

Clean resubmission:

```text
74841
```

Final status:

```text
8/8 array tasks completed
exit code 0:0
final W&B step = 2,999,040
```

Artifact note:

```text
128 GIFs total
64 at step1000960
64 at step2000640
0 at step3000320
```

The final step stopped just below the next GIF trigger, so no 3M GIFs were
written. Contact sheets inspected:

```text
/tmp/humanoid_74841_1m_top_contact.jpg
/tmp/humanoid_74841_2m_top_contact.jpg
```

Visual read: top policies were physically sane, stayed above the floor, kept the
goal in view, and spent time near the ball. They did not show reliable strict
goal contact/reaching.

Final best-by-broad-near-goal score:

```text
progress100_easy25_dist0p5_heading1_stand0
W&B: https://wandb.ai/rl-power/pixel-goal/runs/2i8835gy

dist_mean = 1.956
success_easy_mean = 0.881
success_mean = 0.000
stand_gate_mean = 0.786
torso_z_mean = 1.141
torso_up_z_mean = 0.938
heading_mean = 0.801
avg episodic return = 11873.6
```

Final best strict-success config:

```text
progress400_easy25_dist0p5_heading0p5_stand2
W&B: https://wandb.ai/rl-power/pixel-goal/runs/fj8dfjvk

dist_mean = 1.804
success_easy_mean = 0.638
success_mean = 0.008
stand_gate_mean = 0.872
torso_z_mean = 1.209
torso_up_z_mean = 0.908
heading_mean = 0.845
avg episodic return = 4126.4
```

Final best distance among top configs:

```text
progress400_easy10_dist0p5_heading0p5_stand0
W&B: https://wandb.ai/rl-power/pixel-goal/runs/4fl7gxxe

dist_mean = 1.771
success_easy_mean = 0.668
success_mean = 0.000
stand_gate_mean = 0.876
torso_z_mean = 1.219
torso_up_z_mean = 0.920
heading_mean = 0.733
avg episodic return = 1592.2
```

Important metric definitions:

```text
success_easy = 1 if dist < 2.0 else 0
success      = 1 if dist < 0.5 else 0
```

Interpretation:

- This sweep is the best humanoid goal setup so far.
- It reliably learns upright near-goal behavior.
- Broad easy-success can get very high (`~0.88`), meaning the humanoid often
  enters the `dist < 2.0` region.
- Strict success remains rare; the best strict-success config only reached
  `success_mean = 0.008`.
- More broad proximity reward is unlikely to solve the last gap.

Next reward direction:

- Keep `terminate_when_unhealthy`, `action_repeat=4`, `backend=spring`,
  `crate_cnn + crate heads + stiefel`.
- Use the best current family as the base, especially:
  `progress400_easy25_dist0p5_heading0p5_stand2`.
- Add a sharper close-range/touch reward around the final radius, e.g. reward
  inside `dist < 1.0` and/or a stronger bonus/ramp as `dist -> 0.5`.
- Consider sweeping strict-success reward and a close-range shaping term rather
  than increasing `success_easy_reward`.

## Humanoid goal close-range reward sweep

Launcher:

```text
scripts/goal/humanoid_goal_close_reward_sweep_16_seed0_stiefel.sh
```

Slurm job:

```text
74988
```

Final status:

```text
8/8 array tasks completed
exit code 0:0
16/16 W&B runs finished
64 GIFs total
16 final GIFs
wall time ~= 1h55m
```

This sweep tested a close-range reward that only activates inside a small radius:

```python
close_fraction = clip((close_radius - dist) / (close_radius - 0.5), 0, 1)
close_reward = close_reward_scale * close_fraction
```

It included one baseline control and 15 close-reward configs.

Best distance config:

```text
close_r0p75_c25_s250_stand0
W&B: https://wandb.ai/rl-power/pixel-goal/runs/j3kdkw6q

progress_reward_scale = 400
success_reward = 250
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 0
healthy_reward = 2
close_reward_scale = 25
close_reward_radius = 0.75
terminate_when_unhealthy = true
gate_progress_by_standing = true

dist_mean = 1.800
success_easy_mean = 0.679
success_mean = 0.000
stand_gate_mean = 0.787
torso_z_mean = 1.187
torso_up_z_mean = 0.850
avg episodic return = 3754.2
```

Control from this sweep:

```text
baseline_current_best
W&B: https://wandb.ai/rl-power/pixel-goal/runs/f20hncq0

progress_reward_scale = 400
success_reward = 50
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 2
healthy_reward = 2
close_reward_scale = 0
close_reward_radius = 1.0
terminate_when_unhealthy = true
gate_progress_by_standing = true

dist_mean = 1.948
success_easy_mean = 0.626
success_mean = 0.000
stand_gate_mean = 0.860
torso_z_mean = 1.214
torso_up_z_mean = 0.899
avg episodic return = 3932.6
```

Interpretation:

- Close shaping improved distance versus the baseline control, but still did
  not produce strict contact.
- The best configs used `standing_reward_scale = 0`; explicit standing reward
  preserved posture but hurt approach distance in this sweep.
- The close reward was often too late because policies rarely entered the
  small close-radius region.

## Humanoid goal exponential distance-potential sweep

Launcher:

```text
scripts/goal/humanoid_goal_exp_distance_sweep_16_seed0_stiefel.sh
```

Slurm job:

```text
75098
```

Final status:

```text
8/8 array tasks completed
exit code 0:0
16/16 W&B runs finished
64 GIFs total
16 final GIFs
wall time ~= 1h55m
```

This sweep tested potential-style exponential distance shaping:

```python
prev_phi = exp(-prev_dist / temperature)
phi = exp(-dist / temperature)
reward_exp_distance = scale * (phi - prev_phi)
```

Shared config for the exp-distance sweep:

```text
env_name = humanoid_goal
backend = spring
action_repeat = 4
total_timesteps = 3,100,000
n_envs = 128
num_steps = 10
encoder_type = crate_cnn
use_crate_head = true
heads_optimizer = stiefel

progress_reward_scale = 400
success_reward = 250
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 0
healthy_reward = 2
close_reward_scale = 0
terminate_when_unhealthy = true
gate_progress_by_standing = true
```

Best run:

```text
exp_s100_t0p5
W&B: https://wandb.ai/rl-power/pixel-goal/runs/kttpkg56

exp_distance_reward_scale = 100
exp_distance_reward_temperature = 0.5

progress_reward_scale = 400
success_reward = 250
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 0
healthy_reward = 2
close_reward_scale = 0
terminate_when_unhealthy = true
gate_progress_by_standing = true

success_mean = 0.0609
dist_mean = 1.574
success_easy_mean = 0.663
stand_gate_mean = 0.826
torso_z_mean = 1.193
torso_up_z_mean = 0.885
reward_exp_distance_mean = 0.191
avg episodic return = 6658.1
SPS = 459
```

Top comparison runs:

```text
exp_s25_t1p5
W&B: https://wandb.ai/rl-power/pixel-goal/runs/3ythsxfu
success_mean = 0.000
dist_mean = 1.787
success_easy_mean = 0.730
avg episodic return = 3156.2

exp_s50_t2p0
W&B: https://wandb.ai/rl-power/pixel-goal/runs/zimp9mi6
success_mean = 0.000
dist_mean = 1.799
success_easy_mean = 0.636
avg episodic return = 3564.8

control_close_r0p75_c25_s250_stand0
W&B: https://wandb.ai/rl-power/pixel-goal/runs/3ucbhhjw
success_mean = 0.000
dist_mean = 1.913
success_easy_mean = 0.639
avg episodic return = 3186.1
```

Interpretation:

- Exponential distance-potential shaping is the best humanoid goal reward tried
  so far.
- `exp_s100_t0p5` is the first setup with nontrivial strict success
  (`success_mean = 0.0609`).
- It clearly beats the close-reward control on strict success, distance, and
  return.
- Bigger scale was not automatically better. Some `scale=200` configs achieved
  high `success_easy` and return but did not produce strict contact.
- The best temperature was relatively sharp (`temperature = 0.5`), suggesting
  the final approach needs a stronger local progress signal.

Next reward direction:

- Use `exp_s100_t0p5` as the new humanoid goal baseline.
- Sweep around the current best:

```text
exp_distance_reward_scale in {75, 100, 125, 150}
exp_distance_reward_temperature in {0.35, 0.5, 0.65, 0.8}
```

- Keep the rest of the reward/architecture fixed unless visual inspection of
  final GIFs shows a clear posture or contact exploit.

## Humanoid goal exponential distance refinement sweep

Launcher:

```text
scripts/goal/humanoid_goal_exp_distance_refine_16_seed0_stiefel.sh
```

Slurm job:

```text
75205
```

Final status:

```text
8/8 array tasks completed
exit code 0:0
16/16 W&B runs finished
64 GIFs total
16 final GIFs
wall time ~= 1h55m
```

This was a focused 4x4 refinement around the prior best
`exp_s100_t0p5`:

```text
exp_distance_reward_scale in {75, 100, 125, 150}
exp_distance_reward_temperature in {0.35, 0.5, 0.65, 0.8}
```

Shared config:

```text
env_name = humanoid_goal
backend = spring
action_repeat = 4
total_timesteps = 3,100,000
n_envs = 128
num_steps = 10
encoder_type = crate_cnn
use_crate_head = true
heads_optimizer = stiefel

progress_reward_scale = 400
success_reward = 250
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 0
healthy_reward = 2
close_reward_scale = 0
terminate_when_unhealthy = true
gate_progress_by_standing = true
```

Best run:

```text
exp_s125_t0p35
W&B: https://wandb.ai/rl-power/pixel-goal/runs/0nbs31wq

exp_distance_reward_scale = 125
exp_distance_reward_temperature = 0.35

progress_reward_scale = 400
success_reward = 250
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 0
healthy_reward = 2
close_reward_scale = 0
terminate_when_unhealthy = true
gate_progress_by_standing = true

success_mean = 0.1992
dist_mean = 1.482
success_easy_mean = 0.628
stand_gate_mean = 0.845
torso_z_mean = 1.199
torso_up_z_mean = 0.933
reward_exp_distance_mean = 0.028
avg episodic return = 13826.0
SPS = 456
```

Top comparison runs:

```text
exp_s75_t0p5
W&B: https://wandb.ai/rl-power/pixel-goal/runs/cuya660g
success_mean = 0.000
dist_mean = 1.818
success_easy_mean = 0.687
avg episodic return = 2878.5

exp_s100_t0p8
W&B: https://wandb.ai/rl-power/pixel-goal/runs/3ntibgnd
success_mean = 0.000
dist_mean = 1.869
success_easy_mean = 0.666
avg episodic return = 3846.8

exp_s150_t0p5
W&B: https://wandb.ai/rl-power/pixel-goal/runs/3bmztykd
success_mean = 0.000
dist_mean = 1.878
success_easy_mean = 0.845
avg episodic return = 6718.5

exp_s100_t0p5
W&B: https://wandb.ai/rl-power/pixel-goal/runs/ee1v8d1l
success_mean = 0.000
dist_mean = 2.007
success_easy_mean = 0.548
avg episodic return = 3150.2
```

Interpretation:

- `exp_s125_t0p35` is now the best humanoid goal reward found.
- It improved strict success from the previous best `0.0609` to `0.1992`.
- It also improved distance from `1.574` to `1.482`.
- The result is not explained by broad easy-success alone: some other configs
  had higher `success_easy` or high return but zero strict success.
- The best setting is sharper and stronger than the prior best, suggesting that
  humanoid goal needs a concentrated final-approach progress signal.

Locked reward for final multi-seed validation:

```text
exp_distance_reward_scale = 125
exp_distance_reward_temperature = 0.35

progress_reward_scale = 400
success_reward = 250
success_easy_reward = 25
distance_reward_scale = 0.5
heading_reward_scale = 0.5
standing_reward_scale = 0
healthy_reward = 2
close_reward_scale = 0
terminate_when_unhealthy = true
gate_progress_by_standing = true
```

Final validation plan:

```text
crate_cnn encoder + crate heads + adam,    6 seeds
crate_cnn encoder + crate heads + stiefel, 6 seeds
cnn encoder       + mlp heads   + adam,    6 seeds
cnn encoder       + mlp heads   + stiefel, 6 seeds
```

Use `10,000,000` total timesteps for validation. At the observed throughput
(`~456-461 SPS/run`) and 2 runs per GPU, budget roughly `6.5h` per 2-run GPU
batch, or about `13h` total wall time for 24 runs on 8 GPUs in two waves.
