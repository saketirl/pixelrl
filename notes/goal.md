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
