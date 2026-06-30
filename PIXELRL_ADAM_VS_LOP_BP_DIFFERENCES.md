# PixelRL Adam Baseline vs `rlopt_lop` BP Slippery Ant

This note compares one effective `rlopt_lop` run launched by
`rlopt_lop/scripts/slippery_ant_bp_l40s.sh` against one Adam-only PixelRL run
from `pixelrl/scripts/CL/brax/humanoid_sweep/ant_lop_final_adam_vs_stiefel_6seeds.sh`.

Seed identity and seed count are intentionally out of scope. The PixelRL
comparison is only the Adam branch where `RUN_KIND=adam`,
`HEADS_OPTIMIZER=adam`, and `BASE_OPTIMIZER=adam`. Stiefel settings passed by
the PixelRL script are inactive in this branch.

## Effective Invocation Profiles

`rlopt_lop` BP launcher:

```bash
uv run python -m rlopt_lop.ppo_nonstationary_bp \
  --env slippery_ant \
  --total_steps 20000000 \
  --change_every 2000000 \
  --num_steps 2048 \
  --num_minibatches 16 \
  --update_epochs 10 \
  --lr 1e-4 \
  --lambda0 0.95 \
  --vf_coeff 1.0 \
  --hidden_sizes 256 256 \
  --track
```

PixelRL Adam branch:

```bash
uv run python ppo_brax.py \
  --env-name ant \
  --backend positional \
  --n-envs 1 \
  --total-timesteps 20000000 \
  --learning-rate 1e-4 \
  --adam-eps 1e-8 \
  --base-optimizer adam \
  --weight-decay 0.0 \
  --num-steps 2048 \
  --num-minibatches 128 \
  --update-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --ent-coef 0.01 \
  --vf-coef 1.0 \
  --max-grad-norm 1000000000 \
  --network-arch lop \
  --heads-optimizer adam \
  --slippery-ant \
  --slippery-change-every 2000000 \
  --actor-mean-tanh \
  --actor-mean-scale 1.0 \
  --bounded-global-logstd \
  --actor-logstd-init -2.0 \
  --actor-logstd-min -5.0 \
  --actor-logstd-max -1.4 \
  --track
```

## Top-Level Config Differences

| Setting | `rlopt_lop` BP | PixelRL Adam branch | Difference |
| --- | --- | --- | --- |
| Environment entry | `--env slippery_ant` | `--env-name ant --slippery-ant` | Both target Slippery Ant, but through different wrappers. |
| Backend | implicit wrapper default `positional` | `--backend positional` | Same effective backend; PixelRL exposes it in the launcher. |
| Total env steps | `20,000,000` | `20,000,000` | Same. |
| Friction change cadence | `2,000,000` | `2,000,000` | Same nominal cadence, different wrapper semantics below. |
| Rollout length | `2048` | `2048` | Same. |
| Parallel envs | `1` by config default | `1` | Same. |
| PPO epochs | `10` | `10` | Same. |
| Minibatches | `16` | `128` | PixelRL makes much smaller minibatches. |
| Minibatch size | `2048 / 16 = 128` | `2048 / 128 = 16` | Active optimizer batch size differs by 8x. |
| Learning rate | `1e-4` | `1e-4` | Same. |
| Adam epsilon | `1e-8` | `1e-8` | Same. |
| Gradient clipping | `1e9` | `1e9` | Same effective global clipping threshold. |
| Weight decay | `0.0` | `0.0` | Same effective value. |
| Reward normalization | disabled | enabled by `ppo_brax.py` default | PixelRL script tags say `no_reward_norm`, but the command does not disable the default. |
| Observation normalization | disabled / unused | disabled by default | Same effective behavior. |

## Network and Policy Parameterization

`rlopt_lop` uses `ActorCritic` from `rlopt_lop/models.py` with
`--hidden_sizes 256 256`. The actor and critic are separate two-hidden-layer
MLPs with LeCun-uniform kernels, zero biases, ReLU activations, and a global
unconstrained `log_std` parameter initialized to zero.

PixelRL uses `--network-arch lop`, which selects an identity trunk plus
`LopActor` and `LopCritic` in `ppo_brax.py`. Those heads are shallow: each has
one hidden layer of width 256 before the actor mean or critic value output.
They also use LeCun-uniform kernels, zero biases, and ReLU activation, but the
policy distribution is parameterized differently:

- The actor mean is bounded as `tanh(mean) * actor_mean_scale`.
- The global log standard deviation is bounded through a sigmoid parameter.
- The effective initial log standard deviation is `-2.0`.
- The effective log standard deviation range is `[-5.0, -1.4]`.

This means the PixelRL Adam baseline is not just an Adam optimizer variant of
the `rlopt_lop` BP model. It has a shallower network and a more constrained
continuous-action policy.

## PPO Objective and Update Differences

The `slippery_ant_bp_l40s.sh` command does not pass `--ppo_objective`, so
`rlopt_lop` uses its default `ppo_objective=lop`:

- value loss is plain mean squared error against the GAE target;
- advantages are normalized once at rollout level before minibatching;
- entropy is set to zero in the loss path;
- `entropy_coeff` remains at its default `0.0`;
- `vf_coeff=1.0` multiplies the full value MSE.

PixelRL's Adam branch uses the CleanRL-style PPO objective implemented in
`ppo_brax.py`:

- advantages are normalized inside each minibatch loss when `norm_adv=True`;
- value loss is clipped against the rollout-time value prediction;
- the value loss includes a `0.5` multiplier before `vf_coef`;
- entropy is included with `--ent-coef 0.01`;
- the actor loss uses the same clipped PPO ratio structure.

The two runs therefore apply different optimization pressure even with the same
learning rate, rollout length, update epochs, gamma, GAE lambda, clip epsilon,
and nominal `vf_coeff`.

## Action and Environment Handling

`rlopt_lop` samples directly from the distrax policy and passes the action into
its nonstationary Brax wrapper. The `slippery_ant` path does not explicitly
clip actions in the training loop or in `load_nonstationary_env`.

PixelRL samples from a distrax Gaussian, computes the log probability of that
sample, then clips the sampled action to `[-max_action, max_action]` before
stepping the environment. Because log probability is computed before clipping,
the stored PPO log probability corresponds to the unclipped sampled action.
PixelRL also bounds the actor mean before sampling, which further constrains
the action distribution.

## Friction Schedule Differences

The two Slippery Ant wrappers are not equivalent.

`rlopt_lop` loads a pickled schedule from `lop-jax/rlopt/frictions` through
`rlopt_lop/slippery_schedule.py`. Its `LOPNonstationaryFrictionBraxWrapper`
sets the initial friction to the first loaded schedule value at reset. It then
advances the friction index only when an episode is done and the elapsed
timestep difference is greater than `change_every`.

PixelRL loads a CSV schedule from `pixelrl/configs/continual/frictions.csv`.
Its `NonstationaryFrictionBraxWrapper` keeps Brax's default friction in phase
0, then switches to the CSV row values according to
`state.info["timestep"] // change_every`. The phase computation is tied to the
Brax timestep counter rather than only to episode completion.

So even with the same nominal `change_every=2_000_000`, the initial friction,
schedule source format, and phase-transition condition can differ.

## Optimizer Plumbing

Both effective runs use Adam with LR `1e-4`, epsilon `1e-8`, beta values
equivalent to Adam defaults, and no effective weight decay.

`rlopt_lop` applies one optimizer chain to the combined `ActorCritic`
parameter tree. It uses `adam_with_param_counts`, whose Adam state stores a
count per parameter leaf.

PixelRL builds a multi-transform optimizer because the same script supports
Stiefel heads. In the Adam branch, all active actor and critic labels resolve
to Adam transforms, so the Stiefel transforms are present in the code but not
selected for the run. The optimizer state is still structured around separate
`network`, `actor`, and `critic` parameter collections.

## Logging and Outputs

`rlopt_lop` JITs chunks of updates grouped by the friction-change schedule,
logs return and representation metrics such as rank, effective rank, approximate
rank, and dead neurons, and saves an Orbax checkpoint-style result tree.

PixelRL runs a Python loop over PPO updates, logs W&B scalar metrics such as
episodic return, losses, entropy, approximate KL, SPS, friction phase, and
matrix-constraint diagnostics. It does not save an equivalent Orbax result tree
in this entrypoint.

## Not Compared

- Seed values, seed counts, or seed-to-schedule choices.
- Stiefel optimizer behavior. The PixelRL Adam branch passes Stiefel-related
  arguments, but `heads_optimizer=adam` makes them inactive.
- Empirical performance or W&B result differences.
