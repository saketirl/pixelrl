# Continual Learning Notes

## 2026-04-20: Brax HalfCheetah constant-LR ReLU collapse

Experiment: state-observation Brax HalfCheetah continual dynamics, using
`scripts/CL/brax/continual_halfcheetah.sh` and `ppo_brax.py`.

Setup:
- Environment: `halfcheetah`
- Backend: `spring`
- Seed: `0`
- Total timesteps: `49,984,000`
- Task schedule: `25` tasks, `1,999,360` env steps per task
- PPO rollout shape: `n_envs=128`, `num_steps=10`, so switches are aligned to rollout boundaries
- Optimizer: Adam
- Learning rate: constant `3e-4`
- LR annealing: disabled; launcher no longer passes `--anneal-lr`
- Action repeat: `4`
- PPO args: `num_minibatches=32`, `update_epochs=4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.1`, `ent_coef=0.0`, `vf_coef=0.5`, `max_grad_norm=0.05`
- Actor/critic hidden activation is the only intended comparison: `swish` vs `relu`
- Shared trunk in `ppo_brax.py` still uses swish; the activation flag only changes actor and critic hidden layers.

Dynamics config:
- Base config: `configs/continual/halfcheetah_dynamics.yaml`
- Run configs:
  - ReLU: `slurm_logs/continual_brax_halfcheetah_25tasks_68943.yaml`
  - Swish: `slurm_logs/continual_brax_halfcheetah_25tasks_68944.yaml`
- Uses `halfcheetah_structured_asymmetric` task sampler.
- Enabled perturbations include slide friction, actuator gear, link mass, gravity, and limb length.
- Task switches are logged as `continual_switch ... values=...`; the ReLU and swish runs use the same task sequence because seed and schedule are identical.

Submitted jobs:
- ReLU actor/critic: job `68943`
  - Command: `sbatch --parsable scripts/CL/brax/continual_halfcheetah.sh continual_brax rl-power 1999360 25 "" 0 relu`
  - Exp name: `ppo_brax_cl_halfcheetah_25tasks_relu_constantlr_s0`
  - Log: `slurm_logs/cl_brax_halfcheetah_68943.out`
- Swish actor/critic: job `68944`
  - Command: `sbatch --parsable scripts/CL/brax/continual_halfcheetah.sh continual_brax rl-power 1999360 25 "" 0 swish`
  - Exp name: `ppo_brax_cl_halfcheetah_25tasks_swish_constantlr_s0`
  - Log: `slurm_logs/cl_brax_halfcheetah_68944.out`

Observed result:
- ReLU actor/critic collapses by the end of the run. Final logged returns are roughly `7.5` to `8.0` with episode length `1000`.
- Swish actor/critic remains non-collapsed. Final logged returns are roughly `904` to `912` with episode length `1000`.
- Both jobs completed successfully:
  - `68943`: `COMPLETED`, elapsed about `02:01:19`
  - `68944`: `COMPLETED`, elapsed about `02:01:29`
- Final SPS is similar: ReLU about `6915`, swish about `6903`, so the collapse does not look like a runtime or scheduling artifact.

Interpretation / next checks:
- This is strong evidence that ReLU actor/critic heads are unstable or much less plastic under the HalfCheetah continual dynamics setup.
- Because LR annealing was disabled for both runs, the collapse is not explained by global LR decay.
- Since task schedule, seed, PPO settings, and dynamics config match, the key controlled difference is actor/critic hidden activation.
- A useful follow-up is to inspect when the ReLU collapse begins relative to `continual_switch` task indices, then compare task-local adaptation curves against swish.

## 2026-04-25: SlipperyAnt throughput winner

Experiment: state-observation SlipperyAnt Brax wrapper, using
`scripts/CL/brax/slippery_ant_wrapper_lr1e4.sh` / `ppo_slippery_brax.py`.

Goal:
- Maximize GPU throughput while still reaching roughly `2000` episodic return by about `5M` global steps.

Current winning config:
- Optimizer: Adam
- Actor/critic activation: `swish`
- `N_ENVS=128`
- `NUM_STEPS=10`
- `NUM_MINIBATCHES=32`
- `UPDATE_EPOCHS=4`
- Batch size: `1280`
- Minibatch size: `40`
- Learning rate: `1e-4`
- Action repeat: `4`

Observed at about `5M` steps in throughput sweep group
`slippery_ant_throughput_swish_adam_stiefel_71774`:
- SPS: about `12,890`
- Episodic return: about `2,343`

Interpretation:
- This baseline-shape Adam/swish config is the fastest tested config that still clears the `~2000` return target by `5M`.
- Larger vectorized Adam/swish configs were faster but failed the performance gate:
  - `256 envs x 10 steps`: about `22,773` SPS, return about `1,278`
  - `256 envs x 20 steps`: about `26,572` SPS, return about `-37`
  - `256 envs x 40 steps`: about `28,870` SPS, return about `-85`
- The high-throughput `256 x 40` Adam/relu hyperparameter sweep did not recover learning; tested LR/epoch variants stayed strongly negative by `5M`.
- Multi-process GPU packing fits in memory but does not improve aggregate SPS; it mainly trades per-run speed for breadth.

Recommended command:

```bash
ACTOR_CRITIC_ACTIVATION=swish HEADS_OPTIMIZER=adam \
N_ENVS=128 NUM_STEPS=10 NUM_MINIBATCHES=32 UPDATE_EPOCHS=4 \
sbatch scripts/CL/brax/slippery_ant_wrapper_lr1e4.sh
```

## 2026-04-26: PixelBrax SlipperyAnt first batch plan

Goal:
- Replicate the state-observation SlipperyAnt setup in PixelBrax without wasting the first full GPU on eight identical baseline seeds.
- Use the same slippery schedule semantics as `scripts/CL/brax/slippery_ant_wrapper_lr1e4.sh`:
  - `PHASE_EVERY_ENV_STEPS=4999936`
  - `NUM_PHASES=20`
  - `N_ENVS=128`
  - `NUM_STEPS=10`
  - `ACTION_REPEAT=4`
  - `SLIPPERY_CHANGE_EVERY=(PHASE_EVERY_ENV_STEPS / N_ENVS) * ACTION_REPEAT = 156248`

Implementation status:
- `ppo_pixelbrax.py` now has optional `--slippery-ant` support using the CSV schedule from `configs/continual/slippery_ant_wrapper.py`.
- New PixelBrax launchers live in `scripts/CL/pixelbrax/`.
- `--debug-repr` should not be used for packed training runs; it caused large extra allocations and made pack-2 fail artificially.

Packing probe:
- Short probes used `total_timesteps=12800` per child, so SPS is compile-dominated.
- Pack 2 without debug metrics completed: per-run SPS about `107-108`.
- Pack 4 without debug metrics completed: per-run SPS about `61-65`.
- Pack 8 without debug metrics completed: per-run SPS about `43-47`.
- Pack 12 failed during GPU library initialization with cuSolver/cuBLAS allocation errors.
- Practical setting for the current default PixelBrax config: `RUNS_PER_GPU=8`.

Recommended first 8-run batch:
- Run one GPU with 8 packed jobs for a clean plain-head Adam LR comparison.
- Proposed jobs:
  - plain-head Adam, LR `1e-4`, seeds `0,1,2,3`
  - plain-head Adam, LR `3e-4`, seeds `0,1,2,3`

Interpretation:
- This directly validates the PixelBrax analogue of the state-Brax winner while giving a fair LR comparison.
- It avoids optimizer/head-architecture confounds in the first batch; test Stiefel and CRATE heads only after seeing which Adam LR is alive.
