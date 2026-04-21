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
