# CNN LeJEPA Humanoid Ablation

## Scope

- Environment: `humanoid`
- Encoder: `cnn`
- WandB project: `scott`
- Probe path: online probe via `linear_probe.py`
- Launcher: `scripts/cnn/cnn_lejepa_humanoid_ablation.sh`

## Submission

```bash
sbatch scripts/cnn/cnn_lejepa_humanoid_ablation.sh
```

Optional:

```bash
sbatch scripts/cnn/cnn_lejepa_humanoid_ablation.sh <wandb_entity> <total_timesteps> <muon|adam>
```

## Variant Matrix

| Variant | Purpose | Key settings |
| --- | --- | --- |
| `baseline` | Vanilla CNN PPO reference | no `--lejepa` |
| `legacy` | Reproduce current LeJEPA exactly | `aux_coef=1.0`, no warmup/ramp, `sigreg_mode=legacy`, `num_slices=64` |
| `auxcoef` | Add outer PPO-safe coefficient only | `aux_coef=1e-3`, no warmup/ramp, `sigreg_mode=legacy` |
| `ramp` | Add warmup and ramp | `aux_coef=1e-3`, `warmup=10`, `ramp=50`, `sigreg_mode=legacy` |
| `sigreg` | Switch to corrected SIGReg | `sigreg_mode=epps_pulley`, `num_slices=256` |
| `view_shift` | Safe-view control | corrected SIGReg, `view_mode=shift` |
| `view_shift_noise` | Mild view perturbation | corrected SIGReg, `view_mode=shift_noise` |
| `layernorm_off` | Projector normalization ablation | corrected SIGReg, `--no-lejepa-use-layernorm` |
| `sweep_aux_*` | Outer-coefficient sweep | `1e-4`, `1e-3`, `1e-2` |
| `sweep_lambda_*` | Internal LeJEPA mix sweep | `0.1`, `0.3`, `0.5` |
| `sweep_slices_*` | SIGReg projection sweep | `256`, `512` |

## Metrics To Compare Against Baseline

- `charts/avg_episodic_return`
- learning speed from return-vs-step curves
- `probe/r2_full_test`
- `probe/r2_at_state_dim`
- `probe/k_at_95pct_r2`
- `repr/*` health metrics
- `grads/encoder_ppo_norm`
- `grads/encoder_lejepa_norm`
- `grads/encoder_ppo_lejepa_cosine`

## Result Summary

| Variant | Return vs baseline | Probe vs baseline | Gradient conflict | Interpretation |
| --- | --- | --- | --- | --- |
| `baseline` | pending | pending | pending | reference |
| `legacy` | pending | pending | pending | pending |
| `auxcoef` | pending | pending | pending | pending |
| `ramp` | pending | pending | pending | pending |
| `sigreg` | pending | pending | pending | pending |
| `view_shift` | pending | pending | pending | pending |
| `view_shift_noise` | pending | pending | pending | pending |
| `layernorm_off` | pending | pending | pending | pending |

## Stopping Rule

If the PPO-safe path with small `lejepa_aux_coef`, warmup/ramp, and corrected SIGReg still underperforms vanilla CNN PPO consistently, stop using LeJEPA as a full PPO training auxiliary and pivot to:

1. probe/pretraining-only LeJEPA, or
2. a more directly control-aligned temporal consistency objective.
