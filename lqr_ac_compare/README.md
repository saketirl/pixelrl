
# LQR Upstairs vs Downstairs Actor–Critic (d=512)

This package runs two learning setups over the **same underlying LQR dynamics**:

- **Upstairs**: observation-based actor–critic with **L=3 deep-linear** actor and critic, updated with **Cayley retraction** (keeps trainable square matrices orthogonal).
- **Downstairs**: state-based shallow actor–critic using low-dimensional effective matrices, updated using the **Theorem-2-style Stiefel-direction coefficients** with a small LR (default `1e-3`).

## Files

- `lqr_env.py` – environment with latent state `s_t` and high-dim observation `o_t = M s_t + noise`
- `upstairs_models.py` – upstairs DLN actor/critic (L=3)
- `upstairs_learning.py` – upstairs learning update (minibatch over timesteps for speed)
- `downstairs_models.py` – downstairs shallow actor/critic
- `downstairs_learning.py` – downstairs update (Theorem-2 coefficients)
- `run_experiment.py` – main runner (imports and executes both loops)

## Run

```bash
python run_experiment.py
```

Defaults are chosen to be runnable quickly. For a longer horizon:

```bash
python run_experiment.py --T 10 --dt 0.01 --horizon_steps 1000
```
