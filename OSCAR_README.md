# PixelBrax + Pixel DDPG (JAX)

This repository runs **pixel-based continuous control experiments** using the
[`trevormcinroe/pixelbrax`](https://github.com/trevormcinroe/pixelbrax) codebase,
with **DDPG implemented in JAX/Flax** and **pixel observations** coming from
`PixelEnv` (`state.pixels`, not `state.obs`).

The setup uses the **Brax source code bundled inside PixelBrax**, not a pip-installed
Brax package.

---

## 1. Create the Conda Environment

```bash
conda create -n pixelbrax python=3.9 -y
conda activate pixelbrax
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install Jax
```
pip install "jax[cuda12_pip]==0.4.30" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Check the jax installation
 ```
python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
PY
```

## 3. Install all dependencies:
```
pip install \
  flax==0.8.2 \
  optax==0.2.2 \
  chex==0.1.90 \
  wandb==0.13.11 \
  tyro \
  numpy \
  scipy \
  pillow \
  imageio \
  tqdm
```

## 4. Install mujoco:
```
pip install mujoco==3.2.6 mujoco-mjx==3.2.6

python - <<'PY'
import mujoco
import mujoco_mjx
print("mujoco:", mujoco.__version__)
print("mujoco_mjx:", mujoco_mjx.__version__)
PY
```

## 5. Go to the pixelbrax code folder and install from source:
```
cd /users/<username>/data/<username>/pixelenvs/pixelbrax
pip install -e .
```
Include the following in the PYTHONPATH to point to brax source:
```
export PYTHONPATH="/users/<username>/data/<username>/pixelenvs/pixelbrax/pixelbrax/brax:${PYTHONPATH}"
```

Test brax works:
```
python - <<'PY'
import brax, inspect
print(inspect.getfile(brax))
PY
```

## 6. Run a test that your code works:
```
python ddpg_pixelbrax_jax.py \
  --env-name walker2d \
  --n-envs 16 \
  --total-timesteps 10000
```
