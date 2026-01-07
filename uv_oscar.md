module load python/3.9

uv init

uv add "jax[cuda12_pip]==0.4.30"

uv add flax==0.8.2 \
  optax==0.2.2 \
  chex==0.1.90 \
  wandb==0.17.2 \
  tyro \
  numpy \
  scipy \
  pillow \
  imageio \
  tqdm

uv add mujoco==3.2.6 mujoco-mjx==3.2.6

export PYTHONPATH="/users/apraka15/arjun/pixelrl/pixelbrax/brax:${PYTHONPATH}"

uv pip install -e .

uv add wandb==0.23.1
uv add jaxopt
uv add jaxtyping
