
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p pixelenvs/pixelrl

cd /home/ubuntu/pixelenvs/pixelrl
git clone https://github.com/saketirl/pixelrl.git .
git submodule update --init --recursive

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

export PYTHONPATH="/home/saket/pixelenvs/pixelrl/pixelbrax/brax:${PYTHONPATH}"

uv run python -m wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

export ENV_NAME=halfcheetah
export SEED=1


uv run ppo_pixelbrax_jax2.py \
  --env-name ${ENV_NAME} \
  --backend spring \
  --n-envs 128 \
  --hw 84 \
  --total-timesteps 10000000 \
  --num-steps 10 \
  --num-minibatches 32 \
  --update-epochs 4 \
  --learning-rate 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.1 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --seed ${SEED} \
  --log-interval 1 \
  --frame-stack 4 \
  --action-repeat 4 \
  --anneal-lr \
  --track

