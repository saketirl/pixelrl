

module load anaconda/2023.09-0-7nso27y
module load cuda/12.2

conda activate pixelbrax

cd /users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/

export PYTHONPATH="/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax:${PYTHONPATH}"

wandb login 9fb4ba17a708de72496774b2e25d219f07de038d

