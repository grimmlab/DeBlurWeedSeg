import subprocess
from sklearn.model_selection import ParameterGrid
from scipy.stats import loguniform
import numpy as np

if __name__ == "__main__":
    training_strategy = "combined"
    seed = 42
    encoder_name = "resnet18"
    lower_lr = 1e-4
    upper_lr = 1e-3
    batch_sizes = [768, 640, 512, 384, 256, 128]
    np.random.seed(seed)
    learning_rates = loguniform.rvs(lower_lr, upper_lr, size=10)
    param_grid = {'learning_rate': learning_rates, 'batch_size': batch_sizes}
    parameters = list(ParameterGrid(param_grid))
    for idx, val in enumerate(parameters):
        print(f"starting run {idx}/{len(parameters)} with {encoder_name}")
        subprocess.run(
            f"python3 train.py --encoder_name {encoder_name} --learning_rate {val['learning_rate']} --batch_size {val['batch_size']} --training_strategy {training_strategy}",
            shell=True, check=True)
