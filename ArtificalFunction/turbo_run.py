import os
import sys
syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)
print(__doc__)


from skopt.benchmarks import branin, hart6

from ArtificalFunction.Benchmark import *

from turbo import Turbo1
import numpy as np


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="smac_optimize")
    parser.add_argument('--func_name', type=str, default='rosenbrock')
    parser.add_argument('--n_dims', type=int, default=6)
    parser.add_argument('--num_run', type=int, default=200)
    parser.add_argument('--n_configs_x_params', type=int, default=2)
    parser.add_argument('--rng', type=int, default=8)
    return parser.parse_args()

def smac_optimize(func_name,
                  n_dims,
                  num_run,
                  n_configs_x_params,
                  rng,
                  ):
    np.random.seed(rng)
    artifical_func = {'ackley': Ackley,
                      'branin': Branin,
                      'griewank': Griewank,
                      'levy': Levy,
                      'rosenbrock': Rosenbrock,
                      'michalewicz': Michalewicz}

    if func_name == 'hart6':
        func = hart6
        ub = np.ones(n_dims)
        lb = np.zeros(n_dims)
    else:
        func = artifical_func[func_name](n_dims)
        ub = func.ub
        lb = func.lb

    turbo1 = Turbo1(
        f=func,  # Handle to objective function
        lb=lb,  # Numpy array specifying lower bounds
        ub=ub,  # Numpy array specifying upper bounds
        n_init=n_configs_x_params * n_dims,  # Number of initial bounds from an Latin hypercube design
        max_evals=num_run,  # Maximum number of evaluations
        batch_size=1,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    turbo1.optimize()

    X = turbo1.X  # Evaluated points
    fX = turbo1.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    output_dir_base = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), 'turbo')

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base, exist_ok=True)

    import json
    result = {'y': np.squeeze(fX).tolist(),
              'x': X.tolist()}

    num_files = 0
    file_name = os.path.join(output_dir_base, 'data_' + str(rng) + '_' + str(num_files) + '.json')
    while os.path.exists(file_name):
        num_files += 1
        file_name = os.path.join(output_dir_base, 'data_' + str(rng) + '_' + str(num_files) + '.json')

    with open(os.path.join(file_name), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    args = vars(parse_args())
    hist = []
    y_list = []
    smac_optimize(**args)



