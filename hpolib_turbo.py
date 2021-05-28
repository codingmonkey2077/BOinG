import os
import sys

print(__doc__)

from ConfigSpace import Configuration


from functools import partial

from smac.epm.util_funcs import get_types

from hpobench_list import benchmarks
from hpobench.benchmarks.rl.cartpole import CartpoleReduced as CartPole
from lamcts import MCTS

import time

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="turbo_opt")
    parser.add_argument('--problem_name', type=str, default='adult')
    parser.add_argument('--num_run', type=int, default=100)
    parser.add_argument('--n_configs_x_params', type=int, default=2)
    parser.add_argument('--rng', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='la_mcts')
    return parser.parse_args()


from turbo import Turbo1
import numpy as np
from pathlib import Path
import json

init_dict = {'lunar': 50,
             'robot': 100,
             'rover': 200}


def smac_optimize(problem_name,
                  num_run,
                  n_configs_x_params,
                  rng,
                  model_type
                  ):

    times = []
    output_dir_base = str(Path.cwd() / 'HPObench' / problem_name / model_type)
    os.makedirs(output_dir_base, exist_ok=True)
    num_files = 0
    opt_file_name = os.path.join(output_dir_base, 'optim_data_' + str(rng) + '_' + str(num_files) + '.json')

    x_opt = []
    y_opt =[]

    y_min = [np.inf]
    def optimization_function_wrapper(b, array, y_min):

        time0 = time.time()
        """ Helper-function: simple wrapper to use the benchmark with smac"""
        # New API ---- Use this
        cs = b.get_configuration_space()
        cfg = Configuration(configuration_space=cs, vector=array)
        result_dict = b.objective_function(cfg,)
        y = -result_dict['function_value']
        print(f"current loss: {y}")
        time1 = time.time()
        times.append([time0, time1])

        if y < y_min[0]:
            x_opt.append(array.tolist())
            y_opt.append(y)

            #y_min[0] = y

            run_traj = {"x": x_opt,
                        'y': y_opt}

            with open(os.path.join(opt_file_name), 'w') as f:
                json.dump(run_traj, f)
        return y

    benchmark = benchmarks[problem_name](rng=rng)
    cs = benchmark.get_configuration_space()

    n_dims = len(cs.get_hyperparameters())

    types, bounds = get_types(cs)

    n_init = init_dict.get(problem_name, n_configs_x_params * n_dims)

    if model_type == 'turbo':
        turbo1 = Turbo1(
            f=partial(optimization_function_wrapper, benchmark, y_min=y_min),  # Handle to objective function
            lb=np.zeros(n_dims),  # Numpy array specifying lower bounds
            ub=np.ones(n_dims),  # Numpy array specifying upper bounds
            n_init=n_init,  # Number of initial bounds from an Latin hypercube design
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
    elif model_type == 'la_mcts':
        agent = MCTS(
            lb=np.zeros(n_dims),  # the lower bound of each problem dimensions
            ub=np.ones(n_dims),  # the upper bound of each problem dimensions
            dims=n_dims,  # the problem dimensions
            ninits=n_init,  # the number of random samples used in initializations
            func=partial(optimization_function_wrapper, benchmark, y_min=y_min),  # function object to be optimized
        )

        agent.search(iterations=num_run)

        X = [sample[0] for sample in agent.samples]  # Evaluated points
        fX = [sample[1] for sample in agent.samples]  # Observed values
        X = np.array(X)
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]
    else:
        raise ValueError("unsopported model")

    output_dir_base = str(Path.cwd() / 'HPObench' / problem_name / model_type)

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base, exist_ok=True)

    result = {'y': np.squeeze(fX).tolist(),
              'x': X.tolist(),
              'time': times}

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
