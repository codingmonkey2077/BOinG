import os
import sys
syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)
print(__doc__)


from skopt.benchmarks import branin, hart6
from ConfigSpace import ConfigurationSpace
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

    def func_wrapper(config, func, hist, y_list):
        x = np.asarray(list((config.get_dictionary().values())))
        y = func(x)
        hist.append(x.tolist())
        y_list.append(y)
        return y

    artifical_func = {'ackley': Ackley,
                      'branin': Branin,
                      'griewank': Griewank,
                      'levy': Levy,
                      'rosenbrock': Rosenbrock,
                      'michalewicz': Michalewicz}

    if func_name == 'hart6':
        func = hart6
        bound_configs = [[0., 1.]] * n_dims
    else:
        func = artifical_func[func_name](n_dims)
        bound_configs = np.vstack([func.lb, func.ub]).transpose().tolist()

    cs = ConfigurationSpace()
    cs.generate_all_continuous_from_bounds(bound_configs)

    samples =cs.sample_configuration(num_run)

    X = []
    fX = []

    for sample in samples:
        X.append(sample.get_array().tolist())
        fX.append(func(np.asarray(list((sample.get_dictionary().values())))))

    output_dir_base = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), 'rs')

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base, exist_ok=True)

    import json
    result = {'y': np.squeeze(fX).tolist(),
              'x': X}

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

