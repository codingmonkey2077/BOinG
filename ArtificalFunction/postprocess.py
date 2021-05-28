import os
import sys
import glob
import json
syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)
print(__doc__)

import numpy as np
np.random.seed(123)

from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel

from skopt.benchmarks import branin, hart6
import matplotlib.patches as patches

from ArtificalFunction.Benchmark import *

from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior

from smac_custom.acqusition_function.max_value_entropy_search import *
from smac.epm.util_funcs import get_types
import matplotlib.pyplot as plt
syspath = os.path.dirname((os.path.abspath(__file__)))

def plot_countour(func_name, samples, num_init, lb, ub, ss_region=None,title_supp='van'):
    fig, ax = plt.subplots()
    artifical_func = {'ackley': Ackley(2),
                      'levy': Levy(2),
                      'rosenbrock': Rosenbrock(2),
                      'griewank': Griewank(2)}

    x1_values = np.linspace(lb[0], ub[0], 100)
    x2_values = np.linspace(lb[1], ub[1], 100)

    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([func(val) for val in vals], (100, 100))

    cs = ax.contour(x1_values, x2_values, fx)

    if func_name == 'branin':
        minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    else:
        minima = np.array([func.global_optimum])

    if ss_region is not None:
        # Create a Rectangle patch
        x_lb = ss_region[0][0] * (ub[0] - lb[0]) + lb[0]
        x_ub = ss_region[0][1] * (ub[0] - lb[0]) + lb[0]
        y_lb = ss_region[1][0] * (ub[1] - lb[1]) + lb[1]
        y_ub = ss_region[1][1] * (ub[1] - lb[1]) + lb[1]

        rect = patches.Rectangle((x_lb, y_lb), x_ub - x_lb, y_ub - y_lb, linewidth=1, edgecolor='red', facecolor='none', label='sub space')

        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=7,
            lw=0, label="Minima")

    ax.plot(samples[num_init: -1, 0], samples[num_init: -1, 1], "kx", markersize=7,
            lw=0, label="Sample")

    ax.plot(samples[-1, 0], samples[-1, 1], "rx", markersize=7,
            lw=0, label="new Sample")

    ax.plot(samples[:num_init, 0], samples[:num_init, 1], "x", color='cyan', markersize=7,
            lw=0, label="Init")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("$X_0$")
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylabel("$X_1$")
    ax.set_ylim(lb[1], ub[1])

    ax.set_title('{} {}'.format(func_name, title_supp))


func_name = 'branin'
bl = 'bi_level'
van = 'vanilla'

acq_func_name = 'EI'

n_dims = 2

files = 'data_*.json'

ss_files = 'ss_data_*.json'

files_bl = glob.glob(os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), bl, acq_func_name, files))
files_van = glob.glob(os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), van, files))

files_ss = glob.glob(os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), bl, acq_func_name, ss_files))

rng = 14


artifical_func = {'ackley': Ackley,
                  'branin': Branin,
                  'griewank': Griewank,
                  'levy': Levy,
                  'rosenbrock': Rosenbrock}


func = artifical_func[func_name](n_dims)
bound_configs = np.vstack([func.lb, func.ub]).transpose().tolist()

cs = ConfigurationSpace()
cs.generate_all_continuous_from_bounds(bound_configs)

bounds, types = get_types(cs)

cont_dims = np.where(np.array(types) == 0)[0]

cov_amp = ConstantKernel(
    2.0,
    constant_value_bounds=(np.exp(-10), np.exp(2)),
    prior=LognormalPrior(mean=0.0, sigma=1.0, rng=np.random.RandomState(rng)),
)

exp_kernel = Matern(
    np.ones([2]),
    [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(2)],
    nu=2.5,
    operate_on=cont_dims,
)


noise_kernel = WhiteKernel(
    noise_level=1e-8,
    noise_level_bounds=(np.exp(-25), np.exp(2)),
    prior=HorseshoePrior(scale=0.1, rng=np.random.RandomState(rng)),
)
kernel = cov_amp*exp_kernel + noise_kernel

model = GaussianProcess(cs,
                        types=types,
                        bounds=bounds,
                        seed=np.random.RandomState(rng).randint(0, 2 ** 20),
                        kernel=kernel,
                        normalize_y=True)

data_bl = []
data_van = []
data_ss = []
for file_name in files_bl:
    with open(file_name) as f:
        data_bl.append(json.load(f)['x'])

for file_name in files_van:
    with open(file_name) as f:
        data_van.append(json.load(f)['x'])

for file_name in files_ss:
    with open(file_name) as f:
        data_ss.append(json.load(f)['ss_union'])

quer = 14
x_num = 20

samples_van = np.array(data_van[quer])[:x_num]
samples_bl = np.array(data_bl[quer])[:x_num]


ss_regopm_orig = np.array(data_ss[quer])[x_num-11]
ss_region = ss_regopm_orig * (func.ub - func.lb) + func.lb

in_ss_dims = np.all(samples_bl <= ss_region[:, 1], axis=1) & \
             np.all(samples_bl >= ss_region[:, 0], axis=1)


samples_bl_in_ss = samples_bl[in_ss_dims]

print(ss_regopm_orig)
y = []
for i in range(samples_van.shape[0]):
    y.append(func(samples_van[i]))
model.train(samples_van, np.array(y))
print(np.exp(model.hypers))

y = []
for i in range(samples_bl_in_ss.shape[0]):
    y.append(func(samples_bl_in_ss[i]))
model.train(samples_bl_in_ss, np.array(y))
print(np.exp(model.hypers))

if func_name is 'branin':
    func = branin
    lb = np.array([-5, 0])
    ub = np.array([10, 15])
else:
    lb = func.lb
    ub = func.ub

plot_countour(func_name, samples_bl_in_ss, 0, ss_region[:, 0], ss_region[:, 1], ss_region=None, title_supp='van')
plt.show()