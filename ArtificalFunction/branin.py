import os
import sys
syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)
print(__doc__)

import numpy as np
np.random.seed(123)

import matplotlib.pyplot as plt
from skopt.benchmarks import branin as branin
from matplotlib.colors import LogNorm


import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn import gaussian_process

import copy
import logging
import typing

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, OrdinalHyperparameter, CategoricalHyperparameter

from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace, Configuration
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import LocalAndSortedRandomSearch
from smac_custom.max_value_entropy_search import *
from smac.epm.util_funcs import get_types
from smac_custom.smbo_bi_level import SMBO_Bilevel
from functools import partial


def plot_branin(samples):
    fig, ax = plt.subplots()

    x1_values = np.linspace(-5, 10, 100)
    x2_values = np.linspace(0, 15, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([branin(val) for val in vals], (100, 100))

    cm = ax.pcolormesh(x_ax, y_ax, fx,
                       norm=LogNorm(vmin=fx.min(),
                                    vmax=fx.max()))

    minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=7,
            lw=0, label="Minima")

    ax.plot(samples[:-1, 0], samples[: -1, 1], "D",color='cyan', markersize=7,
            lw=0, label="Sample")
    ax.plot(samples[-1, 0], samples[ -1, 1], "kx", markersize=7,
            lw=0, label="NewPoints")

    cb = fig.colorbar(cm)
    cb.set_label("f(x)")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("$X_0$")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("$X_1$")
    ax.set_ylim([0, 15])
    plt.show()

hist = np.zeros(shape=(0, 2))

y_list = []
y_min = np.inf
def branin_wrapper(config):
    global y_min, y_list, hist
    x0 = config['x0']
    x1 = config['x1']
    x = np.array([x0, x1])
    y = branin(x)
    hist = np.vstack([hist, x])
    #y_min = min(y, y_min)
    y_list.append(y)
    #plot_branin(hist)
    return y

bi_level = True
rng = 4

ouput_dir_base = os.path.join(syspath,'./ArtificalFunction/branin')


if not bi_level:
    #logging.basicConfig(level=logging.DEBUG)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=2)
    x1 = UniformFloatHyperparameter("x1", 0, 15, default_value=2)
    cs.add_hyperparameters([x0, x1])

    # Scenario object
    smac_dir = os.path.join(ouput_dir_base, 'vanilla')

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "limit_resources": False,
                         "runcount-limit": 100,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         "output_dir": smac_dir,
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    #def_value = shekel(cs.get_default_configuration())
    #print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    initial_design_kwargs = {'n_configs_x_params': 4}
    smac = SMAC4BO(scenario=scenario,
                   run_id=rng,
                   rng=np.random.RandomState(rng),
                   tae_runner=branin_wrapper,
                   initial_design_kwargs=initial_design_kwargs,
                   model_type='gp_mcmc',
                   )
    smac.optimize()
else:
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    """
    x0_list = np.linspace(-5., 10., num=20)
    x1_list = np.linspace(0., 15., num=20)
    x0 = CategoricalHyperparameter('x0', choices=x0_list)
    x1 = OrdinalHyperparameter('x1', sequence=x1_list)
    x2 = OrdinalHyperparameter('x2', sequence=x1_list)
    x3 = CategoricalHyperparameter('x3', choices=x0_list)
    x4 = UniformFloatHyperparameter("x4", -5, 10, default_value=2)
    x5 = UniformFloatHyperparameter("x5", 0, 15, default_value=2)

    #cs.add_hyperparameters([x0, x1, x2, x3, x4, x5])
    """
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=2)
    x1 = UniformFloatHyperparameter("x1", 0, 15, default_value=2)

    cs.add_hyperparameters([x0, x1])
    #"""
    smac_dir = os.path.join(ouput_dir_base, 'bi_level')
    scenario = Scenario({"run_obj": "quality",
                         "cs": cs,
                         "limit_resources": False,
                         "runcount-limit": 100,
                         # "output_dir": "smac_grid" if grid else "smac_random",
                         # "output_dir": "smac_grid_office" if grid else "smac_random_office",
                         "output_dir": smac_dir,
                         "deterministic": True,
                         #"maxR": 15,
                         })

    types, bounds = get_types(cs,instance_features= None)

    cont_dims = np.where(np.array(types) == 0)[0]
    cat_dims = np.where(np.array(types) != 0)[0]

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=np.random.RandomState(rng)),
    )

    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )

    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
            operate_on=cat_dims,
        )


    assert (len(cont_dims) + len(cat_dims)) == len(scenario.cs.get_hyperparameters())

    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=np.random.RandomState(rng)),
    )
    kernel = cov_amp * exp_kernel + noise_kernel

    n_mcmc_walkers = 3 * len(kernel.theta)
    if n_mcmc_walkers % 2 == 1:
        n_mcmc_walkers += 1

    model_inner = GaussianProcessMCMC(cs,
                                      types=types,
                                      bounds=bounds,
                                        seed=0,
                                        kernel=kernel,
                                        chain_length=250,
                                        burnin_steps=250,
                                        n_mcmc_walkers=n_mcmc_walkers,
                                        mcmc_sampler='emcee'
                                        )
    """
    acq_func_inner = IntegratedMES(model=model_inner, acquisition_function=MaxValueEntropySearchG(model_inner))
    maxmizer_inner = MESMaximazier(acquisition_function=acq_func_inner,
                                           config_space=cs,
                                           rng=None)
    """
    acq_func_inner = IntegratedAcquisitionFunction(model=model_inner, acquisition_function=EI(model_inner))
    maxmizer_inner = LocalAndSortedRandomSearch(acquisition_function=acq_func_inner,
                                                config_space=cs,
                                                rng=None)

    smbo = partial(SMBO_Bilevel,
                   model_inner=model_inner,
                   acq_optimizer_inner=maxmizer_inner,
                   acquisition_func_inner=acq_func_inner,
                   max_configs_inner_fracs=0.25,
                   min_configs_inner=10,)

    initial_design_kwargs = {'n_configs_x_params': 4}
    smac = SMAC4HPO(scenario=scenario,
                    run_id=rng,
                    rng=np.random.RandomState(rng),
                    smbo_class=smbo,
                    tae_runner=branin_wrapper,
                    initial_design_kwargs=initial_design_kwargs,
                    )

    smac.optimize()

import matplotlib

fig = plt.figure(figsize=(7, 5))
matplotlib.rcParams.update({'font.size': 16})
plt.plot(y_list - branin(np.array([-np.pi, 12.275])), 'b.', ms=10)  # Plot all evaluated points as blue dots
plt.plot(np.minimum.accumulate(y_list) - branin(np.array([-np.pi, 12.275])), 'r', lw=3)  # Plot cumulative minimum as a red line
plt.xlim([0, len(y_list)])
plt.yscale('log')

plt.tight_layout()
plt.show()

import json
hist = hist.tolist()
result = {'y': y_list,
          'x': hist}

num_files = 0
file_name = os.path.join(smac_dir, 'data_' + str(rng) + '_' + str(num_files) + '.json')
while os.path.exists(file_name):
    num_files += 1
    file_name = os.path.join(smac_dir, 'data_' + str(rng) + '_' + str(num_files) + '.json')

with open(os.path.join(file_name), 'w') as f:
    json.dump(result, f)
