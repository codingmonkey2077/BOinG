import os
import sys

syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)
print(__doc__)

import numpy as np

from skopt.benchmarks import branin, hart6
from matplotlib import pyplot as plt
from functools import partial

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace, Configuration
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel

from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.epm.util_funcs import get_types
from smac.optimizer.acquisition import EI, PI, LCB
from smac.optimizer.ei_optimization import LocalAndSortedRandomSearch
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.initial_design.latin_hypercube_design import LHDesign

from ArtificalFunction.Benchmark import *

from smac_custom.acqusition_function.max_value_entropy_search import *
from smac_custom.bibo.smbo_bi_level import SMBO_Bilevel
from smac_custom.bibo.rh2epm_bi import RunHistory2EPM4LogCostBi, RunHistory2EPM4CostBi

from smac_custom.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess
from smac_custom.epm.gaussian_process_gyptorch import GaussianProcessGPyTorch

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.priors import LogNormalPrior, UniformPrior
import torch

import time


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="smac_optimize")
    parser.add_argument('--func_name', type=str, default='branin')
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--bi_level', action='store_true')
    parser.add_argument('--acq_func_name', type=str, default='EI')
    parser.add_argument('--num_run', type=int, default=100)
    parser.add_argument('--n_configs_x_params', type=int, default=4)
    parser.add_argument('--rng', type=int, default=99)
    parser.add_argument('--use_rf', action='store_true')
    parser.add_argument('--local_gp', type=str, default='psgp')
    return parser.parse_args()


def smac_optimize(func_name,
                  n_dims,
                  bi_level,
                  acq_func_name,
                  num_run,
                  n_configs_x_params,
                  rng,
                  local_gp,
                  use_rf,
                  hist,
                  y_list
                  ):
    torch.manual_seed(rng)
    initial_design_kwargs = {'n_configs_x_params': n_configs_x_params,
                             'max_config_fracs': 0.25
                             }

    def func_wrapper(config, func, hist, y_list, times):
        time0 = time.time()
        x = np.asarray(list((config.get_dictionary().values())))
        y = func(x)
        hist.append(x.tolist())
        y_list.append(y)
        time1 = time.time()
        times.append([time0, time1])
        return y

    times = []

    artifical_func = {'ackley': Ackley,
                      'branin': Branin,
                      'griewank': Griewank,
                      'levy': Levy,
                      'eggholder': Eggholder,
                      'rosenbrock': Rosenbrock,
                      'michalewicz': Michalewicz}

    if func_name == 'hart6':
        func = hart6
        bound_configs = [[0., 1.]] * n_dims
    else:
        func = artifical_func[func_name](n_dims)
        bound_configs = np.vstack([func.lb, func.ub]).transpose().tolist()

    cs = ConfigurationSpace(rng)
    cs.generate_all_continuous_from_bounds(bound_configs)

    ouput_dir_base = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims))

    scenario_kwarg = {"run_obj": "quality",  # we optimize quality (alternatively runtime)
                      "limit_resources": False,
                      "runcount-limit": num_run,
                      # max. number of function evaluations; for this example set to a low number
                      "cs": cs,  # configuration space
                      "deterministic": "true"}

    smac_kwargs = {'run_id': rng,
                   'rng': np.random.RandomState(rng),
                   'tae_runner': partial(func_wrapper, func=func, hist=hist, y_list=y_list, times=times),
                   'initial_design': LHDesign,
                   'initial_design_kwargs': initial_design_kwargs}

    acq_func_dict = {'EI': EI,
                     'PI': PI,
                     'LCB': LCB}
    acq_func = acq_func_dict.get(acq_func_name, None)

    types, bounds = get_types(cs)

    cont_dims = np.where(np.array(types) == 0)[0]
    cat_dims = np.where(np.array(types) != 0)[0]

    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
                                  transform=None,
                                  initial_value=1.0
                              ),
                              ard_num_dims=cont_dims.shape[-1])

    base_covar = ScaleKernel(exp_kernel,
                             outputscale_constraint=Interval(
                                 np.exp(-10.),
                                 np.exp(2.),
                                 transform=None,
                                 initial_value=2.0
                             ),
                             outputscale_prior=LogNormalPrior(0.0, 1.0))
    if use_rf:
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        if acq_func_name == 'EI':
            from smac.optimizer.acquisition import LogEI
            smac_kwargs['acquisition_function'] = LogEI
        smac_dir = os.path.join(ouput_dir_base, 'rf', acq_func_name)
        scenario_kwarg['output_dir'] = smac_dir
        scenario = Scenario(scenario_kwarg)
        smac = SMAC4HPO(scenario=scenario,
                        **smac_kwargs)
    else:
        if not bi_level:
            smac_dir = os.path.join(ouput_dir_base, 'fullGP_GPytorch', acq_func_name)
            min_configs_inner = int(np.iinfo(np.int32).max)
        else:
            if local_gp == 'psgp':
                smac_dir = os.path.join(ouput_dir_base, 'BOinG', acq_func_name)
            elif local_gp == 'full_gp_local':
                smac_dir = os.path.join(ouput_dir_base, 'fullGP_local', acq_func_name)
            elif local_gp == 'fitc':
                smac_dir = os.path.join(ouput_dir_base, 'fitc', acq_func_name)
            elif local_gp == 'local_gp':
                smac_dir = os.path.join(ouput_dir_base, 'local_gp', acq_func_name)
            else:
                raise ValueError("Unsupported local gp model")
            min_configs_inner = 5 * n_dims

        scenario_kwarg['output_dir'] = smac_dir
        scenario = Scenario(scenario_kwarg)

        model_inner = GaussianProcessGPyTorch(configspace=cs,
                                              types=types,
                                              bounds=bounds,
                                              bounds_cont=np.array(bounds)[cont_dims],
                                              bounds_cat=np.array(bounds)[cat_dims],
                                              seed=rng,
                                              kernel=base_covar)

        if isinstance(model_inner, GaussianProcessMCMC):
            if acq_func is None:
                acq_func_inner = IntegratedAcquisitionFunction(model=model_inner, acquisition_function=EI(model_inner))
            else:
                acq_func_inner = IntegratedAcquisitionFunction(model=model_inner,
                                                               acquisition_function=acq_func(model_inner))
        else:
            if acq_func is None:
                acq_func_inner = EI(model_inner)
            else:
                acq_func_inner = acq_func(model_inner)

        maxmizer_inner = LocalAndSortedRandomSearch(acquisition_function=acq_func_inner,
                                                    config_space=cs,
                                                    rng=None)

        ss_data = {'ss_union': [], 'num_data': [], 'data_in_ss': []}

        smbo = partial(SMBO_Bilevel,
                       model_inner=model_inner,
                       local_gp=local_gp,
                       acq_optimizer_inner=maxmizer_inner,
                       acquisition_func_inner=acq_func_inner,
                       max_configs_inner_fracs=0.25,
                       min_configs_inner=min_configs_inner,
                       ss_data=ss_data,
                       )

        random_configuration_chooser_kwargs = {'prob': 0.05}
        from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4LogCost
        from smac.optimizer.acquisition import LogEI
        np.seterr(over='raise')

        #model_kwargs = {'log_y': False}

        smac = SMAC4HPO(scenario=scenario,
                        smbo_class=smbo,
                        runhistory2epm=RunHistory2EPM4LogCostBi,
                        acquisition_function=LogEI if isinstance(acq_func_inner, EI) else acq_func,
                        #acquisition_function=acq_func,
                        # runhistory2epm=RunHistory2EPM4CostBi,
                        random_configuration_chooser_kwargs=random_configuration_chooser_kwargs,
                        #model_kwargs=model_kwargs,
                        **smac_kwargs)
    smac.optimize()
    """
    import matplotlib
    
    fig = plt.figure(figsize=(7, 5))
    matplotlib.rcParams.update({'font.size': 16})
    plt.plot(y_list, 'b.', ms=10)  # Plot all evaluated points as blue dots
    plt.plot(np.minimum.accumulate(y_list), 'r', lw=3)  # Plot cumulative minimum as a red line
    plt.xlim([0, len(y_list)])
    plt.title("2D  {} function".format(func_name))
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    """

    import json
    result = {'y': y_list,
              'x': hist,
              'time': times}
    num_files = 0
    file_name = os.path.join(smac_dir, 'data_' + str(rng) + '_' + str(num_files) + '.json')
    while os.path.exists(file_name):
        num_files += 1
        file_name = os.path.join(smac_dir, 'data_' + str(rng) + '_' + str(num_files) + '.json')

    with open(os.path.join(file_name), 'w') as f:
        json.dump(result, f)

    if bi_level:
        file_name_ss = os.path.join(smac_dir, 'ss_data_' + str(rng) + '_' + str(num_files) + '.json')
        with open(os.path.join(file_name_ss), 'w') as f:
            json.dump(ss_data, f)


if __name__ == '__main__':
    args = vars(parse_args())
    hist = []
    y_list = []
    smac_optimize(**args, hist=hist, y_list=y_list)
