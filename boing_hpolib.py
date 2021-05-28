import os
import pathlib

from functools import partial

from smac.optimizer.acquisition import EI, PI, LCB
from smac.optimizer.ei_optimization import LocalAndSortedRandomSearch
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO

from smac.scenario.scenario import Scenario
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.initial_design.latin_hypercube_design import LHDesign

from hpobench.benchmarks.rl.cartpole import CartpoleReduced as CartPole

from smac_custom.acqusition_function.max_value_entropy_search import *
from smac_custom.bibo.smbo_bi_level import SMBO_Bilevel
from smac_custom.bibo.rh2epm_bi import RunHistory2EPM4LogCostBi, RunHistory2EPM4CostBi
from smac_custom.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess
from smac_custom.epm.gaussian_process_gyptorch import GaussianProcessGPyTorch

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.priors import LogNormalPrior, UniformPrior
import torch

from hpobench_list import benchmarks, optimization_function_wrapper


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="smac_optimize")
    parser.add_argument('--problem_name', type=str, default='robot')
    parser.add_argument('--bi_level', action='store_true')
    parser.add_argument('--acq_func_name', type=str, default='EI')
    parser.add_argument('--num_run', type=int, default=7500)
    parser.add_argument('--n_configs_x_params', type=int, default=2)
    parser.add_argument('--rng', type=int, default=40)
    parser.add_argument('--use_rf', action='store_true')
    return parser.parse_args()


def smac_optimize(bi_level,
                  problem_name,
                  acq_func_name,
                  num_run,
                  n_configs_x_params,
                  rng,
                  use_rf,
                  y_list
                  ):
    # torch.manual_seed(rng)

    hist = []

    benchmark = benchmarks[problem_name](rng=rng)
    cs = benchmark.get_configuration_space()

    if problem_name == 'lunar':
        initial_design_kwargs = {'init_budget': 50}
        num_init = 50
    elif problem_name == 'robot':
        initial_design_kwargs = {'init_budget': 100}
        num_init = 100
    elif problem_name == 'rover':
        initial_design_kwargs = {'init_budget': 200}
        num_init = 200
    else:
        initial_design_kwargs = {'n_configs_x_params': n_configs_x_params,
                                 'max_config_fracs': 0.25}
        num_init = min(n_configs_x_params * len(cs.get_hyperparameters()), 0.25 * num_run)


    ouput_dir_base = str(pathlib.Path.cwd() / 'HPObench' / problem_name)

    scenario_kwarg = {"run_obj": "quality",  # we optimize quality (alternatively runtime)
                      "limit_resources": False,
                      "runcount-limit": num_run,
                      "cs": cs,  # configuration space
                      "deterministic": "true"}


    smac_kwargs = {'run_id': rng,
                   'rng': np.random.RandomState(rng),
                   'tae_runner': partial(optimization_function_wrapper, benchmark, y_list=y_list, hist=hist),
                   #'initial_design': LHDesign,
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
        smac_dir = os.path.join(ouput_dir_base, 'rf', acq_func_name)
        scenario_kwarg['output_dir'] = smac_dir
        scenario = Scenario(scenario_kwarg)
        if acq_func_name == 'EI':
            from smac.optimizer.acquisition import LogEI
            smac_kwargs['acquisition_function'] = LogEI
        smac = SMAC4HPO(scenario=scenario,
                        **smac_kwargs)
    else:
        if not bi_level:
            smac_dir = os.path.join(ouput_dir_base, 'fullGP_GPytorch', acq_func_name)
            min_configs_inner = int(np.iinfo(np.int32).max)
        else:
            smac_dir = os.path.join(ouput_dir_base, 'BOinG_vanilla', acq_func_name)
            min_configs_inner = len(cs.get_hyperparameters()) * 5

        scenario_kwarg['output_dir'] = smac_dir
        scenario = Scenario(scenario_kwarg)
        print(smac_dir)
        model_inner = GaussianProcessGPyTorch(configspace=cs,
                                              types=types,
                                              bounds=bounds,
                                              bounds_cont=np.array(bounds)[cont_dims],
                                              bounds_cat=np.array(bounds)[cat_dims],
                                              seed=rng,
                                              kernel=base_covar)

        if acq_func is None:
            acq_func_inner = EI(model_inner)
        else:
            acq_func_inner = acq_func(model_inner)

        maxmizer_inner = LocalAndSortedRandomSearch(acquisition_function=acq_func_inner,
                                                    config_space=cs,
                                                    rng=smac_kwargs.get("rng", None))

        ss_data = {'ss_union': [], 'num_data': [], 'data_in_ss': []}
        if num_run > 5000:
            smac_kwargs['initial_design_kwargs'] = {'init_budget': num_init}
            boing_kwargs = {'prob_varing_radnom_search': True,
                           'n_turbo_init': num_init}
        else:
            boing_kwargs = {'prob_varing_radnom_search': False}

        smbo = partial(SMBO_Bilevel,
                       model_inner=model_inner,
                       acq_optimizer_inner=maxmizer_inner,
                       acquisition_func_inner=acq_func_inner,
                       max_configs_inner_fracs=0.25,
                       min_configs_inner=min_configs_inner,
                       ss_data=ss_data,
                       local_gp='psgp',
                       boing_kwargs=boing_kwargs,
                       )

        random_configuration_chooser_kwargs = {'prob': 1.0}
        from smac.optimizer.acquisition import LogEI
        np.seterr(over='raise')

        rng_smac = smac_kwargs.get("rng", None)
        # torch.manual_seed(rng_smac.randint(0, 2 ** 20))
        # rng_smac.randint(0, 2 **20 )
        model_kwargs = {'log_y': False}

        smac_kwargs.update({'smbo_class' : smbo,
                            'runhistory2epm': RunHistory2EPM4LogCostBi,
                            'acquisition_function': LogEI if isinstance(acq_func_inner, EI) else acq_func,
                            'random_configuration_chooser_kwargs': random_configuration_chooser_kwargs})
        """
        if num_run > 500:
            smac_kwargs.update({'runhistory2epm': RunHistory2EPM4CostBi,
                                'acquisition_function': acq_func,
                                'model_kwargs': model_kwargs,
                                })
        else:
            smac_kwargs.update({'runhistory2epm': RunHistory2EPM4LogCostBi,
                                'acquisition_function': LogEI if isinstance(acq_func_inner, EI) else acq_func,
                                })
        """
        #smac_kwargs.update({'smbo_class': smbo,
        #                    'runhistory2epm': RunHistory2EPM4CostBi,
        #                    'acquisition_function': acq_func,
        #                    'model_kwargs': model_kwargs,
        #                    'random_configuration_chooser_kwargs': random_configuration_chooser_kwargs,
        #                    })


        smac = SMAC4HPO(scenario=scenario,
                        #smbo_class=smbo,
                        #runhistory2epm=RunHistory2EPM4LogCostBi,
                        #acquisition_function=LogEI if isinstance(acq_func_inner, EI) else acq_func,
                        #acquisition_function=acq_func,
                        #runhistory2epm=RunHistory2EPM4CostBi,
                        #model_kwargs=model_kwargs,
                        #random_configuration_chooser_kwargs=random_configuration_chooser_kwargs,
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
    result = {'y': y_list, "x": hist}
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
    y_list = []
    smac_optimize(**args, y_list=y_list)
