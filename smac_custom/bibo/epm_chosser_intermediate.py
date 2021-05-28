import typing
import copy
import itertools
import time

import numpy as np
from scipy.stats import chisquare

from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, \
    UniformIntegerHyperparameter, Constant, UniformFloatHyperparameter

from smac.configspace import Configuration, ConfigurationSpace
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.util_funcs import get_types
from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, RandomSearch, ChallengerList
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.optimizer.random_configuration_chooser import RandomConfigurationChooser, ChooserNoCoolDown
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT

from smac_custom.acqusition_function.max_value_entropy_search import AbstractMES, IntegratedMES
from smac_custom.bibo.rh2epm_bi import RunHistory2EPM4LogCostBi
from smac_custom.util.configspace_custom import get_one_exchange_neighbourhood
from smac_custom.bibo.subspace import SubSpace
from smac_custom.epm.gaussian_process_gyptorch import GaussianProcessGPyTorch

import copy
from torch.quasirandom import SobolEngine
import torch
import gpytorch

from turbo import Turbo1
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube


class EPMChooserIntermediate(EPMChooser):
    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 runhistory2epm: RunHistory2EPM4LogCostBi,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 model_inner: AbstractEPM,
                 local_gp: str,
                 acq_optimizer_inner: AcquisitionFunctionMaximizer,
                 acquisition_func_inner: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: typing.Union[RandomConfigurationChooser] = ChooserNoCoolDown(2.0),
                 predict_x_best: bool = True,
                 min_samples_model: int = 1,
                 batch_size: int = 1,
                 max_configs_inner_fracs: float = 0.5,
                 min_configs_inner: typing.Optional[int] = None,
                 max_spaces_sub: int = 5,
                 ss_data: typing.Optional[typing.Dict] = None,
                 prob_varing_radnom_search: bool = False,
                 n_turbo_init: int=100,
                 ):
        """
        Interface to train the EPM and generate next configurations with outer loop level and inner loop level

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: smac.stats.stats.Stats
            statistics object with configuration budgets
        runhistory: smac.runhistory.runhistory.RunHistory
            runhistory with all runs so far
        model: smac.epm.rf_with_instances.RandomForestWithInstances
            empirical performance model (right now, we support only
            RandomForestWithInstances) in the outer loop
        acq_optimizer: smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
            Optimizer of acquisition function in the outer loop
        model_inner: AbstractEPM,
            empirical performance model (right now, we support only
            RandomForestWithInstances) in the inner loop
        acq_optimizer_inner: AcquisitionFunctionMaximizer,
            Optimizer of acquisition function in the inner loop
        acquisition_func_inner: AbstractAcquisitionFunction,
            acquisition function in the inner loop
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)

        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        batch_size: int
            batch size to be sampled
        min_samples_model: int
-            Minimum number of samples to build a model
        max_configs_inner_fracs : float
            Maximal number of fractions of samples to be included in the inner loop. If the number of samples in the
            subsapce is beyond this value and n_min_config_inner, the subspace will be cropped to fit the requirement
        min_configs_inner: int,
            Minimum number of samples included in the inner loop model
        n_spaces_sub: int
            Maximal number of built subspace
        """
        # initialize the original EPM_Chooser
        super(EPMChooserIntermediate, self).__init__(scenario, stats, runhistory, runhistory2epm, model,
                                                     acq_optimizer, acquisition_func, rng,
                                                     restore_incumbent, random_configuration_chooser, predict_x_best,
                                                     min_samples_model)
        self.model_inner = model_inner
        self.local_gp = local_gp
        self.acq_optimizer_inner = acq_optimizer_inner

        self.acquisition_func_inner = acquisition_func_inner
        self.max_configs_inner_fracs = max_configs_inner_fracs
        self.min_configs_inner = min_configs_inner if min_configs_inner is not None \
            else max(min(5 * len(scenario.cs.get_hyperparameters()), 50), 10)  # clip the number of poitns to be 10-50
        self.batch_size = batch_size
        self.max_spaces_sub = max_spaces_sub

        types, bounds = get_types(self.scenario.cs, instance_features=None)

        self.types = types
        self.bounds = bounds
        self.cat_dims = np.where(np.array(types) != 0)[0]
        self.cont_dims = np.where(np.array(types) == 0)[0]
        self.config_space = scenario.cs

        self.ss_data = ss_data
        self.frac_to_start_bi = 0.8
        self.split_count = np.zeros(len(types))
        self.ss_data['dims_interested'] = []
        self.ss_data['count_splitions'] = []
        self.prob_varing_radnom_search = prob_varing_radnom_search
        self.random_search_upper_log = 1

        self.optimal_value = np.inf
        self.optimal_config = None

        self.ss_threshold = 0.1 ** len(self.scenario.cs.get_hyperparameters())
        if self.prob_varing_radnom_search:
            self.run_turBO = False
            self.failcount_BOinG = 0
            self.failcount_TurBO = 0
            self.lb_TurBO = np.zeros(len(self.cont_dims))
            self.ub_TurBO = np.ones(len(self.cont_dims))
            self.n_turbo_init = n_turbo_init
            self.n_turbo_length_min = 0.5 ** 4

            self.turbo = Turbo1(
                f=None,
                lb=self.lb_TurBO,
                ub=self.ub_TurBO,
                n_init=n_turbo_init,
                max_evals=np.iinfo(np.int32).max,
                batch_size=batch_size,  # We need to update this later
                verbose=False,
            )
            # we don't need the ss to be too small as we can further exploitat with PSGP
            self.turbo.length_min = self.n_turbo_length_min 
            self.X_init_turbo = []
            self.restart_turbo()

            #self.restart_turbo()

    def restart_turbo(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        if self.turbo.n_init == 0:
            self.X_init_turbo = []
        else:
            X_init = latin_hypercube(self.turbo.n_init, len(self.cont_dims))
            self.X_init_turbo = from_unit_cube(X_init, self.lb_TurBO, self.ub_TurBO)

    def turbo_suggestion(self):
        if len(self.X_init_turbo) > 0:
            new_challenger_array = copy.deepcopy(self.X_init_turbo[0, :])
            self.X_init_turbo = self.X_init_turbo[1:, :]  # Remove these pending points
            challenger = Configuration(self.scenario.cs, vector=new_challenger_array)
            challenger.origin = "TurBO init"

        else:
            if len(self.turbo._X) > 0:  # Use random points if we can't fit a GP
                X = to_unit_cube(copy.deepcopy(self.turbo._X), self.lb_TurBO, self.ub_TurBO)
                fX = copy.deepcopy(self.turbo._fX).ravel()
                X_cand, y_cand, _ = self.turbo._create_candidates(
                    X, fX, length=self.turbo.length, n_training_steps=100, hypers={}
                )
                new_challenger_array = self.turbo._select_candidates(X_cand, y_cand)[0, :]
                challenger = Configuration(self.scenario.cs, vector=new_challenger_array)
                challenger.origin = 'TurBO'
            else:
                challenger = self.scenario.cs.sample_configuration(1)
                challenger.origin = 'Randnom Search TurBO'
        return challenger

    def restart_TurBOinG(self, X, Y, Y_raw, train_model=False):
        if train_model:
            self.model.train(X, Y)
        num_samples = 20
        union_ss = []
        union_indices = []
        rand_samples = self.config_space.sample_configuration(num_samples)
        for sample in rand_samples:
            sample_array = sample.get_array()
            union_bounds_cont, _, ss_data_indices = subspace_extraction_intersect(X=X,
                                                                                  challenger=sample_array,
                                                                                  model=self.model,
                                                                                  num_min=self.min_configs_inner,
                                                                                  num_max=MAXINT,
                                                                                  bounds=self.bounds,
                                                                                  cont_dims=self.cont_dims,
                                                                                  cat_dims=self.cat_dims)
            union_ss.append(union_bounds_cont)
            union_indices.append(ss_data_indices)
        union_ss = np.asarray(union_ss)
        volume_ss = np.product(union_ss[:, :, 1] - union_ss[:, :, 0], axis=1)
        ss_idx = np.argmax(volume_ss)
        ss_turbo = union_ss[ss_idx]
        ss_points_indices = union_indices[ss_idx]
        print("TurBO SS")
        print(ss_turbo)
        self.turbo = Turbo1(f=None,
                            lb=ss_turbo[:,0],
                            ub=ss_turbo[:,1],
                            #lb=self.lb_TurBO,
                            #ub=self.ub_TurBO,
                            n_init=max(1, self.n_turbo_init - len(ss_points_indices)),
                            max_evals=np.iinfo(np.int32).max,
                            batch_size=self.batch_size,  # We need to update this later
                            verbose=False,
                            )
        self.restart_turbo()
        self.turbo.length_min = self.n_turbo_length_min 
        self.turbo._X = copy.deepcopy(X[ss_points_indices])
        self.turbo._fX = copy.deepcopy(Y_raw[ss_points_indices])
        self.turbo.X = copy.deepcopy(X[ss_points_indices])
        self.turbo.fX = copy.deepcopy(Y_raw[ss_points_indices])

    def choose_next(self, incumbent_value: float = None) -> typing.Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The
        suggested configurations depend on the argument ``acq_optimizer_outer`` and ''acq_optimizer_inner'' to
        the ``SMBO`` class.

        Parameters
        ----------
        incumbent_value: float
            Cost value of incumbent configuration (required for acquisition function);
            If not given, it will be inferred from runhistory or predicted;
            if not given and runhistory is empty, it will raise a ValueError.

        Returns
        -------
        Iterator
        """
        self.logger.debug("Search for next configuration")
        X, Y, Y_raw, X_configurations = self._collect_data_to_train_model()
        # X, Y, X_configurations = self._collect_data_to_train_model()
        # Y_raw = Y
        if self.prob_varing_radnom_search:
            if self.run_turBO:
                challenger_list = []
                X_new = X[-self.batch_size:]
                Y_new = Y_raw[-self.batch_size:]

                if len(self.turbo._fX) >= self.turbo.n_init:
                    self.turbo._adjust_length(Y_new)
                self.turbo.n_evals += self.batch_size

                self.turbo._X = np.vstack((self.turbo._X, copy.deepcopy(X_new)))
                self.turbo._fX = np.vstack((self.turbo._fX, copy.deepcopy(Y_new)))
                self.turbo.X = np.vstack((self.turbo.X, copy.deepcopy(X_new)))
                self.turbo.fX = np.vstack((self.turbo.fX, copy.deepcopy(Y_new)))

                # Check for a restart
                if self.turbo.length < self.turbo.length_min:
                    optimal_turbo = np.min(self.turbo.fX)
                    #self.optimal_value = Y_raw[-1].item()
                    #self.optimal_config = X[-1]

                    print(f'Best Found value by TurBO: {optimal_turbo}')

                    increment = optimal_turbo - self.optimal_value
                    print(f'increment: {increment}')
                    if increment < 0:
                        min_idx = np.argmin(Y_raw)
                        self.optimal_value = Y_raw[min_idx].item()
                        cfg_diff = X[min_idx] - self.optimal_config
                        self.optimal_config = X[min_idx]
                        if increment < -1e-3 * np.abs(self.optimal_value) or np.abs(np.product(cfg_diff)) >= self.ss_threshold:
                            self.failcount_TurBO -= 1
                            # switch to BOinG as TurBO found a better model
                            self.failcount_BOinG = self.failcount_BOinG // 2
                            self.run_turBO = False
                            #print(f'Prob To BOinG: {prob_to_BOinG:4f}')
                            #print(f'Prob rand: {rand_value:4f}')
                            print('Swich to BOinG!')

                    else:
                        self.failcount_TurBO += 1
                    if self.failcount_TurBO < 4:
                        prob_to_BOinG = 0.5 ** (4 - self.failcount_TurBO)
                    else:
                        prob_to_BOinG = 1 - 0.5 ** (self.failcount_TurBO - 4)


                    print(f'faiout TurBO :{self.failcount_TurBO}')
                    rand_value = self.rng.random()
                    if rand_value < prob_to_BOinG:
                        self.failcount_BOinG = self.failcount_BOinG // 2
                        self.run_turBO = False
                        print(f'Prob To BOinG: {prob_to_BOinG:4f}')
                        print(f'Prob rand: {rand_value:4f}')
                        print('Swich to BOinG!')
                    else:
                        self.restart_TurBOinG(X=X,Y=Y,Y_raw=Y_raw,train_model=True)

                challenger = self.turbo_suggestion()
                challenger_list.append(challenger)
                return iter(challenger_list)


        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )
        # if the number of points is not big enough, we simply build one subspace and
        # fits the model with that single model
        if X.shape[0] < (self.min_configs_inner / self.frac_to_start_bi):
            self.model_inner.train(X, Y_raw)
            if incumbent_value is not None:
                best_observation = incumbent_value
                x_best_array = None  # type: typing.Optional[np.ndarray]
            else:
                if self.runhistory.empty():
                    raise ValueError("Runhistory is empty and the cost value of "
                                     "the incumbent is unknown.")
                x_best_array, best_observation = self._get_x_best(True, X_configurations, use_inner_model=True)

            acq_func_inner_kwargs = {'model': self.model_inner,
                                     'incumbent_array': x_best_array,
                                     'num_data': len(self._get_evaluated_configs()),
                                     'eta': best_observation}
            if isinstance(self.acquisition_func_inner, IntegratedMES) or isinstance(self.acquisition_func_inner,
                                                                                    AbstractMES):
                config_samples = self.scenario.cs.sample_configuration(size=10000)
                acq_func_inner_kwargs['X_dis'] = convert_configurations_to_array(config_samples)

            self.acquisition_func_inner.update(**acq_func_inner_kwargs)
            challengers = self.acq_optimizer_inner.maximize(runhistory=self.runhistory,
                                                            stats=self.stats,
                                                            num_points=1000,
                                                            # type: ignore[attr-defined] # noqa F821
                                                            #random_configuration_chooser=self.random_configuration_chooser,
                                                            )
            return challengers

        # train the outer model
        self.model.train(X, Y)

        if incumbent_value is not None:
            best_observation = incumbent_value
            x_best_array = None  # type: typing.Optional[np.ndarray]
        else:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            x_best_array, best_observation = self._get_x_best(self.predict_x_best, X_configurations)

        self.acquisition_func.update(
            model=self.model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(self._get_evaluated_configs()),
            X=X_configurations,
        )
        list_sub_space = []
        
        if self.prob_varing_radnom_search:
            print(f"failout BOiNG {self.failcount_BOinG}")
            self.failcount_BOinG += 1
            increment = Y_raw[-1].item() - self.optimal_value
            if increment < 0:
                if self.optimal_config is not None:
                    cfg_diff = X[-1] - self.optimal_config
                    if increment < -1e-2 * np.abs(self.optimal_value) or np.abs(np.product(cfg_diff)) >= self.ss_threshold:
                        self.failcount_BOinG -= X.shape[-1]
                    self.optimal_value = Y_raw[-1].item()
                    self.optimal_config = X[-1]
                else:
                    # restart
                    idx_min = np.argmin(Y_raw)
                    print(f"Better value found by BOiNG, restart BOinG")
                    self.optimal_value = Y_raw[idx_min].item()
                    self.optimal_config = X[idx_min]
                    #self.failcount_BOinG -= X.shape[-1]
                    self.failcount_BOinG = 0

            amplify_param = self.failcount_BOinG // (X.shape[-1] * 1)

            if self.failcount_BOinG % (X.shape[-1] * 1) == 0:
                if amplify_param > 4:
                    prob_to_TurBO = 1 - 0.5 ** min(amplify_param - 4, 3)
                else:
                    if amplify_param < 1:
                        prob_to_TurBO = 0
                    else:
                        prob_to_TurBO = 0.5 ** max(4 - amplify_param, self.random_search_upper_log)

                rand_value = self.rng.random()
                if rand_value < prob_to_TurBO:
                    # self.failcount_BOinG = 0
                    self.run_turBO = True
                    print(f'Prob To TurBO: {prob_to_TurBO:4f}')
                    print(f'Prob rand: {rand_value:4f}')
                    print('Switch To TurBO')
                    self.failcount_TurBO = self.failcount_TurBO // 2
                    self.restart_TurBOinG(X=X, Y=Y, Y_raw=Y_raw, train_model=False)


        #    self.random_configuration_chooser.prob = 1 - 0.5 ** min(amplify_param - 5, 3)

            #self.random_configuration_chooser.prob = 0.5 ** max(3 - amplify_param, self.random_search_upper_log)
            #if amplify_param > 5:
            #    self.random_configuration_chooser.prob = 1 - 0.5 ** min(amplify_param - 5, 3)

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory,
            stats=self.stats,
            num_points=self.scenario.acq_opt_challengers,  # type: ignore[attr-defined] # noqa F821
            random_configuration_chooser=self.random_configuration_chooser
        )
        challengers_list = []
        while len(list_sub_space) < self.max_spaces_sub and len(challengers_list) < self.batch_size:
            cfg_challenger = next(challengers)
            challenger = cfg_challenger.get_array()
            # arg_min = np.argmin(Y)
            # inc = X[arg_min]
            # challenger = inc
            new_subspace = True
            for sub_space in list_sub_space:
                # check if the new challenger in the sub space
                challanger_in_ss = check_points_in_ss(challenger,
                                                      cont_dims=self.cont_dims,
                                                      cat_dims=self.cat_dims,
                                                      bounds_cont=sub_space.bounds_ss_cont,
                                                      bounds_cat=sub_space.bounds_ss_cat)
                if challanger_in_ss.size != 0:
                    new_subspace = False
                    challenger = ss.suggest()
                    while challenger in challengers_list:
                        challenger = ss.suggest()
                    challengers_list.append(challenger)
                    break

            if new_subspace:
                trees = self.model.rf.get_all_trees()

                num_max_configs = int(X.shape[0] * self.max_configs_inner_fracs)
                """
                if num_max_configs <= self.min_configs_inner:
                    ss_node_indices = self.model.rf.collect_data_nodes_from_leaf(challenger,
                                                                                 self.min_configs_inner,
                                                                                 MAXINT)
                else:
                    ss_node_indices = self.model.rf.collect_data_nodes_from_leaf(challenger,
                                                                                 self.min_configs_inner,
                                                                                 num_max_configs)
                num_trees = self.model.rf.num_trees()

                bounds_subspaces = [() for _ in range(num_trees)]
                splitions = []
                for i, (ss_node_idx, tree) in enumerate(zip(ss_node_indices, trees)):
                    bounds_subspaces[i] = tree.get_subspace_by_node(bounds=self.bounds, node_idx=ss_node_idx)
                    split_fatures = get_split_features(tree, ss_node_idx)
                    if len(split_fatures) > 0:
                        splitions.append(get_split_features(tree, ss_node_idx))

                union_bounds_cont, union_bounds_cat = union_subspace(bounds_subspaces,
                                                                     cont_dims=self.cont_dims,
                                                                     cat_dims=self.cat_dims,
                                                                     types=self.types)
                """
                union_bounds_cont, union_bounds_cat, ss_data_indices = subspace_extraction_intersect(X=X,
                                                                                                     challenger=challenger,
                                                                                                     model=self.model,
                                                                                                     num_min=self.min_configs_inner,
                                                                                                     num_max=MAXINT if num_max_configs <= 2 * self.min_configs_inner else num_max_configs,
                                                                                                     bounds=self.bounds,
                                                                                                     cont_dims=self.cont_dims,
                                                                                                     cat_dims=self.cat_dims)

                # splitions = sum(splitions, [])
                num_features = len(self.types)
                splitions = get_split_features(trees)
                count_splitions = np.bincount(splitions, minlength=num_features)

                self.split_count = count_splitions + (0.9 * self.split_count).astype(int)

                dims_of_interest = extract_dims_of_interest(self.split_count, num_features)
                print("dims of interest ")
                print(dims_of_interest)
                """
                ss_data_indices = check_points_in_ss(X,
                                                     cont_dims=self.cont_dims,
                                                     cat_dims=self.cat_dims,
                                                     bounds_cont=union_bounds_cont,
                                                     bounds_cat=union_bounds_cat)
                """

                """
                if ss_data_indices.size > Y.size * self.frac_to_start_bi:
                    # Here only continuous bounds are dealt as categorical hps will not have over exploitation problem
                    union_bounds_cont = np.full((np.size(self.cont_dims), 2), [0, 1])
                    ss_data_indices = check_points_in_ss(X,
                                                         cont_dims=self.cont_dims,
                                                         cat_dims=self.cat_dims,
                                                         bounds_cont=union_bounds_cont,
                                                         bounds_cat=union_bounds_cat)
                """
                ss_data_indices = np.where(ss_data_indices)[0]
                self.ss_data['ss_union'].append(union_bounds_cont.tolist())
                #self.ss_data['num_data'].append([ss_data_indices.size, Y.size])
                #self.ss_data['data_in_ss'].append(ss_data_indices[0].tolist())
                #self.ss_data['dims_interested'].append(dims_of_interest.tolist())
                # self.ss_data['count_splitions'].append(count_splitions.tolist())

                #if cfg_challenger.origin != 'Random Search' or not self.prob_varing_radnom_search:
                if True:
                    # return iter(self.scenario.cs.sample_configuration(10))
                    ss = SubSpace(config_space=self.scenario.cs,
                                  bounds=self.bounds,
                                  bounds_ss_cont=union_bounds_cont,
                                  bounds_ss_cat=union_bounds_cat,
                                  hps_types=self.types,
                                  cont_dims=self.cont_dims,
                                  cat_dims=self.cat_dims,
                                  rng=self.rng,
                                  X=X,
                                  y=Y_raw,
                                  rf_incumbent=challenger,
                                  local_gp=self.local_gp,
                                  # acq_optimizer=self.acq_optimizer_inner,
                                  acq_func=self.acquisition_func_inner,
                                  reduce_redundant=False,
                                  dims_of_interest=dims_of_interest,
                                  )
                    list_sub_space.append(ss)

                    challenger = ss.suggest()
                    while challenger in challengers_list:
                        challenger = ss.suggest()
                        challenger.origin = 'BOinG'
                    challengers_list.append(challenger)
                else:
                    """
                    X_incum = X[np.argmin(Y_raw)]
                    new_challengers = self.scenario.cs.sample_configuration(10)
                    if np.all(X_incum <= union_bounds_cont[:, 1]) & np.all(X_incum >= union_bounds_cont[:, 0]):
                        for new_challenger in new_challengers:
                            union_bounds_cont, _, _ = subspace_extraction_intersect(X=X,
                                                                                    challenger=new_challenger,
                                                                                    model=self.model,
                                                                                    num_min=self.min_configs_inner,
                                                                                    num_max=MAXINT if num_max_configs <= 2 * self.min_configs_inner else num_max_configs,
                                                                                    bounds=self.bounds,
                                                                                    cont_dims=self.cont_dims,
                                                                                    cat_dims=self.cat_dims)
                            if (np.all(X_incum <= union_bounds_cont[:, 1]) & np.all(X_incum >= union_bounds_cont[:, 0], axis=1)):
                                break
                    """

                    if isinstance(self.model_inner, GaussianProcessGPyTorch):
                        model = self.model_inner
                    else:
                        from gpytorch.constraints.constraints import Interval
                        exp_kernel = gpytorch.kernels.MaternKernel(2.5,
                                                  lengthscale_constraint=Interval(
                                                      torch.tensor(
                                                          np.exp(-6.754111155189306).repeat(self.cont_dims.shape[-1])),
                                                      torch.tensor(
                                                          np.exp(0.0858637988771976).repeat(self.cont_dims.shape[-1])),
                                                      transform=None,
                                                      initial_value=1.0
                                                  ),
                                                  ard_num_dims=dims_of_interest.shape[-1])

                        base_covar = gpytorch.kernels.ScaleKernel(exp_kernel,
                                                 outputscale_constraint=Interval(
                                                     np.exp(-10.),
                                                     np.exp(2.),
                                                     transform=None,
                                                     initial_value=2.0
                                                 ),
                                                 outputscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0))
                        model = GaussianProcessGPyTorch(configspace=self.scenario.cs,
                                                          types=self.types,
                                                          bounds=np.array(self.bounds)[self.cont_dims].tolist(),
                                                          bounds_cont=np.array(self.bounds)[self.cont_dims],
                                                          bounds_cat=np.array(self.bounds)[self.cat_dims],
                                                          seed=np.random.randint(0, 2**20),
                                                          kernel=base_covar)


                    ss_lb = union_bounds_cont[:, 0]
                    ss_ub = union_bounds_cont[:, 1]

                    X_in_normalize = (X[ss_data_indices] - ss_lb) / (ss_ub-ss_lb)
                    n_dims = len(self.cont_dims)
                    n_candidate = min(100 * n_dims, 5000)

                    y_in = Y_raw[ss_data_indices]
                    model.train(X_in_normalize, y_in)

                    length = 0.8 * 0.5 ** min(max(amplify_param - 1, 0), 4)

                    sobol = SobolEngine(n_dims, scramble=True, seed=self.rng.randint(MAXINT))
                    cands = sobol.draw(n_candidate).cpu().double().detach().numpy()

                    mu, var = model.predict(cands)
                    lcb_param = np.sqrt(2 * np.log((n_dims * len(ss_data_indices)**2)))
                    lcb = mu - lcb_param * np.sqrt(var)
                    x_center = cands[lcb.argmin().item()][None, :]


                    #x_center = X_in_normalize[y_in.argmin().item(), :][None, :]
                    weights = model.gp_model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
                    weights = weights / weights.mean()  # This will make the next line more stable
                    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
                    lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
                    ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

                    # Draw a Sobolev sequence in [lb, ub]
                    # sobol = SobolEngine(n_dims, scramble=True, seed=self.rng.randint(MAXINT))
                    pert = sobol.draw(n_candidate).cpu().detach().numpy()
                    pert = lb + (ub - lb) * pert

                    # Create a perturbation mask
                    prob_perturb = min(20.0 / n_dims, 1.0)
                    mask = np.random.rand(n_candidate, n_dims) <= prob_perturb
                    ind = np.where(np.sum(mask, axis=1) == 0)[0]
                    mask[ind, np.random.randint(0, n_dims - 1, size=len(ind))] = 1

                    # Create candidate points
                    X_cand = x_center.copy() * np.ones((n_candidate, n_dims))
                    X_cand[mask] = pert[mask]

                    # We use Lanczos for sampling if we have enough data
                    with torch.no_grad(), gpytorch.settings.max_cholesky_size(2000):
                        y_cand = model.sample_functions(X_cand, 1)
                    y_cand = model._untransform_y(y_cand)
                    X_cand = X_cand * (ss_ub - ss_lb) + ss_lb

                    challenger = Configuration(self.scenario.cs, vector=X_cand[np.argmin(y_cand)])

                    while challenger in challengers_list:
                        y_cand[np.argmin(y_cand)] = np.inf
                        challenger = Configuration(self.scenario.cs, X_cand[np.argmin(y_cand)])
                        challenger.origin = "TurBO Random"
                    challengers_list.append(challenger)

                print('union_bounds_cont')
                print(union_bounds_cont)
                print('contained {0} data of {1}'.format(ss_data_indices.size, Y.size))

        return iter(challengers_list)

    def _get_x_best(self, predict: bool, X: np.ndarray, use_inner_model: bool = False) -> typing.Tuple[
        float, np.ndarray]:
        """Get value, configuration, and array representation of the "best" configuration.

        The definition of best varies depending on the argument ``predict``. If set to ``True``,
        this function will return the stats of the best configuration as predicted by the model,
        otherwise it will return the stats for the best observed configuration.

        Parameters
        ----------
        predict : bool
            Whether to use the predicted or observed best.

        Return
        ------
        float
        np.ndarry
        Configuration
        """
        if predict:
            if use_inner_model:
                costs = list(map(
                    lambda x: (
                        self.model_inner.predict_marginalized_over_instances(x.reshape((1, -1)))[0][0][0],
                        x,
                    ),
                    X,
                ))
            else:
                costs = list(map(
                    lambda x: (
                        self.model.predict_marginalized_over_instances(x.reshape((1, -1)))[0][0][0],
                        x,
                    ),
                    X,
                ))
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            all_configs = self.runhistory.get_all_configs_per_budget(budget_subset=self.currently_considered_budgets)
            x_best = self.incumbent
            x_best_array = convert_configurations_to_array(all_configs)
            best_observation = self.runhistory.get_cost(x_best)
            best_observation_as_array = np.array(best_observation).reshape((1, 1))
            # It's unclear how to do this for inv scaling and potential future scaling.
            # This line should be changed if necessary
            best_observation = self.rh2EPM.transform_response_values(best_observation_as_array)
            best_observation = best_observation[0][0]

        return x_best_array, best_observation

    def _collect_data_to_train_model(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # if we use a float value as a budget, we want to train the model only on the highest budget
        available_budgets = []
        for run_key in self.runhistory.data.keys():
            available_budgets.append(run_key.budget)

        # Sort available budgets from highest to lowest budget
        available_budgets = sorted(list(set(available_budgets)), reverse=True)

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y, Y_raw = self.rh2EPM.transform(self.runhistory, budget_subset=[b, ])
            if X.shape[0] >= self.min_samples_model:
                self.currently_considered_budgets = [b, ]
                configs_array = self.rh2EPM.get_configurations(
                    self.runhistory, budget_subset=self.currently_considered_budgets)
                return X, Y, Y_raw, configs_array

        return np.empty(shape=[0, 0]), np.empty(shape=[0, ]), np.empty(shape=[0, ]), np.empty(shape=[0, 0])


def expand_subspace(bounds: np.ndarray,
                    bounds_ss_cont: np.ndarray,
                    bounds_ss_cat: typing.List[typing.List[typing.Tuple]],
                    tree,
                    parent_idx: int,
                    cont_dims: np.ndarray,
                    cat_dims: np.ndarray):
    """
    expand the subspace with trees
    Parameters
    ----------
    bounds: bounds of the entire region
    bounds_ss_cont: continuous bounds of the subspace to be expanded
    bounds_ss_cat: categorical bounds of the subspace to be expanded
    tree: binary_full_tree_rss
    parent_idx: index of the parent
    cont_dims: continuous dimensions
    cat_dims: categorical dimensions

    Returns
    -------
    bounds_ss_cont: continuous bounds of the subspace
    bounds_ss_cat: categorical bounds of the subspace
    is_expanded: if the subspace is expanded
    """
    parent_node = tree.get_node(tree.get_node(parent_idx).parent())
    par_feature_idx = parent_node.get_feature_index()
    par_feature_idx_cont = np.where(par_feature_idx == cont_dims)[0]

    is_expanded = True

    if par_feature_idx_cont.size == 0:
        par_feature_idx_cat = np.where(par_feature_idx == cat_dims)[0][0]
        # categorical features
        bound_parent = tree.get_subspace_by_node(bounds=bounds, node_idx=parent_idx)
        if set(bound_parent[par_feature_idx]) == bounds_ss_cat[par_feature_idx_cat]:
            is_expanded = False
        bounds_ss_cat[par_feature_idx_cat] = bound_parent[par_feature_idx]
    else:
        split_value = parent_node.get_num_split_value()
        bound_parent = bounds_ss_cont[par_feature_idx_cont][0]
        if split_value < bound_parent[0]:
            bound_parent = np.array([split_value, bound_parent[1]])
        elif split_value > bound_parent[1]:
            bound_parent = np.array([bound_parent[0], split_value])
        else:
            is_expanded = False
        bounds_ss_cont[par_feature_idx] = bound_parent
    return bounds_ss_cont, bounds_ss_cat, is_expanded


def subspace_extraction_intersect(X: np.ndarray,
                                  challenger: np.ndarray,
                                  model: RandomForestWithInstances,
                                  num_min: int,
                                  num_max: int,
                                  bounds: np.ndarray,
                                  cat_dims: np.ndarray,
                                  cont_dims: np.ndarray):
    """
    extract a subspace that contains at least num_min but no more than num_max
    Parameters
    ----------
    X: points used to train the model
    challenger: the challenger where the subspace would grow
    model: a rf model
    num_min: minimal number of points to be included in the subspace
    num_max: maximal number of points to be included in the subspace
    bounds: bounds of the entire space
    cat_dims: categorical dimensions
    cont_dims: continuous dimensions

    Returns
    -------
    union_bounds_cont: np.ndarray, the continuous bounds of the subregion
    union_bounds_cat, List[Tuple], the categorical bounds of the subregion
    in_ss_dims: indices of the points that lie inside the subregion
    """
    trees = model.rf.get_all_trees()
    num_trees = len(trees)
    node_indices = [0] * num_trees

    indices_trees = np.arange(num_trees)
    np.random.shuffle(indices_trees)
    ss_indices = np.full(X.shape[0], True)

    stop_update = [False] * num_trees

    ss_bounds = np.array(bounds)

    if cat_dims.size == 0:
        ss_bounds_cat = [()]
    else:
        ss_bounds_cat = [() for _ in range(len(cat_dims))]
        for i, cat_dim in enumerate(cat_dims):
            ss_bounds_cat[i] = np.arange(ss_bounds[cat_dim][0])

    if cont_dims.size == 0:
        ss_bounds_cont = np.array([])
    else:
        ss_bounds_cont = ss_bounds[cont_dims]

    def traverse_forest(check_num_min=True):
        nonlocal ss_indices
        np.random.shuffle(indices_trees)
        for i in indices_trees:
            if stop_update[i]:
                continue
            tree = trees[int(i)]
            node_idx = node_indices[i]
            node = tree.get_node(node_idx)

            if node.is_a_leaf():
                stop_update[i] = True
                continue

            feature_idx = node.get_feature_index()
            cont_feature_idx = np.where(feature_idx == cont_dims)[0]
            if cont_feature_idx.size == 0:
                cat_feature_idx = np.where(feature_idx == cat_dims)[0][0]
                split_value = node.get_cat_split()
                intersect = np.intersect1d(ss_bounds_cat[cat_feature_idx], split_value, assume_unique=True)

                if len(intersect) == len(ss_bounds_cat[cat_feature_idx]):
                    temp_child_idx = 0
                    node_indices[i] = node.get_child_index(temp_child_idx)
                elif len(intersect) == 0:
                    temp_child_idx = 1
                    node_indices[i] = node.get_child_index(temp_child_idx)
                else:
                    if challenger[feature_idx] in intersect:
                        temp_child_idx = 0
                        temp_node_indices = ss_indices & np.in1d(X[:, feature_idx], split_value)
                        temp_bound_ss = intersect
                    else:
                        temp_child_idx = 1
                        temp_node_indices = ss_indices & np.in1d(X[:, feature_idx], split_value, invert=True)
                        temp_bound_ss = np.setdiff1d(ss_bounds_cat[cat_feature_idx], split_value)
                    if sum(temp_node_indices) > num_min:
                        # number of points inside subspace is still greater than num_min
                        ss_bounds_cat[cat_feature_idx] = temp_bound_ss
                        ss_indices = temp_node_indices
                        node_indices[i] = node.get_child_index(temp_child_idx)
                    else:
                        if check_num_min:
                            stop_update[i] = True
                        else:
                            node_indices[i] = node.get_child_index(temp_child_idx)
            else:
                split_value = node.get_num_split_value()
                cont_feature_idx = cont_feature_idx.item()
                if ss_bounds_cont[cont_feature_idx][0] <= split_value <= ss_bounds_cont[cont_feature_idx][1]:
                    # the subspace can be further split
                    if challenger[feature_idx] >= split_value:
                        temp_bound_ss = np.array([split_value, ss_bounds_cont[cont_feature_idx][1]])
                        temp_node_indices = ss_indices & (X[:, feature_idx] >= split_value)
                        temp_child_idx = 1
                    else:
                        temp_bound_ss = np.array([ss_bounds_cont[cont_feature_idx][0], split_value])
                        temp_node_indices = ss_indices & (X[:, feature_idx] <= split_value)
                        temp_child_idx = 0
                    if sum(temp_node_indices) > num_min:
                        # number of points inside subspace is still greater than num_min
                        ss_bounds_cont[cont_feature_idx] = temp_bound_ss
                        ss_indices = temp_node_indices
                        node_indices[i] = node.get_child_index(temp_child_idx)
                    else:
                        if check_num_min:
                            stop_update[i] = True
                        else:
                            node_indices[i] = node.get_child_index(temp_child_idx)
                else:
                    temp_child_idx = 1 if challenger[feature_idx] >= split_value else 0
                    node_indices[i] = node.get_child_index(temp_child_idx)

    while sum(stop_update) < num_trees:
        traverse_forest()

    if sum(ss_indices) > num_max:
        stop_update = [False] * num_trees
        while sum(stop_update) < num_trees:
            traverse_forest(False)

    return ss_bounds_cont, ss_bounds_cat, ss_indices


def subspace_extraction(X: np.ndarray,
                        challenger: np.ndarray,
                        model: RandomForestWithInstances,
                        num_min: int,
                        num_max: int,
                        bounds: np.ndarray,
                        cat_dims: np.ndarray,
                        cont_dims: np.ndarray):
    """
    extract a subspace that contains at least num_min but no more than num_max
    Parameters
    ----------
    X: points used to train the model
    challenger: the challenger where the subspace would grow
    model: a rf model
    num_min: minimal number of points to be included in the subspace
    num_max: maximal number of points to be included in the subspace
    bounds: bounds of the entire space
    cat_dims: categorical dimensions
    cont_dims: continuous dimensions

    Returns
    -------
    union_bounds_cont: np.ndarray, the continuous bounds of the subregion
    union_bounds_cat, List[Tuple], the categorical bounds of the subregion
    in_ss_dims: indices of the points that lie inside the subregion
    """
    trees = model.rf.get_all_trees()
    num_trees = len(trees)
    node_indices = [0] * num_trees
    bounds_subspaces = [[] for _ in range(num_trees)]
    indices_trees = np.arange(num_trees)
    np.random.shuffle(indices_trees)

    for i in indices_trees:
        tree = trees[int(i)]
        leaf_idx = tree.find_leaf_index(challenger)
        node_indices[i] = leaf_idx
        bounds_subspaces[i] = tree.get_subspace_by_node(bounds=bounds, node_idx=leaf_idx)

    union_bounds_cont, union_bounds_cat = union_subspace(bounds_subspaces,
                                                         cont_dims=cont_dims,
                                                         cat_dims=cat_dims, )

    in_ss_dims = check_points_in_ss(X=X,
                                    cont_dims=cont_dims,
                                    cat_dims=cat_dims,
                                    bounds_cont=union_bounds_cont,
                                    bounds_cat=union_bounds_cat)

    stop_update = [False] * num_trees

    while in_ss_dims.sum() < num_min:
        if sum(stop_update) == num_trees:
            break
        np.random.shuffle(indices_trees)
        for i in indices_trees:
            if stop_update[i]:
                continue
            tree = trees[int(i)]
            parent_idx = tree.get_node(node_indices[i]).parent()
            union_bounds_cont_temp, union_bounds_cat_temp, is_expanded = expand_subspace(bounds,
                                                                                         union_bounds_cont,
                                                                                         union_bounds_cat,
                                                                                         tree,
                                                                                         parent_idx,
                                                                                         cont_dims, cat_dims)

            if not is_expanded:
                node_indices[i] = parent_idx
                if parent_idx == 0:
                    stop_update[i] = True
                continue
            in_ss_dims_temp = check_points_in_ss(X=X,
                                                 cont_dims=cont_dims,
                                                 cat_dims=cat_dims,
                                                 bounds_cont=union_bounds_cont_temp,
                                                 bounds_cat=union_bounds_cat_temp)

            if in_ss_dims_temp.sum() > num_max or parent_idx == 0:
                stop_update[i] = True
                continue
            else:
                union_bounds_cont = union_bounds_cont_temp
                union_bounds_cat = union_bounds_cat_temp
                node_indices[i] = parent_idx
                in_ss_dims = in_ss_dims_temp
    return union_bounds_cont, union_bounds_cat, in_ss_dims


def check_points_in_ss(X: np.ndarray,
                       cont_dims: np.ndarray,
                       cat_dims: np.ndarray,
                       bounds_cont: np.ndarray,
                       bounds_cat: typing.List[typing.List[typing.Tuple]],
                       ):
    """
    check which points will be included in the subspace
    Parameters
    ----------
    X: np.ndarray(N,D),
        points to be checked
    cont_dims: np.ndarray(D_cont)
        dimensions of the continuous hyperparameters
    cat_dims: np.ndarray(D_cat)
        dimensions of the categorical hyperparameters
    bounds_cont: typing.List[typing.Tuple]
        subspaces bounds of categorical hyperparameters, its length is the number of categorical hyperparameters
    bounds_cat: np.ndarray(D_cont, 2)
        subspaces bounds of continuous hyperparameters, its length is the number of categorical hyperparameters
    Return
    ----------
    indices_in_ss:np.ndarray(N)
        indices of data that included in subspaces
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]

    if cont_dims.size != 0:
        in_ss_dims = np.all(X[:, cont_dims] <= bounds_cont[:, 1], axis=1) & \
                     np.all(X[:, cont_dims] >= bounds_cont[:, 0], axis=1)

        #bound_left = bounds_cont[:, 0] - np.min(X[in_ss_dims][:, cont_dims] - bounds_cont[:, 0], axis=0)
        #bound_right = bounds_cont[:, 1] + np.min(bounds_cont[:, 1] - X[in_ss_dims][:, cont_dims], axis=0)
        #in_ss_dims = np.all(X[:, cont_dims] <= bound_right, axis=1) & \
        #             np.all(X[:, cont_dims] >= bound_left, axis=1)
    else:
        in_ss_dims = np.ones(X.shape[-1], dtype=bool)

    for bound_cat, cat_dim in zip(bounds_cat, cat_dims):
        in_ss_dims &= np.in1d(X[:, cat_dim], bound_cat)

    # indices_in_ss = np.where(in_ss_dims)[0]
    return in_ss_dims


from pyrfr.regression import binary_full_tree_rss, bindary_node


def get_split_features(trees: typing.List[binary_full_tree_rss]) -> typing.List[int]:
    """
    get the feature indices that are split to arrive the node
    Parameters
    ----------
    trees: typing.List[binary_full_tree_rss]
        trees to be checked
    Return
    ----------
    split_features: List[int]:
    a list containing the split feature indices
    """
    split_features = []
    for tree in trees:
        depth = tree.depth()
        node = tree.get_node(0)
        current_depth = 0

        nodes_to_visit = []

        child_0 = node.get_child_index(0)
        child_1 = node.get_child_index(1)
        nodes_to_visit.extend([child_0, child_1])

        split_feature = [node.get_feature_index()]

        while (current_depth < depth // 2):
            children_to_visit = []
            while len(nodes_to_visit) != 0:
                node_idx = nodes_to_visit.pop()
                node = tree.get_node(node_idx)
                child_0 = node.get_child_index(0)
                child_1 = node.get_child_index(1)
                children_to_visit.extend([i for i in [child_0, child_1] if i != 0])
                split_feature.append(node.get_feature_index())

            nodes_to_visit = children_to_visit
            current_depth += 1
        split_features.extend(split_feature)
    return split_features


def extract_dims_of_interest(split_count_history: np.ndarray,
                             num_features: int):
    """
    extract the dimensions of interest with a chi-test
    Parameters
    ----------
    split_count_history: np.ndarray(num_features)
    split history, number of split for each features
    num_features: int
    number of features
    Return
    ----------
    dims_of_interest: np.ndarray(num_features)
    dimensions of interest
    """
    return np.arange(num_features)
    sum_splits_count = np.sum(split_count_history)
    f_exp = np.ones(num_features) * (sum_splits_count / num_features)  # uniformly distributed samples
    _, p_value = chisquare(split_count_history, np.ceil(f_exp))
    print(f"P_VALUE!!!{p_value:.3f}")
    print(split_count_history)
    if p_value < 0.05:
        # the uniform hypothesis is rejected
        count_argsort = np.argsort(split_count_history)[::-1]
        count_selected_features = 0
        dims_of_interest = []
        count_threshold = 0.8 * sum_splits_count
        for i in count_argsort:
            dims_of_interest.append(i)
            count_selected_features += split_count_history[i]
            if count_selected_features > count_threshold:
                break
        return np.sort(dims_of_interest)
    else:
        return np.arange(num_features)


def union_subspace(bounds_subspaces: typing.List[typing.List[typing.Tuple]],
                   cont_dims: np.ndarray,
                   cat_dims: np.ndarray,
                   types: typing.Optional[typing.List[int]] = None):
    """
    find the unions of all subspaces
    Parameters
    ----------
    bounds_subspaces: typing.List[typing.List[typing.Tuple]]
        bounds of all the subspaces, of shape [num_trees, num_hps, 2(for cont_hps) or n(for cat_hps)]
    cont_dims: np.ndarray(D_cont)
        dimensions of the continuous hyperparameters
    cat_dims: np.ndarray(D_cat)
        dimensions of the categorical hyperparameters
    check_redundant: bool
        check if redundant dimensions exit in the space
    types: typing.Optional[typing.List[int]]
        types of the hyperparameters
    Return
    ----------
    union_bounds_cont: np.ndarray(D_cont, 2)
        continuous subspaces bounds
    union_bounds_cat: typing.List[typing.List[typing.Tuple]]
        categorical subspaces bounds

    """
    bounds_array = np.array(bounds_subspaces)
    bounds_cont = bounds_array[:, cont_dims].tolist()

    """
    bounds_cont = np.asarray(bounds_cont)
    lb_cont = bounds_cont[:, :, 0]
    ub_cont = bounds_cont[:, :, 1]
    lb_cont = np.zeros(lb_cont.shape[1])
    ub_cont = np.ones(ub_cont.shape[1])
    union_bounds_cont = np.vstack([lb_cont, ub_cont])
    union_bounds_cont = np.transpose(union_bounds_cont)
    return  union_bounds_cont, union_bounds_cat
    """
    # """
    bounds_cont = np.asarray(bounds_cont)  # bounds is of shape (num_trees, D_cont, 2)
    lb_cont = bounds_cont[:, :, 0]
    ub_cont = bounds_cont[:, :, 1]

    union_cont_lb_mean = np.mean(lb_cont, axis=0)
    union_cont_lb_std = np.std(lb_cont, axis=0)
    union_cont_lb = union_cont_lb_mean - union_cont_lb_std
    union_cont_lb_min = np.min(lb_cont, axis=0)
    union_cont_lb = np.where(union_cont_lb > union_cont_lb_min, union_cont_lb, union_cont_lb_min)

    union_cont_ub_mean = np.mean(ub_cont, axis=0)
    union_cont_ub_std = np.std(ub_cont, axis=0)
    union_cont_ub = union_cont_ub_mean + union_cont_ub_std
    union_cont_ub_max = np.max(ub_cont, axis=0)
    union_cont_ub = np.where(union_cont_ub < union_cont_ub_max, union_cont_ub, union_cont_ub_max)

    union_bounds_cont = np.vstack([union_cont_lb, union_cont_ub])
    union_bounds_cont = np.transpose(union_bounds_cont)

    if cat_dims.size == 0:
        union_bounds_cat = [()]
    else:
        # union_bounds_cat = [np.array(bound)[cat_dims].tolist() for bound in bounds_subspaces]
        # union_bounds_cat = [bounds_subspaces[:, cat_dim] for cat_dim in cat_dims]
        union_bounds_cat = [bounds_array[:, cat_dim].tolist() for cat_dim in cat_dims]

        # https://stackoverflow.com/a/18408559
        union_bounds_cat = [tuple(set(sum(cat, ()))) for cat in union_bounds_cat]
    return union_bounds_cont, union_bounds_cat
