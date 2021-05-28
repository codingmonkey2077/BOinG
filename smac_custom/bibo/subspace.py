import typing
import copy
import itertools
import time
import math

import numpy as np

from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, \
    UniformIntegerHyperparameter, Constant, UniformFloatHyperparameter

from smac.configspace import Configuration, ConfigurationSpace
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.configspace import get_one_exchange_neighbourhood

from smac_custom.acqusition_function.max_value_entropy_search import AbstractMES, IntegratedMES
from smac.optimizer.acquisition import PI
from smac_custom.bibo.rh2epm_bi import RunHistory2EPM4LogCostBi
from smac_custom.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess
#from smac_custom.util.configspace_custom import get_one_exchange_neighbourhood
from smac.epm.util_funcs import get_types


from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.priors import LogNormalPrior, UniformPrior
import torch


class SubSpace(object):
    def __init__(self,
                 config_space: ConfigurationSpace,
                 bounds: typing.List[typing.Tuple[float, float]],
                 # bounds_ss_cont_outer: np.ndarray,
                 bounds_ss_cont: np.ndarray,
                 bounds_ss_cat: typing.List[typing.Tuple],
                 hps_types: typing.List[int],
                 cont_dims: np.ndarray,
                 cat_dims: np.ndarray,
                 # cats_freq: typing.List[typing.List],
                 rng: np.random.RandomState,
                 X: np.ndarray,
                 y: np.ndarray,
                 rf_incumbent: np.ndarray,
                 #model: AbstractEPM,
                 local_gp:str,
                 # acq_optimizer: AcquisitionFunctionMaximizer,
                 acq_func: AbstractAcquisitionFunction,
                 reduce_redundant=False,
                 dims_of_interest: typing.Optional[np.ndarray] = None,
                 ):
        """
        initialize a new subspace,
        Parameters
        ----------
        config_space: ConfigurationSpace
            ConfigurationSpaece of whole search space
        bounds: typing.List[typing.Tuple[float, float]]
            bounds of configuration space with length D
        bounds_ss_cont: np.ndarray(D_cont, 2)
            subspaces bounds of continuous hyperparameters, its length is the number of categorical hyperparameters
        bounds_ss_cat: typing.List[typing.Tuple]
            subspaces bounds of categorical hyperparameters, its length is the number of categorical hyperparameters
        hps_types: typing.List[int],
            types of the hyperparameters
        cont_dims: np.ndarray(D_cont)
            which hyperparameters are continuous
        cat_dims: np.ndarray(D_cat)
            which hyperparameters are categorical
        rng: np.random.RandomState
            random state
        X: np.ndarray(N, D)
            evaluated configuration to feed to this subspace for fitting the model of subspace
        y: np.ndarray(N,)
            performance of configurations X
        model: ~smac.epm.base_epm.AbstractEPM
            model in subspace
        acq_optimizer: ~smac.optimizer.acquisition.AcquisitionFunctionMaximizer
            subspace acquisition function maximizer
        acq_func: ~smac.optimizer.ei_optimization.AbstractAcquisitionFunction
            acquisition function
        reduce_redundant: bool
            if the redundant dimesnions selected by RF is not optimized
        dims_of_interest: typing.Optional[np.ndarray()]
            dimensions of features that are important to the output
        """
        self.config_space = config_space
        #self.model = copy.deepcopy(model)  # avoid other subspace modify the trained models
        # self.acq_optimizer = acq_optimizer
        # self.acq_func = acq_func
        self.rng = rng
        self.reduce_redundant = reduce_redundant
        # self.bounds_ss_cont_outer = bounds_ss_cont_outer

        n_hypers = len(config_space.get_hyperparameters())
        model_types = copy.deepcopy(hps_types)
        model_bounds = copy.deepcopy(bounds)

        if dims_of_interest is not None:
            dims_of_interest_cont = np.intersect1d(dims_of_interest, cont_dims)
            dims_of_interest_cat = np.intersect1d(dims_of_interest, cat_dims)

            if reduce_redundant:
                if cat_dims.size == 0:
                    self.dims_of_interest = dims_of_interest_cont
                    dims_of_interest_cat = np.empty((0,))
                elif cont_dims.size == 0:
                    self.dims_of_interest = dims_of_interest_cat
                    dims_of_interest_cont = np.empty((0,))
                else:
                    self.dims_of_interest = np.hstack([dims_of_interest_cont, dims_of_interest_cat])
            else:
                dims_of_interest_cont = cont_dims
                dims_of_interest_cat = cat_dims
                self.dims_of_interest = np.hstack([cat_dims, cont_dims])
        else:
            dims_of_interest_cont = cont_dims
            dims_of_interest_cat = cat_dims
            self.dims_of_interest = np.hstack([cat_dims, cont_dims])

        self.cont_dims = dims_of_interest_cont
        self.cat_dims = dims_of_interest_cat

        # we noramlize the non-CategoricalHyperparameter by x = (x-lb)*scale
        lbs = np.full(n_hypers, 0.)
        scales = np.full(n_hypers, 1.)

        hps = config_space.get_hyperparameters()

        # deal with categorical hyperaprameters
        for i, cat_idx in enumerate(cat_dims):
            hp_cat = hps[cat_idx]
            parents = config_space.get_parents_of(hp_cat.name)
            if len(parents) == 0:
                can_be_inactive = False
            else:
                can_be_inactive = True
            n_cats = len(bounds_ss_cat[i])
            if can_be_inactive:
                n_cats = n_cats + 1
            model_types[cat_idx] = n_cats
            model_bounds[cat_idx] = (int(n_cats), np.nan)

        # store the dimensions of numerical hyperparameters, UniformFloatHyperparameter and UniformIntegerHyperparameter
        dims_cont_num = []
        idx_cont_num = []
        dims_cont_ord = []
        idx_cont_ord = []
        # deal with ordinary hyperaprameters
        for i, cont_idx in enumerate(cont_dims):
            param = hps[cont_idx]
            if isinstance(param, OrdinalHyperparameter):
                parents = config_space.get_parents_of(param.name)
                if len(parents) == 0:
                    can_be_inactive = False
                else:
                    can_be_inactive = True
                n_cats = bounds_ss_cont[i][1] - bounds_ss_cont[i][0] + 1
                if can_be_inactive:
                    model_bounds[cont_idx] = (0, int(n_cats))
                else:
                    model_bounds[cont_idx] = (0, int(n_cats) - 1)
                lbs[cont_idx] = bounds_ss_cont[i][0]  # in subapce, it should start from 0
                dims_cont_ord.append(cont_idx)
                idx_cont_ord.append(i)
            else:
                dims_cont_num.append(cont_idx)
                idx_cont_num.append(i)

        self.bounds_ss_cont = bounds_ss_cont
        self.bounds_ss_cat = bounds_ss_cat
        self.model_bounds = model_bounds
        self.model_types = model_types
        self.dims_cont_ord = np.array(dims_cont_ord)
        self.idx_cont_ord = np.array(idx_cont_ord)

        lbs[dims_cont_num] = bounds_ss_cont[idx_cont_num, 0]
        # rescale numerical hyperparameters to [0., 1.]
        scales[dims_cont_num] = 1. / (bounds_ss_cont[idx_cont_num, 1] - bounds_ss_cont[idx_cont_num, 0])

        self.lbs = lbs
        self.scales = scales

        X_normalized = self._noramlize_X(X=X,
                                         bounds_cat=bounds_ss_cat,
                                         model_bounds=model_bounds,
                                         lbs=lbs,
                                         scales=scales,
                                         cat_dims=cat_dims)

        indices_in_ss = np.all(X_normalized[:, dims_of_interest_cont] <= 1.0, axis=1) & \
                        np.all(X_normalized[:, dims_of_interest_cont] >= 0.0, axis=1)

        X_normalized = X_normalized[:, dims_of_interest]

        self.cs_inner = ConfigurationSpace()
        hp_list = []
        idx_cont = 0
        idx_cat = 0

        hps = config_space.get_hyperparameters()

        for idx in self.dims_of_interest:
            param = hps[idx]
            if isinstance(param, CategoricalHyperparameter):
                choices = [param.choices[int(choice_idx)] for choice_idx in bounds_ss_cat[idx_cat]]
                # cat_freq_arr = np.array((cats_freq[idx_cat]))
                # weights = cat_freq_arr / np.sum(cat_freq_arr)
                hp_new = CategoricalHyperparameter(param.name, choices=choices)  # , weights=weights)

                idx_cat += 1

            elif isinstance(param, OrdinalHyperparameter):
                hp_new = OrdinalHyperparameter(param.name, sequence=np.arrange(model_bounds[idx]))
                idx_cont += 1

            elif isinstance(param, Constant):
                hp_new = copy.deepcopy(param)

            elif isinstance(param, UniformFloatHyperparameter):
                lower = param.lower
                upper = param.upper
                if param.log:
                    lower_log = np.log(lower)
                    upper_log = np.log(upper)
                    hp_new_lower = np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][0] + lower_log)
                    hp_new_upper = np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][1] + lower_log)
                    hp_new = UniformFloatHyperparameter(name=param.name,
                                                        lower=max(hp_new_lower, lower),
                                                        upper=min(hp_new_upper, upper),
                                                        log=True)
                else:
                    hp_new_lower = (upper - lower) * bounds_ss_cont[idx_cont][0] + lower
                    hp_new_upper = (upper - lower) * bounds_ss_cont[idx_cont][1] + lower
                    hp_new = UniformFloatHyperparameter(name=param.name,
                                                        lower=max(hp_new_lower, lower),
                                                        upper=min(hp_new_upper, upper),
                                                        log=False)
                idx_cont += 1
            elif isinstance(param, UniformIntegerHyperparameter):
                lower = param.lower
                upper = param.upper
                if param.log:
                    lower_log = np.log(lower)
                    upper_log = np.log(upper)
                    hp_new_lower = int(
                        math.floor(np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][0] + lower_log)))
                    hp_new_upper = int(
                        math.ceil(np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][1] + lower_log)))
                    hp_new = UniformIntegerHyperparameter(name=param.name,
                                                          lower=max(hp_new_lower, lower),
                                                          upper=min(hp_new_upper, upper),
                                                          log=True)
                else:
                    hp_new_lower = int(math.floor((upper - lower) * bounds_ss_cont[idx_cont][0])) + lower
                    hp_new_upper = int(math.ceil((upper - lower) * bounds_ss_cont[idx_cont][1])) + lower
                    hp_new = UniformIntegerHyperparameter(name=param.name,
                                                          lower=max(hp_new_lower, lower),
                                                          upper=min(hp_new_upper, upper),
                                                          log=False)
                idx_cont += 1
            hp_list.append(hp_new)

        self.cs_inner.add_hyperparameters(hp_list)
        self.cs_inner.add_conditions(config_space.get_conditions())
        self.cs_inner.add_forbidden_clauses(config_space.get_forbiddens())

        self.cont_dims = cont_dims
        self.X_normalized = X_normalized[indices_in_ss]
        self.stored_configs = []

        hps_types, _ = get_types(self.cs_inner)

        for i in range(self.X_normalized.shape[0]):
            self.stored_configs.append(Configuration(configuration_space=self.cs_inner, vector=self.X_normalized[i]))
        self.y = self.cost_transform(y[indices_in_ss])
        y_trans = self.cost_transform(y)
        self.model_x = X_normalized
        self.model_y = y

        self.incumbent_rf = rf_incumbent

        exp_kernel = MaternKernel(2.5,
                                  lengthscale_constraint=Interval(
                                      torch.tensor(np.exp(-6.754111155189306).repeat(self.cont_dims.shape[-1])),
                                      torch.tensor(np.exp(0.0858637988771976).repeat(self.cont_dims.shape[-1])),
                                      transform=None,
                                      initial_value=torch.tensor(1.0),
                                  ),
                                  ard_num_dims=self.cont_dims.shape[-1])

        base_covar = ScaleKernel(exp_kernel,
                                 outputscale_constraint=Interval(
                                     np.exp(-10.),
                                     np.exp(2.),
                                     transform=None,
                                     initial_value=2.0
                                 ),
                                 outputscale_prior=LogNormalPrior(0.0, 1.0))

        num_inducing_points = min(max(min(2 * len(self.cont_dims), 10), X.shape[0] // 20), 50)

        self.model = PartialSparseGaussianProcess(configspace=self.cs_inner,
                                                  types=hps_types,
                                                  bounds=np.array(bounds)[self.cont_dims].tolist(),
                                                  bounds_cont=np.array(
                                                      [[0, 1.] for _ in range(len(dims_of_interest_cont))]),
                                                  bounds_cat=bounds_ss_cat,
                                                  local_gp=local_gp,
                                                  seed=np.random.randint(0, 2**20),
                                                  kernel=base_covar,
                                                  num_inducing_points=num_inducing_points)

        self.model.train(X_normalized, y_trans)

        mu, _ = self.model.predict(X_normalized)

        idx_eta = np.argmin(mu)

        # if self.pure_local_search:
        #    self.incumbent_rf = np.array([X_noramlized[idx_eta]])

        self.acq_func_kwargs = {'model': self.model,
                                'incumbent_array': X_normalized[idx_eta],
                                'num_data': self.model.num_points if isinstance(self.model, PartialSparseGaussianProcess)
                                else y_trans.size,
                                'eta': mu[idx_eta]}
        self.update_acq_func(acq_func=acq_func)
        self.generate_challengers()

    def update_acq_func(self, acq_func: AbstractAcquisitionFunction):
        if isinstance(acq_func, IntegratedMES) or isinstance(acq_func, AbstractMES):
            samples = self.cs_inner.sample_configuration(size=10000)
            samples = convert_configurations_to_array(samples)
            self.acq_func = acq_func
            self.acq_func.update(X_dis=samples, **self.acq_func_kwargs)
        else:
            self.acq_func = acq_func
            self.acq_func.update(**self.acq_func_kwargs)

    def update_rf_incumbent(self, new_incumbent):
        self.incumbent_rf = self._noramlize_X(X=new_incumbent,
                                              bounds_cat=self.bounds_ss_cat,
                                              model_bounds=self.model_bounds,
                                              lbs=self.lbs,
                                              scales=self.scales,
                                              cat_dims=self.cat_dims)

    def generate_challengers(self):
        self.challengers = self._generate_challengers()
        print(self.challengers[0])
        print("array of challenge:")
        print(self.challengers[0][1].get_array())
        print("!"*50)
        self.challengers_iterator = ChallengerList(cs_in=self.cs_inner,
                                                   cs_out=self.config_space,
                                                   challengers=self.challengers,
                                                   dims_of_interest=self.dims_of_interest,
                                                   rf_incumbent=self.incumbent_rf)

    @staticmethod
    def cost_transform(y):
        return y

    def suggest(self):
        return next(self.challengers_iterator)

    def _generate_challengers(self):
        """
        generate new challengers list for this subspace
        """
        num_random_samples = 10000
        num_init_points = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 10,
            6: 10,
            7: 8,
            8: 6,
        }.get(len(self.cs_inner.get_hyperparameters()), 5)

        vectorization_min_obtain = 2
        vectorization_max_obtain = 64
        n_steps_plateau_walk = 5
        """
        if self.pure_local_search:
            if np.all(self.incumbent_rf[:, self.cont_dims] <= 1.0, axis=1) & np.all(
                    self.incumbent_rf[:, self.cont_dims] >= 0.0, axis=1):
                incumbents = []
                for i in range(self.incumbent_rf.shape[0]):
                    incumbents.append(Configuration(self.cs_inner, vector=self.incumbent_rf[i]))
                acq_values = self.acq_func(incumbents)
                candidates_challenge = [(acq_value, incumbent) for acq_value, incumbent in zip(acq_values, incumbents)]
            else:
                candidates_challenge = []
            init_points_local = self._get_init_points(num_init_points=3, additional_start_points=candidates_challenge)
            candidates_local = self._do_search(start_points=init_points_local,
                                               vectorization_min_obtain=vectorization_min_obtain,
                                               vectorization_max_obtain=vectorization_max_obtain,
                                               n_steps_plateau_walk=n_steps_plateau_walk
                                               )
            next_configs_by_acq_value = (candidates_local + candidates_challenge)
        else:
        """
        samples_random = self.cs_inner.sample_configuration(size=num_random_samples)
        acq_values_random = self.acq_func(samples_random)

        random = self.rng.rand(len(acq_values_random))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values_random.flatten()))
        candidates_random = [(acq_values_random[ind], samples_random[ind]) for ind in indices]

        init_points_local = self._get_init_points(num_init_points=num_init_points,
                                                  additional_start_points=candidates_random)
        candidates_local = self._do_search(start_points=init_points_local,
                                           vectorization_min_obtain=vectorization_min_obtain,
                                           vectorization_max_obtain=vectorization_max_obtain,
                                           n_steps_plateau_walk=n_steps_plateau_walk
                                           )

        next_configs_by_acq_value = (
                candidates_random + candidates_local
        )

        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])

        return next_configs_by_acq_value

    def _noramlize_X(self,
                     X: np.ndarray,
                     bounds_cat: typing.List[typing.Tuple],
                     model_bounds: typing.List[typing.Tuple[float, float]],
                     lbs: np.ndarray,
                     scales: np.ndarray,
                     cat_dims: np.ndarray,
                     ):
        """
        noramlize X to fit the subspace
        Parameters
        ----------
        X: np.ndarray(N,D)
            input X, configurations arrays
        bounds_cat: typing.List[typing.Tuple]
            bounds of subspace (unnormalized)
        model_bounds: typing.List[typing.Tuple[float, float]]
            model bounds (normalized), used to clp the output arrays
        lbs: np.ndarray(D)
            lower bounds of subspace
        scales: np.ndarray(D)
            scale of subspace
        cat_dims:np.array(D_{cat})
            index of categorical dimensions
        Returns
        -------
        X_normalized: np.ndarray(N,D)
            normalized input X
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # normalize X
        X_normalized = (X - lbs) * scales
        # normalize categorical function, for instance, if bounds_subspace[i] is a categorical bound contains elements
        # [1, 3, 5], then we map 1->0, 3->1, 5->2
        for cat_idx, cat_bound in zip(cat_dims, bounds_cat):
            X_i = X_normalized[:, cat_idx]
            cond_list = [X_i == cat for cat in cat_bound]
            choice_list = np.arange(len(cat_bound))
            X_i = np.select(cond_list, choice_list)
            X_normalized[:, cat_idx] = X_i

        # clip the values to avoid overflow
        # model_bounds = np.array(model_bounds)
        # X_normalized = np.clip(X_normalized, model_bounds[:, 0], model_bounds[:, -1])
        return X_normalized

    def _get_init_points(self,
                         num_init_points: int,
                         additional_start_points: typing.Optional[typing.List[typing.Tuple[float, Configuration]]]):
        if len(self.stored_configs) == 0:
            init_points = self.cs_inner.sample_configuration(size=num_init_points)
        else:
            acq_values_previous = self.acq_func(self.stored_configs)
            random = self.rng.rand(len(acq_values_previous))
            # Last column is primary sort key!
            indices = np.lexsort((random.flatten(), acq_values_previous.flatten()))
            configs_previous_runs_sorted = [self.stored_configs[ind] for ind in indices[::-1]][:num_init_points]

            costs, _ = self.model.predict_marginalized_over_instances(self.X_normalized)
            assert len(self.X_normalized) == len(costs), (self.X_normalized.shape, costs.shape)

            if len(costs.shape) == 2 and costs.shape[1] > 1:
                weights = np.array([self.rng.rand() for _ in range(costs.shape[1])])
                weights = weights / np.sum(weights)
                costs = costs @ weights

            # From here
            # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
            random = self.rng.rand(len(costs))
            # Last column is primary sort key!
            indices = np.lexsort((random.flatten(), costs.flatten()))

            # Cannot use zip here because the indices array cannot index the
            # rand_configs list, because the second is a pure python list
            configs_previous_runs_sorted_by_cost = [self.stored_configs[ind] for ind in indices][:num_init_points]

            if additional_start_points is not None:
                additional_start_points = [asp[1] for asp in additional_start_points[:num_init_points]]
            else:
                additional_start_points = []

            init_points = []
            init_points_as_set = set()  # type: Set[Configuration]
            for cand in itertools.chain(
                    configs_previous_runs_sorted,
                    configs_previous_runs_sorted_by_cost,
                    additional_start_points,
            ):
                if cand not in init_points_as_set:
                    init_points.append(cand)
                    init_points_as_set.add(cand)
        return init_points

    def _do_search(
            self,
            start_points: typing.List[Configuration],
            vectorization_min_obtain: int = 2,
            vectorization_max_obtain: int = 64,
            n_steps_plateau_walk: int = 10,
    ) -> typing.List[typing.Tuple[float, Configuration]]:
        # Gather data strucuture for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        candidates = start_points
        # Compute the acquisition value of the candidates
        num_candidates = len(candidates)
        acq_val_candidates = self.acq_func(candidates)
        if num_candidates == 1:
            acq_val_candidates = [acq_val_candidates[0][0]]
        else:
            acq_val_candidates = [a[0] for a in acq_val_candidates]

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_candidates
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_candidates
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_candidates
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_candidates
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_candidates
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        obtain_n = [vectorization_min_obtain] * num_candidates
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(candidates):
            neighborhood_iterators.append(get_one_exchange_neighbourhood(
                inc, seed=self.rng.randint(low=0, high=100000)))
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[]] * num_candidates  # type: List[List[Configuration]]

        num_iters = 0
        while np.any(active):

            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_candidates
            # Used to request a new neighborhood for the candidates of the i-th local search
            new_neighborhood = [False] * num_candidates

            # gather all neighbors
            neighbors = []
            for i, neighborhood_iterator in enumerate(neighborhood_iterators):
                if active[i]:
                    neighbors_for_i = []
                    for j in range(obtain_n[i]):
                        try:
                            n = next(neighborhood_iterator)
                            neighbors_generated[i] += 1
                            neighbors_for_i.append(n)
                        except StopIteration:
                            obtain_n[i] = len(neighbors_for_i)
                            new_neighborhood[i] = True
                            break
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                start_time = time.time()
                acq_val = self.acq_func(neighbors)
                end_time = time.time()
                times.append(end_time - start_time)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = [acq_val]

                # Comparing the acquisition function of the neighbors with the acquisition value of the candidate
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_candidates):
                    if not active[i]:
                        continue
                    # And for each local search we know how many neighbors we obtained
                    for j in range(obtain_n[i]):
                        # The next line is only true if there was an improvement and we basically need to iterate to
                        # the i+1-th local search
                        if improved[i]:
                            acq_index += 1
                        else:
                            neighbors_looked_at[i] += 1

                            # Found a better configuration
                            if acq_val[acq_index] > acq_val_candidates[i]:
                                candidates[i] = neighbors[acq_index]
                                acq_val_candidates[i] = acq_val[acq_index]
                                new_neighborhood[i] = True
                                improved[i] = True
                                local_search_steps[i] += 1
                                neighbors_w_equal_acq[i] = []
                                obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_candidates[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_candidates):
                if not active[i]:
                    continue
                if obtain_n[i] == 0 or improved[i]:
                    obtain_n[i] = 2
                else:
                    obtain_n[i] = obtain_n[i] * 2
                    obtain_n[i] = min(obtain_n[i], vectorization_max_obtain)
                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            candidates[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        candidates[i], seed=self.rng.randint(low=0, high=100000),
                    )
        return [(a, i) for a, i in zip(acq_val_candidates, candidates)]


class ChallengerList(typing.Iterator):
    def __init__(
            self,
            cs_in: ConfigurationSpace,
            cs_out: ConfigurationSpace,
            challengers: typing.List[typing.Tuple[float, Configuration]],
            dims_of_interest: typing.Optional[np.ndarray]=None,
            rf_incumbent: typing.Optional[np.ndarray]=None,
    ):
        self.cs_in = cs_in
        self.challengers = challengers
        self.cs_out = cs_out
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self.dim_of_interests = dims_of_interest
        self.reduce_redudant = (len(cs_out.get_hyperparameters()) != len(cs_in.get_hyperparameters()))
        self.rf_incumbent = rf_incumbent

    def __next__(self) -> Configuration:
        if self.challengers is not None and self._index == len(self.challengers):
            raise StopIteration
        challenger = self.challengers[self._index][1]
        self._index += 1
        value = challenger.get_dictionary()
        if self.reduce_redudant:
            rf_incumbent = Configuration(configuration_space=self.cs_out, vector=self.rf_incumbent).get_dictionary()
            # we replace the cooresponding value in rf incumbent with the value suggested by our optimizer
            for k in value.keys():
                rf_incumbent[k] = value[k]
            config = Configuration(configuration_space=self.cs_out, values=rf_incumbent)
        else:
            config = Configuration(configuration_space=self.cs_out, values=value)
        return config

    def __len__(self) -> int:
        if self.challengers is None:
            self.challengers = self.challengers_callback()
        return len(self.challengers) - self._index
