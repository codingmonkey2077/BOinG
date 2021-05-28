import typing
from collections import OrderedDict

import numpy as np
from scipy import optimize

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel, InducingPointKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import UniformPrior, HorseshoePrior
from gpytorch.utils.errors import NanError

from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from botorch.optim.utils import _scipy_objective_and_grad, _get_extra_mll_args

from smac.configspace import ConfigurationSpace
from smac.utils.constants import VERY_SMALL_NUMBER
from smac.epm.base_gp import BaseModel

from psgp.partial_sparse_kernel import PartialSparseKernel
from psgp.partial_sparse_mean import PartialSparseMean
from torch.utils.data import TensorDataset, DataLoader

import pyDOE

import time


class PartailSparseGPModel(ExactGP):
    def __init__(self, in_x, in_y, out_x, out_y, likelihood, base_covar_kernel, inducing_points,
                 batch_shape=torch.Size(), ):
        in_x = in_x.unsqueeze(-1) if in_x.ndimension() == 1 else in_x
        out_x = out_x.unsqueeze(-1) if out_x.ndimension() == 1 else out_x
        inducing_points = inducing_points.unsqueeze(-1) if inducing_points.ndimension() == 1 else inducing_points
        assert inducing_points.shape[-1] == in_x.shape[-1] == out_x.shape[-1]
        super(PartailSparseGPModel, self).__init__(in_x, in_y, likelihood)

        self.base_covar = base_covar_kernel

        self.covar_module = PartialSparseKernel(self.base_covar, inducing_points=inducing_points,
                                                outer_points=out_x, outer_y=out_y, likelihood=likelihood)
        self.mean_module = PartialSparseMean(covar_module=self.covar_module)
        self._mean_module = ZeroMean()


        #lower_inducing_points = torch.min(torch.cat([in_x, out_x, torch.zeros([*batch_shape, 1, in_x.shape[-1]])]),
        #                                  dim=-2).values
        #upper_inducing_points = torch.max(torch.cat([in_x, out_x, torch.ones([*batch_shape, 1, in_x.shape[-1]])]),
        #                                  dim=-2).values
        """
        lower_inducing_points = torch.zeros([*batch_shape, in_x.shape[-1]])
        upper_inducing_points = torch.ones([*batch_shape,  in_x.shape[-1]])

        self.covar_module.register_constraint(param_name="inducing_points",
                                              constraint=Interval(lower_inducing_points.repeat(inducing_points.shape[0]),
                                                                  upper_inducing_points.repeat(inducing_points.shape[0]),
                                                                  transform=None,
                                                                  ),
                                              )
        """
        self.optimize_kernel_hps = True

    def deactivate_kernel_grad(self):
        self.optimize_kernel_hps = False
        for p_name, t in self.named_parameters():
            if p_name == 'covar_module.inducing_points':
                t.requires_grad = True
            else:
                t.requires_grad = False

    def deactivate_inducing_points_grad(self):
        if not self.optimize_kernel_hps:
            raise ValueError("inducing_points will only be inactivate if self.optimize_kernel_hps is set True")
        for p_name, t in self.named_parameters():
            if p_name == 'covar_module.inducing_points':
                t.requires_grad = False
            else:
                t.requires_grad = True

    def forward(self, x):
        if self.training:
            if self.optimize_kernel_hps:
                covar_x = self.base_covar(x)
                mean_x = self._mean_module(x)
            else:
                covar_x = self.covar_module(x)
                mean_x = self.mean_module(x)
        else:
            covar_x = self.covar_module(x)
            mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, kernel, inducing_points):
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

        kernel_length = torch.squeeze(kernel.base_kernel.raw_lengthscale)

        shape_inducing_points = inducing_points.shape
        #lower_inducing_points = torch.ones([shape_inducing_points[-1]]) * -kernel_length
        #upper_inducing_points = torch.ones([shape_inducing_points[-1]]) * (1. + kernel_length)
        lower_inducing_points = torch.zeros([shape_inducing_points[-1]])
        upper_inducing_points = torch.ones([shape_inducing_points[-1]])
        #"""
        self.variational_strategy.register_constraint(param_name="inducing_points",
                                                      constraint=Interval(lower_inducing_points.repeat(shape_inducing_points[0]),
                                                                          upper_inducing_points.repeat(shape_inducing_points[0]),
                                                                          transform=None),
                                                    )
        #"""
        self.double()

        for p_name, t in self.named_hyperparameters():
            if p_name != "variational_strategy.inducing_points":
                t.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, base_covar_kernel, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = base_covar_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PartialSparseGaussianProcess(BaseModel):
    def __init__(self,
                 configspace: ConfigurationSpace,
                 types: typing.List[int],
                 bounds: typing.List[typing.Tuple[float, float]],
                 bounds_cont: np.ndarray,
                 bounds_cat: typing.List[typing.List[typing.Tuple]],
                 seed: int,
                 local_gp: str,
                 kernel: Kernel,
                 num_inducing_points: int,
                 likelihood: typing.Optional[FixedNoiseGaussianLikelihood] = None,
                 normalize_y: bool = True,
                 n_opt_restarts: int = 10,
                 instance_features: typing.Optional[np.ndarray] = None,
                 pca_components: typing.Optional[int] = None,
                 ):
        super(PartialSparseGaussianProcess, self).__init__(configspace,
                                                           types,
                                                           bounds,
                                                           seed,
                                                           kernel,
                                                           instance_features,
                                                           pca_components,
                                                           )
        self.kernel = kernel
        if likelihood is None:
            noise_prior = HorseshoePrior(0.1)
            likelihood = GaussianLikelihood(
                #noise=1e-8 * torch.ones(1),
                #learn_additional_noise=True,
                noise_prior=noise_prior,
                noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
            ).double()
        hypers = {}
        #hypers["noise"] = torch.exp(torch.tensor([-24.],dtype=torch.float64)).double()
        likelihood.initialize(**hypers)
        self.likelihood = likelihood
        self.bound_cont = bounds_cont
        self.bound_cat = bounds_cat
        self.num_inducing_points = num_inducing_points

        self.cat_dims = np.where(np.array(types) != 0)[0]
        self.cont_dims = np.where(np.array(types) == 0)[0]

        self.normalize_y = normalize_y
        self.n_opt_restarts = n_opt_restarts

        self.hypers = np.empty((0,))
        self.property_dict = OrderedDict()
        self.is_trained = False
        self._n_ll_evals = 0

        self.num_points = 0

        self.local_gp = local_gp

        #self._set_has_conditions()

    def set_inducing_pooints(self, num_inducing_points: int):
        self.num_inducing_points = num_inducing_points

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True) -> 'GaussianProcess':
        """
        Computes the Cholesky decomposition of the covariance of X and
        estimates the GP hyperparameters by optimizing the marginal
        loglikelihood. The prior mean of the GP is set to the empirical
        mean of X.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """

        X = self._impute_inactive(X)
        if len(y.shape) == 1:
            self.n_objectives_ = 1
        else:
            self.n_objectives_ = y.shape[1]
        if self.n_objectives_ == 1:
            y = y.flatten()

        ss_data_indices = check_points_in_ss(X,
                                             cont_dims=self.cont_dims,
                                             cat_dims=self.cat_dims,
                                             bounds_cont=self.bound_cont,
                                             bounds_cat=self.bound_cat)
        if self.local_gp == 'psgp' or self.local_gp == 'fitc':
            if np.sum(ss_data_indices) > np.shape(y)[0] - self.num_inducing_points:
                if self.normalize_y:
                    y = self._normalize_y(y)
                self.num_points = np.shape(y)[0]
                get_gp_kwargs = {'in_x': X, 'in_y': y, 'out_x': None, 'out_y': None}
            else:
                in_x = X[ss_data_indices]
                in_y = y[ss_data_indices]
                out_x = X[~ss_data_indices]
                out_y = y[~ss_data_indices]
                self.num_points = np.shape(in_y)[0]
                if self.normalize_y:
                    in_y = self._normalize_y(in_y)
                out_y = (out_y - self.mean_y_) / self.std_y_
                get_gp_kwargs = {'in_x': in_x, 'in_y': in_y, 'out_x': out_x, 'out_y': out_y}
        elif self.local_gp == 'local_gp':
            in_x = X[ss_data_indices]
            in_y = y[ss_data_indices]
            self.num_points = np.shape(in_y)[0]
            if self.normalize_y:
                in_y = self._normalize_y(in_y)
            get_gp_kwargs = {'in_x': in_x, 'in_y': in_y, 'out_x': None, 'out_y': None}
        elif self.local_gp == 'full_gp_local':
            if self.normalize_y:
                y = self._normalize_y(y)
            self.num_points = np.shape(y)[0]
            get_gp_kwargs = {'in_x': X, 'in_y': y, 'out_x': None, 'out_y': None}
        else:
            raise ValueError("Unsupported local gp model, local gp only support psgp, fitc, local_gp, full_gp_local")

        n_tries = 10
        time0 = time.time()

        for i in range(n_tries):
            try:
                self.gp = self._get_gp(**get_gp_kwargs)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e
                # Assume that the last entry of theta is the noise
                # theta = np.exp(self.kernel.theta)
                # theta[-1] += 1
                # self.kernel.theta = np.log(theta)

        if do_optimize:
            #self._all_priors = self._get_all_priors(add_bound_priors=False)
            self.hypers = self._optimize()
            self.gp = set_params_with_array(self.gp, self.hypers, self.property_dict)

            time1 = time.time()
            print("!"*50)
            print(f"time used for kernel optimizing: {(time1 - time0):.4f}")
            if isinstance(self.gp.model, PartailSparseGPModel):
                self.gp.model.deactivate_kernel_grad()
                """
                x0, property_dict, bounds = module_to_array(module=self.gp)

                p0 = [x0]
                while len(p0) < self.n_opt_restarts:
                    try:
                        self.gp.pyro_sample_from_prior()
                        sample, _, _ = module_to_array(module=self.gp)
                        self.gp = set_params_with_array(self.gp, self.hypers, self.property_dict)
                        self.gp.model.deactivate_kernel_grad()
                        p0.append(sample.astype(np.float64))
                    except Exception as e:
                        continue

                bounds = np.asarray(bounds).transpose().tolist()

                theta_star = x0
                f_opt_star = np.inf
                for i, start_point in enumerate(p0):
                    try:
                        theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(
                            _scipy_objective_and_grad,
                            start_point,
                            args=(self.gp, property_dict),
                            maxiter=50,
                            bounds=bounds,
                        )

                    except RuntimeError as e:
                        self.logger.warning(f"Fail to optimize as an Error occurs: {e}")
                        continue
                    if f_opt < f_opt_star:
                        f_opt_star = f_opt
                        theta_star = theta

                self.gp = set_params_with_array(self.gp, theta_star, property_dict)
                """
                #"""

                inducing_points = torch.from_numpy(pyDOE.lhs(n=out_x.shape[-1], samples=self.num_inducing_points))

                kernel = self.gp.model.base_covar
                var_gp = VariationalGPModel(kernel, inducing_points=inducing_points)

                out_x_ = torch.from_numpy(out_x)
                out_y_ = torch.from_numpy(out_y)
                #out_y_ = (out_y_ - torch.mean(out_y_)) / torch.std(out_y_)

                #train_dataset = TensorDataset(out_x_, out_y_)
                #train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

                variational_ngd_optimizer = gpytorch.optim.NGD(var_gp.variational_parameters(), num_data=out_y_.size(0),
                                                               lr=0.1)

                var_gp.train()
                likelihood = GaussianLikelihood().double()
                likelihood.train()

                mll_func = gpytorch.mlls.GammaRobustVariationalELBO

                var_mll = mll_func(likelihood, var_gp, num_data=out_y_.size(0))

                for t in var_gp.variational_parameters():
                    t.requires_grad = False

                x0, property_dict, bounds = module_to_array(module=var_mll)
                for t in var_gp.variational_parameters():
                    t.requires_grad = True
                bounds = np.asarray(bounds).transpose().tolist()
                
                start_points = [x0]

                inducing_idx = 0
                inducing_size = out_x.shape[-1] * self.num_inducing_points
                for p_name, attrs in property_dict.items():
                    if p_name != "model.variational_strategy.inducing_points":
                        # Construct the new tensor
                        if len(attrs.shape) == 0:  # deal with scalar tensors
                            inducing_idx = inducing_idx + 1
                        else:
                            inducing_idx = inducing_idx + np.prod(attrs.shape)
                    else:
                        break
                while len(start_points) < 3:
                    new_start_point = np.random.rand(*x0.shape)
                    new_inducing_points = torch.from_numpy(pyDOE.lhs(n=out_x.shape[-1], samples=self.num_inducing_points)).flatten()
                    new_start_point[inducing_idx: inducing_idx+inducing_size] = new_inducing_points
                    #new_start_point[0] = x0[0]
                    start_points.append(new_start_point)

                def sci_opi_wrapper(x, mll, property_dict, train_inputs, train_targets):
                    variational_ngd_optimizer.zero_grad()

                    mll = set_params_with_array(mll, x, property_dict)
                    mll.zero_grad()
                    try:  # catch linear algebra errors in gpytorch
                        output = mll.model(train_inputs)
                        args = [output, train_targets] + _get_extra_mll_args(mll)
                        loss = -mll(*args).sum()
                    except RuntimeError as e:
                        if isinstance(e, NanError) or "singular" in e.args[0]:
                            return float("nan"), np.full_like(x, "nan")
                        else:
                            raise e  # pragma: nocover
                    loss.backward()
                    variational_ngd_optimizer.step()
                    param_dict = OrderedDict(mll.named_parameters())
                    grad = []
                    for p_name in property_dict:
                        t = param_dict[p_name].grad
                        if t is None:
                            # this deals with parameters that do not affect the loss
                            grad.append(np.zeros(property_dict[p_name].shape.numel()))
                        else:
                            grad.append(t.detach().view(-1).cpu().double().clone().numpy())
                    mll.zero_grad()
                    return loss.item(), np.concatenate(grad)
                """
                hyperparameter_optimizer = torch.optim.Adam([
                    {'params': var_gp.hyperparameters()}
                ], lr=0.01)
                num_epochs = 50
                for i in range(num_epochs):
                    for x_batch, y_batch in train_loader:
                        ### Perform NGD step to optimize variational parameters
                        variational_ngd_optimizer.zero_grad()
                        hyperparameter_optimizer.zero_grad()
                        output = var_gp(x_batch)
                        loss = -var_mll(output, y_batch)
                        loss.backward()
                        #import pdb
                        #pdb.set_trace()
                        print(list(var_gp.variational_parameters()))
                        print(var_gp.variational_strategy.inducing_points)
                        variational_ngd_optimizer.step()
                        hyperparameter_optimizer.step()
                """
                theta_star = x0
                f_opt_star = np.inf
                for start_point in start_points:
                    theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(
                        sci_opi_wrapper,
                        start_point,
                        args=(var_mll, property_dict, out_x_, out_y_),
                        bounds=bounds,
                        maxiter=50,
                    )

                    if f_opt < f_opt_star:
                        f_opt_star = f_opt
                        theta_star = theta


                start_idx = 0
                for p_name, attrs in property_dict.items():
                    if p_name != "model.variational_strategy.inducing_points":
                        # Construct the new tensor
                        if len(attrs.shape) == 0:  # deal with scalar tensors
                            start_idx = start_idx + 1
                        else:
                            start_idx = start_idx + np.prod(attrs.shape)
                    else:
                        end_idx = start_idx + np.prod(attrs.shape)
                        inducing_points = torch.tensor(
                            theta_star[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
                        ).view(*attrs.shape)
                        break

                #inducing_points = dict(var_gp.named_hyperparameters())['variational_strategy.inducing_points']

                self.gp_model.initialize(**{'covar_module.inducing_points': inducing_points})


                #"""
                """
                optimizer = torch.optim.Adam([{"params": self.gp.model.parameters()}], lr=0.01)


                for _ in range(50):
                    optimizer.zero_grad()
                    output = self.gp.model(self.gp.model.train_inputs[0])
                    loss = -self.gp(output, self.gp.model.train_targets)
                    loss.backward()
                    optimizer.step()
                """
            print("@"*50)
            time2 = time.time()
            print(f"time used for inducingPoints optimizing: {(time2 - time1):.4f}")
        else:
            self.hypers, self.property_dict, _ = module_to_array(module=self.gp)
        time4 = time.time()
        print("#"*50)
        print(f"time used for hps optimizing: {(time4 - time0):.4f}")
        self.is_trained = True
        return self

    def _get_gp(self,
                in_x: typing.Optional[np.ndarray] = None,
                in_y: typing.Optional[np.ndarray] = None,
                out_x: typing.Optional[np.ndarray] = None,
                out_y: typing.Optional[np.ndarray] = None) -> typing.Optional[ExactMarginalLogLikelihood]:
        if in_x is None:
            return None

        in_x = torch.from_numpy(in_x)
        in_y = torch.from_numpy(in_y)
        if out_x is None:
            self.gp_model = ExactGPModel(in_x, in_y, likelihood=self.likelihood, base_covar_kernel=self.kernel).double()
        else:
            #self.gp_model = ExactGPModel(in_x, in_y, likelihood=self.likelihood, base_covar_kernel=self.kernel)

            out_x = torch.from_numpy(out_x)
            out_y = torch.from_numpy(out_y)
            # weights = torch.ones(out_y.shape[0]) / out_y.shape[0]
            # inducing_points = out_x[torch.multinomial(weights, self.num_inducing_points)]
            if self.local_gp == 'fitc':
                weights = torch.ones(out_y.shape[0]) / out_y.shape[0]
                inducing_points = out_x[torch.multinomial(weights, self.num_inducing_points)]
                inducing_points = torch.cat([inducing_points, in_x])
                kernel = InducingPointKernel(self.kernel, inducing_points=torch.cat(([inducing_points, in_x])),
                                             likelihood=self.likelihood)
                self.gp_model = ExactGPModel(torch.cat([in_x, out_x]), torch.cat([in_y, out_y]),
                                             likelihood=self.likelihood,
                                             base_covar_kernel=kernel).double()
            else:
                if self.num_inducing_points <= in_y.shape[0]:
                    weights = torch.ones(in_y.shape[0]) / in_y.shape[0]
                    inducing_points = in_x[torch.multinomial(weights, self.num_inducing_points)]
                else:
                    weights = torch.ones(out_y.shape[0]) / out_y.shape[0]
                    inducing_points = out_x[torch.multinomial(weights, self.num_inducing_points - in_y.shape[0])]
                    inducing_points = torch.cat([inducing_points, in_x])
                self.gp_model = PartailSparseGPModel(in_x, in_y, out_x, out_y,
                                                     likelihood=self.likelihood,
                                                     base_covar_kernel=self.kernel,
                                                     inducing_points=inducing_points).double()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        mll.double()
        return mll

    def _optimize(self) -> np.ndarray:
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """
        if isinstance(self.gp_model, PartailSparseGPModel):
            self.gp_model.deactivate_inducing_points_grad()
        x0, property_dict, bounds = module_to_array(module=self.gp)
        self.property_dict = property_dict

        bounds = np.asarray(bounds).transpose().tolist()
        #x0 = x0.astype(np.float64)


        p0 = [x0]

        while len(p0) < self.n_opt_restarts:
            try:
                self.gp.pyro_sample_from_prior()
                sample, _, _ = module_to_array(module=self.gp)
                p0.append(sample.astype(np.float64))
            except Exception as e:
                continue

        self.gp_model.train()
        self.likelihood.train()

        theta_star = x0
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            try:
                theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(
                    _scipy_objective_and_grad,
                    start_point,
                    args=(self.gp, property_dict),
                    bounds=bounds,
                )

            except RuntimeError as e:
                self.logger.warning(f"Fail to optimize as an Error occurs: {e}")
                continue
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta

        return theta_star

    def _predict(self, X_test: np.ndarray,
                 cov_return_type: typing.Optional[str] = 'diagonal_cov') \
            -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        cov_return_type: typing.Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) or None
            predictive variance or standard deviation

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        X_test = torch.from_numpy(self._impute_inactive(X_test))
        self.likelihood.eval()
        self.gp_model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(X_test))

            mu = observed_pred.mean.numpy()
            if cov_return_type is None:
                var = None

                if self.normalize_y:
                    mu = self._untransform_y(mu)

            else:
                if cov_return_type != 'full_cov':
                    var = observed_pred.stddev.numpy()
                    var = var ** 2  # since we get standard deviation for faster computation
                else:
                    # output full covariance
                    var = observed_pred.covariance_matrix().numpy()

                # Clip negative variances and set them to the smallest
                # positive float value
                var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

                if self.normalize_y:
                    mu, var = self._untransform_y(mu, var)

                if cov_return_type == 'diagonal_std':
                    var = np.sqrt(var)  # converting variance to std deviation if specified

        return mu, var

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        X_test = torch.from_numpy(self._impute_inactive(X_test))
        with torch.no_grad:
            funcs = self.likelihood(self.gp_model(X_test)).sample(torch.Size([n_funcs])).t().cpu().detach().numpy()

        if self.normalize_y:
            funcs = self._untransform_y(funcs)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs



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

        bound_left = bounds_cont[:, 0] - np.min(X[in_ss_dims][:, cont_dims] - bounds_cont[:, 0], axis=0)
        bound_right = bounds_cont[:, 1] + np.min(bounds_cont[:, 1] - X[in_ss_dims][:, cont_dims], axis=0)
        in_ss_dims = np.all(X[:, cont_dims] <= bound_right, axis=1) & \
                     np.all(X[:, cont_dims] >= bound_left, axis=1)
    else:
        in_ss_dims = np.ones(X.shape[-1], dtype=bool)

    for bound_cat, cat_dim in zip(bounds_cat, cat_dims):
        in_ss_dims &= np.in1d(X[:, cat_dim], bound_cat)

    # indices_in_ss = np.where(in_ss_dims)[0]
    return in_ss_dims
