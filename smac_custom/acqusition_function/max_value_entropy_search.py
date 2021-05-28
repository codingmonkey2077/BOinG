"""
This file is the python implementation of Max Value Entropy Search:
Max-value Entropy Search for Efficient Bayesian Optimization (Wang and Jegelka, in ICML 2017)
https://arxiv.org/pdf/1703.01968.pdf
and the code is implemented based on its matlab implementation:
https://github.com/zi-w/Max-value-Entropy-Search
"""
from typing import List, Tuple, Any, Dict, Callable, Union
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import copy

from smac.configspace import Configuration, ConfigurationSpace
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunHistory
from smac.epm.base_epm import AbstractEPM
from smac.epm.gaussian_process import GaussianProcess
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer
from smac.optimizer.acquisition import AbstractAcquisitionFunction, IntegratedAcquisitionFunction
from smac.configspace.util import convert_configurations_to_array
from smac.epm.util_funcs import get_types

from scipy.optimize import minimize_scalar


class AbstractMES(AbstractAcquisitionFunction):
    """
        Python implementation of Max-value entropy search acquisition function,
        here we sample y_star from the approximate Gumble distribution

        Attributes
        ----------
        nK: int
            the number of samples of the optimal values y_star
        eta: float
            the best evaluated value
        num_mgrid : int
            number of samples to build mgrid
        """

    def __init__(self,
                 model: GaussianProcess,
                 nK: int = 100):
        """
        # TODO decide when to sample the optimal values instead of sampling from
        """
        super(AbstractMES, self).__init__(model)
        self.long_name = 'Max-value entropy search'
        self.nK = nK
        self.eta = None
        self.y_star = None
        self.sigma_0 = None
        self._required_updates = ('model', 'eta')

    def update(self, X_dis: np.ndarray, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        here the value of y_star needs to be estimated here

        Parameters
        ----------
        X_dis: np.ndarray(N, D), The input points to discretize the space. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        kwargs
        """
        for key in self._required_updates:
            if key not in kwargs:
                raise ValueError(
                    'Acquisition function %s needs to be updated with key %s, but only got '
                    'keys %s.'
                    % (self.__class__.__name__, key, list(kwargs.keys()))
                )
        for key in kwargs:
            if key in self._required_updates:
                setattr(self, key, kwargs[key])

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the entropy reduction values.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            approximated entropy reduction when sampling X
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        m, v = self.model.predict(X)
        m = -m
        # we set v to be a very small value instead setting it as 1 when variance is 0 or negative
        v1 = np.where(v <= self.sigma_0 + 1e-9, self.sigma_0 + 1e-9, v)
        s = np.sqrt(v1)
        N = m.shape[0]
        # y_star_nk, m, s are of size N*K
        y_star_nk = np.tile(- self.y_star, (N, 1))
        m_tile = np.tile(m, (1, self.nK))
        s_tile = np.tile(s, (1, self.nK))
        gamma = (y_star_nk - m_tile) / s_tile

        pdf_gamma = norm.pdf(gamma)
        cdf_gamma = norm.cdf(gamma)
        f = np.mean(gamma * pdf_gamma / (2 * cdf_gamma) - np.log(cdf_gamma), axis=1, keepdims=True)

        return f


class MaxValueEntropySearchG(AbstractMES):
    """
    Python implementation of Max-value entropy search acquisition function,
    here we sample y_star from the approximate Gumble distribution

    Attributes
    ----------
    nK: int
        the number of samples of the optimal values y_star
    eta: float
        the best evaluated value
    num_mgrid : int
        number of samples to build mgrid
    """
    def __init__(self,
                 model: GaussianProcess,
                 nK: int = 100,
                 num_mgrid: int = 100):
        """
        # TODO decide when to sample the optimal values instead of sampling from
        """
        super(MaxValueEntropySearchG, self).__init__(model)
        self.long_name = 'Max-value entropy search'
        self.nK = nK
        self.eta = None
        self.num_mgrid = num_mgrid
        self.y_star = None
        self.sigma_0 = None
        self._required_updates = ('model', 'eta')

    def update(self, X_dis: np.ndarray, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        here the value of y_star needs to be estimated here

        Parameters
        ----------
        X_dis: np.ndarray(N, D), The input points to discretize the space. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        kwargs
        """
        super(MaxValueEntropySearchG, self).update(X_dis, **kwargs)
        if len(X_dis.shape) == 1:
            X_dis = X_dis[np.newaxis, :]
        sigma_0 = np.exp(self.model.hypers[-1]) * self.model.std_y_
        self.sigma_0 = sigma_0

        x_train = self.model.gp.X_train_

        m, v = self.model.predict(np.vstack([X_dis, x_train]))
        m = - m
        # we set v to be a very small value instead setting it as 1 when variance is 0 or negative
        v1 = np.where(v <= sigma_0+1e-9, sigma_0+1e-9, v)
        s = np.sqrt(v1)

        #y_star = np.empty([self.nK, ])
        # TODO define mk to store the computed values, consider if use maximizer(n points here or in maximizer)...
        # TODO after selecting the max value, considering if re optimization with GP Process
        left = - self.eta
        ncdf = lambda y: np.prod(norm.cdf((y - m) / s))
        if ncdf(left) < 0.25:
            right = np.max(m + 5 * s)
            # normalized cdf
            while ncdf(right) < 0.75:
                right = right + right - left
            med = minimize_scalar(fun=lambda x: (ncdf(x)-0.5)**2, bounds=[left, right],
                                  method='Bounded', tol=None, options={'maxiter': 1000, 'xatol': 0.01}).x
            q1 = minimize_scalar(fun=lambda x: (ncdf(x)-0.25)**2, bounds=[left, right],
                                 method='Bounded', tol=None, options={'maxiter': 1000, 'xatol': 0.01}).x
            q2 = minimize_scalar(fun=lambda x: (ncdf(x)-0.75)**2, bounds=[left, right],
                                 method='Bounded', tol=None, options={'maxiter': 1000, 'xatol': 0.01}).x
            beta = (q1 - q2) / (np.log(np.log(4 / 3)) - np.log(np.log(4)))
            alpha = med + beta * np.log(np.log(2))
            # Sample from the Gumbel distribution.
            y_star = - np.log(-np.log(np.random.rand(1, self.nK))) * beta + alpha
            y_star = np.where(y_star < left + 5 * np.sqrt(sigma_0), left + 5 * np.sqrt(sigma_0), y_star)
            self.y_star = - np.squeeze(y_star)
        else:
            y_star = np.full(self.nK, left + 5 * np.sqrt(sigma_0))
            self.y_star = - y_star


class MaxValueEntropySearchR(AbstractMES):
    """
    Python implementation of Max-value entropy search acquisition function,
    here we sample y_star from the approximate Thompson Sampling

    TODO:  with smac.optimizer.ei_optimization.IntegratedAcquisitionFunction
     and smac.epm.guassian_process_mcmc

    Attributes
    ----------
    nK: int
        the number of samples of the optimal values y_star
    eta: float
        the best evaluated value
    num_mgrid : int
        number of samples to build mgrid
    """
    def __init__(self,
                 model: GaussianProcess,
                 nK: int = 100,
                 n_features: int = 100):
        """
        # TODO decide when to sample the optimal values instead of sampling from
        """
        super(MaxValueEntropySearchR, self).__init__(model)
        self.long_name = 'Max-value entropy search R'
        self.nK = nK
        self.eta = None
        self.n_features = n_features
        self.y_star = None
        self.sigma_0 = None
        self._required_updates = ('model', 'eta')

    def update(self, X_dis: np.ndarray, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        here the value of y_star needs to be estimated here

        Parameters
        ----------
        X: np.ndarray(N, D), The input points to discretize the space. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        kwargs
        """
        super(MaxValueEntropySearchR, self).update(X_dis, **kwargs)
        #if len(X.shape) == 1:
        #    X = X[np.newaxis, :]

        x_train = self.model.gp.X_train_
        y_train = - self.model.gp.y_train_

        y_train_max = np.max(y_train)
        y_train = np.expand_dims(y_train, axis=-1)
        d = x_train.shape[-1]

        hypers = np.exp(self.model.hypers)
        sigma = hypers[0]
        sigma_0 = hypers[-1]
        l = hypers[1:-1]

        W = np.random.normal(size=(self.nK, self.n_features, d)) * np.sqrt(l)
        b = 2 * np.pi * np.random.rand(self.nK, self.n_features, 1)
        Z = np.sqrt(2*sigma/self.n_features) * np.cos(W @ np.transpose(x_train) + b)

        noise = np.random.normal(size=(self.nK, self.n_features, 1))
        if x_train.shape[0] < self.n_features:
            A = np.transpose(Z, (0, 2, 1)) @ Z + sigma_0 * np.eye(x_train.shape[0])
            inv_tmp = np.linalg.inv(np.linalg.cholesky(A))
            A_inv = np.transpose(inv_tmp, (0, 2, 1)) @ inv_tmp
            mu = Z @ A_inv @ y_train
            w, v = np.linalg.eig(A)
            w = np.expand_dims(w, axis=-1)
            R = 1. / (np.sqrt(w) * (np.sqrt(w) + np.sqrt(sigma_0)))
            theta = noise - (Z @  (v @ (R * (np.transpose(v, (0, 2, 1)) @ (np.transpose(Z, (0, 2, 1)) @ noise))))) + mu
        else:
            A = Z @ np.transpose(Z, (0, 2, 1)) / sigma_0 + np.eye(self.n_features)
            inv_tmp = np.linalg.inv(np.linalg.cholesky(A))
            A_inv = np.transpose(inv_tmp, (0, 2, 1)) @ inv_tmp
            mu = A_inv @ Z @ y_train / sigma_0
            theta = mu + np.linalg.cholesky(A_inv) @ noise

        tagert_vector = lambda x: \
            np.squeeze(np.sqrt(2 * sigma / self.n_features) * np.transpose(theta, (0, 2, 1)) @
                       np.cos(W @ np.transpose(x, (0, 2, 1)) + b), axis=1)
        target_grident = lambda x: \
            np.squeeze((np.transpose(theta, (0, 2, 1)) @ -np.sqrt(2 * sigma / self.n_features) *
              np.tile(np.sin(W @ np.transpose(x)) + b), (1, 1, d)) * W, axis=1)
        xgrid = np.random.rand(self.nK, 1000, d)
        ygrid = tagert_vector(xgrid)

        y_star = np.min(ygrid, axis=1)

        y_star = np.where(y_star > y_train_max - 5 * np.sqrt(sigma_0), y_train_max - 5 * np.sqrt(sigma_0), y_star)
        y_star = y_star * self.model.std_y_ + self.model.mean_y_

        self.sigma_0 = sigma_0
        self.y_star = y_star


class IntegratedMES(IntegratedAcquisitionFunction):
    def __init__(self,
                 model: AbstractEPM,
                 acquisition_function: AbstractMES,
                 nK: int = 100,
                 **kwargs: Any):
        super(IntegratedMES, self).__init__(model, acquisition_function, **kwargs)
        self.nK = nK
        self.y_star = np.empty([1, self.nK])

    def update(self, X_dis: np.ndarray, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        here the value of y_star needs to be estimated here

        Parameters
        ----------
        X_dis: np.ndarray(N, D), The input points to discretize the space. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        kwargs
        """
        model = kwargs['model']
        del kwargs['model']
        if not hasattr(model, 'models') or len(model.models) == 0:
            raise ValueError('IntegratedAcquisitionFunction requires at least one model to integrate!')
        if len(self._functions) == 0 or len(self._functions) != len(model.models):
            self._functions = [copy.deepcopy(self.acq) for _ in model.models]
        for submodel, func in zip(model.models, self._functions):
            func.update(X_dis=X_dis, model=submodel, **kwargs)


    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the configuration that maximize the entropy reduction

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self._functions is None:
            raise ValueError('Need to call update first!')
        f = np.array([func._compute(X) for func in self._functions])
        self.y_star = np.vstack([func.y_star for func in self._functions])

        return f.mean(axis=0)


class MESMaximazier(AcquisitionFunctionMaximizer):
    # TODO decide when to evaluate the value that maximize the posterior rahter than the acqusition funciton
    def __init__(
            self,
            acquisition_function: IntegratedMES,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            num_samples: int = 1000,
    ):
        super(MESMaximazier, self).__init__(acquisition_function, config_space, rng)
        self.num_samples = num_samples
        _, bounds = get_types(config_space=config_space, instance_features=None)
        self.bounds = bounds

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int) -> List[Tuple[float, Configuration]]:
        candidates = []
        acq_val_candidates = []

        samples = self.config_space.sample_configuration(size=self.num_samples) # TODO: modify here to be sobel sequent?
        f = self.acquisition_function(samples)
        idx_max = np.argmax(f)
        max_value = f[idx_max]
        x_start = convert_configurations_to_array([samples[idx_max]])
        neg_ac_func = lambda x: -self.acquisition_function._compute(x)[0]
        kwargs_optimizer = {'maxiter': 100}
        """
        res = minimize(neg_ac_func, x0=x_start, method='L-BFGS-B', bounds=self.bounds,
                       options=kwargs_optimizer)
        if -res.fun < max_value:
            optimum = x_start
            optimum = optimum.flatten()
        else:
            optimum = res.x
            optimum = optimum.flatten()
        """
        optimum = x_start

        optimum = Configuration(self.config_space, vector=optimum)
        candidates.append(optimum)
        acq_val_candidates.append(f)

        return [(0, candidates[i]) for i in range(len(candidates))]
