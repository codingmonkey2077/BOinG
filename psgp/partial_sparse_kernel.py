import copy
import math

import torch

from gpytorch.kernels.kernel import Kernel
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify
#from gpytorch.mlls import InducingPointKernelAddedLossTerm
from gpytorch.mlls import AddedLossTerm
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.constraints.constraints import Interval

class InducingPointKernelAddedLossTerm(AddedLossTerm):
    def __init__(self, variational_dist, prior_dist, likelihood):
        self.prior_dist = prior_dist
        self.variational_dist = variational_dist
        self.likelihood = likelihood

    def loss(self, *params):
        prior_covar = self.prior_dist.lazy_covariance_matrix
        variational_covar = self.variational_dist.lazy_covariance_matrix
        diag = prior_covar.diag() - variational_covar.diag()
        shape = prior_covar.shape[:-1]
        noise_diag = self.likelihood._shaped_noise_covar(shape, ).diag()
        return 0.5 * (diag / noise_diag).sum()


class PartialSparseKernel(Kernel):
    def __init__(self, base_kernel, inducing_points, likelihood, outer_points, outer_y, active_dims=None):
        super(PartialSparseKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.likelihood = likelihood

        if inducing_points.ndimension() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        self.outer_points = outer_points
        self.outer_y = outer_y
        self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        self.register_added_loss_term("inducing_point_loss_term")

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        if hasattr(self, "_cached_inducing_sigma"):
            del self._cached_inducing_sigma
        if hasattr(self, "_cached_poster_mean_mat"):
            del self._cached_poster_mean_mat
        if hasattr(self, "_train_cached_k_u1"):
            del self._train_cached_k_u1
        if hasattr(self, "_train_cached_inducing_sigma_inv_root"):
            del self._train_cached_inducing_sigma_inv_root
        if hasattr(self, "_train_cached_lambda_diag_inv"):
            del self._train_cached_lambda_diag_inv
        if hasattr(self, "_cached_posterior_mean"):
            del self._cached_posterior_mean
        return super(PartialSparseKernel, self).train(mode)

    @property
    def _inducing_mat(self):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = delazify(self.base_kernel(self.inducing_points, self.inducing_points))
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_mat, upper=True, jitter=settings.cholesky_jitter.value())
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            if not self.training:
                self._cached_kernel_inv_root = res
            return res

    @property
    def _k_u1(self):
        if not self.training and hasattr(self, "_cached_k_u1"):
            return self._cached_k_u1
        else:
            res = delazify(self.base_kernel(self.inducing_points, self.outer_points))
            if not self.training:
                self._cached_k_u1 = res
            else:
                self._train_cached_k_u1 = res.clone()
            return res

    @property
    def _lambda_diag_inv(self):
        if not self.training and hasattr(self, "_cached_lambda_diag_inv"):
            return self._cached_lambda_diag_inv
        else:
            diag_k11 = delazify(self.base_kernel(self.outer_points, diag=True))

            diag_q11 = delazify(RootLazyTensor(self._k_u1.transpose(-1, -2).matmul(self._inducing_inv_root))).diag()

            # Diagonal correction for predictive posterior
            correction = (diag_k11 - diag_q11).clamp(0, math.inf)

            sigma = self.likelihood._shaped_noise_covar(correction.shape).diag()

            res = delazify(DiagLazyTensor((correction + sigma).reciprocal()))


            if not self.training:
                self._cached_lambda_diag_inv = res
            else:
                self._train_cached_lambda_diag_inv = res.clone()
            return res

    @property
    def _inducing_sigma(self):
        if not self.training and hasattr(self, "_cached_inducing_sigma"):
            return self._cached_inducing_sigma
        else:
            k_u1 = self._k_u1
            res = PsdSumLazyTensor(self._inducing_mat, MatmulLazyTensor(k_u1, MatmulLazyTensor(self._lambda_diag_inv,
                                                                                               k_u1.transpose(-1, -2))))
            res = delazify(res)
            if not self.training:
                self._cached_inducing_sigma = res

            return res

    @property
    def _inducing_sigma_inv_root(self):
        if not self.training and hasattr(self, "_cached_inducing_sigma_inv_root"):
            return self._cached_inducing_sigma_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_sigma, upper=True, jitter=settings.cholesky_jitter.value())

            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]
            res = inv_root
            if not self.training:
                self._cached_inducing_sigma_inv_root = res
            else:
                self._train_cached_inducing_sigma_inv_root = res.clone()
            return res

    @property
    def _poster_mean_mat(self):
        if not self.training and hasattr(self, "_cached_poster_mean_mat"):
            return self._cached_poster_mean_mat
        else:
            inducing_sigma_inv_root = self._inducing_sigma_inv_root
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._k_u1
            lambda_diag_inv = self._lambda_diag_inv

            res_mat = delazify(MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))

            res = torch.matmul(res_mat, self.outer_y)

            if not self.training:
                self._cached_poster_mean_mat = res
            return res

    def _get_covariance(self, x1, x2):
        k_x1x2 = self.base_kernel(x1, x2)
        k_x1u = delazify(self.base_kernel(x1, self.inducing_points))
        inducing_inv_root = self._inducing_inv_root
        inducing_sigma_inv_root = self._inducing_sigma_inv_root
        if torch.equal(x1, x2):
            q_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_inv_root))

            s_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_sigma_inv_root))
        else:
            k_x2u = delazify(self.base_kernel(x2, self.inducing_points))
            q_x1x2 = MatmulLazyTensor(
                k_x1u.matmul(inducing_inv_root), k_x2u.matmul(inducing_inv_root).transpose(-1, -2)
            )
            s_x1x2 = MatmulLazyTensor(
                k_x1u.matmul(inducing_sigma_inv_root), k_x2u.matmul(inducing_sigma_inv_root).transpose(-1, -2)
            )
        covar = PsdSumLazyTensor(k_x1x2, -1. * q_x1x2, s_x1x2)

        if self.training:
            k_iu = self.base_kernel(x1, self.inducing_points)
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._train_cached_k_u1 if hasattr(self, "_train_cached_k_u1") else self._k_u1
            lambda_diag_inv = self._train_cached_lambda_diag_inv \
                if hasattr(self, "_train_cached_lambda_diag_inv") else self._lambda_diag_inv

            mean = torch.matmul(delazify(MatmulLazyTensor(k_iu, MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))), self.outer_y)

            self._cached_posterior_mean = mean
        return covar

    def _covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = delazify(self.base_kernel(inputs, diag=True))
        return DiagLazyTensor(covar_diag)

    def posterior_mean(self, inputs):
        if self.training and hasattr(self, "_cached_posterior_mean"):
            return self._cached_posterior_mean
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        k_iu = delazify(self.base_kernel(inputs, self.inducing_points))
        poster_mean = self._poster_mean_mat
        res = torch.matmul(k_iu, poster_mean)
        return res

    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)
        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")

            zero_mean = torch.zeros_like(x1.select(-1, 0))
            new_added_loss_term = InducingPointKernelAddedLossTerm(
                MultivariateNormal(zero_mean, DiagLazyTensor(self.base_kernel(self.outer_points, self.outer_points, diag=True))),
                MultivariateNormal(zero_mean, RootLazyTensor(self._k_u1.transpose(-1, -2).matmul(self._inducing_inv_root))),
                self.likelihood,
            )

            #self.update_added_loss_term("inducing_point_loss_term", new_added_loss_term)

        if diag:
            return covar.diag()
        else:
            return covar

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def __deepcopy__(self, memo):
        replace_inv_root = False
        replace_kernel_mat = False
        replace_k_u1 = False
        replace_lambda_diag_inv = False
        replace_inducing_sigma = False
        replace_inducing_sigma_inv_root = False
        replace_poster_mean = False

        if hasattr(self, "_cached_kernel_inv_root"):
            replace_inv_root = True
            kernel_inv_root = self._cached_kernel_inv_root
        if hasattr(self, "_cached_kernel_mat"):
            replace_kernel_mat = True
            kernel_mat = self._cached_kernel_mat
        if hasattr(self, "_cached_k_u1"):
            replace_k_u1 = True
            k_u1 = self._cached_k_u1
        if hasattr(self, "_cached_lambda_diag_inv"):
            replace_lambda_diag_inv = True
            lambda_diag_inv = self._cached_lambda_diag_inv
        if hasattr(self, "_cached_inducing_sigma"):
            replace_inducing_sigma = True
            inducing_sigma = self._cached_inducing_sigma
        if hasattr(self, "_cached_inducing_sigma_inv_root"):
            replace_inducing_sigma_inv_root = True
            inducing_sigma_inv_root = self._cached_inducing_sigma_inv_root
        if hasattr(self, "_cached_poster_mean_mat"):
            replace_poster_mean = True
            poster_mean_mat = self._cached_poster_mean_mat

        cp = self.__class__(
            base_kernel=copy.deepcopy(self.base_kernel),
            inducing_points=copy.deepcopy(self.inducing_points),
            likelihood=self.likelihood,
            active_dims=self.active_dims,
        )

        if replace_inv_root:
            cp._cached_kernel_inv_root = kernel_inv_root

        if replace_kernel_mat:
            cp._cached_kernel_mat = kernel_mat

        if replace_k_u1:
            cp._cached_k_u1 = k_u1

        if replace_lambda_diag_inv:
            cp._cached_lambda_diag_inv = lambda_diag_inv

        if replace_inducing_sigma:
            cp._inducing_sigma = inducing_sigma

        if replace_inducing_sigma_inv_root:
            cp._cached_inducing_sigma_inv_root = inducing_sigma_inv_root

        if replace_poster_mean:
            cp._cached_poster_mean_mat = poster_mean_mat

        return cp