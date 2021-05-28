import torch

from psgp.partial_sparse_kernel import PartialSparseKernel
from gpytorch.means.mean import Mean
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify
from gpytorch.constraints.constraints import Interval

class PartialSparseMean(Mean):
    def __init__(self, covar_module: PartialSparseKernel, prior=None, batch_shape=torch.Size(), **kwargs):
        super(PartialSparseMean, self).__init__()
        self.covar_module = covar_module
        self.batch_shape = batch_shape
        self.covar_module = covar_module
        #self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        #self.register_constraint(param_name="constant", constraint=Interval(-2, 2))
        #if prior is not None:
        #    self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        res = self.covar_module.posterior_mean(input).detach()
        return res
