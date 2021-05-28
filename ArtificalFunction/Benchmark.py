import numpy as np


class TestFunc():
    def __init__(self, dim, lb, ub):
        self.dim = dim
        self.lb = lb * np.ones(dim)
        self.ub = ub * np.ones(dim)

    def __call__(self, x):
        self._check_input(x)
        return self._compute(x)

    def _check_input(self, x):
        assert x.size == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

    def _compute(self, x):
        raise NotImplementedError


class Griewank(TestFunc):
    def __init__(self, dim=2, lb=-600., ub=600.):
        super(Griewank, self).__init__(dim, lb, ub)
        self.global_optimum = np.zeros(dim)

    def _compute(self, x):
        val = np.sum(x ** 2) / 4000 - np.prod(np.cos(x) / (np.arange(self.dim) + 1)) + 1
        return val


class Ackley(TestFunc):
    def __init__(self, dim=2, lb=-32.768, ub=32.768):
        super(Ackley, self).__init__(dim, lb, ub)
        self.global_optimum = np.zeros(dim)

    def _compute(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi

        pow1 = - b * np.linalg.norm(x) / np.sqrt(self.dim)
        pow2 = np.mean(np.cos(c*x))
        val = -a * np.exp(pow1) - np.exp(pow2) + a + np.exp(1)
        return val


class Levy(TestFunc):
    def __init__(self, dim=2, lb=-5., ub=10.):
        super(Levy, self).__init__(dim, lb, ub)
        self.global_optimum = np.ones(dim)

    def _compute(self, x):
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
              np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
              (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1]) ** 2)
        return val


class Rosenbrock(TestFunc):
    def __init__(self, dim=2, lb=-5., ub=10.):
        super(Rosenbrock, self).__init__(dim, lb, ub)
        self.global_optimum = np.ones(dim)

    def _compute(self, x):
        x_i1 = np.array(x[1:])
        x_i = np.array(x[:-1])
        val = np.sum(100 * (x_i1 - x_i ** 2) ** 2 + (x_i - 1)**2)
        return val


class BukinN6(TestFunc):
    def __init__(self, dim=2, lb=None, ub=None):
        assert dim == 2
        self.dim = 2
        self.lb = np.array([-15, -3])
        self.ub = np.array([-5, 3])
        self.global_optimum = np.array([-10, 1])

    def _compute(self, x):
        return 100 * np.sqrt(np.abs(x[1] - x[0])) + 0.01 * np.abs(x[0] + 1)


class Michalewicz(TestFunc):
    def __init__(self, dim=2, lb=0., ub=np.pi):
        super(Michalewicz, self).__init__(dim, lb, ub)
        self.m = 10
        self.global_optimum = np.array([2.20, 1.57]) if dim is 2 else None

    def _compute(self, x):
        i = np.arange(self.dim) + 1
        return - np.sum(np.sin(x) * np.power(np.sin(i*x/np.pi), 2 * self.m))


class Eggholder(TestFunc):
    def __init__(self, dim=2, lb=-512., ub=512.):
        super(Eggholder, self).__init__(dim, lb, ub)
        self.global_optimum = np.array([512., 404.2319]) if dim is 2 else None

    def _compute(self, x):
        return -(x[1] + 47)*np.sin(np.sqrt(np.abs(x[1] + 0.5 * x[0] + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47)))


class Branin(TestFunc):
    def __init__(self, dim=2, lb=None, ub=None):
        #assert dim == 2
        self.dim = dim
        if dim == 2:
            self.lb = np.array([-5, 0])
            self.ub = np.array([10, 15])
        else:
            self.lb = np.hstack([np.array([-5, 0]), np.zeros(dim - 2)])
            self.ub = np.hstack([np.array([10, 15]), np.ones(dim - 2)])
        self.global_optimum = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])

    def _compute(self, x):
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5. / np.pi
        r = 6
        s = 10
        t = 1. / (8 * np.pi)
        return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s




