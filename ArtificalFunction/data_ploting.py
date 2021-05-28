import os
import sys

syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)

import json
import numpy as np
import os
import glob
from skopt.benchmarks import branin, hart6
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from ArtificalFunction.Benchmark import *
from turbo import Turbo1


def plot_heatmap(func_name, samples, num_init, ss_region=None, title_supp='van'):
    fig, ax = plt.subplots()
    artifical_func = {'ackley': Ackley(2),
                      'levy': Levy(2),
                      'rosenbrock': Rosenbrock(2),
                      'griewank': Griewank(2),
                      'michalewicz': Michalewicz(2)}

    if func_name is 'branin':
        func = branin
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
    else:
        func = artifical_func[func_name]
        lb = func.lb
        ub = func.ub

    x1_values = np.linspace(lb[0], ub[0], 100)
    x2_values = np.linspace(lb[1], ub[1], 100)

    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([func(val) for val in vals], (100, 100))

    cm = ax.pcolormesh(x_ax, y_ax, fx,
                       norm=LogNorm(vmin=fx.min(),
                                    vmax=fx.max()),
                       )
    if func_name == 'branin':
        minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    else:
        minima = np.array([func.global_optimum])

    if ss_region is not None:
        # Create a Rectangle patch
        x_lb = ss_region[0][0] * (ub[0] - lb[0]) + lb[0]
        x_ub = ss_region[0][1] * (ub[0] - lb[0]) + lb[0]
        y_lb = ss_region[1][0] * (ub[1] - lb[1]) + lb[1]
        y_ub = ss_region[1][1] * (ub[1] - lb[1]) + lb[1]

        rect = patches.Rectangle((x_lb, y_lb), x_ub - x_lb, y_ub - y_lb, linewidth=1, edgecolor='red', facecolor='none',
                                 label='sub space')

        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=7,
            lw=0, label="Minima")

    ax.plot(samples[num_init: -1, 0], samples[num_init: -1, 1], "kx", markersize=7,
            lw=0, label="Sample")

    ax.plot(samples[-1, 0], samples[-1, 1], "rx", markersize=7,
            lw=0, label="new Sample")

    ax.plot(samples[:num_init, 0], samples[:num_init, 1], "x", color='cyan', markersize=7,
            lw=0, label="Init")

    cb = fig.colorbar(cm)
    cb.set_label("sampled points")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("$X_0$")
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylabel("$X_1$")
    ax.set_ylim(lb[1], ub[1])

    ax.set_title('{} {}'.format(func_name, title_supp))


def plot_countour(func_name, samples, num_init, ss_region=None, title_supp='van'):
    fig, ax = plt.subplots()
    artifical_func = {'ackley': Ackley(2),
                      'levy': Levy(2),
                      'rosenbrock': Rosenbrock(2),
                      'griewank': Griewank(2)}

    if func_name is 'branin':
        func = branin
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
    else:
        func = artifical_func[func_name]
        lb = func.lb
        ub = func.ub

    x1_values = np.linspace(lb[0], ub[0], 100)
    x2_values = np.linspace(lb[1], ub[1], 100)

    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([func(val) for val in vals], (100, 100))

    cs = ax.contour(x1_values, x2_values, fx)

    if func_name == 'branin':
        minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    else:
        minima = np.array([func.global_optimum])

    if ss_region is not None:
        # Create a Rectangle patch
        x_lb = ss_region[0][0] * (ub[0] - lb[0]) + lb[0]
        x_ub = ss_region[0][1] * (ub[0] - lb[0]) + lb[0]
        y_lb = ss_region[1][0] * (ub[1] - lb[1]) + lb[1]
        y_ub = ss_region[1][1] * (ub[1] - lb[1]) + lb[1]

        rect = patches.Rectangle((x_lb, y_lb), x_ub - x_lb, y_ub - y_lb, linewidth=1, edgecolor='red', facecolor='none',
                                 label='sub space')

        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.plot(minima[:, 0], minima[:, 1], "r.", markersize=7,
            lw=0, label="Minima")

    ax.plot(samples[num_init: -1, 0], samples[num_init: -1, 1], "kx", markersize=7,
            lw=0, label="Sample")

    ax.plot(samples[-1, 0], samples[-1, 1], "rx", markersize=7,
            lw=0, label="new Sample")

    ax.plot(samples[:num_init, 0], samples[:num_init, 1], "x", color='cyan', markersize=7,
            lw=0, label="Init")

    ax.legend(loc="best", numpoints=1)

    ax.set_xlabel("$X_0$")
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylabel("$X_1$")
    ax.set_ylim(lb[1], ub[1])

    ax.set_title('{} {}'.format(func_name, title_supp))


def plot_subregion(func_name, samples, num_init, ss_regions):
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.075)

    artifical_func = {'ackley': Ackley(2),
                      'levy': Levy(2),
                      'rosenbrock': Rosenbrock(2),
                      'griewank': Griewank(2),
                      'michalewicz': Michalewicz(2),
                      'eggholder': Eggholder(2)}

    if func_name is 'branin':
        func = branin
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
    else:
        func = artifical_func[func_name]
        lb = func.lb
        ub = func.ub

    x1_values = np.linspace(lb[0], ub[0], 100)
    x2_values = np.linspace(lb[1], ub[1], 100)

    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([func(val) for val in vals], (100, 100))

    cm = ax.contour(x_ax, y_ax, fx, np.exp(np.linspace(np.log(fx.min()), np.log(fx.max()), 15)),
                       norm=LogNorm(vmin=fx.min(),
                                    vmax=fx.max()),
                       )
    if func_name == 'branin':
        minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    else:
        minima = np.array([func.global_optimum])

    for i, ss_region in enumerate(ss_regions):
        # Create a Rectangle patch
        x_lb = ss_region[0][0] * (ub[0] - lb[0]) + lb[0]
        x_ub = ss_region[0][1] * (ub[0] - lb[0]) + lb[0]
        y_lb = ss_region[1][0] * (ub[1] - lb[1]) + lb[1]
        y_ub = ss_region[1][1] * (ub[1] - lb[1]) + lb[1]

        rect = patches.Rectangle((x_lb, y_lb), x_ub - x_lb, y_ub - y_lb, linewidth=2,
                                 edgecolor=(0., 0., 0.), facecolor='red', alpha=i / (3 * len(ss_regions)),
                                 label="subregions" if i == len(ss_regions) - 1 else None)

        # Add the patch to the Axes
        ax.add_patch(rect)
        rect = patches.Rectangle((x_lb, y_lb), x_ub - x_lb, y_ub - y_lb, linewidth=2,
                                 edgecolor=(0., 0., 0.), fill=False, alpha=0.4)
        ax.add_patch(rect)

    # samples = samples[0]

    for i in range(len(samples) // 10):
        ax.plot(samples[10 * i: 10 * i + 10, 0], samples[10 * i: 10 * i + 10, 1], ".", color='black',
                alpha=i / (len(samples) // 10.), markersize=20,
                lw=0, label="Samples" if i == len(samples) // 10 - 1 else None)

    ax.plot(samples[:num_init, 0], samples[:num_init, 1], ".", color='cyan', markersize=20,
            lw=0, label="Init")

    ax.plot(minima[:, 0], minima[:, 1], "rX", markersize=25,
            lw=0, label="Minima")
    cb = fig.colorbar(cm)
    """
    ax.legend(loc="best", numpoints=1)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 2.4, .102), loc='lower left',
                       ncol=4, mode="expand", borderaxespad=0.)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend_traj.png', dpi='figure', bbox_inches=bbox)
    exit()
    """

    # ax.set_xlabel("$X_0$")
    # ax.set_xlim(lb[0], ub[0])
    # ax.set_ylabel("$X_1$")
    # ax.set_ylim(lb[1], ub[1])

    # ax.set_title('{}'.format(func_name))


# """
def plot_fraction_volume(func):
    bl = 'BOinG'
    file_bl = glob.glob(os.path.join(syspath, func, 'dim_10', bl, 'EI', 'data_42_0.json'))
    files_ss = glob.glob(os.path.join(syspath, func, 'dim_10', bl, 'EI', 'ss_data_42_0.json'))
    data_bl = []
    for file_name in files_ss:
        with open(file_name) as f:
            data_bl.append(json.load(f))
    data = np.array(data_bl[0]['ss_union'])

    data_bl_y = []
    for file_name in file_bl:
        with open(file_name) as f:
            data_bl_y.append(json.load(f))
    data_y = np.array(data_bl_y[0]['y'])

    ss_union = data[:, :, 1] - data[:, :, 0]
    ss_union_frac = np.prod(ss_union, axis=1)
    plt.plot(ss_union_frac, label='fraction, volume of subspace')
    y_acu = np.minimum.accumulate(data_y)
    x_incumbent_change = np.where((y_acu[20:] - y_acu[19:-1]) < 0)[0] + 1

    # plt.step(x=np.arange(np.size(y_acu[20:])) + 1, y=y_acu[20:], label='loss')
    for i in range(np.size(x_incumbent_change)):
        plt.axvline(x_incumbent_change[i], c='orange')

    plt.title('volume_subspace / whole_space (points are selected with GP in subspace) on {}'.format(func))
    plt.yscale('log')
    plt.legend()
    plt.rcParams['font.size'] = '15'
    plt.show()


def plot_fraction_data_points(func):
    bl = 'BOinG'
    plt.rcParams['font.size'] = '15'
    file_bl = glob.glob(os.path.join(syspath, func, 'dim_10', bl, 'EI', 'data_42_0.json'))
    files_ss = glob.glob(os.path.join(syspath, func, 'dim_10', bl, 'EI', 'ss_data_42_0.json'))
    data_bl = []
    for file_name in files_ss:
        with open(file_name) as f:
            data_bl.append(json.load(f))
    data = np.array(data_bl[0]['num_data'])
    data_ss = data[:, 0]
    data_all = data[:, 1]
    data_frac = data_ss / data_all

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('num of evaluations')
    ax1.set_ylabel('num_points', color=color)
    ax1.plot(data_ss, color=color, label='number of data in subspace')
    ax1.plot(data_all, label='number of all points')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('fraction of points in the subspace', color=color)  # we already handled the x-label with ax1
    ax2.plot(data_frac, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    leg = plt.legend()
    plt.yscale('log')
    plt.title('points in subspace versus all sampled points on {}'.format(func))
    plt.show()


def step_plot(data, num_eval, optimum, opt_name):
    data = data[:num_eval, :]
    data = np.minimum.accumulate(data - optimum, axis=1)
    data[data<=0] = 1e-12
    data_log = np.log(data)
    data_log[data_log<=0]
    print(opt_name)
    print(data[:, -1])
    print(np.mean(data[:, -1]))

    median = np.quantile(data, 0.5, axis=0)
    median = np.mean(data, axis=0)
    import scipy.stats as stats
    stand_err = stats.sem(data_log, axis=0)
    meadian_log = np.log(median)

    #upper = median + stand_err
    #lower = median - stand_err
    #median =np.exp(median)
    upper = meadian_log + stand_err
    lower = meadian_log - stand_err
    upper = np.exp(upper)
    lower = np.exp(lower)
    quan25 = np.quantile(data, 0.25, axis=0)
    quan75 = np.quantile(data, 0.75, axis=0)
    quan25 = upper
    quan75 = lower
    print(stand_err)
    x_values = np.arange(median.size) + 1

    plt.step(x=x_values, y=median, label="{}".format(opt_name))

    plt.fill_between(x=x_values, y1=quan25, y2=quan75, alpha=0.3, step='pre', interpolate=False, )
    plt.xlim(0, data_length)


# plot_fraction_volume('ackley')
# plot_fraction_data_points('ackley')
# exit()
# """


def plot_gp_prediction(func_name, samples, ss_region):
    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
    from smac.optimizer.acquisition import EI
    from smac_custom.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess
    from smac_custom.epm.gaussian_process_gyptorch import GaussianProcessGPyTorch
    from smac.epm.util_funcs import get_types

    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.constraints.constraints import Interval
    from gpytorch.priors import LogNormalPrior, UniformPrior
    import torch
    import copy

    torch.manual_seed(0)

    func = {'ackley': Ackley,
            'branin': Branin,
            'griewank': Griewank,
            'levy': Levy,
            'rosenbrock': Rosenbrock,
            'michalewicz': Michalewicz}[func_name](dim=2)

    bound_configs = np.vstack([func.lb, func.ub]).transpose().tolist()

    cs = ConfigurationSpace()
    cs.generate_all_continuous_from_bounds(bound_configs)

    types, bounds = get_types(cs)

    cont_dims = np.where(np.array(types) == 0)[0]
    cat_dims = np.where(np.array(types) != 0)[0]

    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
                                  transform=None,
                                  initial_value=1.0
                                  #initial_value=torch.tensor([[0.1863, 0.5024]], dtype=torch.float64)
                              ),
                              ard_num_dims=cont_dims.shape[-1])

    base_covar = ScaleKernel(exp_kernel,
                             outputscale_constraint=Interval(
                                 np.exp(-10.),
                                 np.exp(2.),
                                 transform=None,
                                 initial_value=2.0
                                 #initial_value=1.5761
                             ),
                             outputscale_prior=LogNormalPrior(0.0, 1.0))

    model = PartialSparseGaussianProcess(configspace=cs, types=types, bounds=bounds, bounds_cont=np.array(bounds),
                                         bounds_cat=np.empty((0,)), local_gp='local_gp', seed=np.random.randint(0, 2 ** 20),
                                         kernel=base_covar, num_inducing_points=4)

    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
                                  transform=None,
                                  initial_value=1.0
                                  #initial_value=torch.tensor([[0.1863, 0.5024]], dtype=torch.float64)
                              ),
                              ard_num_dims=cont_dims.shape[-1])

    base_covar = ScaleKernel(exp_kernel,
                             outputscale_constraint=Interval(
                                 np.exp(-10.),
                                 np.exp(2.),
                                 transform=None,
                                 initial_value=2.0
                                 #initial_value=1.5761
                             ),
                             outputscale_prior=LogNormalPrior(0.0, 1.0))

    model_fullGP = GaussianProcessGPyTorch(configspace=cs, types=types, bounds=bounds,
                                           bounds_cont=np.array(bounds), bounds_cat=np.empty((0,)),
                                           seed=np.random.randint(0, 2 ** 20),
                                           kernel=base_covar)

    cs_inner = ConfigurationSpace()
    ub = func.ub
    lb = func.lb

    x_lb = ss_region[0][0] * (ub[0] - lb[0]) + lb[0]
    x_ub = ss_region[0][1] * (ub[0] - lb[0]) + lb[0]
    y_lb = ss_region[1][0] * (ub[1] - lb[1]) + lb[1]
    y_ub = ss_region[1][1] * (ub[1] - lb[1]) + lb[1]

    cs_inner.generate_all_continuous_from_bounds([[x_lb, x_ub], [y_lb, y_ub]])


    samples_norm = (samples - func.lb) / (func.ub - func.lb)
    X = (samples_norm - ss_region[:,0] ) / (ss_region[:, 1] -ss_region[:,0])
    y = np.empty([X.shape[0], 1])
    for i in range(len(X)):
        y[i][0] = func(samples[i])
    in_ss_dims = np.all(X[:, cont_dims] <= 1, axis=1) & \
                 np.all(X[:, cont_dims] >= 0, axis=1)
    in_ss_dims = np.where(in_ss_dims)[0]


    model.train(X, y)

    mu_train_psgp, _ = model.predict(X)

    model_fullGP.train(X, y)
    mu_train_fullGP, _ = model.predict(X)

    resolustion = 150

    x1_values = np.linspace(0, 1, resolustion)
    x2_values = np.linspace(0, 1, resolustion)

    x_ax, y_ax = np.meshgrid(x1_values, x2_values)

    X = np.dstack([x_ax, y_ax]).reshape([-1, 2])

    X = (X - ss_region[:,0] ) / (ss_region[:, 1] -ss_region[:,0])

    mu_psgp, var_psgp = model.predict(X)
    if model_fullGP.is_trained:
        mu_fullgp, var_fullgp = model_fullGP.predict(X)
    else:
        mu_fullgp = np.empty([resolustion, resolustion])
        var_fullgp = np.empty([resolustion, resolustion])

    mu_psgp = mu_psgp.reshape([resolustion, resolustion])
    var_psgp = var_psgp.reshape([resolustion, resolustion])

    if hasattr(model.gp_model.covar_module, "inducing_points"):
        inducing_points = model.gp_model.covar_module.inducing_points.detach().numpy()
        inducing_points = inducing_points * (ss_region[:, 1] -ss_region[:,0]) + ss_region[:,0]


    mu_fullgp = mu_fullgp.reshape([resolustion, resolustion])
    var_fullgp = var_fullgp.reshape([resolustion, resolustion])

    from smac.optimizer.acquisition import EI
    from scipy.stats import norm

    def ei(mu, var, eta):
        std = np.sqrt(var)
        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            s_copy = np.copy(std)
            std[s_copy == 0.0] = 1.0
            z = (eta - mu) / std
            f = (eta - mu) * norm.cdf(z) + std * norm.pdf(z)
        else:
            z = (eta - mu) / std
            f = (eta - mu) * norm.cdf(z) + std * norm.pdf(z)
        f[f < 1e-8] = 1e-8
        return f


    ei_psgp = {'mu': mu_psgp,
               'var':var_psgp,
               'eta': np.min(mu_train_psgp[in_ss_dims])}

    ei_fullGP = {'mu': mu_fullgp,
                 'var': var_fullgp,
                 'eta': np.min(mu_train_fullGP[in_ss_dims])}

    y = ei(**ei_psgp)
    #y = mu_psgp


    y[y < 1e-8] = 1e-8


    fig, ax = plt.subplots()

    cm = ax.pcolormesh(x_ax, y_ax, y.reshape([resolustion, resolustion]),
                       norm=LogNorm(y.min(), y.max())
                       )
    cb = fig.colorbar(cm)
    cb.ax.set_title('Loss')

    rect = patches.Rectangle((ss_region[0][0], ss_region[1][0]), ss_region[0,1] - ss_region[0,0], ss_region[1,1] -
                             ss_region[1,0], linewidth=1,
                             edgecolor='red', fill=False)

    ax.add_patch(rect)

    if hasattr(model.gp_model.covar_module, "inducing_points"):
        ax.plot(inducing_points[:,0], inducing_points[:, 1], "x", color='cyan', markersize=7, lw=0, label='inducing points')

    ax.plot(samples_norm[:, 0], samples_norm[:, 1], "x", color='r', markersize=7,
            lw=0, label="Samples")

    ax.legend()
    ax.set_title("PSGP")
    plt.subplots_adjust(left=0.05, right=1.0, top=0.95, bottom=0.05)

    plt.show()

    exit()


def plot_trajectory(func_name='branin', n_dims=2, acq_func_name='EI'):
    bl = 'BOinG'
    van = 'vanilla',
    turbo = 'turbo'
    run = 19
    run_num = 0
    files = f'data_{run}_{run_num}.json'

    ss_files = f'ss_data_{run}_{run_num}.json'

    files_GP = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), "fullGP_GPytorch", acq_func_name, files)

    file_bl = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), bl, acq_func_name, files)

    file_turbo = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), turbo, files)

    file_ss = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), bl, acq_func_name,
                           ss_files)
    file_ss_turbo = os.path.join(syspath, 'ArtificalFunction', func_name, 'dim_{}'.format(n_dims), turbo, ss_files)

    data_bl = []
    data_ss = []
    data_van = []

    data_turbo = []
    data_turbo_ss = []

    with open(file_bl) as f:
        data_bl.append(json.load(f)['x'])
    with open(file_ss) as f:
        data_ss.append(json.load(f)['ss_union'])
        # data_ss.append(json.load(f))

    with open(files_GP) as f:
        data_van.append(json.load(f)['x'])

    # with open(file_turbo) as f:
    #    data = json.load(f)
    #    data_turbo.append(data['data_trained'])
    #    data_turbo_ss.append(data['ss_union'])

    quer = 0
    x_num = 29
    matplotlib.rcParams.update({'font.size': 30})
    #plot_gp_prediction(func_name=func_name, samples=np.array(data_bl[quer])[:x_num], ss_region=np.array(data_ss[quer])[x_num - 13])
    # plt.rcParams['font.size'] = '30'

    #plot_subregion(func_name=func_name, samples=np.array(data_bl[quer]), num_init=8, ss_regions=np.array(data_ss[quer][::10]))
    plot_subregion(func_name=func_name, samples=np.array(data_van[quer]), num_init=8, ss_regions=[])
    plt.show()
    exit()

    import pdb
    pdb.set_trace()

    for x_num in range(15, 99):
        plot_heatmap(func_name=func_name, samples=np.array(data_bl[quer])[:x_num], num_init=8,
                     ss_region=np.array(data_ss[quer])[x_num - 14], title_supp='bi-level')
        plt.savefig(os.path.join(syspath, 'tmp', 'sample_contour_bl_{}_st_{}.png'.format(quer, x_num)))
        # plt.show()
        plt.cla()


plot_trajectory('branin')
exit()

"""
for x_num in range(0, 99):
    plot_heatmap(func_name=func_name, samples=np.array(data_turbo[quer][x_num+1][:]),num_init=8, ss_region=np.array(data_turbo_ss[quer])[x_num], title_supp='turbo')
    plt.savefig(os.path.join(syspath, 'tmp', 'sample_contour_turbo_{}_at_{}.png'.format(quer, x_num+8)))
    plt.cla()

exit()
#"""

hist = np.zeros(shape=(0, 2))

syspath = os.path.dirname((os.path.abspath(__file__)))

func_name = 'levy'

smac_models = ['BOinG', 'fullGP', 'rf']
other_models = ['turbo', 'MCTS', 'rs']
smac_models = ['BOinG', 'local_gp', "fullGP_local"]
other_models= []
acq_func_names = ['LCB', 'EI', 'PI']
acq_func_names = ['EI']


n_dims = 10

if n_dims <= 2:
    data_length = 100
elif n_dims <= 6:
    data_length = 200
else:
    data_length = 400

files = 'data_*_0.json'

ss_files = 'ss_data_*_0.json'
files_smac = {}
files_others = {}

for model in smac_models:
    files_smac[model] = {}
    for acq_func in acq_func_names:
        if model == "fullGP":
            files_smac[model][acq_func] = glob.glob(
                os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), "fullGP_GPytorch", acq_func, files))
        elif model == "local_gp":
            files_smac[model][acq_func] = glob.glob(
                os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), "local_gp", acq_func, files))
        else:
            files_smac[model][acq_func] = glob.glob(
                os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), model, acq_func, files))


for model in other_models:
    files_others[model] = glob.glob(os.path.join(syspath, func_name, 'dim_{}'.format(n_dims), model, files))

func_kwargs = {'dim': n_dims, }
artifical_func = {'ackley': Ackley(**func_kwargs),
                  'levy': Levy(**func_kwargs),
                  'eggholder': Eggholder(**func_kwargs),
                  'rosenbrock': Rosenbrock(**func_kwargs),
                  'griewank': Griewank(**func_kwargs),
                  'michalewicz': Michalewicz(**func_kwargs),
                  'hart6': hart6,
                  'branin': branin
                  }
func = artifical_func[func_name]


def load_data(files, data_size):
    data = []
    for file_name in files:
        with open(file_name) as f:
            y = np.array(json.load(f)['y'])[:data_size]
        if len(y) < data_size:
            y = np.concatenate([y, y[-1] * np.ones(data_size - len(y))])
        data.append(y)
    data = np.asarray(data)
    return data


data_smac = {}
data_others = {}
num_eval = []

for model in smac_models:
    data_smac[model] = {}
    for acq_func in acq_func_names:
        data_smac[model][acq_func] = load_data(files_smac[model][acq_func], data_size=data_length)
        num_eval.append(data_smac[model][acq_func].shape[0])

for model in other_models:
    data = load_data(files_others[model], data_size=data_length)
    if model == "MCTS":
        data *= -1
    data_others[model] = data
    num_eval.append(data.shape[0])

fig = plt.figure(figsize=(8, 6))
matplotlib.rcParams.update({'font.size': 30})

num_eval = min(num_eval)

if func_name is 'branin':
    optimum = branin([9.42478, 2.475])
elif func_name is 'hart6':
    optimum = hart6([0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573])
elif func_name is 'michalewicz':
    if n_dims == 2:
        optimum = -1.8013
    elif n_dims == 5:
        optimum = -4.687658
    elif n_dims == 10:
        optimum = -9.66015
    else:
        raise ValueError('unsuopported value and n_dims')
elif func_name is 'eggholder':
    eh = Eggholder()
    optimum = -959.6407
else:
    optimum = 0

for model in smac_models:
    for acq_func in acq_func_names:
        step_plot(data_smac[model][acq_func], num_eval, optimum, f"{model}_{acq_func}")

for model in other_models:
    step_plot(data_others[model], num_eval, optimum, f'{model}')

leg = plt.legend()
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('Number of function evaluations')
plt.tight_layout()
plt.title("{}, dim {}".format(func_name, n_dims))
plt.subplots_adjust(left=0.075, right=0.98, top=0.95, bottom=0.09)
# plt.savefig(os.path.join(syspath, 'perf', '{}{}{}.png'.format(func_name, n_dims, acq_func_names)))
plt.show()
