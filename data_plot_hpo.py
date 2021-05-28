import os
import sys
syspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(syspath)

from scipy import stats

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


syspath = os.path.dirname((os.path.abspath(__file__)))


func_name = 'robot'
bl = 'BOinG'
van = 'full_gp'
turbo = 'turbo'
la_mcts = 'la_mcts'
rf = 'rf'
rs = 'rs'

HPObench = 'HPObench'

#acq_func_names = ['LCB', 'PI', 'EI']
acq_func_names = ['EI']
if func_name == 'robot':
    data_length = 3000
elif func_name == 'rover':
    data_length = 10000
elif func_name == 'lunar':
    data_length = 1500
else:
    data_length = 100

files = 'data_*_0.json'

ss_files = 'ss_data_*_0.json'

if func_name in ['lunar', 'robot', 'rover']:
    smac_models = ['BOinG','rf',]
    other_models = ['turbo', 'la_mcts','rs']
    #other_models = ['turbo','rs']
    #smac_models = ["BOinG", "BOinG_TurBO","BOinG_TurBO4_1",'BOinG_4_TurBO', 'BOinG_5_TurBO', 'BOinG_Backup']
    #smac_models = ["BOinG", "rf"]
    #other_models = ['turbo', 'turbo_4']
    #other_models = ['turbo', 'rs', 'la_mcts']
else:
    smac_models = ['BOinG', 'fullGP', 'rf']
    other_models = ['turbo', 'la_mcts', 'rs']

#smac_models = ['BOinG']
#other_models = ['turbo', 'la_mcts']
#smac_models = ['BOinG', 'local_gp', 'fullGP_local']
#other_models = []
#other_models = ['turbo']

files_smac = {}
files_others = {}

num_init_dict = {'cartpole': 14,
                'adult': 16,
                'higgs': 16,
                'robot': 100,
                'lunar': 50,
                'askl': 100}


num_init = num_init_dict[func_name]

for model in smac_models:
    files_smac[model] = {}
    for acq_func in acq_func_names:
        if model == "BOinG":
            files_smac[model][acq_func] = glob.glob(os.path.join(syspath, HPObench, func_name, model, acq_func, 'data_*_0.json'))
        elif model == 'fullGP':
            files_smac[model][acq_func] = glob.glob(os.path.join(syspath, HPObench, func_name, "fullGP_GPytorch", acq_func, 'data_*_0.json'))
        else:
            files_smac[model][acq_func] = glob.glob(os.path.join(syspath, HPObench, func_name, model, acq_func, files))

for model in other_models:
    if model == 'la_mcts':
        files_others[model] = glob.glob(os.path.join(syspath, HPObench, func_name, model, "optim_data_*_0.json"))
        #files_others[model] = glob.glob(os.path.join(syspath, HPObench, func_name, model, files))
    elif model == 'rs':
        files_others[model] = glob.glob(os.path.join(syspath, HPObench, func_name,model, 'data_*_0.json'))
    else:
        files_others[model] = glob.glob(os.path.join(syspath, HPObench, func_name,model, files))


def load_data(files, data_size):
    data = []
    for file_name in files:
        with open(file_name) as f:
            y = json.load(f)['y'][:data_size]
            data.append(-1*np.array(y))
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
    if model == "la_mcts" and func_name != 'robot':
        data *= -1
    data_others[model] = data
    num_eval.append(data.shape[0])

num_eval = 20 if func_name in ['lunar', 'robot', 'rover'] else 30

fig = plt.figure(figsize=(16, 9))
matplotlib.rcParams.update({'font.size': 30})

def step_plot(data, num_eval, optimum, opt_name):
    num_init = 50
    data = data[:num_eval, num_init:]
    data = np.maximum.accumulate(data - optimum, axis=1)
    median = np.mean(data, axis=0)

    #median = np.quantile(data, 0.5, axis=0)
    lower = np.quantile(data, 0.25, axis=0)
    upper = np.quantile(data, 0.75, axis=0)

    std = np.std(data, axis=0)
    sem = stats.sem(data, axis=0)

    err = sem
    lower = median - err
    upper = median + err

    x_values = np.arange(num_init, data_length) + 1
    plt.step(x=x_values, y=median, label="{}".format(opt_name))
    plt.fill_between(x=x_values, y1=lower, y2=upper, alpha=0.3, step='pre', interpolate=False)
    plt.xlim(num_init, data_length)
    plt.ylim(0)
    print(opt_name)
    print(data[:,-1])
    print(median[-1])
    print(err[-1])

optimum= 0

#smac_models = ['BOinG', 'fullGP', 'rf']
#other_models = ['turbo', 'la_mcts', 'rs']

opt_names = {"BOinG": "BOinG_EI",
             "fullGP": "fullGP_EI",
             "rf": "RF_EI",
             "turbo":"TurBO",
             'la_mcts': "LA-MCTS",
             "rs": "Random Search",
             'local_gp': "Local GP",
             'fullGP_local': "Full GP Local"}

for model in smac_models:
    for acq_func in acq_func_names:
        step_plot(data_smac[model][acq_func], num_eval, optimum, opt_names.get(model, model))

for model in other_models:
    step_plot(data_others[model], num_eval, optimum, opt_names.get(model, model))


leg = plt.legend()
plt.tight_layout()
#plt.title(func_name)
#plt.title(f'{func_name.capitalize()}')
plt.title(f'Robot Pushing')
plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.09)
plt.ylabel('Accuracy' if func_name in ['adult', 'higgs'] else 'Reward')
plt.xlabel("Number of Evaluations")
#plt.yscale('log')
#plt.xscale('log')
#plt.savefig(os.path.join(syspath, 'perf', '{}_{}_{}D_{}evals.png'.format(acq_func_name, func_name, n_dims, num_eval)))
plt.show()
