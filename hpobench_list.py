from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetAdultOnStepsBenchmark as AdultBench
from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetMnistOnStepsBenchmark as MinistBench
from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetHiggsOnStepsBenchmark as HiggsBench
from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetLetterOnTimeBenchmark as LetterBench

from hpobench_custom.cartpole_bench import CartpoleModified
from hpobench_custom.lunar import LunarLandBench
from hpobench_custom.robot_push import RobotPushBench


benchmarks = {'adult': AdultBench,
              'minist': MinistBench,
              'higgs': HiggsBench,
              'letter': LetterBench,
              'cartpole': CartpoleModified,
              'lunar': LunarLandBench,
              'robot': RobotPushBench,
              }


def optimization_function_wrapper(b, cfg, y_list, hist=None):
    """ Helper-function: simple wrapper to use the benchmark with smac"""
    # New API ---- Use this
    result_dict = b.objective_function(cfg, )
    y = -result_dict['function_value']
    y_list.append(y)
    if hist is not None:
        hist.append(cfg.get_array().tolist())
    return y

