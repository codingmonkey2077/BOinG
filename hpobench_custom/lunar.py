from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper

import ConfigSpace as CS
import time

from typing import Union, Dict
import numpy as np
import gym
import copy


class LunarLandBench(AbstractBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, num_init_states: int = 50, num_max_steps: int= 1000):
        super(LunarLandBench, self).__init__()

        self.rng = rng_helper.get_rng(rng=rng)
        #np.random.seed(0)
        self.env = gym.make("LunarLander-v2")
        self.num_dims = 12
        self.num_init_states = num_init_states
        self.env.seed(self.rng.randint(1, 100000))
        self.num_max_steps = num_max_steps

        self.defaults = {f"w{i}": 1.0 for i in range(12)}

    @staticmethod
    def heuristic_Controller(s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the configuration space for this benchmark
        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
        CS.ConfigurationSpace -
            Containing the benchmark's hyperparameter
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(f"w{i}", lower=0.0, default_value=1.0, upper=2.0, log=False) for i in range(12)
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Defines the available fidelity parameters as a "fidelity space" for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the fidelity space.
        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's fidelity parameters
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.Constant("constant_fidelity", 1.0),
        ])
        return fidel_space

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[Dict, CS.Configuration],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Use a heuristic controller to control a lunar lander

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : inverse of the final reward
            cost : time to run all agents
            info : Dict
                max_episodes : the maximum length of an episode
                budget : number of agents used
                all_runs : the episode length of all runs of all agents
                fidelity : the used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        self.env.seed(self.rng.randint(1, 100000))
        np.random.seed(self.rng.randint(1, 100000))

        # fill in missing entries with default values for 'incomplete/reduced' configspaces
        new_config = self.defaults
        new_config.update(configuration)
        configuration = new_config

        start_time = time.time()

        x = np.zeros(self.num_dims)
        for i in range(self.num_dims):
            x[i] = configuration[f"w{i}"]

        reward_episodes = []

        for i in range(self.num_init_states):
            total_reward = 0
            steps = 0
            state = self.env.reset()
            while steps < self.num_max_steps:
                action = self.heuristic_Controller(state, x)
                state, r, done, info = self.env.step(action)
                total_reward += r

                # if steps % 20 == 0 or done:
                #    print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                #    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1
                if done:
                    break
            reward_episodes.append(total_reward)
        function_value = np.mean(reward_episodes)
        cost = time.time() - start_time

        return {'function_value': function_value,
                'cost': cost,
                'info': {'fidelity': fidelity}
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the cartpole benchmark. Use the full budget.
        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : average episode length
            cost : time to run all agents
            info : Dict
                max_episodes : the maximum length of an episode
                budget : number of agents used
                all_runs : the episode length of all runs of all agents
                fidelity : the used fidelities in this evaluation
        """

        return self.objective_function(configuration=configuration, fidelity=fidelity, rng=rng,
                                       **kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {'name': 'LunarLand',
                'references': ['@InProceedings{eriksson-neurips-19,'
                               'title = {Scalable Global Optimization via Local {Bayesian} Optimization},'
                               'url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-'
                               'local-bayesian-optimization.pdf},'
                               'author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, '
                               'Ryan D and Poloczek, Matthias},'
                               'booktitle = {Advances in Neural Information Processing Systems},'
                               'pages = {5496--5507},'
                               'year      = {2019}}'],
                'note': 'This benchmark is not deterministic, since the gym environment is not deterministic.'}
