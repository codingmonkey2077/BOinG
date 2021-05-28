from hpobench.benchmarks.rl.cartpole import CartpoleReduced, CartpoleBase

import time
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
import tensorflow as tf
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper


class CartpoleModified(CartpoleReduced):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, defaults: Union[Dict, None] = None,
                 max_episodes: Union[int, None] = 200):
       super(CartpoleModified, self).__init__(rng,
                                              defaults,
                                              max_episodes)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[Dict, CS.Configuration],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        A modified version of Cartpole from HPOBench, we compute the average reward of the policy

        The budget describes how often the agent is trained on the experiment.
        It returns the average number of the length of episodes.

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
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        tf.random.set_random_seed(self.rng.randint(1, 100000))
        np.random.seed(self.rng.randint(1, 100000))

        # fill in missing entries with default values for 'incomplete/reduced' configspaces
        new_config = self.defaults
        new_config.update(configuration)
        configuration = new_config

        start_time = time.time()

        network_spec = [{'type': 'dense',
                         'size': configuration["n_units_1"],
                         'activation': configuration['activation_1']},
                        {'type': 'dense',
                         'size': configuration["n_units_2"],
                         'activation': configuration['activation_2']}]

        converged_episodes = []

        for _ in range(fidelity["budget"]):
            agent = PPOAgent(states=self.env.states,
                             actions=self.env.actions,
                             network=network_spec,
                             update_mode={'unit': 'episodes', 'batch_size': configuration["batch_size"]},
                             step_optimizer={'type': configuration["optimizer_type"],
                                             'learning_rate': configuration["learning_rate"]},
                             optimization_steps=configuration["optimization_steps"],
                             discount=configuration["discount"],
                             baseline_mode=configuration["baseline_mode"],
                             baseline={"type": "mlp",
                                       "sizes": [configuration["baseline_n_units_1"],
                                                 configuration["baseline_n_units_2"]]},
                             baseline_optimizer={"type": "multi_step",
                                                 "optimizer": {"type": configuration["baseline_optimizer_type"],
                                                               "learning_rate":
                                                                   configuration["baseline_learning_rate"]},
                                                 "num_steps": configuration["baseline_optimization_steps"]},
                             likelihood_ratio_clipping=configuration["likelihood_ratio_clipping"]
                             )

            def episode_finished(record):
                # Check if we have converged
                return np.mean(record.episode_rewards[-self.avg_n_episodes:]) != 200

            runner = Runner(agent=agent, environment=self.env)
            runner.run(num_timesteps=6000, episodes=self.max_episodes, max_episode_timesteps=200, episode_finished=episode_finished)
            runner.run(episodes=self.avg_n_episodes, max_episode_timesteps=200,
                       episode_finished=episode_finished, testing=False)
            converged_episodes.append(np.mean(runner.episode_rewards[-self.avg_n_episodes:]))

        cost = time.time() - start_time

        return {'function_value': np.mean(converged_episodes),
                'cost': cost,
                'info': {'max_episodes': self.max_episodes,
                         'all_runs': converged_episodes,
                         'fidelity': fidelity
                         }
                }

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        meta_information = CartpoleBase.get_meta_information()
        meta_information['description'] = 'cartpole evaluating  with reduced configuration space'
        return meta_information

