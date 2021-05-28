from collections import deque
import copy
from typing import Union, Dict, Generator, Optional
from operator import itemgetter

import numpy as np  # type: ignore
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter, NumericalHyperparameter
import ConfigSpace.c_util


def get_one_exchange_neighbourhood(
        configuration: Configuration,
        seed: int,
        num_neighbors: int = 4,
        stdev: float = 0.2,
        dims_of_interest:Optional[np.ndarray]=None) -> Generator[Configuration, None, None]:
    """
    Return all configurations in a one-exchange neighborhood.

    The method is implemented as defined by:
    Frank Hutter, Holger H. Hoos and Kevin Leyton-Brown
    Sequential Model-Based Optimization for General Algorithm Configuration
    In Proceedings of the conference on Learning and Intelligent
    Optimization(LION 5)

    Parameters
    ----------
    configuration : :class:`~ConfigSpace.configuration_space.Configuration`
        for this Configuration object ``num_neighbors`` neighbors are computed
    seed : int
        Sets the random seed to a fixed value
    num_neighbors : (int, optional)
        number of configurations, which are sampled from the neighbourhood
        of the input configuration
    stdev : (float, optional)
        The standard deviation is used to determine the neigbours of
        :class:`~ConfigSpace.hyperparameters.UniformFloatHyperparameter` and
        :class:`~ConfigSpace.hyperparameters.UniformIntegerHyperparameter`.

    Returns
    -------
    Generator
         It contains configurations, with values being situated around
         the given configuration.

    """
    random = np.random.RandomState(seed)

    hyperparameters_list = list(
        list(configuration.configuration_space._hyperparameters.keys())
    )
    hyperparameters_list_length = len(hyperparameters_list)

    if dims_of_interest is None:
        indices_hps = np.arange(hyperparameters_list_length)
        hyperparameters_used = [hp.name for hp in configuration.configuration_space.get_hyperparameters()
                                if hp.get_num_neighbors(configuration.get(hp.name)) == 0 and
                                configuration.get(hp.name) is not None]
        number_of_usable_hyperparameters = sum(np.isfinite(configuration.get_array()))
        n_neighbors_per_hp = {
            hp.name: num_neighbors if
            isinstance(hp, NumericalHyperparameter) and hp.get_num_neighbors(configuration.get(hp.name)) > num_neighbors
            else hp.get_num_neighbors(configuration.get(hp.name))
            for hp in configuration.configuration_space.get_hyperparameters()
        }
    else:
        indices_hps = dims_of_interest
        hyperparameters_used = [hp.name for hp in itemgetter(*dims_of_interest)(configuration.configuration_space.get_hyperparameters())
                                if hp.get_num_neighbors(configuration.get(hp.name)) == 0 and
                                configuration.get(hp.name) is not None]
        number_of_usable_hyperparameters = sum(np.isfinite(configuration.get_array()[dims_of_interest]))
        n_neighbors_per_hp = {
            hp.name: num_neighbors if
            isinstance(hp, NumericalHyperparameter) and hp.get_num_neighbors(configuration.get(hp.name)) > num_neighbors
            else hp.get_num_neighbors(configuration.get(hp.name))
            for hp in itemgetter(*dims_of_interest)(configuration.configuration_space.get_hyperparameters())
        }

    finite_neighbors_stack = {}  # type: Dict
    configuration_space = configuration.configuration_space  # type: ConfigSpace


    while len(hyperparameters_used) < number_of_usable_hyperparameters:
        index = int(random.choice(indices_hps))
        hp_name = hyperparameters_list[index]
        if n_neighbors_per_hp[hp_name] == 0:
            continue

        else:
            neighbourhood = []
            number_of_sampled_neighbors = 0
            array = configuration.get_array()  # type: np.ndarray
            value = array[index]  # type: float

            # Check for NaNs (inactive value)
            if value != value:
                continue

            iteration = 0
            hp = configuration_space.get_hyperparameter(hp_name)  # type: Hyperparameter
            num_neighbors_for_hp = hp.get_num_neighbors(configuration.get(hp_name))
            while True:
                # Obtain neigbors differently for different possible numbers of
                # neighbors
                if num_neighbors_for_hp == 0:
                    break
                # No infinite loops
                elif iteration > 100:
                    break
                elif np.isinf(num_neighbors_for_hp):
                    if number_of_sampled_neighbors >= 1:
                        break
                    if isinstance(hp, UniformFloatHyperparameter):
                        neighbor = hp.get_neighbors(value, random, number=1, std=stdev)[0]
                    else:
                        neighbor = hp.get_neighbors(value, random, number=1)[0]
                else:
                    if iteration > 0:
                        break
                    if hp_name not in finite_neighbors_stack:
                        if isinstance(hp, UniformIntegerHyperparameter):
                            neighbors = hp.get_neighbors(
                                value, random,
                                number=n_neighbors_per_hp[hp_name], std=stdev,
                            )
                        else:
                            neighbors = hp.get_neighbors(value, random)
                        random.shuffle(neighbors)
                        finite_neighbors_stack[hp_name] = neighbors
                    else:
                        neighbors = finite_neighbors_stack[hp_name]
                    neighbor = neighbors.pop()

                # Check all newly obtained neigbors
                new_array = array.copy()
                new_array = ConfigSpace.c_util.change_hp_value(
                    configuration_space=configuration_space,
                    configuration_array=new_array,
                    hp_name=hp_name,
                    hp_value=neighbor,
                    index=index)
                try:
                    # Populating a configuration from an array does not check
                    #  if it is a legal configuration - check this (slow)
                    new_configuration = Configuration(configuration_space,
                                                      vector=new_array)  # type: Configuration
                    # Only rigorously check every tenth configuration (
                    # because moving around in the neighborhood should
                    # just work!)
                    if np.random.random() > 0.95:
                        new_configuration.is_valid_configuration()
                    else:
                        configuration_space._check_forbidden(new_array)
                    neighbourhood.append(new_configuration)
                except ForbiddenValueError:
                    pass

                iteration += 1
                if len(neighbourhood) > 0:
                    number_of_sampled_neighbors += 1

            # Some infinite loop happened and no valid neighbor was found OR
            # no valid neighbor is available for a categorical
            if len(neighbourhood) == 0:
                hyperparameters_used.append(hp_name)
                n_neighbors_per_hp[hp_name] = 0
                hyperparameters_used.append(hp_name)
            else:
                if hp_name not in hyperparameters_used:
                    n_ = neighbourhood.pop()
                    n_neighbors_per_hp[hp_name] -= 1
                    if n_neighbors_per_hp[hp_name] == 0:
                        hyperparameters_used.append(hp_name)
                    yield n_

