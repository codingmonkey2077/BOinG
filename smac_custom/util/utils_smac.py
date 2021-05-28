import copy
import numpy as np
from itertools import chain, combinations

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from smac.configspace import Configuration


def RangeSubSpace(length: Union[int, float],
                  interval: Union[int, float],
                  lower: Union[int, float, None] = None,
                  upper: Union[int, float, None] = None,
                  log: bool = False):
    """
    builds a dict that wraps all the information about ranges of subspaces
    Parameters
    ----------
    length: Union[int, float],
        range value
    interval: Union[int, float],
        interval between neighbouring grid center
    lower: Union[int, float, None],
        lower bound of the subspace, can be None
    upper: Union[int, float, None],
        upper bound of the subspace, can be None
    log: bool
        if the hyperparameter is sampled in the log scale
    Returns
    ----------
    A dict containing all the information of ranges of subspaces
    """
    # TODO Rewrite with class so that we can generate the grid points here and move them based on the
    if (lower is not None) & (upper is not None):
        assert upper > lower, "upper value must be greater than lower value"
    range_sub = {'length': length,
                 'lower': lower,
                 'upper': upper,
                 'log': log}
    if length < 0:
        raise ValueError('length must be a positive value')
    if log and (length < 1.0):
        raise ValueError('under log scale the length must be greater than 1.0')
    if log:
        dis2overlapping = interval / length
    else:
        dis2overlapping = interval - length
    if dis2overlapping < length:
        Warning('the length is too small to cover the whole search space')
    range_sub.update({'dis2overlapping': dis2overlapping})
    return range_sub


# https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def get_neighbors_overlapping(config_inner: Dict[str, Any],
                              config_outer: Configuration,
                              configs_range_sub: Dict[str, Dict],
                              ) -> List[Optional[Configuration]]:
    """
    get a list of configurations which contain the configuration sampled from the inner loop
    Parameters
    ----------
    config_inner: Dict[str, Any]
        configuration sampled from inner subspace, stored in the form of Dict
    config_outer: Configuration
        configuration in the outer loop which config_inner is sampled from
    configs_range_sub: Dict[str, RangeSubSpace]
        map the name of the hyperparameter to the range of the hyperparameter in the subspace
    Returns
    ----------:
    neighbours_overlap: List[Optional[Configuration]]
        a list of configurations whose subspaces contain config_inner, can bee empty
    """
    neighbours_overlap = []
    configs_overlap = {}
    values_outer = config_outer.get_dictionary()
    for hp_name, hp_inner in config_inner.items():
        range_sub = configs_range_sub.get(hp_name)
        if range_sub is None:
            continue  # here some bool or categorical parameters may exist
        hp_outer = values_outer.get(hp_name)
        if range_sub['log']:
            if hp_inner > hp_outer * range_sub['dis2overlapping']:
                configs_overlap.update({hp_name: True})
            elif hp_inner < hp_outer / range_sub['dis2overlapping']:
                configs_overlap.update({hp_name: False})
        else:
            if hp_inner > hp_outer + range_sub['dis2overlapping']:
                configs_overlap.update({hp_name: True})
            elif hp_inner < hp_outer - range_sub['dis2overlapping']:
                configs_overlap.update({hp_name: False})

    neighbor_configs = list(powerset(configs_overlap))  # get all the subsets of configs_overlap
    for neighbor_config in neighbor_configs:
        directions = dict((x, configs_overlap[x]) for x in neighbor_config)  # rebuild Dict
        neighbor_configuration, new_configuration = get_neighbor_by_names(config_outer, directions)
        if new_configuration:
            neighbours_overlap.append(neighbor_configuration)
    return neighbours_overlap


def get_neighbor_by_names(incumbent: Configuration,
                          directions: Dict[str, bool]) -> Tuple[Configuration, bool]:
    """
    if one configuration sampled from inner loop locates in the overlapping region, we need to find all the
    configurations whose subspace also contains this configuration
    Parameters
    ----------
    incumbent: Configuration
        incumbent configuration
    directions: Dict[str, bool]
        a dict indicating which configurations are involved, the second term indicates if new neighbourhood should be
        sampled greater or smaller than the incumbent
    Returns
    ----------
    neighbor_configuration: Configuration
        a configuration
    new_configuration: bool
       if a new configuration is generated
    """
    values_new = copy.deepcopy(incumbent.get_dictionary())
    new_configuration = False
    for hp_name, ascend in directions.items():
        hp = incumbent.configuration_space.get_hyperparameter(hp_name)
        value = incumbent.get_dictionary()[hp_name]

        if not hp.has_neighbors():
            Warning('the hyperparameter {} has no neighborhood but we ask a neighborhood of it'.format(hp_name))
        neighbor = hp.get_neighbors(value, rs=np.random.RandomState(0), number=1, transform=True)
        if len(neighbor) == 2:
            value = neighbor[1] if ascend else neighbor[0]
            new_configuration = True
        else:  # the hyperparameter only has one neighbourhood, it can be the maximal or minimize value in the sequence
            if ascend:
                if neighbor[0] > value:
                    # the hyperparameter is the smallest value and we need the neighbourhood greater than it
                    value = neighbor[0]
                    new_configuration = True
            else:
                if neighbor[0] < value:
                    # the hyperparameter is the largest value and we need th neighbourhood smaller than it
                    value = neighbor[0]
                    new_configuration = True
        values_new[hp_name] = value
    try:
        neighbor_configuration = Configuration(incumbent.configuration_space, values=values_new)
    except ValueError:
        neighbor_configuration = incumbent
        new_configuration = True
    return neighbor_configuration, new_configuration


def find_between(val: float,
                 func: Callable,
                 func_values: np.ndarray,
                 mgrid: np.ndarray,
                 thres: float):
    """
    https://github.com/zi-w/Max-value-Entropy-Search/blob/master/utils/find_between.m
    finds rese such that func(res) = value via binary search
    Parameters
    ----------
    val: int
        the value to reach
    func: Callable
        the function to be called
    func_values: np.darray(N, )
        function values on sampled points
    mgrid: np.ndarray(N,)
        a vector such that min(mgrid) < res < max(mgrid).
    thres: float
        a threshold
    Returns
    -------
    float
        values that satisfy the equation above
    """
    indics_min = np.argmin(np.abs(func_values-val), axis=0)
    if np.abs(np.take_along_axis(func_values, np.expand_dims(indics_min, axis=0), axis=0) - val) < thres:
        res = mgrid[indics_min]
        return res
    if np.take_along_axis(func_values, np.expand_dims(indics_min, axis=0), axis=0) > val:
        left = mgrid[indics_min-1]
        right = mgrid[indics_min]
    else:
        left = mgrid[indics_min]
        right = mgrid[indics_min+1]

    mid = (left + right)/2
    midval = func(mid)
    cnt = 1
    while np.abs(midval - val) > thres:
        if midval > val:
            right = mid
        else:
            left = mid
        mid = (left + right)/2
        midval = func(mid)
        cnt = cnt+1
        if cnt > 1000:
            break
    return mid
