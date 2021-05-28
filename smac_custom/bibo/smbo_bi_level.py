import os
import logging
import numpy as np
import time
import typing

from smac.configspace import Configuration
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.base_epm import AbstractEPM
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer, RunInfoIntent
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, RandomConfigurationChooser
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.base import BaseRunner
from smac.callbacks import IncorporateRunResultCallback


from smac.optimizer.smbo import SMBO
from smac_custom.bibo.epm_chosser_intermediate import EPMChooserIntermediate

__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


class SMBO_Bilevel(SMBO):
    """Interface that contains the main Bayesian optimization loop

    Attributes
    ----------
    logger
    incumbent
    scenario
    config_space
    stats
    initial_design
    runhistory
    intensifier
    num_run
    rng
    initial_design_configs
    epm_chooser
    """
    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: AbstractRacer,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 model_inner: AbstractEPM,
                 local_gp: str,
                 acq_optimizer_inner: AcquisitionFunctionMaximizer,
                 acquisition_func_inner: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 tae_runner: BaseRunner,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: typing.Union[RandomConfigurationChooser] = ChooserNoCoolDown(2.0),
                 predict_x_best: bool = True,
                 min_samples_model: int = 1,
                 max_configs_inner_fracs: float = 0.5,
                 min_configs_inner: typing.Optional[int] = None,
                 ss_data= None,
                 boing_kwargs: typing.Dict = {}):

        """
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        initial_design: InitialDesign
            initial sampling design
        runhistory: RunHistory
            runhistory with all runs so far
        runhistory2epm : AbstractRunHistory2EPM
            Object that implements the AbstractRunHistory2EPM to convert runhistory
            data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration
            (probably with some kind of racing on the instances)
        num_run: int
            id of this run (used for pSMAC)
        model: RandomForestWithInstances
            empirical performance model (right now, we support only RandomForestWithInstances)
        acq_optimizer: AcquisitionFunctionMaximizer
            Optimizer of acquisition function.
        acquisition_func : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill criterion for acq_optimizer)
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        min_samples_model: int
-            Minimum number of samples to build a model
        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.incumbent = restore_incumbent

        self.scenario = scenario
        self.config_space = scenario.cs  # type: ignore[attr-defined] # noqa F821
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.num_run = num_run
        self.rng = rng
        self._min_time = 10 ** -5
        self.tae_runner = tae_runner

        self.initial_design_configs = []  # type: typing.List[Configuration]

        self.ss_data = ss_data

        # initialize the chooser to get configurations from the EPM
        self.epm_chooser = EPMChooserIntermediate(scenario=scenario,
                                                  stats=stats,
                                                  runhistory=runhistory,
                                                  runhistory2epm=runhistory2epm,
                                                  model=model,
                                                  acq_optimizer=acq_optimizer,
                                                  acquisition_func=acquisition_func,
                                                  model_inner=model_inner,
                                                  local_gp=local_gp,
                                                  acq_optimizer_inner=acq_optimizer_inner,
                                                  acquisition_func_inner=acquisition_func_inner,
                                                  rng=rng,
                                                  restore_incumbent=restore_incumbent,
                                                  random_configuration_chooser=random_configuration_chooser,
                                                  predict_x_best=predict_x_best,
                                                  min_samples_model=min_samples_model,
                                                  batch_size=1,
                                                  max_configs_inner_fracs=max_configs_inner_fracs,
                                                  min_configs_inner=min_configs_inner,
                                                  max_spaces_sub=1,
                                                  ss_data=self.ss_data,
                                                  **boing_kwargs,
                                                  )

        # Internal variable - if this is set to True it will gracefully stop SMAC
        self._stop = False

        # Callbacks. All known callbacks have a key. If something does not have a key here, there is
        # no callback available.
        self._callbacks = {
            '_incorporate_run_results': list()
        }  # type: typing.Dict[str, typing.List[typing.Callable]]
        self._callback_to_key = {
            IncorporateRunResultCallback: '_incorporate_run_results',
        }  # type: typing.Dict[typing.Type, str]


