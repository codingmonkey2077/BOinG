import numpy as np
import typing
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, RunHistory2EPM4Cost, RunHistory2EPM4LogCost

from smac.runhistory.runhistory import RunHistory, RunKey, RunValue
from smac.configspace import convert_configurations_to_array
from smac.epm.base_imputor import BaseImputor
from smac.utils import constants


class RunHistory2EPM4CostBi(AbstractRunHistory2EPM):
    """TODO"""

    def transform(
        self,
        runhistory: RunHistory,
        budget_subset: typing.Optional[typing.List] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Returns vector representation of runhistory; if imputation is
        disabled, censored (TIMEOUT with time < cutoff) will be skipped

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            Runhistory containing all evaluated configurations/instances
        budget_subset : list of budgets to consider

        Returns
        -------
        X: numpy.ndarray
            configuration vector x instance features
        Y: numpy.ndarray
            cost values
        """
        self.logger.debug("Transform runhistory into X,y format")

        s_run_dict = self._get_s_run_dict(runhistory, budget_subset)
        X, Y, Y_raw = self._build_matrix(run_dict=s_run_dict, runhistory=runhistory,
                                  store_statistics=True)

        # Get real TIMEOUT runs
        t_run_dict = self._get_t_run_dict(runhistory, budget_subset)
        # use penalization (e.g. PAR10) for EPM training
        store_statistics = True if np.isnan(self.min_y) else False
        tX, tY, tY_raw = self._build_matrix(run_dict=t_run_dict, runhistory=runhistory,
                                    store_statistics=store_statistics)

        # if we don't have successful runs,
        # we have to return all timeout runs
        if not s_run_dict:
            return tX, tY, tY_raw

        if self.impute_censored_data:
            # Get all censored runs
            if budget_subset is not None:
                c_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                              if runhistory.data[run].status in self.impute_state
                              and runhistory.data[run].time < self.cutoff_time
                              and run.budget in budget_subset}
            else:
                c_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                              if runhistory.data[run].status in self.impute_state
                              and runhistory.data[run].time < self.cutoff_time}

            if len(c_run_dict) == 0:
                self.logger.debug("No censored data found, skip imputation")
                # If we do not impute, we also return TIMEOUT data
                X = np.vstack((X, tX))
                Y = np.concatenate((Y, tY))
                Y_raw = np.concatenate((Y_raw, tY_raw))
            else:

                # better empirical results by using PAR1 instead of PAR10
                # for censored data imputation
                cen_X, cen_Y, cen_Y_raw = self._build_matrix(run_dict=c_run_dict,
                                                             runhistory=runhistory,
                                                             return_time_as_y=True,
                                                             store_statistics=False,)

                # Also impute TIMEOUTS
                tX, tY, tY_raw = self._build_matrix(run_dict=t_run_dict,
                                                    runhistory=runhistory,
                                                    return_time_as_y=True,
                                                    store_statistics=False,)
                self.logger.debug("%d TIMEOUTS, %d CAPPED, %d SUCC" %
                                  (tX.shape[0], cen_X.shape[0], X.shape[0]))
                cen_X = np.vstack((cen_X, tX))
                cen_Y = np.concatenate((cen_Y, tY))
                cen_Y_raw = np.concatenate((cen_Y_raw, tY_raw))

                # return imp_Y in PAR depending on the used threshold in imputor
                assert isinstance(self.imputor, BaseImputor)  # please mypy
                imp_Y = self.imputor.impute(censored_X=cen_X, censored_y=cen_Y,
                                            uncensored_X=X, uncensored_y=Y)

                imp_Y_raw = self.imputor.impute(censored_X=cen_X, censored_y=cen_Y_raw,
                                                uncensored_X=X, uncensored_y=Y_raw)

                # Shuffle data to mix censored and imputed data
                X = np.vstack((X, cen_X))
                Y = np.concatenate((Y, imp_Y))
                Y_raw = np.concatenate((Y_raw, imp_Y_raw))
        else:
            # If we do not impute, we also return TIMEOUT data
            X = np.vstack((X, tX))
            Y = np.concatenate((Y, tY))
            Y_raw = np.concatenate((Y_raw, tY_raw))

        self.logger.debug("Converted %d observations" % (X.shape[0]))
        return X, Y, Y_raw

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory,
                      return_time_as_y: bool = False,
                      store_statistics: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"Builds X,y matrixes from selected runs from runhistory

        Parameters
        ----------
        run_dict: dict: RunKey -> RunValue
            dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            runhistory object
        return_time_as_y: bool
            Return the time instead of cost as y value. Necessary to access the raw y values for imputation.
        store_statistics: bool
            Whether to store statistics about the data (to be used at subsequent calls)

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """

        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_dict)
        n_cols = self.num_params
        X = np.ones([n_rows, n_cols + self.n_feats]) * np.nan
        y = np.ones([n_rows, 1])

        # Then populate matrix
        for row, (key, run) in enumerate(run_dict.items()):
            # Scaling is automatically done in configSpace
            conf = runhistory.ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]
            if self.n_feats:
                feats = self.instance_features[key.instance_id]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector
            # run_array[row, -1] = instances[row]
            if return_time_as_y:
                y[row, 0] = run.time
            else:
                y[row, 0] = run.cost

        if y.size > 0:
            if store_statistics:
                self.perc = np.percentile(y, self.scale_perc)
                self.min_y = np.min(y)
                self.max_y = np.max(y)
            y_trans = self.transform_response_values(values=y)
            return X, y_trans, y
        else:
            return X, y, y

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Returns the input values.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return values


class RunHistory2EPM4LogCostBi(RunHistory2EPM4CostBi):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transforms the response values by using a log transformation.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        # ensure that minimal value is larger than 0
        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        if min_y == self.max_y:
            # prevent diving by zero
            min_y *= 1 - 10 ** -10
        values_return = (values - min_y) / (self.max_y - min_y)
        values_return = np.log(values_return)
        return values_return