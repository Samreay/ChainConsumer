import logging
import numpy as np


class Chain(object):
    def __init__(self, chain, parameters, name, weights=None, posterior=None, walkers=None,
                 grid=False, num_free_params=None, num_eff_data_points=None, color=None, linewidth=None,
                 linestyle=None, kde=None, shade_alpha=None):
        self.chain = chain
        self.parameters = parameters
        self.name = name

        if weights is None:
            weights = np.ones(chain.shape[0])
        weights = weights.squeeze()

        if posterior is not None:
            posterior = posterior.squeeze()

        self.weights = weights
        self.posterior = posterior
        self.walkers = walkers
        self.grid = grid
        self.num_free_params = num_free_params
        self.num_eff_data_points = num_eff_data_points
        self._logger = logging.getLevelName(self.__class__.__name__)

        # Storing config overrides
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.kde = kde
        self.shade_alpha = shade_alpha

        self.summaries = {}
        self.config = {}

        self.validate_chain()
        self.validated_params = set()

    def validate_chain(self):
        # So many people request help when the pass in junk data without realising it.
        # Let's try and flag this as quickly as we can.
        # Defensive coding; engage!

        assert isinstance(self.name, str), "Chain name needs to be a string. It is %s" % type(self.name)
        assert np.all(np.isfinite(self.weights)), "Chain %s has weights which are NaN or inf!" % self.name
        assert len(self.weights.shape) == 1, "Weights should be a 1D array, have instead %s" % str(self.weights.shape)
        assert self.weights.size == self.chain.shape[0], "Chain %s has %d steps but %d weights" % \
                                                         (self.name, self.weights.size, self.chain.shape[0])
        if self.walkers is not None:
            assert int(self.walkers) == self.walkers, "Walkers should be an integer!"
            assert self.chain.shape[0] % self.walkers == 0, \
                "Chain %s has %d walkers and %d steps... which aren't divisible. They need to be!" % \
                (self.name, self.walkers, self.chain.shape[0])
        assert isinstance(self.grid, bool), "Chain %s has %s for grid, should be a bool" % (self.name, type(self.grid))
        assert self.parameters is not None, "Chain %s has parameter list of None. Please give names" % self.name
        assert len(self.parameters) == self.chain.shape[1], "Chain %s has %d parameters but data has %d columns" % \
                                                            (self.name, len(self.parameters), self.chain.shape[1])
        for i, p in enumerate(self.parameters):
            assert isinstance(p, str), "Param index %d, which is %s, needs to be a string!" % (i, p)
        if self.posterior is not None:
            assert len(self.posterior.shape) == 1, "posterior should be a 1D array, have instead %s" % str(self.posterior.shape)
            assert self.posterior.size == self.chain.shape[0], "Chain %s has %d steps but %d log-posterior values" % \
                                                               (self.name, self.posterior.size, self.chain.shape[0])
            assert np.all(np.isfinite(self.posterior)), "Chain %s has NaN or inf in the log-posterior" % self.name
        if self.num_free_params is not None:
            assert isinstance(self.num_free_params, (int, float)), \
                "Chain %s has num_free_params which is not an integer, its %s" % (self.name, type(self.num_free_params))
            assert np.isfinite(self.num_free_params), "num_free_params is either infinite or NaN"
            assert self.num_free_params > 0, "num_free_params must be positive"
        if self.num_eff_data_points is not None:
            assert isinstance(self.num_eff_data_points, (int, float)), \
                "Chain %s has num_eff_data_points which is not an a number, its %s" % (self.name, type(self.num_eff_data_points))
            assert np.isfinite(self.num_eff_data_points), "num_eff_data_points is either infinite or NaN"
            assert self.num_eff_data_points > 0, "num_eff_data_points must be positive"

    def reset_config(self):
        self.config = {}
        self.summaries = {}
        self.validated_params = set()

    def get_summary(self, param, callback):
        if param in self.summaries:
            return self.summaries[param]
        result = callback(self, param)
        self.summaries[param] = result
        return result

    def get_color_data(self):
        color_param = self.config.get("color_params")
        color_data = None
        if color_param in self.parameters:
            color_data = self.get_data(color_param)
        elif color_param == "weights":
            color_data = self.weights
        elif color_param == "log_weights":
            color_data = np.log(self.weights)
        elif color_param == "posterior":
            color_data = self.posterior
        return color_data

    def get_data(self, params):
        if not isinstance(params, list):
            params = [params]

        params = [self.parameters[param] if isinstance(param, int) else param for param in params]
        for p in params:
            self.validate_parameter(p)
        indexes = [self.parameters.index(param) for param in params]
        return np.squeeze(self.chain[:, indexes])

    def validate_parameter(self, param):
        if param not in self.validated_params:
            index = self.parameters.index(param)
            data = self.chain[:, index]
            msg = "Data for chain %s, parameter %s is being used, but has either NaNs or infs in it!"
            assert np.all(np.isfinite(data)), msg % (self.name, param)
            self.validated_params.add(param)
