import numpy as np
import logging
from scipy.interpolate import interp1d
import matplotlib.cm as cm


from chainconsumer.comparisons import Comparison
from chainconsumer.diagnostic import Diagnostic
from chainconsumer.plotter import Plotter
from chainconsumer.helpers import get_extents, get_bins
from chainconsumer.analysis import Analysis

__all__ = ["ChainConsumer"]


class ChainConsumer(object):
    """ A class for consuming chains produced by an MCMC walk

    """
    __version__ = "0.17.0"

    def __init__(self):
        logging.basicConfig()
        self._logger = logging.getLogger(__name__)
        self._all_colours = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107",
                            "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        self._cmaps = ["viridis", "inferno", "hot", "Blues", "Greens", "Greys"]
        self._linestyles = ["-", '--', ':']
        self._chains = []
        self._walkers = []
        self._weights = []
        self._posteriors = []
        self._names = []
        self._parameters = []
        self._all_parameters = []
        self._grids = []
        self._num_free = []
        self._num_data = []
        self._default_parameters = None
        self._init_params()
        self._gauss_mode = 'reflect'
        self._configured = False

        self.plotter = Plotter(self)
        self.diagnostic = Diagnostic(self)
        self.comparison = Comparison(self)
        self.analysis = Analysis(self)

    def _init_params(self):
        self.config = {}
        self.config_truth = {}
        self._configured = False
        self._configured_truth = False

    def add_chain(self, chain, parameters=None, name=None, weights=None, posterior=None, walkers=None,
                  grid=False, num_free_params=None, num_eff_data_points=None):
        """ Add a chain to the consumer.

        Parameters
        ----------
        chain : str|ndarray|dict
            The chain to load. Normally a ``numpy.ndarray``. If a string is found, it
            interprets the string as a filename and attempts to load it in. If a ``dict``
            is passed in, it assumes the dict has keys of parameter names and values of
            an array of samples. Notice that using a dictionary puts the order of
            parameters in the output under the control of the python ``dict.keys()`` function.
            If you passed ``grid`` is set, you can pass in the parameter ranges in list form.
        parameters : list[str], optional
            A list of parameter names, one for each column (dimension) in the chain. This parameter
            should remain ``None`` if a dictionary is given as ``chain``, as the parameter names
            are taken from the dictionary keys.
        name : str, optional
            The name of the chain. Used when plotting multiple chains at once.
        weights : ndarray, optional
            If given, uses this array to weight the samples in chain
        posterior : ndarray, optional
            If given, records the log posterior for each sample in the chain
        walkers : int, optional
            How many walkers went into creating the chain. Each walker should
            contribute the same number of steps, and should appear in contiguous
            blocks in the final chain.
        grid : boolean, optional
            Whether the input is a flattened chain from a grid search instead of a Monte-Carlo
            chains. Note that when this is set, `walkers` should not be set, and `weights` should
            be set to the posterior evaluation for the grid point. **Be careful** when using
            a coarse grid of setting a high smoothing value, as this may oversmooth the posterior
            surface and give unreasonably large parameter bounds.
        num_eff_data_points : int|float, optional
            The number of effective (independent) data points used in the model fitting. Not required
            for plotting, but required if loading in multiple chains to perform model comparison.
        num_free_params : int, optional
            The number of degrees of freedom in your model. Not required for plotting, but required if
            loading in multiple chains to perform model comparison.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        is_dict = False
        assert chain is not None, "You cannot have a chain of None"
        if isinstance(chain, str):
            if chain.endswith("txt"):
                chain = np.loadtxt(chain)
            else:
                chain = np.load(chain)
        elif isinstance(chain, dict):
            assert parameters is None, \
                "You cannot pass a dictionary and specify parameter names"
            is_dict = True
            parameters = list(chain.keys())
            chain = np.array([chain[p] for p in parameters]).T
        elif isinstance(chain, list):
            chain = np.array(chain).T

        if grid:
            assert walkers is None, "If grid is set, walkers should not be"
            assert weights is not None, "If grid is set, you need to supply weights"
            if len(weights.shape) > 1:
                assert not is_dict, "We cannot construct a meshgrid from a dictionary, as the parameters" \
                                    "are no longer ordered. Please pass in a flattened array instead."
                self._logger.info("Constructing meshgrid for grid results")
                meshes = np.meshgrid(*[u for u in chain.T], indexing="ij")
                chain = np.vstack([m.flatten() for m in meshes]).T
                weights = weights.flatten()
                assert weights.size == chain[:, 0].size, "Error, given weight array size disagrees with parameter sampling"

        if len(chain.shape) == 1:
            chain = chain[None].T
        self._chains.append(chain)
        self._names.append(name)
        self._posteriors.append(posterior)
        assert walkers is None or chain.shape[0] % walkers == 0, \
            "The number of steps in the chain cannot be split evenly amongst the number of walkers"
        self._walkers.append(walkers)
        if weights is None:
            self._weights.append(np.ones(chain.shape[0]))
        else:
            self._weights.append(weights)
        if self._default_parameters is None and parameters is not None:
            self._default_parameters = parameters

        self._grids.append(grid)

        if parameters is None:
            if self._default_parameters is not None:
                assert chain.shape[1] == len(self._default_parameters), \
                    "Chain has %d dimensions, but default parameters have %d dimensions" \
                    % (chain.shape[1], len(self._default_parameters))
                parameters = self._default_parameters
                self._logger.debug("Adding chain using default parameters")
            else:
                self._logger.debug("Adding chain with no parameter names")
                parameters = [x for x in range(chain.shape[1])]
        else:
            self._logger.debug("Adding chain with defined parameters")
            assert len(parameters) <= chain.shape[1], \
                "Have only %d columns in chain, but have been given %d parameters names! " \
                "Please double check this." % (chain.shape[1], len(parameters))
        for p in parameters:
            if p not in self._all_parameters:
                self._all_parameters.append(p)
        self._parameters.append(parameters)

        self._num_data.append(num_eff_data_points)
        self._num_free.append(num_free_params)

        self._init_params()
        return self

    def remove_chain(self, chain=-1):
        """
        Removes a chain from ChainConsumer. Calling this will require any configurations set to be redone!

        Parameters
        ----------
        chain : int|str, list[str|int]
            The chain(s) to remove. You can pass in either the chain index, or the chain name, to remove it.
            By default removes the last chain added.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        if isinstance(chain, str) or isinstance(chain, int):
            chain = [chain]

        chain = sorted([self._get_chain(c) for c in chain])[::-1]
        assert len(chain) == len(list(set(chain))), "Error, you are trying to remove a chain more than once."

        for index in chain:
            parameters = self._parameters[index]

            del self._chains[index]
            del self._names[index]
            del self._weights[index]
            del self._posteriors[index]
            del self._parameters[index]
            del self._grids[index]
            del self._num_free[index]
            del self._num_data[index]

            # Recompute all_parameters
            for p in parameters:
                has = False
                for ps in self._parameters:
                    if p in ps:
                        has = True
                        break
                if not has:
                    i = self._all_parameters.index(p)
                    del self._all_parameters[i]

        # Need to reconfigure
        self._init_params()

        return self

    def configure(self, statistics="max", max_ticks=5, plot_hists=True, flip=True,
                  serif=True, sigmas=None, summary=None, bins=None, rainbow=None,
                  colors=None, linestyles=None, linewidths=None, kde=False, smooth=None,
                  cloud=None, shade=None, shade_alpha=None, bar_shade=None, num_cloud=None,
                  color_params=None, plot_color_params=False, cmaps=None, usetex=True,
                  diagonal_tick_labels=True, label_font_size=14, tick_font_size=12, spacing=None):  # pragma: no cover
        r""" Configure the general plotting parameters common across the bar
        and contour plots.

        If you do not call this explicitly, the :func:`plot`
        method will invoke this method automatically.

        Please ensure that you call this method *after* adding all the relevant data to the
        chain consumer, as the consume changes configuration values depending on
        the supplied data.

        Parameters
        ----------
        statistics : string|list[str], optional
            Which sort of statistics to use. Defaults to `"max"` for maximum likelihood
            statistics. Other available options are `"mean"` and `"cumulative"`. In the
            very, very rare case you want to enable different statistics for different
            chains, you can pass in a list of strings.
        max_ticks : int, optional
            The maximum number of ticks to use on the plots
        plot_hists : bool, optional
            Whether to plot marginalised distributions or not
        flip : bool, optional
            Set to false if, when plotting only two parameters, you do not want it to
            rotate the histogram so that it is horizontal.
        sigmas : np.array, optional
            The :math:`\sigma` contour levels to plot. Defaults to [0, 1, 2, 3] for a single chain
            and [0, 1, 2] for multiple chains. The leading zero is required if you don't want
            your surfaces to have a hole in them.
        serif : bool, optional
            Whether to display ticks and labels with serif font.
        summary : bool, optional
            If overridden, sets whether parameter summaries should be set as axis titles.
            Will not work if you have multiple chains
        bins : int|float,list[int|float], optional
            The number of bins to use. By default uses :math:`\frac{\sqrt{n}}{10}`, where
            :math:`n` are the number of data points. Giving an integer will set the number
            of bins to the given value. Giving a float will scale the number of bins, such
            that giving ``bins=1.5`` will result in using :math:`\frac{1.5\sqrt{n}}{10}` bins.
            Note this parameter is most useful if `kde=False` is also passed, so you
            can actually see the bins and not a KDE.
        rainbow : bool|list[bool], optional
            Set to True to force use of rainbow colours
        colors : str(hex)|list[str(hex)], optional
            Provide a list of colours to use for each chain. If you provide more chains
            than colours, you *will* get the rainbow colour spectrum. If you only pass
            one colour, all chains are set to this colour. This probably won't look good.
        linestyles : str|list[str], optional
            Provide a list of line styles to plot the contours and marginalsied
            distributions with. By default, this will become a list of solid lines. If a
            string is passed instead of a list, this style is used for all chains.
        linewidths : float|list[float], optional
            Provide a list of line widths to plot the contours and marginalsied
            distributions with. By default, this is a width of 1. If a float
            is passed instead of a list, this width is used for all chains.
        kde : bool|list[bool], optional
            Whether to use a Gaussian KDE to smooth marginalised posteriors. If false, uses
            bins and linear interpolation, so ensure you have plenty of samples if your
            distribution is highly non-gaussian. Due to the slowness of performing a
            KDE on all data, it is often useful to disable this before producing final
            plots.
        smooth : int|list[int], optional
            Defaults to 3. How much to smooth the marginalised distributions using a gaussian filter.
            If ``kde`` is set to true, this parameter is ignored. Setting it to either
            ``0``, ``False`` disables smoothing. For grid data, smoothing
            is set to 0 by default, not 3.
        cloud : bool|list[bool], optional
            If set, overrides the default behaviour and plots the cloud or not
        shade : bool|list[bool] optional
            If set, overrides the default behaviour and plots filled contours or not. If a list of
            bools is passed, you can turn shading on or off for specific chains.
        shade_alpha : float|list[float], optional
            Filled contour alpha value override. Default is 1.0. If a list is passed, you can set the
            shade opacity for specific chains.
        bar_shade : bool|list[bool], optional
            If set to true, shades in confidence regions in under histogram. By default
            this happens if you less than 3 chains, but is disabled if you are comparing
            more chains. You can pass a list if you wish to shade some chains but not others.
        num_cloud : int|list[int], optional
            The number of scatter points to show when enabling `cloud` or setting one of the parameters
            to colour scatter. Defaults to 15k per chain.
        color_params : str|list[str], optional
            The name of the parameter to use for the colour scatter. Defaults to none, for no colour. If set 
            to 'weights' or 'posterior' (without the quotes), and that is not a parameter in the chain, it will 
            use the weights or posterior, respectively, to colour the points.
        plot_color_params : bool|list[bool], optional
            Whether or not the colour parameter should also be plotted as a posterior surface.
        cmaps : str|list[str]
            The matplotlib colourmap to use in the `colour_param`. If you have multiple `color_param`s, you can
            specific a different cmap for each variable. By default ChainConsumer will cycle between several
            cmaps.
        usetex : bool, optional
            Whether or not to parse text as LaTeX in plots.
        diagonal_tick_labels : bool, optional
            Whether to display tick labels on a 45 degree angle.
        label_font_size : int|float, optional
            The font size for plot axis labels and axis titles if summaries are configured to display.
        tick_font_size : int|float, optional
            The font size for the tick labels in the plots.
        spacing : float, optional
            The amount of spacing to add between plots. Defaults to `None`, which equates to 1.0 for less 
            than 6 dimensions and 0.0 for higher dimensions.
            
        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        num_chains = len(self._chains)

        assert rainbow is None or colors is None, \
            "You cannot both ask for rainbow colours and then give explicit colours"

        # Determine statistics
        assert statistics is not None, "statistics should be a string or list of strings!"
        if isinstance(statistics, str):
            statistics = [statistics.lower()] * len(self._chains)
        elif isinstance(statistics, list):
            for i, l in enumerate(statistics):
                statistics[i] = l.lower()
        else:
            raise ValueError("statistics is not a string or a list!")
        for s in statistics:
            assert s in ["max", "mean", "cumulative"], \
                "statistics %s not recognised. Should be max, mean or cumulative" % s

        # Determine bins
        if bins is None:
            bins = get_bins(self._chains)
        elif isinstance(bins, list):
            bins = [b2 if isinstance(b2, int) else np.floor(b2 * b1) for b1, b2 in zip(get_bins(self._chains), bins)]
        elif isinstance(bins, float):
            bins = [np.floor(b * bins) for b in get_bins(self._chains)]
        elif isinstance(bins, int):
            bins = [bins] * len(self._chains)
        else:
            raise ValueError("bins value is not a recognised class (float or int)")

        # Determine KDEs
        if isinstance(kde, bool):
            kde = [False if g else kde for g in self._grids]

        # Determine smoothing
        if smooth is None:
            smooth = [0 if g or k else 3 for g, k in zip(self._grids, kde)]
        else:
            if smooth is not None and not smooth:
                smooth = 0
            if isinstance(smooth, list):
                smooth = [0 if k else s for s, k in zip(smooth, kde)]
            else:
                smooth = [0 if k else smooth for k in kde]

        # Determine color parameters
        if color_params is None:
            color_params = [None] * num_chains
        else:
            if isinstance(color_params, str):
                color_params = [color_params if color_params in ps + ["weights", "posterior"] else None for ps in self._parameters]
                color_params = [None if c == "posterior" and self._posteriors[i] is None else c for i, c in enumerate(color_params)]
            elif isinstance(color_params, list) or isinstance(color_params, tuple):
                for c, p in zip(color_params, self._parameters):
                    if c is not None:
                        assert c in p, "Color parameter %s not in parameters %s" % (c, p)
        # Determine if we should plot color parameters
        if isinstance(plot_color_params, bool):
            plot_color_params = [plot_color_params] * len(color_params)

        # Determine cmaps
        if cmaps is None:
            param_cmaps = {}
            cmaps = []
            i = 0
            for cp in color_params:
                if cp is None:
                    cmaps.append(None)
                elif cp in param_cmaps:
                    cmaps.append(param_cmaps[cp])
                else:
                    param_cmaps[cp] = self._cmaps[i]
                    cmaps.append(self._cmaps[i])
                    i = (i + 1) % len(self._cmaps)

        # Determine colours
        if colors is None:
            if rainbow:
                colors = cm.rainbow(np.linspace(0, 1, num_chains))
            else:
                if num_chains > len(self._all_colours):
                    num_needed_colours = np.sum([c is None for c in color_params])
                    colour_list = cm.rainbow(np.linspace(0, 1, num_needed_colours))
                else:
                    colour_list = self._all_colours
                colors = []
                ci = 0
                for c in color_params:
                    if c:
                        colors.append('#000000')
                    else:
                        colors.append(colour_list[ci])
                        ci += 1
        elif isinstance(colors, str):
                colors = [colors] * len(self._chains)

        # Determine linestyles
        if linestyles is None:
            i = 0
            linestyles = []
            for c in color_params:
                if c is None:
                    linestyles.append(self._linestyles[0])
                else:
                    linestyles.append(self._linestyles[i])
                    i = (i + 1) % len(self._linestyles)
        elif isinstance(linestyles, str):
            linestyles = [linestyles] * len(self._chains)

        # Determine linewidths
        if linewidths is None:
            linewidths = [1.0] * len(self._chains)
        elif isinstance(linewidths, float) or isinstance(linewidths, int):
            linewidths = [linewidths] * len(self._chains)

        # Determine clouds
        if cloud is None:
            cloud = False
        cloud = [cloud or c is not None for c in color_params]

        # Determine cloud points
        if num_cloud is None:
            num_cloud = 30000
        if isinstance(num_cloud, int) or isinstance(num_cloud, float):
            num_cloud = [int(num_cloud)] * num_chains

        # Modify shade alpha based on how many chains we have
        if shade_alpha is None:
            if num_chains == 1:
                shade_alpha = 1.0
            else:
                shade_alpha = np.sqrt(1.0 / num_chains)
        # Decrease the shading amount if there are colour scatter points
        if isinstance(shade_alpha, float) or isinstance(shade_alpha, int):
            shade_alpha = [shade_alpha if c is None else 0.25 * shade_alpha for c in color_params]

        # Should we shade the contours
        if shade is None:
            shade = num_chains <= 2
        if isinstance(shade, bool):
            # If not overridden, do not shade chains with colour scatter points
            shade = [shade and c is None for c in color_params]

        # Figure out if we should display parameter summaries
        if summary is not None:
            summary = summary and len(self._chains) == 1

        # Figure out bar shading
        if bar_shade is None:
            bar_shade = len(self._chains) <= 2
        if isinstance(bar_shade, bool):
            bar_shade = [bar_shade] * len(self._chains)

        # Figure out how many sigmas to plot
        if sigmas is None:
            if num_chains == 1:
                sigmas = np.array([0, 1, 2, 3])
            else:
                sigmas = np.array([0, 1, 2])
        sigmas = np.sort(sigmas)

        # List options
        self.config["shade"] = shade[:num_chains]
        self.config["shade_alpha"] = shade_alpha[:num_chains]
        self.config["bar_shade"] = bar_shade[:len(self._chains)]
        self.config["bins"] = bins
        self.config["kde"] = kde
        self.config["cloud"] = cloud
        self.config["linewidths"] = linewidths
        self.config["linestyles"] = linestyles
        self.config["colors"] = colors
        self.config["smooth"] = smooth
        self.config["color_params"] = color_params
        self.config["plot_color_params"] = plot_color_params
        self.config["cmaps"] = cmaps
        self.config["num_cloud"] = num_cloud

        # Verify we have enough options entered.
        for key in self.config.keys():
            val = self.config[key]
            assert len(val) >= num_chains, \
                "Only have %d options for %s, but have %d chains!" % (len(val), key, num_chains)

        # Non list options
        self.config["sigmas"] = sigmas
        self.config["statistics"] = statistics
        self.config["summary"] = summary
        self.config["flip"] = flip
        self.config["serif"] = serif
        self.config["plot_hists"] = plot_hists
        self.config["max_ticks"] = max_ticks
        self.config["usetex"] = usetex
        self.config["diagonal_tick_labels"] = diagonal_tick_labels
        self.config["label_font_size"] = label_font_size
        self.config["tick_font_size"] = tick_font_size
        self.config["spacing"] = spacing

        self._configured = True
        return self

    def configure_truth(self, **kwargs):  # pragma: no cover
        """ Configure the arguments passed to the ``axvline`` and ``axhline``
        methods when plotting truth values.

        If you do not call this explicitly, the :func:`plot` method will
        invoke this method automatically.

        Recommended to set the parameters ``linestyle``, ``color`` and/or ``alpha``
        if you want some basic control.

        Default is to use an opaque black dashed line.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments to unwrap when calling ``axvline`` and ``axhline``.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        if kwargs.get("ls") is None and kwargs.get("linestyle") is None:
            kwargs["ls"] = "--"
            kwargs["dashes"] = (3, 3)
        if kwargs.get("color") is None:
            kwargs["color"] = "#000000"
        self.config_truth = kwargs
        self._configured_truth = True
        return self

    def divide_chain(self, chain=0):
        """
        Returns a ChainConsumer instance containing all the walks of a given chain
        as individual chains themselves.

        This method might be useful if, for example, your chain was made using
        MCMC with 4 walkers. To check the sampling of all 4 walkers agree, you could
        call this to get a ChainConumser instance with one chain for ech of the
        four walks. If you then plot, hopefully all four contours
        you would see agree.

        Parameters
        ----------
        chain : int|str, optional
            The index or name of the chain you want divided

        Returns
        -------
        ChainConsumer
            A new ChainConsumer instance with the same settings as the parent instance, containing
            ``num_walker`` chains.
        """
        if isinstance(chain, str):
            assert chain in self._names, "No chain with name %s found" % chain
            i = self._names.index(chain)
        elif isinstance(chain, int):
            i = chain
        else:
            raise ValueError("Type %s not recognised. Please pass in an int or a string" % type(chain))
        assert self._walkers[i] is not None, "The chain you have selected was not added with any walkers!"
        num_walkers = self._walkers[i]
        cs = np.split(self._chains[i], num_walkers)
        ws = np.split(self._weights[i], num_walkers)
        con = ChainConsumer()
        for j, (c, w) in enumerate(zip(cs, ws)):
            con.add_chain(c, weights=w, name="Chain %d" % j, parameters=self._parameters[i])
        return con

    def _get_chain(self, chain):
        if isinstance(chain, str):
            assert chain in self._names, "Chain %s not found!" % chain
            index = self._names.index(chain)
        elif isinstance(chain, int):
            assert chain < len(self._chains), "Chain index %d not found!" % chain
            index = chain
        else:
            raise ValueError("Type %s not recognised for chain" % type(chain))
        return index

    def _get_chain_name(self, index):
        return self._names[index] or index

    # Deprecated methods
    def plot(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.plotter.plot instead")
        return self.plotter.plot(*args, **kwargs)

    def plot_walks(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.plotter.plot_walks instead")
        return self.plotter.plot_walks(*args, **kwargs)

    def get_latex_table(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_latex_table instead")
        return self.analysis.get_latex_table(*args, **kwargs)

    def get_parameter_text(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_parameter_text instead")
        return self.analysis.get_parameter_text(*args, **kwargs)

    def get_summary(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_summary instead")
        return self.analysis.get_summary(*args, **kwargs)

    def get_correlations(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_correlations instead")
        return self.analysis.get_correlations(*args, **kwargs)

    def get_correlation_table(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_correlation_table instead")
        return self.analysis.get_correlation_table(*args, **kwargs)

    def get_covariance(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_covariance instead")
        return self.analysis.get_covariance(*args, **kwargs)

    def get_covariance_table(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.analysis.get_covariance_table instead")
        return self.analysis.get_covariance_table(*args, **kwargs)

    def diagnostic_gelman_rubin(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.diagnostic.gelman_rubin instead")
        return self.diagnostic.gelman_rubin(*args, **kwargs)

    def diagnostic_geweke(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.diagnostic.geweke instead")
        return self.diagnostic.geweke(*args, **kwargs)

    def comparison_aic(self):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.comparison.aic instead")
        return self.comparison.aic()

    def comparison_bic(self):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.comparison.bic instead")
        return self.comparison.bic()

    def comparison_dic(self):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.comparison.dic instead")
        return self.comparison.dic()

    def comparison_table(self, *args, **kwargs):  # pragma: no cover
        self._logger.warning("This method is deprecated. Please use chainConsumer.comparison.comparison_table instead")
        return self.comparison.comparison_table(*args, **kwargs)


