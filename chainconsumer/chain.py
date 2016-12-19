import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.interpolate import interp1d, griddata
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.cm as cm
import statsmodels.api as sm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import normaltest
from statsmodels.regression.linear_model import yule_walker

__all__ = ["ChainConsumer"]


class ChainConsumer(object):
    """ A class for consuming chains produced by an MCMC walk

    """
    __version__ = "0.15.4"

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
        self._summaries = {
            "max": self._get_parameter_summary_max,
            "mean": self._get_parameter_summary_mean,
            "cumulative": self._get_parameter_summary_cumulative
        }
        self._gauss_mode = 'reflect'

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
        chain : int|str, list[str]
            The chain(s) to remove. You can pass in either the chain index, or the chain name, to remove it.
            By default removes the last chain added.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        if isinstance(chain, str) or isinstance(chain, int):
            chain = [chain]
        elif isinstance(chain, list):
            for c in chain:
                assert isinstance(c, str), "If you specify a list, " \
                                           "you must specify chain names, not indexes." \
                                           "This is to avoid confusion when specifying," \
                                           "for example, [0,0]. As this might be an error," \
                                           "or a request to remove the first two chains."
        for c in chain:
            index = self._get_chain(c)
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
                  color_params=None, plot_color_params=False, cmaps=None, usetex=True):  # pragma: no cover
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
            The name of the parameter to use for the colour scatter. Defaults to none, for no colour.
        plot_color_params : bool|list[bool], optional
            Whether or not the colour parameter should also be plotted as a posterior surface.
        cmaps : str|list[str]
            The matplotlib colourmap to use in the `colour_param`. If you have multiple `color_param`s, you can
            specific a different cmap for each variable. By default ChainConsumer will cycle between several
            cmaps.
        usetex : bool, optional
            Whether or not to parse text as LaTeX in plots.

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
            bins = self._get_bins()
        elif isinstance(bins, list):
            bins = [b2 if isinstance(b2, int) else np.floor(b2 * b1) for b1, b2 in zip(self._get_bins(), bins)]
        elif isinstance(bins, float):
            bins = [np.floor(b * bins) for b in self._get_bins()]
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
                color_params = [color_params if color_params in ps else None for ps in self._parameters]
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
                shade_alpha = np.sqrt(1 / num_chains)
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

        self._configured = True
        return self

    def configure_general(self, **kwargs):  # pragma: no cover
        """ Deprecated method. Left in only to provide a more useful error message. See `configure()`."""
        raise DeprecationWarning("Individual configurations have all be moved into the configure function")

    def configure_contour(self, **kwargs):  # pragma: no cover
        """ Deprecated method. Left in only to provide a more useful error message. See `configure()`."""
        raise DeprecationWarning("Individual configurations have all be moved into the configure function")

    def configure_bar(self, **kwargs):  # pragma: no cover
        """ Deprecated method. Left in only to provide a more useful error message. See `configure()`."""
        raise DeprecationWarning("Individual configurations have all be moved into the configure function")

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

    def get_summary(self, squeeze=True):
        """  Gets a summary of the marginalised parameter distributions.

        Parameters
        ----------
        squeeze : bool, optional
            Squeeze the summaries. If you only have one chain, squeeze will not return
            a length one list, just the single summary. If this is false, you will
            get a length one list.

        Returns
        -------
        list of dictionaries
            One entry per chain, parameter bounds stored in dictionary with parameter as key
        """
        results = []
        for ind, (chain, parameters, weights, g) in enumerate(zip(self._chains,
                                                                  self._parameters, self._weights, self._grids)):
            res = {}
            for i, p in enumerate(parameters):
                summary = self._get_parameter_summary(chain[:, i], weights, p, ind, grid=g)
                res[p] = summary
            results.append(res)
        if squeeze and len(results) == 1:
            return results[0]
        return results

    def get_latex_table(self, parameters=None, transpose=False, caption=None,
                        label="tab:model_params", hlines=True, blank_fill="--"):  # pragma: no cover
        """ Generates a LaTeX table from parameter summaries.

        Parameters
        ----------
        parameters : list[str], optional
            A list of what parameters to include in the table. By default, includes all parameters
        transpose : bool, optional
            Defaults to False, which gives each column as a parameter, each chain (framework)
            as a row. You can swap it so that you have a parameter each row and a framework
            each column by setting this to True
        caption : str, optional
            If you want to generate a caption for the table through Python, use this.
            Defaults to an empty string
        label : str, optional
            If you want to generate a label for the table through Python, use this.
            Defaults to an empty string
        hlines : bool, optional
            Inserts ``\\hline`` before and after the header, and at the end of table.
        blank_fill : str, optional
            If a framework does not have a particular parameter, will fill that cell of
            the table with this string.

        Returns
        -------
        str
            the LaTeX table.
        """
        if parameters is None:
            parameters = self._all_parameters
        for i, name in enumerate(self._names):
            assert name is not None, \
                "Generating a LaTeX table requires all chains to have names." \
                " Ensure you have `name=` in your `add_chain` call"
        for p in parameters:
            assert isinstance(p, str), \
                "Generating a LaTeX table requires all parameters have labels"
        num_parameters = len(parameters)
        num_chains = len(self._chains)
        fit_values = self.get_summary(squeeze=False)
        if label is None:
            label = ""
        if caption is None:
            caption = ""

        end_text = " \\\\ \n"
        if transpose:
            column_text = "c" * (num_chains + 1)
        else:
            column_text = "c" * (num_parameters + 1)

        center_text = ""
        hline_text = "\\hline\n"
        if hlines:
            center_text += hline_text + "\t\t"
        if transpose:
            center_text += " & ".join(["Parameter"] + self._names) + end_text
            if hlines:
                center_text += "\t\t" + hline_text
            for p in parameters:
                arr = ["\t\t" + p]
                for chain_res in fit_values:
                    if p in chain_res:
                        arr.append(self.get_parameter_text(*chain_res[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        else:
            center_text += " & ".join(["Model"] + parameters) + end_text
            if hlines:
                center_text += "\t\t" + hline_text
            for name, chain_res in zip(self._names, fit_values):
                arr = ["\t\t" + name]
                for p in parameters:
                    if p in chain_res:
                        arr.append(self.get_parameter_text(*chain_res[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        if hlines:
            center_text += "\t\t" + hline_text
        final_text = self._get_latex_table(caption, label) % (column_text, center_text)

        return final_text

    def get_parameter_text(self, lower, maximum, upper, wrap=False):
        """ Generates LaTeX appropriate text from marginalised parameter bounds.

        Parameters
        ----------
        lower : float
            The lower bound on the parameter
        maximum : float
            The value of the parameter with maximum probability
        upper : float
            The upper bound on the parameter
        wrap : bool
            Wrap output text in dollar signs for LaTeX

        Returns
        -------
        str
            The formatted text given the parameter bounds
        """
        if lower is None or upper is None:
            return ""
        upper_error = upper - maximum
        lower_error = maximum - lower
        resolution = min(np.floor(np.log10(np.abs(upper_error))),
                         np.floor(np.log10(np.abs(lower_error))))
        factor = 0
        fmt = "%0.1f"
        r = 1
        if np.abs(resolution) > 2:
            factor = -resolution
        if resolution == 2:
            fmt = "%0.0f"
            factor = -1
            r = 0
        if resolution == 1:
            fmt = "%0.0f"
        if resolution == -1:
            fmt = "%0.2f"
            r = 2
        elif resolution == -2:
            fmt = "%0.3f"
            r = 3
        upper_error *= 10 ** factor
        lower_error *= 10 ** factor
        maximum *= 10 ** factor
        upper_error = round(upper_error, r)
        lower_error = round(lower_error, r)
        maximum = round(maximum, r)
        if maximum == -0.0:
            maximum = 0.0
        if resolution == 2:
            upper_error *= 10 ** -factor
            lower_error *= 10 ** -factor
            maximum *= 10 ** -factor
            factor = 0
            fmt = "%0.0f"
        upper_error_text = fmt % upper_error
        lower_error_text = fmt % lower_error
        if upper_error_text == lower_error_text:
            text = r"%s\pm %s" % (fmt, "%s") % (maximum, lower_error_text)
        else:
            text = r"%s^{+%s}_{-%s}" % (fmt, "%s", "%s") % \
                   (maximum, upper_error_text, lower_error_text)
        if factor != 0:
            text = r"\left( %s \right) \times 10^{%d}" % (text, -factor)
        if wrap:
            text = "$%s$" % text
        return text

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

    def plot(self, figsize="GROW", parameters=None, extents=None, filename=None,
             display=False, truth=None, legend=None):  # pragma: no cover
        """ Plot the chain!

        Parameters
        ----------
        figsize : str|tuple(float)|float, optional
            The figure size to generate. Accepts a regular two tuple of size in inches,
            or one of several key words. The default value of ``COLUMN`` creates a figure
            of appropriate size of insertion into an A4 LaTeX document in two-column mode.
            ``PAGE`` creates a full page width figure. ``GROW`` creates an image that
            scales with parameters (1.5 inches per parameter). String arguments are not
            case sensitive. If you pass a float, it will scale the default ``GROW`` by
            that amount, so ``2.0`` would result in a plot 3 inches per parameter.
        parameters : list[str]|int, optional
            If set, only creates a plot for those specific parameters (if list). If an
            integer is given, only plots the fist so many parameters.
        extents : list[tuple[float]] or dict[str], optional
            Extents are given as two-tuples. You can pass in a list the same size as
            parameters (or default parameters if you don't specify parameters),
            or as a dictionary.
        filename : str, optional
            If set, saves the figure to this location
        display : bool, optional
            If True, shows the figure using ``plt.show()``.
        truth : list[float] or dict[str], optional
            A list of truth values corresponding to parameters, or a dictionary of
            truth values indexed by key
        legend : bool, optional
            If true, creates a legend in your plot using the chain names.

        Returns
        -------
        figure
            the matplotlib figure

        """

        if not self._configured:
            self.configure()
        if not self._configured_truth:
            self.configure_truth()

        if legend is None:
            legend = len(self._chains) > 1

        # Get all parameters to plot, taking into account some of them
        # might be excluded colour parameters
        color_params = self.config["color_params"]
        plot_color_params = self.config["plot_color_params"]
        all_parameters = []
        for cp, ps, pc in zip(color_params, self._parameters, plot_color_params):
            for p in ps:
                if (p != cp or pc) and p not in all_parameters:
                    all_parameters.append(p)

        # Calculate cmap extents
        unique_color_params = list(set(color_params))
        num_cax = len(unique_color_params)
        if None in unique_color_params:
            num_cax -= 1
        color_param_extents = {}
        for u in unique_color_params:
            umin, umax = np.inf, -np.inf
            for i, cp in enumerate(color_params):
                if cp is not None and u == cp:
                    data = self._chains[i][:, self._parameters[i].index(cp)]
                    umin = min(umin, data.min())
                    umax = max(umax, data.max())
            color_param_extents[u] = (umin, umax)

        if parameters is None:
            parameters = all_parameters
        elif isinstance(parameters, int):
            parameters = self._all_parameters[:parameters]
        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()
        if truth is not None and isinstance(truth, list):
            truth = truth[:len(parameters)]
        if isinstance(figsize, float):
            grow_size = figsize
            figsize = "GROW"
        else:
            grow_size = 1.0

        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5 + (1 if num_cax > 0 else 0), 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            elif figsize.upper() == "GROW":
                figsize = (grow_size * len(parameters) + num_cax * 1.0, grow_size * len(parameters))

            else:
                raise ValueError("Unknown figure size %s" % figsize)
        elif isinstance(figsize, float):
                figsize = (figsize * grow_size * len(parameters), figsize * grow_size * len(parameters))

        assert truth is None or isinstance(truth, dict) or \
               (isinstance(truth, list) and len(truth) == len(parameters)), \
            "Have a list of %d parameters and %d truth values" % (len(parameters), len(truth))

        assert extents is None or isinstance(extents, dict) or \
               (isinstance(extents, list) and len(extents) == len(parameters)), \
            "Have a list of %d parameters and %d extent values" % (len(parameters), len(extents))

        if truth is not None and isinstance(truth, list):
            truth = dict((p, t) for p, t in zip(parameters, truth))

        if extents is not None and isinstance(extents, list):
            extents = dict((p, e) for p, e in zip(parameters, extents))

        plot_hists = self.config["plot_hists"]
        flip = (len(parameters) == 2 and plot_hists and self.config["flip"])

        fig, axes, params1, params2, extents = self._get_figure(parameters, figsize=figsize,
                                                                       flip=flip, external_extents=extents)
        axl = axes.ravel().tolist()
        summary = self.config["summary"]
        fit_values = self.get_summary(squeeze=False)

        if summary is None:
            summary = len(parameters) < 5 and len(self._chains) == 1
        if len(self._chains) == 1:
            self._logger.debug("Plotting surfaces for chain of dimension %s" %
                               (self._chains[0].shape,))
        else:
            self._logger.debug("Plotting surfaces for %d chains" % len(self._chains))
        cbar_done = []
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j:
                    continue
                ax = axes[i, j]
                do_flip = (flip and i == len(params1) - 1)
                if plot_hists and i == j:
                    max_val = None
                    for ii, (chain, weights, parameters, fit, grid) in \
                            enumerate(zip(self._chains, self._weights, self._parameters, fit_values, self._grids)):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)

                        m = self._plot_bars(ii, ax, p1, chain[:, index], weights, grid=grid, fit_values=fit[p1], flip=do_flip,
                                            summary=summary, truth=truth, extents=extents[p1])
                        if max_val is None or m > max_val:
                            max_val = m
                    if do_flip:
                        ax.set_xlim(0, 1.1 * max_val)
                    else:
                        ax.set_ylim(0, 1.1 * max_val)

                else:
                    for ii, (chain, parameters, fit, weights, grid) in \
                            enumerate(zip(self._chains, self._parameters, fit_values, self._weights, self._grids)):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        color_data = None
                        extent = None
                        if color_params[ii] is not None:
                            color_index = parameters.index(color_params[ii])
                            color_data = chain[:, color_index]
                            extent = color_param_extents[color_params[ii]]
                        h = self._plot_contour(ii, ax, chain[:, i2], chain[:, i1], weights, p1, p2,
                                           grid, truth=truth, color_data=color_data, color_extent=extent)
                        if h is not None and color_params[ii] not in cbar_done:
                            cbar_done.append(color_params[ii])
                            aspect = figsize[1] / 0.15
                            fraction = 0.85 / figsize[0]
                            cbar = fig.colorbar(h, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                            cbar.set_label(color_params[ii], fontsize=14)
                            cbar.solids.set(alpha=1)

        colors = self.config["colors"]
        linestyles = self.config["linestyles"]
        linewidths = self.config["linewidths"]
        if self._names is not None and legend:
            ax = axes[0, -1]
            artists = [plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw)
                       for n, c, ls, lw in zip(self._names, colors, linestyles, linewidths) if n is not None]
            location = "center" if len(parameters) > 1 else 1
            ax.legend(artists, self._names, loc=location, frameon=False)
        fig.canvas.draw()
        for ax in axes[-1, :]:
            offset = ax.get_xaxis().get_offset_text()
            ax.set_xlabel('{0} {1}'.format(ax.get_xlabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
            offset.set_visible(False)
        for ax in axes[:, 0]:
            offset = ax.get_yaxis().get_offset_text()
            ax.set_ylabel('{0} {1}'.format(ax.get_ylabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
            offset.set_visible(False)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def plot_walks(self, parameters=None, truth=None, extents=None, display=False,
                   filename=None, chains=None, convolve=None, figsize=None,
                   plot_weights=True, plot_posterior=True, log_weight=None): # pragma: no cover
        """ Plots the chain walk; the parameter values as a function of step index.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains are well behaved, or if certain parameters are suspect
        or require a greater burn in period.

        The desired outcome is to see an unchanging distribution along the x-axis of the plot.
        If there are obvious tails or features in the parameters, you probably want
        to investigate.

        See :class:`.dessn.chain.demoWalk.DemoWalk` for example usage.

        Parameters
        ----------
        parameters : list[str]|int, optional
            Specifiy a subset of parameters to plot. If not set, all parameters are plotted.
            If an integer is given, only the first so many parameters are plotted.
        truth : list[float]|dict[str], optional
            A list of truth values corresponding to parameters, or a dictionary of
            truth values keyed by the parameter.
        extents : list[tuple]|dict[str], optional
            A list of two-tuples for plot extents per parameter, or a dictionary of
            extents keyed by the parameter.
        display : bool, optional
            If set, shows the plot using ``plt.show()``
        filename : str, optional
            If set, saves the figure to the filename
        chains : int|str, list[str|int], optional
            Used to specify which chain to show if more than one chain is loaded in.
            Can be an integer, specifying the
            chain index, or a str, specifying the chain name.
        convolve : int, optional
            If set, overplots a smoothed version of the steps using ``convolve`` as
            the width of the smoothing filter.
        figsize : tuple, optional
            If set, sets the created figure size.
        plot_weights : bool, optional
            If true, plots the weight if they are available
        plot_posterior : bool, optional
            If true, plots the log posterior if they are available
        log_weight : bool, optional
            Whether to display weights in log space or not. If None, the value is
            inferred by the mean weights of the plotted chains.

        Returns
        -------
        figure
            the matplotlib figure created

        """
        if not self._configured:
            self.configure()
        if not self._configured_truth:
            self.configure_truth()

        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()

        if chains is None:
            chains = list(range(len(self._chains)))
        else:
            if isinstance(chains, str) or isinstance(chains, int):
                chains = [chains]
            chains = [self._get_chain(c) for c in chains]

        all_parameters2 = [p for i in chains for p in self._parameters[i]]
        all_parameters = []
        for p in all_parameters2:
            if p not in all_parameters:
                all_parameters.append(p)

        if parameters is None:
            parameters = all_parameters
        elif isinstance(parameters, int):
            parameters = self.all_parameters[:parameters]

        if truth is not None and isinstance(truth, list):
            truth = truth[:len(parameters)]

        assert truth is None or isinstance(truth, dict) or \
               (isinstance(truth, list) and len(truth) == len(parameters)), \
            "Have a list of %d parameters and %d truth values" % (len(parameters), len(truth))

        assert extents is None or isinstance(extents, dict) or \
               (isinstance(extents, list) and len(extents) == len(parameters)), \
            "Have a list of %d parameters and %d extent values" % (len(parameters), len(extents))

        if truth is not None and isinstance(truth, list):
            truth = dict((p, t) for p, t in zip(parameters, truth))
        if truth is None:
            truth = {}

        if extents is not None and isinstance(extents, list):
            extents = dict((p, e) for p, e in zip(parameters, extents))
        if extents is None:
            extents = {}

        n = len(parameters)
        extra = 0
        if plot_weights:
            plot_weights = plot_weights and np.any([np.any(self._weights[c] != 1.0) for c in chains])

        plot_posterior = plot_posterior and np.any([self._posteriors[c] is not None for c in chains])

        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            figsize = (8, 0.75 + (n + extra))

        fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)
        colors = self.config["colors"]

        if self.config["usetex"]:
            plt.rc('text', usetex=True)
        if self.config["serif"]:
            plt.rc('font', family='serif')
        else:
            plt.rc('font', family='sans-serif')
        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = parameters[i - n]
                for index in chains:
                    if p in self._parameters[index]:
                        chain_row = self._chains[index][:, self._parameters[index].index(p)]
                        self._plot_walk(ax, p, chain_row, truth=truth.get(p),
                                        extents=extents.get(p), convolve=convolve, color=colors[index])
                        truth[p] = None
            else:
                if i == 0 and plot_posterior:
                    for index in chains:
                        if self._posteriors[index] is not None:
                            self._plot_walk(ax, "$\log(P)$", self._posteriors[index] - self._posteriors[index].max(),
                                            convolve=convolve, color=colors[index])
                else:
                    if log_weight is None:
                        log_weight = np.any([self._weights[index].mean() < 0.1 for index in chains])
                    if log_weight:
                        for index in chains:
                            self._plot_walk(ax, r"$\log_{10}(w)$", np.log10(self._weights[index]),
                                            convolve=convolve, color=colors[index])
                    else:
                        for index in chains:
                            self._plot_walk(ax, "$w$", self._weights[index],
                                            convolve=convolve, color=colors[index])

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def diagnostic_gelman_rubin(self, chain=None, threshold=0.05):
        r""" Runs the Gelman Rubin diagnostic on the supplied chains.

        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        threshold : float, optional
            The maximum deviation permitted from 1 for the final value
            :math:`\hat{R}`

        Returns
        -------
        float
            whether or not the chains pass the test

        Notes
        -----

        I follow PyMC in calculating the Gelman-Rubin statistic, where,
        having :math:`m` chains of length :math:`n`, we compute

        .. math::

            B = \frac{n}{m-1} \sum_{j=1}^{m} \left(\bar{\theta}_{.j} - \bar{\theta}_{..}\right)^2

            W = \frac{1}{m} \sum_{j=1}^{m} \left[ \frac{1}{n-1} \sum_{i=1}^{n} \left( \theta_{ij} - \bar{\theta_{.j}}\right)^2 \right]

        where :math:`\theta` represents each model parameter. We then compute
        :math:`\hat{V} = \frac{n_1}{n}W + \frac{1}{n}B`, and have our convergence ratio
        :math:`\hat{R} = \sqrt{\frac{\hat{V}}{W}}`. We check that for all parameters,
        this ratio deviates from unity by less than the supplied threshold.
        """
        if chain is None:
            keys = [n if n is not None else i for i, n in enumerate(self._names)]
            return np.all([self.diagnostic_gelman_rubin(k, threshold=threshold) for k in keys])
        index = self._get_chain(chain)
        num_walkers = self._walkers[index]
        parameters = self._parameters[index]
        name = self._names[index] if self._names[index] is not None else "%d" % index
        chain = self._chains[index]
        chains = np.split(chain, num_walkers)
        assert num_walkers > 1, "Cannot run Gelman-Rubin statistic with only one walker"
        m = 1.0 * len(chains)
        n = 1.0 * chains[0].shape[0]
        all_mean = np.mean(chain, axis=0)
        chain_means = np.array([np.mean(c, axis=0) for c in chains])
        chain_std = np.array([np.std(c, axis=0) for c in chains])
        b = n / (m - 1) * ((chain_means - all_mean)**2).sum(axis=0)
        w = (1 / m) * chain_std.sum(axis=0)
        var = (n - 1) * w / n + b / n
        R = np.sqrt(var / w)
        passed = np.abs(R - 1) < threshold
        print("Gelman-Rubin Statistic values for chain %s" % name)
        for p, v, pas in zip(parameters, R, passed):
            param = "Param %d" % p if isinstance(p, int) else p
            print("%s: %7.5f (%s)" % (param, v, "Passed" if pas else "Failed"))
        return np.all(passed)

    def diagnostic_geweke(self, chain=None, first=0.1, last=0.5, threshold=0.05):
        """ Runs the Geweke diagnostic on the supplied chains.

        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        first : float, optional
            The amount of the start of the chain to use
        last : float, optional
            The end amount of the chain to use
        threshold : float, optional
            The p-value to use when testing for normality.

        Returns
        -------
        float
            whether or not the chains pass the test

        """
        if chain is None:
            keys = [n if n is not None else i for i, n in enumerate(self._names)]
            return np.all([self.diagnostic_geweke(k, threshold=threshold) for k in keys])
        index = self._get_chain(chain)
        num_walkers = self._walkers[index]
        assert num_walkers is not None and num_walkers > 0, \
            "You need to specify the number of walkers to use the Geweke diagnostic."
        name = self._names[index] if self._names[index] is not None else "%d" % index
        chain = self._chains[index]
        chains = np.split(chain, num_walkers)
        n = 1.0 * chains[0].shape[0]
        n_start = int(np.floor(first * n))
        n_end = int(np.floor((1 - last) * n))
        mean_start = np.array([np.mean(c[:n_start, i])
                               for c in chains for i in range(c.shape[1])])
        var_start = np.array([self._spec(c[:n_start, i])/c[:n_start, i].size
                              for c in chains for i in range(c.shape[1])])
        mean_end = np.array([np.mean(c[n_end:, i])
                             for c in chains for i in range(c.shape[1])])
        var_end = np.array([self._spec(c[n_end:, i])/c[n_end:, i].size
                            for c in chains for i in range(c.shape[1])])
        zs = (mean_start - mean_end) / (np.sqrt(var_start + var_end))
        stat, pvalue = normaltest(zs)
        print("Gweke Statistic for chain %s has p-value %e" % (name, pvalue))
        return pvalue > threshold

    def comparison_aic(self):
        r""" Returns the corrected Akaike Information Criterion (AICc) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the AIC is defined as

        .. math::
            AIC \equiv -2\ln(P) + 2k,

        where :math:`P` represents the posterior, and :math:`k` the number of model parameters. The AICc
        is then defined as

        .. math::
            AIC_c \equiv AIC + \frac{2k(k+1)}{N-k-1},

        where :math:`N` represents the number of independent data points used in the model fitting.
        The AICc is a correction for the AIC to take into account finite chain sizes.

        Returns
        -------
        list[float]
            A list of all the AICc values - one per chain, in the order in which the chains were added.
        """
        aics = []
        aics_bool = []
        for i, (p, n_data, n_free) in enumerate(zip(self._posteriors, self._num_data, self._num_free)):
            if p is None or n_data is None or n_free is None:
                aics_bool.append(False)
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                self._logger.warn("You need to set %s for chain %s to get the AIC" %
                                  (missing[:-2], self._get_chain_name(i)))
            else:
                aics_bool.append(True)
                c_cor = (1.0 * n_free * (n_free + 1) / (n_data - n_free - 1))
                aics.append(2.0 * (n_free + c_cor - np.max(p)))
        if len(aics) > 0:
            aics -= np.min(aics)
        aics_fin = []
        i = 0
        for b in aics_bool:
            if not b:
                aics_fin.append(None)
            else:
                aics_fin.append(aics[i])
                i += 1
        return aics_fin

    def comparison_bic(self):
        r""" Returns the corrected Bayesian Information Criterion (BIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the BIC is defined as

        .. math::
            BIC \equiv -2\ln(P) + k \ln(N),

        where :math:`P` represents the posterior, :math:`k` the number of model parameters and :math:`N`
        the number of independent data points used in the model fitting.

        Returns
        -------
        list[float]
            A list of all the BIC values - one per chain, in the order in which the chains were added.
        """
        bics = []
        bics_bool = []
        for i, (p, n_data, n_free) in enumerate(zip(self._posteriors, self._num_data, self._num_free)):
            if p is None or n_data is None or n_free is None:
                bics_bool.append(False)
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                self._logger.warn("You need to set %s for chain %s to get the BIC" %
                                  (missing[:-2], self._get_chain_name(i)))
            else:
                bics_bool.append(True)
                bics.append(n_free * np.log(n_data) - 2 * np.max(p))
        if len(bics) > 0:
            bics -= np.min(bics)
        bics_fin = []
        i = 0
        for b in bics_bool:
            if not b:
                bics_fin.append(None)
            else:
                bics_fin.append(bics[i])
                i += 1
        return bics_fin

    def comparison_dic(self):
        r""" Returns the corrected Deviance Information Criterion (DIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, this method will return `None` for that chain. **Note that
        the DIC metric is only valid on posterior surfaces which closely resemble multivariate normals!**
        Formally, we follow Liddle (2007) and first define *Bayesian complexity* as

        .. math::
            p_D = \bar{D}(\theta) - D(\bar{\theta}),

        where :math:`D(\theta) = -2\ln(P(\theta)) + C` is the deviance, where :math:`P` is the posterior
        and :math:`C` a constant. From here the DIC is defined as

        .. math::
            DIC \equiv D(\bar{\theta}) + 2p_D = \bar{D}(\theta) + p_D.

        Returns
        -------
        list[float]
            A list of all the DIC values - one per chain, in the order in which the chains were added.

        References
        ----------
        [1] Andrew R. Liddle, "Information criteria for astrophysical model selection", MNRAS (2007)
        """
        dics = []
        dics_bool = []
        for i, p in enumerate(self._posteriors):
            if p is None:
                dics_bool.append(False)
                self._logger.warn("You need to set the posterior for chain %s to get the DIC" %
                                  self._get_chain_name(i))
            else:
                dics_bool.append(True)
                chain = self._chains[i]
                num_params = chain.shape[1]
                means = np.array([np.average(chain[:, ii], weights=self._weights[i]) for ii in range(num_params)])
                d = -2 * p
                d_of_mean = griddata(chain, d, means, method='nearest')[0]
                mean_d = np.average(d, weights=self._weights[i])
                p_d = mean_d - d_of_mean
                dic = mean_d + p_d
                dics.append(dic)
        if len(dics) > 0:
            dics -= np.min(dics)
        dics_fin = []
        i = 0
        for b in dics_bool:
            if not b:
                dics_fin.append(None)
            else:
                dics_fin.append(dics[i])
                i += 1
        return dics_fin

    def comparison_table(self, caption=None, label="tab:model_comp", hlines=True,
                         aic=True, bic=True, dic=True, sort="bic", descending=True):  # pragma: no cover
        """
        Return a LaTeX ready table of model comparisons.

        Parameters
        ----------
        caption : str, optional
            The table caption to insert.
        label : str, optional
            The table label to insert.
        hlines : bool, optional
            Whether to insert hlines in the table or not.
        aic : bool, optional
            Whether to include a column for AICc or not.
        bic : bool, optional
            Whether to include a column for BIC or not.
        dic : bool, optional
            Whether to include a column for DIC or not.
        sort : str, optional
            How to sort the models. Should be one of "bic", "aic" or "dic".
        descending : bool, optional
            The sort order.

        Returns
        -------
        str
            A LaTeX table to be copied into your document.
        """

        if sort == "bic":
            assert bic, "You cannot sort by BIC if you turn it off"
        if sort == "aic":
            assert aic, "You cannot sort by AIC if you turn it off"
        if sort == "dic":
            assert dic, "You cannot sort by DIC if you turn it off"

        if caption is None:
            caption = ""
        if label is None:
            label = ""

        base_string = self._get_latex_table(caption, label)
        end_text = " \\\\ \n"
        num_cols = 1 + (1 if aic else 0) + (1 if bic else 0)
        column_text = "c" * (num_cols + 1)
        center_text = ""
        hline_text = "\\hline\n"
        if hlines:
            center_text += hline_text
        center_text += "\tModel" + (" & AIC" if aic else "") + (" & BIC " if bic else "") \
                       + (" & DIC " if dic else "") + end_text
        if hlines:
            center_text += "\t" + hline_text
        if aic:
            aics = self.comparison_aic()
        else:
            aics = np.zeros(len(self._chains))
        if bic:
            bics = self.comparison_bic()
        else:
            bics = np.zeros(len(self._chains))
        if dic:
            dics = self.comparison_dic()
        else:
            dics = np.zeros(len(self._chains))

        if sort == "bic":
            to_sort = bics
        elif sort == "aic":
            to_sort = aics
        elif sort == "dic":
            to_sort = dics
        else:
            raise ValueError("sort %s not recognised, must be dic, aic or dic" % sort)

        good = [i for i, t in enumerate(to_sort) if t is not None]
        names = [self._names[g] for g in good]
        aics = [aics[g] for g in good]
        bics = [bics[g] for g in good]
        to_sort = bics if sort == "bic" else aics

        indexes = np.argsort(to_sort)

        if descending:
            indexes = indexes[::-1]

        for i in indexes:
            line = "\t" + names[i]
            if aic:
                line += "  &  %5.1f  " % aics[i]
            if bic:
                line += "  &  %5.1f  " % bics[i]
            if dic:
                line += "  &  %5.1f  " % dics[i]
            line += end_text
            center_text += line
        if hlines:
            center_text += "\t" + hline_text

        return base_string % (column_text, center_text)

    # Method of estimating spectral density following PyMC.
    # See https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py
    def _spec(self, x, order=2):
        beta, sigma = yule_walker(x, order)
        return sigma ** 2 / (1. - np.sum(beta)) ** 2

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

    def _plot_walk(self, ax, parameter, data, truth=None, extents=None,
                   convolve=None, color=None):  # pragma: no cover
        if extents is not None:
            ax.set_ylim(extents)
        assert convolve is None or isinstance(convolve, int), \
            "Convolve must be an integer pixel window width"
        x = np.arange(data.size)
        ax.set_xlim(0, x[-1])
        ax.set_ylabel(parameter)
        if color is None:
            color = "#0345A1"
        ax.scatter(x, data, c=color, s=2, marker=".", edgecolors="none", alpha=0.5)
        max_ticks = self.config["max_ticks"]
        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))

        if convolve is not None:
            color2 = self._scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=':', color=color2, alpha=1)
        if truth is not None:
            ax.axhline(truth, **self.config_truth)

    def _plot_bars(self, iindex, ax, parameter, chain_row, weights, flip=False, summary=False, fit_values=None,
                   truth=None, extents=None, grid=False):  # pragma: no cover

        # Get values from config
        kde = self.config["kde"][iindex]
        colour = self.config["colors"][iindex]
        linestyle = self.config["linestyles"][iindex]
        bar_shade = self.config["bar_shade"][iindex]
        linewidth = self.config["linewidths"][iindex]
        bins = self.config["bins"][iindex]
        smooth = self.config["smooth"][iindex]

        bins, smooth = self._get_smoothed_bins(smooth, bins)
        if grid:
            bins = self._get_grid_bins(chain_row)
        else:
            bins = np.linspace(extents[0], extents[1], bins)
        hist, edges = np.histogram(chain_row, bins=bins, normed=True, weights=weights)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        if smooth:
            hist = gaussian_filter(hist, smooth, mode=self._gauss_mode)
        if kde:
            assert np.all(weights == 1.0), "You can only use KDE if your weights are all one. " \
                                           "If you would like weights, please vote for this issue: " \
                                           "https://github.com/scikit-learn/scikit-learn/issues/4394"
            pdf = sm.nonparametric.KDEUnivariate(chain_row)
            pdf.fit()
            xs = np.linspace(extents[0], extents[1], 100)
            if flip:
                ax.plot(pdf.evaluate(xs), xs, color=colour, ls=linestyle, lw=linewidth)
            else:
                ax.plot(xs, pdf.evaluate(xs), color=colour, ls=linestyle, lw=linewidth)
            interpolator = pdf.evaluate
        else:
            if smooth:
                if flip:
                    ax.plot(hist, edge_center, color=colour, ls=linestyle, lw=linewidth)
                else:
                    ax.plot(edge_center, hist, color=colour, ls=linestyle, lw=linewidth)
            else:
                if flip:
                    orientation = "horizontal"
                else:
                    orientation = "vertical"
                ax.hist(edge_center, weights=hist, bins=edges, histtype="step",
                        color=colour, orientation=orientation, ls=linestyle, lw=linewidth)
            interp_type = "linear" if smooth else "nearest"
            interpolator = interp1d(edge_center, hist, kind=interp_type)

        if bar_shade and fit_values is not None:
            lower = fit_values[0]
            upper = fit_values[2]
            if lower is not None and upper is not None:
                if lower < edge_center.min():
                    lower = edge_center.min()
                if upper > edge_center.max():
                    upper = edge_center.max()
                x = np.linspace(lower, upper, 1000)
                if flip:
                    ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x),
                                     color=colour, alpha=0.2)
                else:
                    ax.fill_between(x, np.zeros(x.shape), interpolator(x),
                                    color=colour, alpha=0.2)
                if summary:
                    if isinstance(parameter, str):
                        ax.set_title(r"$%s = %s$" % (parameter.strip("$"),
                                                     self.get_parameter_text(*fit_values)), fontsize=14)
                    else:
                        ax.set_title(r"$%s$" % (self.get_parameter_text(*fit_values)), fontsize=14)
        if truth is not None:
            truth_value = truth.get(parameter)
            if truth_value is not None:
                if flip:
                    ax.axhline(truth_value, **self.config_truth)
                else:
                    ax.axvline(truth_value, **self.config_truth)
        return hist.max()

    def _plot_contour(self, iindex, ax, x, y, w, px, py, grid, truth=None, color_data=None, color_extent=None):  # pragma: no cover

        levels = 1.0 - np.exp(-0.5 * self.config["sigmas"] ** 2)
        h = None
        cloud = self.config["cloud"][iindex]
        smooth = self.config["smooth"][iindex]
        colour = self.config["colors"][iindex]
        bins = self.config["bins"][iindex]
        shade = self.config["shade"][iindex]
        shade_alpha = self.config["shade_alpha"][iindex]
        linestyle = self.config["linestyles"][iindex]
        linewidth = self.config["linewidths"][iindex]
        cmap = self.config["cmaps"][iindex]

        if grid:
            binsx = self._get_grid_bins(x)
            binsy = self._get_grid_bins(y)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)
        else:
            bins, smooth = self._get_smoothed_bins(smooth, bins, marginalsied=False)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins, weights=w)

        colours = self._scale_colours(colour, len(levels))
        colours2 = [self._scale_colour(colours[0], 0.7)] + \
                   [self._scale_colour(c, 0.8) for c in colours[:-1]]

        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        if smooth:
            hist = gaussian_filter(hist, smooth, mode=self._gauss_mode)
        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if cloud:
            n = self.config["num_cloud"][iindex]
            skip = max(1, int(x.size / n))
            kwargs = {"c": colours[1], "alpha": 0.3}
            if color_data is not None:
                kwargs["c"] = color_data[::skip]
                kwargs["cmap"] = cmap
                if color_extent is not None:
                    kwargs["vmin"] = color_extent[0]
                    kwargs["vmax"] = color_extent[1]

            h = ax.scatter(x[::skip], y[::skip], s=10, marker=".", edgecolors="none", **kwargs)
            if color_data is None:
                h = None

        if shade:
            ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours,
                        alpha=shade_alpha)
        ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2,
                   linestyles=linestyle, linewidths=linewidth)

        if truth is not None:
            truth_value = truth.get(px)
            if truth_value is not None:
                ax.axhline(truth_value, **self.config_truth)
            truth_value = truth.get(py)
            if truth_value is not None:
                ax.axvline(truth_value, **self.config_truth)
        return h

    def _get_extent2(self, data, weight):  # pragma: no cover
        mean = np.average(data, weights=weight)
        std = np.sqrt(np.average((data - mean) ** 2, weights=weight))
        max_sigma = np.array(self.config["sigmas"]).max()
        sigma_extent = max(3, max_sigma + 1)
        min_prop = mean - sigma_extent * std
        max_prop = mean + sigma_extent * std
        return min_prop, max_prop

    def _get_extent(self, data, weight):
        hist, be = np.histogram(data, weights=weight, bins=1000, normed=True)
        bc = 0.5 * (be[1:] + be[:-1])
        cdf = hist.cumsum()
        cdf = cdf / cdf.max()
        icdf = (1 - cdf)[::-1]
        threshold = 1e-3
        i1 = np.where(cdf > threshold)[0][0]
        i2 = np.where(icdf > threshold)[0][0]

        return bc[i1], bc[bc.size - i2]

    def _get_figure(self, all_parameters, flip, figsize=(5, 5),
                    external_extents=None):  # pragma: no cover
        n = len(all_parameters)
        max_ticks = self.config["max_ticks"]
        plot_hists = self.config["plot_hists"]
        if not plot_hists:
            n -= 1

        if n == 2 and plot_hists and flip:
            gridspec_kw = {'width_ratios': [3, 1], 'height_ratios': [1, 3]}
        else:
            gridspec_kw = {}
        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
        if self.config["usetex"]:
            plt.rc('text', usetex=True)
        if self.config["serif"]:
            plt.rc('font', family='serif')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=0.05)

        formatter = ScalarFormatter(useOffset=False) # useMathText=True
        # formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))

        extents = {}
        for p in all_parameters:
            min_val = None
            max_val = None
            if external_extents is not None and p in external_extents:
                min_val, max_val = external_extents[p]
            else:
                for i, (chain, parameters, w) in enumerate(zip(self._chains, self._parameters, self._weights)):
                    if p not in parameters:
                        continue
                    index = parameters.index(p)
                    if self._grids[i]:
                        min_prop = chain[:, index].min()
                        max_prop = chain[:, index].max()
                    else:
                        min_prop, max_prop = self._get_extent(chain[:, index], w)
                    if min_val is None or min_prop < min_val:
                        min_val = min_prop
                    if max_val is None or max_prop > max_val:
                        max_val = max_prop
            extents[p] = (min_val, max_val)

        if plot_hists:
            params1 = all_parameters
            params2 = all_parameters
        else:
            params1 = all_parameters[1:]
            params2 = all_parameters[:-1]
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                ax = axes[i, j]
                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    if i != n - 1 or (flip and j == n - 1):
                        ax.set_xticks([])
                    else:
                        display_x_ticks = True
                        if isinstance(p2, str):
                            ax.set_xlabel(p2, fontsize=14)
                    if j != 0 or (plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(p1, fontsize=14)
                    if display_x_ticks:
                        [l.set_rotation(45) for l in ax.get_xticklabels()]
                        ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                        ax.xaxis.set_major_formatter(formatter)
                    if display_y_ticks:
                        [l.set_rotation(45) for l in ax.get_yticklabels()]
                        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                        ax.yaxis.set_major_formatter(formatter)
                    if i != j or not plot_hists:
                        ax.set_ylim(extents[p1])
                    elif flip and i == 1:
                        ax.set_ylim(extents[p1])
                    ax.set_xlim(extents[p2])

        return fig, axes, params1, params2, extents

    def _get_bins(self):
        proposal = [max(20, np.floor(1.0 * np.power(chain.shape[0] / chain.shape[1], 0.25)))
                    for chain in self._chains]
        return proposal

    def _clamp(self, val, minimum=0, maximum=255):  # pragma: no cover
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val

    def _scale_colours(self, colour, num):  # pragma: no cover
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        scales = np.logspace(np.log(0.9), np.log(1.3), num)
        colours = [self._scale_colour(colour, scale) for scale in scales]
        return colours

    def _scale_colour(self, colour, scalefactor):  # pragma: no cover
        if isinstance(colour, np.ndarray):
            r, g, b = colour[:3]*255.0
        else:
            hex = colour.strip('#')
            if scalefactor < 0 or len(hex) != 6:
                return hex
            r, g, b = int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:], 16)
        r = self._clamp(int(r * scalefactor))
        g = self._clamp(int(g * scalefactor))
        b = self._clamp(int(b * scalefactor))
        return "#%02x%02x%02x" % (r, g, b)

    def _convert_to_stdev(self, sigma):  # pragma: no cover
        # From astroML
        shape = sigma.shape
        sigma = sigma.ravel()
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)

        sigma_cumsum = 1.0* sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]

        return sigma_cumsum[i_unsort].reshape(shape)

    def _get_smoothed_bins(self, smooth, bins, marginalsied=True):
        if smooth is None or not smooth or smooth == 0:
            return bins, 0
        else:
            return ((3 if marginalsied else 2) * smooth * bins), smooth

    def _get_grid_bins(self, data):
        bin_c = sorted(np.unique(data))
        delta = 0.5 * (bin_c[1] - bin_c[0])
        bins = np.concatenate((bin_c - delta, [bin_c[-1] + delta]))
        return bins

    def _get_smoothed_histogram(self, data, weights, chain_index, grid):
        smooth = self.config["smooth"][chain_index]
        if grid:
            bins = self._get_grid_bins(data)
        else:
            bins = self.config['bins'][chain_index]
            bins, smooth = self._get_smoothed_bins(smooth, bins)
        hist, edges = np.histogram(data, bins=bins, normed=True, weights=weights)
        edge_centers = 0.5 * (edges[1:] + edges[:-1])
        xs = np.linspace(edge_centers[0], edge_centers[-1], 10000)
        if smooth:
            hist = gaussian_filter(hist, smooth, mode=self._gauss_mode)

        if self.config["kde"][chain_index]:
            kde_xs = np.linspace(edge_centers[0], edge_centers[-1], max(100, int(bins)))
            assert np.all(weights == 1.0), "You can only use KDE if your weights are all one. " \
                                           "If you would like weights, please vote for this issue: " \
                                           "https://github.com/scikit-learn/scikit-learn/issues/4394"
            pdf = sm.nonparametric.KDEUnivariate(data)
            pdf.fit()
            ys = interp1d(kde_xs, pdf.evaluate(kde_xs), kind="cubic")(xs)
        else:
            ys = interp1d(edge_centers, hist, kind="linear")(xs)
        cs = ys.cumsum()
        cs /= cs.max()
        return xs, ys, cs

    def _get_parameter_summary(self, data, weights, parameter, chain_index, **kwargs):
        if not self._configured:
            self.configure()
        method = self._summaries[self.config["statistics"][chain_index]]
        return method(data, weights, parameter, chain_index, **kwargs)

    def _get_parameter_summary_mean(self, data, weights, parameter, chain_index, desired_area=0.6827, grid=False):
        xs, ys, cs = self._get_smoothed_histogram(data, weights, chain_index, grid)
        vals = [0.5 - desired_area / 2, 0.5, 0.5 + desired_area / 2]
        bounds = interp1d(cs, xs)(vals)
        bounds[1] = 0.5 * (bounds[0] + bounds[2])
        return bounds

    def _get_parameter_summary_cumulative(self, data, weights, parameter, chain_index, desired_area=0.6827, grid=False):
        xs, ys, cs = self._get_smoothed_histogram(data, weights, chain_index, grid)
        vals = [0.5 - desired_area / 2, 0.5, 0.5 + desired_area / 2]
        bounds = interp1d(cs, xs)(vals)
        return bounds

    def _get_parameter_summary_max(self, data, weights, parameter, chain_index, desired_area=0.6827, grid=False):
        xs, ys, cs = self._get_smoothed_histogram(data, weights, chain_index, grid)
        n_pad = 1000
        x_start = xs[0] * np.ones(n_pad)
        x_end = xs[-1] * np.ones(n_pad)
        y_start = np.linspace(0, ys[0], n_pad)
        y_end = np.linspace(ys[-1], 0, n_pad)
        xs = np.concatenate((x_start, xs, x_end))
        ys = np.concatenate((y_start, ys, y_end))
        cs = ys.cumsum()
        cs = cs / cs.max()
        startIndex = ys.argmax()
        maxVal = ys[startIndex]
        minVal = 0
        threshold = 0.001

        x1 = None
        x2 = None
        count = 0
        while x1 is None:
            mid = (maxVal + minVal) / 2.0
            count += 1
            try:
                if count > 50:
                    raise Exception("Failed to converge")
                i1 = startIndex - np.where(ys[:startIndex][::-1] < mid)[0][0]
                i2 = startIndex + np.where(ys[startIndex:] < mid)[0][0]
                area = cs[i2] - cs[i1]
                deviation = np.abs(area - desired_area)
                if deviation < threshold:
                    x1 = xs[i1]
                    x2 = xs[i2]
                elif area < desired_area:
                    maxVal = mid
                elif area > desired_area:
                    minVal = mid
            except:
                self._logger.warn("Parameter %s is not constrained" % parameter)
                return [None, xs[startIndex], None]

        return [x1, xs[startIndex], x2]

    def _get_latex_table(self, caption, label):  # pragma: no cover
        base_string = r"""\begin{table}
    \centering
    \caption{%s}
    \label{%s}
    \begin{tabular}{%s}
        %s    \end{tabular}
\end{table}"""
        return base_string % (caption, label, "%s", "%s")

    def _get_chain_name(self, index):
        return self._names[index] or index