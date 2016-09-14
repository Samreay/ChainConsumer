import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.interpolate import interp1d
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
    __version__ = "0.11.1"

    def __init__(self):
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.all_colours = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107",
                            "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        self.chains = []
        self.walkers = []
        self.weights = []
        self.posteriors = []
        self.names = []
        self.parameters = []
        self.all_parameters = []
        self.default_parameters = None
        self._configured_bar = False
        self._configured_contour = False
        self._configured_truth = False
        self._configured_general = False
        self.parameters_contour = {}
        self.parameters_bar = {}
        self.parameters_truth = {}
        self.parameters_general = {}
        self.summaries = {
            "max": self._get_parameter_summary_max,
            "mean": self._get_parameter_summary_mean,
            "cumulative": self._get_parameter_summary_cumulative
        }

    def add_chain(self, chain, parameters=None, name=None, weights=None, posterior=None, walkers=None):
        """ Add a chain to the consumer.

        Parameters
        ----------
        chain : str|ndarray|dict
            The chain to load. Normally a ``numpy.ndarray``. If a string is found, it
            interprets the string as a filename and attempts to load it in. If a ``dict``
            is passed in, it assumes the dict has keys of parameter names and values of
            an array of samples. Notice that using a dictionary puts the order of
            parameters in the output under the control of the python ``dict.keys()`` function.
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

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        assert chain is not None, "You cannot have a chain of None"
        if isinstance(chain, str):
            if chain.endswith("txt"):
                chain = np.loadtxt(chain)
            else:
                chain = np.load(chain)
        elif isinstance(chain, dict):
            assert parameters is None, \
                "You cannot pass a dictionary and specify parameter names"
            parameters = list(chain.keys())
            chain = np.array([chain[p] for p in parameters]).T
        elif isinstance(chain, list):
            chain = np.array(chain)
        if len(chain.shape) == 1:
            chain = chain[None].T
        self.chains.append(chain)
        self.names.append(name)
        self.posteriors.append(posterior)
        assert walkers is None or chain.shape[0] % walkers == 0, \
            "The number of steps in the chain cannot be split evenly amongst the number of walkers"
        self.walkers.append(walkers)
        if weights is None:
            self.weights.append(np.ones(chain.shape[0]))
        else:
            self.weights.append(weights)
        if self.default_parameters is None and parameters is not None:
            self.default_parameters = parameters

        if parameters is None:
            if self.default_parameters is not None:
                assert chain.shape[1] == len(self.default_parameters), \
                    "Chain has %d dimensions, but default parameters have %d dimensions" \
                    % (chain.shape[1], len(self.default_parameters))
                parameters = self.default_parameters
                self.logger.debug("Adding chain using default parameters")
            else:
                self.logger.debug("Adding chain with no parameter names")
                parameters = [x for x in range(chain.shape[1])]
        else:
            self.logger.debug("Adding chain with defined parameters")
        for p in parameters:
            if p not in self.all_parameters:
                self.all_parameters.append(p)
        self.parameters.append(parameters)

        self._configured_bar = False
        self._configured_contour = False
        self._configured_general = False
        self._configured_truth = False
        return self

    def configure_general(self, statistics="max", bins=None, flip=True, rainbow=None,
                          colours=None, linestyles=None, linewidths=None, serif=True,
                          plot_hists=True, max_ticks=5, kde=False, smooth=3):  # pragma: no cover
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
        bins : int|float, optional
            The number of bins to use. By default uses :math:`\frac{\sqrt{n}}{10}`, where
            :math:`n` are the number of data points. Giving an integer will set the number
            of bins to the given value. Giving a float will scale the number of bins, such
            that giving ``bins=1.5`` will result in using :math:`\frac{1.5\sqrt{n}}{10}` bins.
            Note this parameter is most useful if `kde=False` is also passed, so you
            can actually see the bins and not a KDE.
        flip : bool, optional
            Set to false if, when plotting only two parameters, you do not want it to
            rotate the histogram so that it is horizontal.
        rainbow : bool, optional
            Set to True to force use of rainbow colours
        colours : str(hex)|list[str(hex)], optional
            Provide a list of colours to use for each chain. If you provide more chains
            than colours, you *will* get the rainbow colour spectrum. If you only pass
            one colour, all chains are set to this colour. This probably won't look good.
        linestyles : str, list[str], optional
            Provide a list of line styles to plot the contours and marginalsied
            distributions with. By default, this will become a list of solid lines. If a
            string is passed instead of a list, this style is used for all chains.
        linewidths : float, list[float], optional
            Provide a list of line widths to plot the contours and marginalsied
            distributions with. By default, this is a width of 1. If a float
            is passed instead of a list, this width is used for all chains.
        serif : bool, optional
            Whether to display ticks and labels with serif font.
        plot_hists : bool, optional
            Whether to plot marginalised distributions or not
        max_ticks : int, optional
            The maximum number of ticks to use on the plots
        kde : bool [optional]
            Whether to use a Gaussian KDE to smooth marginalised posteriors. If false, uses
            bins and linear interpolation, so ensure you have plenty of samples if your
            distribution is highly non-gaussian. Due to the slowness of performing a
            KDE on all data, it is often useful to disable this before producing final
            plots.
        smooth : int, optional
            How much to smooth the marginalised distributions using a gaussian filter.
            If ``kde`` is set to true, this parameter is ignored. Setting it to either
            ``0``, ``False`` or ``None`` disables smoothing.


        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        assert rainbow is None or colours is None, \
            "You cannot both ask for rainbow colours and then give explicit colours"

        assert statistics is not None, "statistics should be a string or list of strings!"
        if isinstance(statistics, str):
            statistics = [statistics.lower()] * len(self.chains)
        elif isinstance(statistics, list):
            for i, l in enumerate(statistics):
                statistics[i] = l.lower()
        else:
            raise ValueError("statistics is not a string or a list!")
        for s in statistics:
            assert s in ["max", "mean", "cumulative"], \
                "statistics %s not recognised. Should be max, mean or cumulative" % s
        self.parameters_general["statistics"] = statistics
        if bins is None:
            bins = self._get_bins()
        elif isinstance(bins, float):
            bins = [np.floor(b * bins) for b in self._get_bins()]
        elif isinstance(bins, int):
            bins = [bins] * len(self.chains)
        else:
            raise ValueError("bins value is not a recognised class (float or int)")
        self.parameters_general["bins"] = bins
        self.parameters_general["max_ticks"] = max_ticks
        self.parameters_general["flip"] = flip
        self.parameters_general["serif"] = serif
        self.parameters_general["rainbow"] = rainbow
        self.parameters_general["plot_hists"] = plot_hists
        self.parameters_general["kde"] = kde
        if not smooth or kde:
            smooth = None
        self.parameters_general["smooth"] = smooth
        if colours is None:
            if self.parameters_general.get("colours") is None:
                self.parameters_general["colours"] = self.all_colours[:len(self.chains)]
        else:
            if isinstance(colours, str):
                colours = [colours] * len(self.chains)
            self.parameters_general["colours"] = colours
        if linestyles is None:
            if self.parameters_general.get("linestyles") is None:
                self.parameters_general["linestyles"] = ["-"] * len(self.chains)
        else:
            if isinstance(linestyles, str):
                linestyles = [linestyles] * len(self.chains)
            self.parameters_general["linestyles"] = linestyles[:len(self.chains)]
        if linewidths is None:
            if self.parameters_general.get("linewidths") is None:
                self.parameters_general["linewidths"] = [1.0] * len(self.chains)
        else:
            if isinstance(linewidths, float) or isinstance(linewidths, int):
                linewidths = [linewidths] * len(self.chains)
            self.parameters_general["linewidths"] = linewidths[:len(self.chains)]
        self._configured_general = True
        return self

    def configure_contour(self, sigmas=None, cloud=None, shade=None,
                          shade_alpha=None):  # pragma: no cover
        """ Configure the default variables for the contour plots.

        If you do not call this explicitly, the :func:`plot` method
        will invoke this method automatically.

        Please ensure that you call this method *after* adding all the relevant data to the
        chain consumer, as the consume changes configuration values depending on
        the supplied data.

        Parameters
        ----------
        sigmas : np.array, optional
            The :math:`\sigma` contour levels to plot. Defaults to [0, 1, 2, 3] for a single chain
            and [0, 1, 2] for multiple chains. The leading zero is required if you don't want
            your surfaces to have a hole in them.
        cloud : bool, optional
            If set, overrides the default behaviour and plots the cloud or not
        shade : bool|list[bool] optional
            If set, overrides the default behaviour and plots filled contours or not. If a list of
            bools is passed, you can turn shading on or off for specific chains.
        shade_alpha : float|list[float], optional
            Filled contour alpha value override. Default is 1.0. If a list is passed, you can set the
            shade opacity for specific chains.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        num_chains = len(self.chains)

        if sigmas is None:
            if num_chains == 1:
                sigmas = np.array([0, 1, 2, 3])
            elif num_chains < 4:
                sigmas = np.array([0, 1, 2])
            else:
                sigmas = np.array([0, 1, 2])
        sigmas = np.sort(sigmas)
        self.parameters_contour["sigmas"] = sigmas
        if cloud is None:
            cloud = False
        self.parameters_contour["cloud"] = cloud

        if shade_alpha is None:
            if num_chains == 1:
                shade_alpha = 1.0
            else:
                shade_alpha = np.sqrt(1 / num_chains)
        if isinstance(shade_alpha, float) or isinstance(shade_alpha, int):
                shade_alpha = [shade_alpha] * num_chains

        if shade is None:
            shade = num_chains <= 2
        if isinstance(shade, bool):
            shade = [shade] * num_chains
        self.parameters_contour["shade"] = shade[:num_chains]
        self.parameters_contour["shade_alpha"] = shade_alpha[:num_chains]

        self._configured_contour = True

        return self

    def configure_bar(self, summary=None, shade=None):  # pragma: no cover
        """ Configure the bar plots showing the marginalised distributions.

        If you do not call this explicitly, the :func:`plot` method will
        invoke this method automatically.

        Please ensure that you call this method *after* adding all the relevant data to the
        chain consumer, as the consume changes configuration values depending on
        the supplied data.

        Parameters
        ----------
        summary : bool, optional
            If overridden, sets whether parameter summaries should be set as axis titles.
            Will not work if you have multiple chains
        shade : bool|list[bool], optional
            If set to true, shades in confidence regions in under histogram. By default
            this happens if you less than 3 chains, but is disabled if you are comparing
            more chains. You can pass a list if you wish to shade some chains but not others.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        if summary is not None:
            summary = summary and len(self.chains) == 1
        self.parameters_bar["summary"] = summary
        if shade is None:
            shade = len(self.chains) <= 2
        if isinstance(shade, bool):
            shade = [shade] * len(self.chains)
        self.parameters_bar["shade"] = shade[:len(self.chains)]
        self._configured_bar = True
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
        self.parameters_truth = kwargs
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
        for ind, (chain, parameters, weights) in enumerate(zip(self.chains,
                                                               self.parameters, self.weights)):
            res = {}
            for i, p in enumerate(parameters):
                summary = self._get_parameter_summary(chain[:, i], weights, p, ind)
                res[p] = summary
            results.append(res)
        if squeeze and len(results) == 1:
            return results[0]
        return results

    def get_latex_table(self, parameters=None, transpose=False, caption=None,
                        label=None, hlines=True, blank_fill="--"):  # pragma: no cover
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
            parameters = self.all_parameters
        for i, name in enumerate(self.names):
            assert name is not None, \
                "Generating a LaTeX table requires all chains to have names." \
                " Ensure you have `name=` in your `add_chain` call"
        for p in parameters:
            assert isinstance(p, str), \
                "Generating a LaTeX table requires all parameters have labels"
        num_parameters = len(parameters)
        num_chains = len(self.chains)
        fit_values = self.get_summary(squeeze=False)
        if label is None:
            label = ""
        if caption is None:
            caption = ""

        base_string = r"""\begin{table}[]
        \centering
        \caption{%s}
        \label{%s}
        \begin{tabular}{%s}
        %s      \end{tabular}
\end{table}"""
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
            center_text += " & ".join(["Parameter"] + self.names) + end_text
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
            for name, chain_res in zip(self.names, fit_values):
                arr = ["\t\t" + name]
                for p in parameters:
                    if p in chain_res:
                        arr.append(self.get_parameter_text(*chain_res[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        if hlines:
            center_text += "\t\t" + hline_text
        final_text = base_string % (caption, label, column_text, center_text)

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
            assert chain in self.names, "No chain with name %s found" % chain
            i = self.names.index(chain)
        elif isinstance(chain, int):
            i = chain
        else:
            raise ValueError("Type %s not recognised. Please pass in an int or a string" % type(chain))
        assert self.walkers[i] is not None, "The chain you have selected was not added with any walkers!"
        num_walkers = self.walkers[i]
        cs = np.split(self.chains[i], num_walkers)
        ws = np.split(self.weights[i], num_walkers)
        con = ChainConsumer()
        con._configured_bar = self._configured_bar
        con._configured_contour = self._configured_contour
        con._configured_truth = self._configured_truth
        con._configured_general = self._configured_general
        con.parameters_contour = self.parameters_contour
        con.parameters_bar = self.parameters_bar
        con.parameters_truth = self.parameters_truth
        con.parameters_general = self.parameters_general
        for j, (c, w) in enumerate(zip(cs, ws)):
            con.add_chain(c, weights=w, name="Chain %d" % j, parameters=self.parameters[i])
        return con

    def plot(self, figsize="GROW", parameters=None, extents=None, filename=None,
             display=False, truth=None, legend=None):  # pragma: no cover
        """ Plot the chain!

        Parameters
        ----------
        figsize : str|tuple(float), optional
            The figure size to generate. Accepts a regular two tuple of size in inches,
            or one of several key words. The default value of ``COLUMN`` creates a figure
            of appropriate size of insertion into an A4 LaTeX document in two-column mode.
            ``PAGE`` creates a full page width figure. ``GROW`` creates an image that
            scales with parameters (1.5 inches per parameter). String arguments are not
            case sensitive.
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

        if not self._configured_general:
            self.configure_general()
        if not self._configured_bar:
            self.configure_bar()
        if not self._configured_contour:
            self.configure_contour()
        if not self._configured_truth:
            self.configure_truth()
        if legend is None:
            legend = len(self.chains) > 1
        if parameters is None:
            parameters = self.all_parameters
        elif isinstance(parameters, int):
            parameters = self.all_parameters[:parameters]
        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()
        if truth is not None and isinstance(truth, list):
            truth = truth[:len(parameters)]

        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5, 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            elif figsize.upper() == "GROW":
                figsize = (1.5 * len(parameters), 1.5 * len(parameters))
            else:
                raise ValueError("Unknown figure size %s" % figsize)

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

        plot_hists = self.parameters_general["plot_hists"]
        flip = (len(parameters) == 2 and plot_hists and self.parameters_general["flip"])

        fig, axes, params1, params2, extents = self._get_figure(parameters, figsize=figsize,
                                                                flip=flip, external_extents=extents)

        num_bins = self.parameters_general["bins"]
        self.logger.info("Plotting surfaces with %s bins" % num_bins)
        fit_values = self.get_summary(squeeze=False)
        colours = self._get_colours(self.parameters_general["colours"],
                                    rainbow=self.parameters_general["rainbow"])
        linestyles = self.parameters_general["linestyles"]
        shades = self.parameters_contour["shade"]
        shade_alphas = self.parameters_contour["shade_alpha"]
        summary = self.parameters_bar["summary"]
        bar_shades = self.parameters_bar["shade"]
        linewidths = self.parameters_general["linewidths"]
        num_chains = len(self.chains)
        assert len(linestyles) == num_chains, \
            "Have %d linestyles and %d chains. Please address." % (len(linestyles), num_chains)
        assert len(linewidths) == num_chains, \
            "Have %d linewidths and %d chains. Please address." % (len(linewidths), num_chains)
        assert len(bar_shades) == num_chains, \
            "Have %d bar_shades and %d chains. Please address." % (len(bar_shades), num_chains)
        assert len(shade_alphas) == num_chains, \
            "Have %d shade_alphas and %d chains. Please address." % (len(shade_alphas), num_chains)
        assert len(shades) == num_chains, \
            "Have %d shades and %d chains. Please address." % (len(shades), num_chains)

        if summary is None:
            summary = len(parameters) < 5 and len(self.chains) == 1
        if len(self.chains) == 1:
            self.logger.debug("Plotting surfaces for chain of dimenson %s" %
                              (self.chains[0].shape,))
        else:
            self.logger.debug("Plotting surfaces for %d chains" % len(self.chains))
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j:
                    continue
                ax = axes[i, j]
                do_flip = (flip and i == len(params1) - 1)
                if plot_hists and i == j:
                    max_val = None
                    for chain, weights, parameters, colour, bins, fit, ls, bs, lw in \
                            zip(self.chains, self.weights, self.parameters, colours,
                                num_bins, fit_values, linestyles, bar_shades, linewidths):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        m = self._plot_bars(ax, p1, chain[:, index], weights, colour, ls, bs, lw, bins=bins,
                                            fit_values=fit[p1], flip=do_flip, summary=summary,
                                            truth=truth, extents=extents[p1])
                        if max_val is None or m > max_val:
                            max_val = m
                    if do_flip:
                        ax.set_xlim(0, 1.1 * max_val)
                    else:
                        ax.set_ylim(0, 1.1 * max_val)

                else:
                    for chain, parameters, bins, colour, ls, s, sa, lw, fit, weights in \
                            zip(self.chains, self.parameters, num_bins, colours, linestyles, shades,
                                shade_alphas, linewidths, fit_values, self.weights):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        self._plot_contour(ax, chain[:, i2], chain[:, i1], weights, p1, p2, colour, ls,
                                           s, sa, lw, bins=bins, truth=truth)

        if self.names is not None and legend:
            ax = axes[0, -1]
            artists = [plt.Line2D((0, 1), (0, 0), color=c)
                       for n, c in zip(self.names, colours) if n is not None]
            location = "center" if len(parameters) > 1 else 1
            ax.legend(artists, self.names, loc=location, frameon=False)
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
                   filename=None, chain=None, convolve=None, figsize=None,
                   plot_weights=True, plot_posterior=True): # pragma: no cover
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
        chain : int|str, optional
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

        Returns
        -------
        figure
            the matplotlib figure created

        """
        if not self._configured_general:
            self.configure_general()
        if not self._configured_bar:
            self.configure_bar()
        if not self._configured_contour:
            self.configure_contour()
        if not self._configured_truth:
            self.configure_truth()

        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()

        if parameters is None:
            parameters = self.all_parameters
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

        if chain is None:
            if len(self.chains) == 1:
                chain = 0
            else:
                raise ValueError("You can only plot walks for one chain at a time. "
                                 "If you have multiple chains, please pass an "
                                 "index or a chain name via the chain parameter")

        if isinstance(chain, str):
            assert chain in self.names, \
                "A chain with name %s is not found in available names: %s" % (chain, self.names)
            chain = self.names.index(chain)

        n = len(parameters)
        extra = 0
        if plot_weights:
            plot_weights = plot_weights and np.any(self.weights[chain] != 1.0)

        plot_posterior = plot_posterior and self.posteriors[chain] is not None

        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            figsize = (8, 0.75 + (n + extra))

        chain_data = self.chains[chain]
        self.logger.debug("Plotting chain of size %s" % (chain_data.shape,))
        chain_parameters = self.parameters[chain]

        fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)

        plt.rc('text', usetex=True)
        if self.parameters_general["serif"]:
            plt.rc('font', family='serif')
        else:
            plt.rc('font', family='sans-serif')
        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = parameters[i - n]
                assert p in chain_parameters, \
                    "Chain does not have parameter %s, it has %s" % (p, chain_parameters)
                chain_row = chain_data[:, chain_parameters.index(p)]
                self._plot_walk(ax, p, chain_row, truth=truth.get(p),
                                extents=extents.get(p), convolve=convolve)
            else:
                if i == 0 and plot_posterior:
                    self._plot_walk(ax, "$\log(P)$", self.posteriors[chain] - self.posteriors[chain].max(), convolve=convolve)
                else:
                    self._plot_walk(ax, "$w$", self.weights[chain], convolve=convolve)

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
            keys = [n if n is not None else i for i, n in enumerate(self.names)]
            return np.all([self.diagnostic_gelman_rubin(k, threshold=threshold) for k in keys])
        index = self._get_chain(chain)
        num_walkers = self.walkers[index]
        parameters = self.parameters[index]
        name = self.names[index] if self.names[index] is not None else "%d" % index
        chain = self.chains[index]
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
        passed = np.abs(var - 1) < threshold
        print("Gelman-Rubin Statistic values for chain %s" % name)
        for p, v, pas in zip(parameters, var, passed):
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
            keys = [n if n is not None else i for i, n in enumerate(self.names)]
            return np.all([self.diagnostic_geweke(k, threshold=threshold) for k in keys])
        index = self._get_chain(chain)
        num_walkers = self.walkers[index]
        name = self.names[index] if self.names[index] is not None else "%d" % index
        chain = self.chains[index]
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

    # Method of estimating spectral density following PyMC.
    # See https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py
    def _spec(self, x, order=2):
        beta, sigma = yule_walker(x, order)
        return sigma ** 2 / (1. - np.sum(beta)) ** 2

    def _get_chain(self, chain):
        if isinstance(chain, str):
            assert chain in self.names, "Chain %s not found!" % chain
            index = self.names.index(chain)
        elif isinstance(chain, int):
            assert chain < len(self.chains), "Chain index %d not found!" % chain
            index = chain
        else:
            raise ValueError("Type %s not recognised for chain" % type(chain))
        return index

    def _plot_walk(self, ax, parameter, data, truth=None, extents=None,
                   convolve=None):  # pragma: no cover
        if extents is not None:
            ax.set_ylim(extents)
        assert convolve is None or isinstance(convolve, int), \
            "Convolve must be an integer pixel window width"
        x = np.arange(data.size)
        ax.set_xlim(0, x[-1])
        ax.set_ylabel(parameter)
        ax.scatter(x, data, c="#0345A1", s=2, marker=".", edgecolors="none", alpha=0.5)
        max_ticks = self.parameters_general["max_ticks"]
        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
        if convolve is not None:
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=':', color="#FF0000", alpha=1)
        if truth is not None:
            ax.axhline(truth, **self.parameters_truth)

    def _plot_bars(self, ax, parameter, chain_row, weights, colour, linestyle, bar_shade,
                   linewidth, bins=25, flip=False, summary=False, fit_values=None,
                   truth=None, extents=None):  # pragma: no cover

        kde = self.parameters_general["kde"]
        smooth = self.parameters_general["smooth"]
        bins, smooth = self._get_smoothed_bins(smooth, bins)

        bins = np.linspace(extents[0], extents[1], bins)
        hist, edges = np.histogram(chain_row, bins=bins, normed=True, weights=weights)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        if smooth:
            hist = gaussian_filter(hist, smooth, mode='constant')
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
                x = np.linspace(lower, upper, 1000)
                if lower > edge_center.min() and upper < edge_center.max():
                    if flip:
                        ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x),
                                         color=colour, alpha=0.2)
                    else:
                        ax.fill_between(x, np.zeros(x.shape), interpolator(x),
                                        color=colour, alpha=0.2)
                if summary and isinstance(parameter, str):
                    ax.set_title(r"$%s = %s$" % (parameter.strip("$"),
                                                 self.get_parameter_text(*fit_values)), fontsize=14)
        if truth is not None:
            truth_value = truth.get(parameter)
            if truth_value is not None:
                if flip:
                    ax.axhline(truth_value, **self.parameters_truth)
                else:
                    ax.axvline(truth_value, **self.parameters_truth)
        return hist.max()

    def _plot_contour(self, ax, x, y, w, px, py, colour, linestyle, shade,
                      shade_alpha, linewidth, bins=25, truth=None):  # pragma: no cover

        levels = 1.0 - np.exp(-0.5 * self.parameters_contour["sigmas"] ** 2)
        smooth = self.parameters_general["smooth"]
        bins, smooth = self._get_smoothed_bins(smooth, bins, marginalsied=False)

        colours = self._scale_colours(colour, len(levels))
        colours2 = [self._scale_colour(colours[0], 0.7)] + \
                   [self._scale_colour(c, 0.8) for c in colours[:-1]]

        hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins, weights=w)
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        if smooth:
            hist = gaussian_filter(hist, smooth, mode='constant')
        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if self.parameters_contour["cloud"]:
            skip = max(1, int(x.size / 50000))
            ax.scatter(x[::skip], y[::skip], s=10, alpha=0.3, c=colours[1],
                       marker=".", edgecolors="none")
        if shade:
            ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours,
                        alpha=shade_alpha)
        ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2,
                   linestyles=linestyle, linewidths=linewidth)

        if truth is not None:
            truth_value = truth.get(px)
            if truth_value is not None:
                ax.axhline(truth_value, **self.parameters_truth)
            truth_value = truth.get(py)
            if truth_value is not None:
                ax.axvline(truth_value, **self.parameters_truth)

    def _get_colours(self, colours, rainbow=False):  # pragma: no cover
        num_chains = len(self.chains)
        if rainbow or num_chains > len(colours):
            colours = cm.rainbow(np.linspace(0, 1, num_chains))
        else:
            colours = colours[:num_chains]
        return colours

    def _get_figure(self, all_parameters, flip, figsize=(5, 5),
                    external_extents=None):  # pragma: no cover
        n = len(all_parameters)
        max_ticks = self.parameters_general["max_ticks"]
        plot_hists = self.parameters_general["plot_hists"]
        max_sigma = np.array(self.parameters_contour["sigmas"]).max()
        sigma_extent = max(3, max_sigma + 1)
        if not plot_hists:
            n -= 1

        if n == 2 and plot_hists and flip:
            gridspec_kw = {'width_ratios': [3, 1], 'height_ratios': [1, 3]}
        else:
            gridspec_kw = {}
        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)

        if self.parameters_general["serif"]:
            plt.rc('text', usetex=True)
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
                for chain, parameters in zip(self.chains, self.parameters):
                    if p not in parameters:
                        continue
                    index = parameters.index(p)
                    # min_val = chain[:, index].min()
                    # max_val = chain[:, index].max()
                    mean = np.mean(chain[:, index])
                    std = np.std(chain[:, index])
                    min_prop = mean - sigma_extent * std
                    max_prop = mean + sigma_extent* std
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
                    for chain in self.chains]
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

    def _get_smoothed_histogram(self, data, weights, chain_index):
        smooth = self.parameters_general["smooth"]
        bins = self.parameters_general['bins'][chain_index]
        bins, smooth = self._get_smoothed_bins(smooth, bins)
        hist, edges = np.histogram(data, bins=bins, normed=True, weights=weights)
        edge_centers = 0.5 * (edges[1:] + edges[:-1])
        xs = np.linspace(edge_centers[0], edge_centers[-1], 10000)
        if smooth:
            hist = gaussian_filter(hist, smooth, mode='constant')

        if self.parameters_general["kde"]:
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
        if not self._configured_general:
            self.configure_general()
        method = self.summaries[self.parameters_general["statistics"][chain_index]]
        return method(data, weights, parameter, chain_index, **kwargs)

    def _get_parameter_summary_mean(self, data, weights, parameter, chain_index, desired_area=0.6827):
        xs, ys, cs = self._get_smoothed_histogram(data, weights, chain_index)
        vals = [0.5 - desired_area / 2, 0.5, 0.5 + desired_area / 2]
        bounds = interp1d(cs, xs)(vals)
        bounds[1] = 0.5 * (bounds[0] + bounds[2])
        return bounds

    def _get_parameter_summary_cumulative(self, data, weights, parameter, chain_index, desired_area=0.6827):
        xs, ys, cs = self._get_smoothed_histogram(data, weights, chain_index)
        vals = [0.5 - desired_area / 2, 0.5, 0.5 + desired_area / 2]
        bounds = interp1d(cs, xs)(vals)
        return bounds

    def _get_parameter_summary_max(self, data, weights, parameter, chain_index, desired_area=0.6827):
        xs, ys, cs = self._get_smoothed_histogram(data, weights, chain_index)
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
                self.logger.warn("Parameter %s is not constrained" % parameter)
                return [None, xs[startIndex], None]

        return [x1, xs[startIndex], x2]


