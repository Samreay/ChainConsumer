import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import statsmodels.api as sm


__all__ = ["ChainConsumer"]


class ChainConsumer(object):
    """ A class for consuming chains produced by an MCMC walk

    """
    __version__ = "0.9.0"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.all_colours = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107",
                            "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        self.chains = []
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

    def add_chain(self, chain, parameters=None, name=None, weights=None, posterior=None):
        """ Add a chain to the consumer.

        Parameters
        ----------
        chain : str|ndarray
            The chain to load. Normally a ``numpy.ndarray``, but can also accept a string.
            If a string is found, it interprets the string as a filename
            and attempts to load it in.
        parameters : list[str], optional
            A list of parameter names, one for each column (dimension) in the chain.
        name : str, optional
            The name of the chain. Used when plotting multiple chains at once.
        weights : ndarray, optional
            If given, uses this array to weight the samples in chain
        posterior : ndarray, optional
            If given, records the log posterior for each sample in the chain

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
        if len(chain.shape) == 1:
            chain = chain[None].T
        self.chains.append(chain)
        self.names.append(name)
        self.posteriors.append(posterior)
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

    def configure_general(self, bins=None, flip=True, rainbow=None, colours=None,
                          serif=True, plot_hists=True, max_ticks=5, kde=False):  # pragma: no cover
        r""" Configure the general plotting parameters common across the bar
        and contour plots. If you do not call this explicitly, the :func:`plot`
        method will invoke this method automatically.

        Parameters
        ----------
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
        colours : list[str(hex)], optional
            Provide a list of colours to use for each chain. If you provide more chains
            than colours, you *will* get the rainbow colour spectrum.
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

        """
        assert rainbow is None or colours is None, \
            "You cannot both ask for rainbow colours and then give explicit colours"

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
        if colours is None:
            self.parameters_general["colours"] = self.all_colours
        else:
            self.parameters_general["colours"] = colours

        self._configured_general = True
        return self

    def configure_contour(self, sigmas=None, cloud=None, contourf=None,
                          contourf_alpha=1.0):  # pragma: no cover
        """ Configure the default variables for the contour plots. If you do not call this
        explicitly, the :func:`plot` method will invoke this method automatically.

        Please ensure that you call this method after adding all the relevant data to the
        chain consumer, as the consume changes configuration values depending on
        the presupplied data.

        Parameters
        ----------
        sigmas : np.array, optional
            The :math:`\sigma` contour levels to plot. Defaults to [0.5, 1, 2, 3].
            Number of contours shown decreases with the number of chains to show.
        cloud : bool, optional
            If set, overrides the default behaviour and plots the cloud or not
        contourf : bool, optional
            If set, overrides the default behaviour and plots filled contours or not
        contourf_alpha : float, optional
            Filled contour alpha value override.
        """
        num_chains = len(self.chains)

        if sigmas is None:
            if num_chains == 1:
                sigmas = np.array([0, 0.5, 1, 1.5, 2])
            elif num_chains < 4:
                sigmas = np.array([0, 0.5, 1, 2])
            else:
                sigmas = np.array([0, 1, 2])
        sigmas = np.sort(sigmas)
        self.parameters_contour["sigmas"] = sigmas
        if cloud is None:
            cloud = False
        self.parameters_contour["cloud"] = cloud

        if contourf is None:
            contourf = num_chains == 1
        self.parameters_contour["contourf"] = contourf
        self.parameters_contour["contourf_alpha"] = contourf_alpha

        self._configured_contour = True

        return self

    def configure_bar(self, summary=None, shade=None):  # pragma: no cover
        """ Configure the bar plots showing the marginalised distributions. If you do not
        call this explicitly, the :func:`plot` method will invoke this method automatically.

        summary : bool, optional
            If overridden, sets whether parameter summaries should be set as axis titles.
            Will not work if you have multiple chains
        shade : bool, optional
            If set to true, shades in confidence regions in under histogram. By default
            this happens if you have a single chain, but is disabled if you are comparing
            multiple chains.
        """
        if summary is not None:
            summary = summary and len(self.chains) == 1
        self.parameters_bar["summary"] = summary
        self.parameters_bar["shade"] = shade if shade is not None else len(self.chains) == 1
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
        """
        if kwargs.get("ls") is None and kwargs.get("linestyle") is None:
            kwargs["ls"] = "--"
            kwargs["dashes"] = (3, 3)
        if kwargs.get("color") is None:
            kwargs["color"] = "#000000"
        self.parameters_truth = kwargs
        self._configured_truth = True
        return self

    def get_summary(self):
        """  Gets a summary of the marginalised parameter distributions.

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
        fit_values = self.get_summary()
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

    def divide_chain(self, i, num_walkers):
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
        """ Plot the chain

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
        fit_values = self.get_summary()
        colours = self._get_colours(self.parameters_general["colours"],
                                    rainbow=self.parameters_general["rainbow"])
        summary = self.parameters_bar["summary"]
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
                    for chain, weights, parameters, colour, bins, fit in \
                            zip(self.chains, self.weights, self.parameters, colours,
                                num_bins, fit_values):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        m = self._plot_bars(ax, p1, chain[:, index], weights, colour, bins=bins,
                                            fit_values=fit[p1], flip=do_flip, summary=summary,
                                            truth=truth, extents=extents[p1])
                        if max_val is None or m > max_val:
                            max_val = m
                    if do_flip:
                        ax.set_xlim(0, 1.1 * max_val)
                    else:
                        ax.set_ylim(0, 1.1 * max_val)

                else:
                    for chain, parameters, bins, colour, fit, weights in \
                            zip(self.chains, self.parameters, num_bins,
                                colours, fit_values, self.weights):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        self._plot_contour(ax, chain[:, i2], chain[:, i1], weights, p1, p2, colour,
                                           bins=bins, truth=truth)

        if self.names is not None and legend:
            ax = axes[0, -1]
            artists = [plt.Line2D((0, 1), (0, 0), color=c)
                       for n, c in zip(self.names, colours) if n is not None]
            location = "center" if len(parameters) > 1 else 1
            ax.legend(artists, self.names, loc=location, frameon=False)

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

    def _plot_bars(self, ax, parameter, chain_row, weights, colour, bins=25, flip=False, summary=False,
                   fit_values=None, truth=None, extents=None):  # pragma: no cover
        bins = np.linspace(extents[0], extents[1], bins + 1)
        hist, edges = np.histogram(chain_row, bins=bins, normed=True, weights=weights)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        kde = self.parameters_general["kde"]
        if kde:
            assert np.all(weights == 1.0), "You can only use KDE if your weights are all one. " \
                                           "If you would like weights, please vote for this issue: " \
                                           "https://github.com/scikit-learn/scikit-learn/issues/4394"
            pdf = sm.nonparametric.KDEUnivariate(chain_row)
            pdf.fit()
            xs = np.linspace(extents[0], extents[1], 100)
            if flip:
                ax.plot(pdf.evaluate(xs), xs, color=colour)
            else:
                ax.plot(xs, pdf.evaluate(xs), color=colour)
            interpolator = pdf.evaluate
        else:
            if flip:
                orientation = "horizontal"
            else:
                orientation = "vertical"
            ax.hist(edge_center, weights=hist, bins=edges, histtype="step",
                    color=colour, orientation=orientation)
            interpolator = interp1d(edge_center, hist, kind="nearest")

        if self.parameters_bar["shade"] and fit_values is not None:
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

    def _plot_contour(self, ax, x, y, w, px, py, colour, bins=25, truth=None):  # pragma: no cover

        levels = 1.0 - np.exp(-0.5 * self.parameters_contour["sigmas"] ** 2)

        colours = self._scale_colours(colour, len(levels))
        colours2 = [self._scale_colour(colours[0], 0.7)] + \
                   [self._scale_colour(c, 0.8) for c in colours[:-1]]

        hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins, weights=w)
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if self.parameters_contour["cloud"]:
            skip = max(1, x.size / 80000)
            ax.scatter(x[::skip], y[::skip], s=10, alpha=0.4, c=colours[1],
                       marker=".", edgecolors="none")
        if self.parameters_contour["contourf"]:
            ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours,
                        alpha=self.parameters_contour["contourf_alpha"])
        ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2)

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
                    min_prop = mean - 3 * std
                    max_prop = mean + 3 * std
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
                    if display_y_ticks:
                        [l.set_rotation(45) for l in ax.get_yticklabels()]
                        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                    if i != j or not plot_hists:
                        ax.set_ylim(extents[p1])
                    elif flip and i == 1:
                        ax.set_ylim(extents[p1])
                    ax.set_xlim(extents[p2])

        return fig, axes, params1, params2, extents

    def _get_bins(self):
        proposal = [max(20, np.floor(1.2 * np.power(chain.shape[0] / chain.shape[1], 0.3)))
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
        scales = np.logspace(np.log(0.8), np.log(1.4), num)
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

    def _get_parameter_summary(self, data, weights, parameter, chain_index, desired_area=0.6827):
        if not self._configured_general:
            self.configure_general()
        bins = self.parameters_general['bins'][chain_index]
        hist, edges = np.histogram(data, bins=bins, normed=True, weights=weights)
        edge_centers = 0.5 * (edges[1:] + edges[:-1])
        xs = np.linspace(edge_centers[0], edge_centers[-1], 10000)
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


