# -*- coding: utf-8 -*-
import numpy as np
import logging

from .comparisons import Comparison
from .diagnostic import Diagnostic
from .plotter import Plotter
from .helpers import get_bins
from .analysis import Analysis
from .colors import Colors
from .chain import Chain

__all__ = ["ChainConsumer"]


class ChainConsumer(object):
    """ A class for consuming chains produced by an MCMC walk. Or grid searches. To make plots, 
    figures, tables, diagnostics, you name it.

    """
    __version__ = "0.27.0"

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger("chainconsumer")
        self.color_finder = Colors()
        self._all_colours = self.color_finder.get_default()
        self._cmaps = ["viridis", "inferno", "hot", "Blues", "Greens", "Greys"]
        self._linestyles = ["-", '--', ':']
        self.chains = []
        self._all_parameters = []
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
        # for c in self.chains:
        #     c.reset_config()

    def get_mcmc_chains(self):
        return [c for c in self.chains if c.mcmc_chain]

    def add_chain(self, chain, parameters=None, name=None, weights=None, posterior=None, walkers=None,
                  grid=False, num_eff_data_points=None, num_free_params=None, color=None, linewidth=None,
                  linestyle=None, kde=None, shade=None, shade_alpha=None, power=None, marker_style=None, marker_size=None,
                  marker_alpha=None, plot_contour=None, plot_point=None, statistics=None, cloud=None,
                  shade_gradient=None, bar_shade=None, bins=None, smooth=None, color_params=None,
                  plot_color_params=None, cmap=None, num_cloud=None):
        r""" Add a chain to the consumer.

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
        color : str(hex), optional
            Provide a colour for the chain. Can be used instead of calling `configure` for convenience.
        linewidth : float, optional
            Provide a line width to plot the contours. Can be used instead of calling `configure` for convenience.
        linestyle : str, optional
            Provide a line style to plot the contour. Can be used instead of calling `configure` for convenience.
        kde : bool|float, optional
            Set the `kde` value for this specific chain. Can be used instead of calling `configure` for convenience.
        shade : booloptional
            If set, overrides the default behaviour and plots filled contours or not. If a list of
            bools is passed, you can turn shading on or off for specific chains.
        shade_alpha : float, optional
            Filled contour alpha value. Can be used instead of calling `configure` for convenience.
        power : float, optional
            The power to raise the posterior surface to. Useful for inflating or deflating uncertainty for debugging.
        marker_style : str|, optional
            The marker style to use when plotting points. Defaults to `'.'`
        marker_size : numeric|, optional
            Size of markers, if plotted. Defaults to `20`.
        marker_alpha : numeric, optional
            The alpha values when plotting markers.
        plot_contour : bool, optional
            Whether to plot the whole contour (as opposed to a point). Defaults to true for less than
            25 concurrent chains.
        plot_point : bool, optional
            Whether to plot a maximum likelihood point. Defaults to true for more then 24 chains.
        statistics : string, optional
            Which sort of statistics to use. Defaults to `"max"` for maximum likelihood
            statistics. Other available options are `"mean"`, `"cumulative"`, `"max_symmetric"`,
            `"max_closest"` and `"max_central"`. In the
            very, very rare case you want to enable different statistics for different
            chains, you can pass in a list of strings.
        cloud : bool, optional
            If set, overrides the default behaviour and plots the cloud or not        shade_gradient :
        bar_shade : bool, optional
            If set to true, shades in confidence regions in under histogram. By default
            this happens if you less than 3 chains, but is disabled if you are comparing
            more chains. You can pass a list if you wish to shade some chains but not others.
        bins : int|float, optional
            The number of bins to use. By default uses :math:`\frac{\sqrt{n}}{10}`, where
            :math:`n` are the number of data points. Giving an integer will set the number
            of bins to the given value. Giving a float will scale the number of bins, such
            that giving ``bins=1.5`` will result in using :math:`\frac{1.5\sqrt{n}}{10}` bins.
            Note this parameter is most useful if `kde=False` is also passed, so you
            can actually see the bins and not a KDE.        smooth : 
        color_params : str, optional
            The name of the parameter to use for the colour scatter. Defaults to none, for no colour. If set
            to 'weights', 'log_weights', or 'posterior' (without the quotes), and that is not a parameter in the chain, 
            it will respectively  use the weights, log weights, or posterior, to colour the points.
        plot_color_params : bool, optional
            Whether or not the colour parameter should also be plotted as a posterior surface.
        cmaps : str, optional
            The matplotlib colourmap to use in the `colour_param`. If you have multiple `color_param`s, you can
            specific a different cmap for each variable. By default ChainConsumer will cycle between several
            cmaps.
        num_cloud : int, optional
            The number of scatter points to show when enabling `cloud` or setting one of the parameters
            to colour scatter. Defaults to 15k per chain.
            
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
                assert weights.size == chain[:,
                                       0].size, "Error, given weight array size disagrees with parameter sampling"

        if len(chain.shape) == 1:
            chain = chain[None].T

        if name is None:
            name = "Chain %d" % len(self.chains)

        if power is not None:
            assert isinstance(power, int) or isinstance(power, float), "Power should be numeric, but is %s" % type(
                power)

        if self._default_parameters is None and parameters is not None:
            self._default_parameters = parameters

        if parameters is None:
            if self._default_parameters is not None:
                assert chain.shape[1] == len(self._default_parameters), \
                    "Chain has %d dimensions, but default parameters have %d dimensions" \
                    % (chain.shape[1], len(self._default_parameters))
                parameters = self._default_parameters
                self._logger.debug("Adding chain using default parameters")
            else:
                self._logger.debug("Adding chain with no parameter names")
                parameters = ["%d" % x for x in range(chain.shape[1])]
        else:
            self._logger.debug("Adding chain with defined parameters")
            assert len(parameters) <= chain.shape[1], \
                "Have only %d columns in chain, but have been given %d parameters names! " \
                "Please double check this." % (chain.shape[1], len(parameters))
        for p in parameters:
            if p not in self._all_parameters:
                self._all_parameters.append(p)

        # Sorry, no KDE for you on a grid.
        if grid:
            kde = None
        if color is not None:
            color = self.color_finder.get_formatted([color])[0]

        c = Chain(chain, parameters, name, weights=weights, posterior=posterior, walkers=walkers,
                  grid=grid, num_free_params=num_free_params, num_eff_data_points=num_eff_data_points,
                  color=color, linewidth=linewidth, linestyle=linestyle, kde=kde, shade_alpha=shade_alpha, power=power,
                  marker_style=marker_style, marker_size=marker_size, marker_alpha=marker_alpha,
                  plot_contour=plot_contour, plot_point=plot_point, statistics=statistics, cloud=cloud,
                  shade=shade, shade_gradient=shade_gradient, bar_shade=bar_shade, bins=bins, smooth=smooth,
                  color_params=color_params, plot_color_params=plot_color_params, cmap=cmap,
                  num_cloud=num_cloud)
        self.chains.append(c)
        self._init_params()
        return self

    def add_covariance(self, mean, covariance, parameters=None, name=None, **kwargs):
        r""" Generate samples as per mean and covariance supplied. Useful for Fisher matrix forecasts.

        Parameters
        ----------
        mean : list|np.ndarray
            The an array of mean values.
        covariance : list|np.ndarray
            The 2D array describing the covariance. Dimensions should agree with the `mean` input.
        parameters : list[str], optional
            A list of parameter names, one for each column (dimension) in the mean array.
        name : str, optional
            The name of the chain. Used when plotting multiple chains at once.
        kwargs :
            Extra arguments about formatting - identical to what you would find in `add_chain`. `linewidth`, `color`,
            etc.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        chain = np.random.multivariate_normal(mean, covariance, size=1000000)
        self.add_chain(chain, parameters=parameters, name=name, **kwargs)
        self.chains[-1].mcmc_chain = False  # So we dont plot this when looking at walks, etc
        return self

    def add_marker(self, location, parameters=None, name=None, color=None, marker_size=None,
                   marker_style=None, marker_alpha=None):
        r""" Add a marker to the plot at the given location.

        Parameters
        ----------
        location : list|np.ndarray
            The coordinates to place the marker
        parameters : list[str], optional
            A list of parameter names, one for each column (dimension) in the mean array.
        name : str, optional
            The name of the chain. Used when plotting multiple chains at once.
        color : str(hex), optional
            Provide a colour for the chain. Can be used instead of calling `configure` for convenience.
        marker_style : str|, optional
            The marker style to use when plotting points. Defaults to `'.'`
        marker_size : numeric|, optional
            Size of markers, if plotted. Defaults to `20`.
        marker_alpha : numeric, optional
            The alpha values when plotting markers.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        chain = np.vstack((location, location))
        posterior = np.array([0, 1])
        self.add_chain(chain, parameters=parameters, posterior=posterior, name=name, color=color, marker_size=marker_size,
                       marker_style=marker_style, marker_alpha=marker_alpha, plot_point=True, plot_contour=False)
        self.chains[-1].mcmc_chain = False  # So we dont plot this when looking at walks, etc
        return self

    def remove_chain(self, chain=-1):
        r""" Removes a chain from ChainConsumer.

        Calling this will require any configurations set to be redone!

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

        chain = sorted([i for c in chain for i in self._get_chain(c)])[::-1]
        assert len(chain) == len(list(set(chain))), "Error, you are trying to remove a chain more than once."

        for index in chain:
            del self.chains[index]

        seen = set()
        self._all_parameters = [p for c in self.chains for p in c.parameters if not (p in seen or seen.add(p))]

        # Need to reconfigure
        self._init_params()

        return self

    def configure(self, statistics="max", max_ticks=5, plot_hists=True, flip=True,
                  serif=True, sigma2d=False, sigmas=None, summary=None, bins=None, rainbow=None,
                  colors=None, linestyles=None, linewidths=None, kde=False, smooth=None,
                  cloud=None, shade=None, shade_alpha=None, shade_gradient=None, bar_shade=None,
                  num_cloud=None, color_params=None, plot_color_params=False, cmaps=None,
                  plot_contour=None, plot_point=None, global_point=True, marker_style=None, marker_size=None, marker_alpha=None,
                  usetex=True, diagonal_tick_labels=True, label_font_size=12, tick_font_size=10,
                  spacing=None, contour_labels=None, contour_label_font_size=10,
                  legend_kwargs=None, legend_location=None, legend_artists=None,
                  legend_color_text=True, watermark_text_kwargs=None, summary_area=0.6827):  # pragma: no cover
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
            statistics. Other available options are `"mean"`, `"cumulative"`, `"max_symmetric"`,
            `"max_closest"` and `"max_central"`. In the
            very, very rare case you want to enable different statistics for different
            chains, you can pass in a list of strings.
        max_ticks : int, optional
            The maximum number of ticks to use on the plots
        plot_hists : bool, optional
            Whether to plot marginalised distributions or not
        flip : bool, optional
            Set to false if, when plotting only two parameters, you do not want it to
            rotate the histogram so that it is horizontal.
        sigma2d: bool, optional
            Defaults to `False`. When `False`, uses :math:`\sigma` levels for 1D Gaussians - ie confidence
            levels of 68% and 95%. When `True`, uses the confidence levels for 2D Gaussians, where 1 and 2
            :math:`\sigma` represents 39% and 86% confidence levels respectively.
        sigmas : np.array, optional
            The :math:`\sigma` contour levels to plot. Defaults to [0, 1, 2, 3] for a single chain
            and [0, 1, 2] for multiple chains.
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
            Provide a list of line styles to plot the contours and marginalised
            distributions with. By default, this will become a list of solid lines. If a
            string is passed instead of a list, this style is used for all chains.
        linewidths : float|list[float], optional
            Provide a list of line widths to plot the contours and marginalised
            distributions with. By default, this is a width of 1. If a float
            is passed instead of a list, this width is used for all chains.
        kde : bool|float|list[bool|float], optional
            Whether to use a Gaussian KDE to smooth marginalised posteriors. If false, uses
            bins and linear interpolation, so ensure you have plenty of samples if your
            distribution is highly non-gaussian. Due to the slowness of performing a
            KDE on all data, it is often useful to disable this before producing final
            plots. If float, scales the width of the KDE bandpass manually.
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
        shade_gradient : float|list[float], optional
            How much to vary colours in different contour levels.
        bar_shade : bool|list[bool], optional
            If set to true, shades in confidence regions in under histogram. By default
            this happens if you less than 3 chains, but is disabled if you are comparing
            more chains. You can pass a list if you wish to shade some chains but not others.
        num_cloud : int|list[int], optional
            The number of scatter points to show when enabling `cloud` or setting one of the parameters
            to colour scatter. Defaults to 15k per chain.
        color_params : str|list[str], optional
            The name of the parameter to use for the colour scatter. Defaults to none, for no colour. If set
            to 'weights', 'log_weights', or 'posterior' (without the quotes), and that is not a parameter in the chain, 
            it will respectively  use the weights, log weights, or posterior, to colour the points.
        plot_color_params : bool|list[bool], optional
            Whether or not the colour parameter should also be plotted as a posterior surface.
        cmaps : str|list[str], optional
            The matplotlib colourmap to use in the `colour_param`. If you have multiple `color_param`s, you can
            specific a different cmap for each variable. By default ChainConsumer will cycle between several
            cmaps.
        plot_contour : bool|list[bool], optional
            Whether to plot the whole contour (as opposed to a point). Defaults to true for less than
            25 concurrent chains.
        plot_point : bool|list[bool], optional
            Whether to plot a maximum likelihood point. Defaults to true for more then 24 chains.
        global_point : bool, optional
            Whether the point which gets plotted is the global posterior maximum, or the marginalised 2D 
            posterior maximum. Note that when you use marginalised 2D maximums for the points, you do not
             get the 1D histograms. Defaults to `True`, for a global maximum value.
        marker_style : str|list[str], optional
            The marker style to use when plotting points. Defaults to `'.'`
        marker_size : numeric|list[numeric], optional
            Size of markers, if plotted. Defaults to `20`.
        marker_alpha : numeric|list[numeric], optional
            The alpha values when plotting markers.
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
        contour_labels : string, optional
            If unset do not plot contour labels. If set to "confidence", label the using confidence
            intervals. If set to "sigma", labels using sigma.
        contour_label_font_size : int|float, optional
            The font size for contour labels, if they are enabled.
        legend_kwargs : dict, optional
            Extra arguments to pass to the legend api.
        legend_location : tuple(int,int), optional
            Specifies the subplot in which to locate the legend. By default, this will be (0, -1),
            corresponding to the top right subplot if there are more than two parameters,
            and the bottom left plot for only two parameters with flip on.
            For having the legend in the primary subplot
            in the bottom left, set to (-1,0).
        legend_artists : bool, optional
            Whether to include hide artists in the legend. If all linestyles and line widths are identical,
            this will default to false (as only the colours change). Otherwise it will be true.
        legend_color_text : bool, optional
            Whether to colour the legend text.
        watermark_text_kwargs : dict, optional
            Options to pass to the fontdict property when generating text for the watermark.
        summary_area : float, optional
            The confidence interval used when generating parameter summaries. Defaults to 1 sigma, aka 0.6827
            
        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        # Dirty way of ensuring overrides happen when requested
        l = locals()
        explicit = []
        for k in l.keys():
            if l[k] is not None:
                explicit.append(k)
                if k.endswith("s"):
                    explicit.append(k[:-1])
        self._init_params()

        num_chains = len(self.chains)

        assert rainbow is None or colors is None, \
            "You cannot both ask for rainbow colours and then give explicit colours"

        # Determine statistics
        assert statistics is not None, "statistics should be a string or list of strings!"
        if isinstance(statistics, str):
            assert statistics in list(Analysis.summaries), "statistics %s not recognised. Should be in %s" % (statistics, Analysis.summaries)
            statistics = [statistics.lower()] * len(self.chains)
        elif isinstance(statistics, list):
            for i, l in enumerate(statistics):
                statistics[i] = l.lower()
        else:
            raise ValueError("statistics is not a string or a list!")

        # Determine KDEs
        if isinstance(kde, bool) or isinstance(kde, float):
            kde = [False if c.grid else kde for c in self.chains]

        kde_override = [c.kde for c in self.chains]
        kde = [c2 if c2 is not None else c1 for c1, c2 in zip(kde, kde_override)]

        # Determine bins
        if bins is None:
            bins = get_bins(self.chains)
        elif isinstance(bins, list):
            bins = [b2 if isinstance(b2, int) else np.floor(b2 * b1) for b1, b2 in zip(get_bins(self.chains), bins)]
        elif isinstance(bins, float):
            bins = [np.floor(b * bins) for b in get_bins(self.chains)]
        elif isinstance(bins, int):
            bins = [bins] * len(self.chains)
        else:
            raise ValueError("bins value is not a recognised class (float or int)")

        # Determine smoothing
        if smooth is None:
            smooth = [0 if c.grid or k else 3 for c, k in zip(self.chains, kde)]
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
                color_params = [
                    color_params if color_params in cs.parameters + ["log_weights", "weights", "posterior"] else None
                    for cs in self.chains]
                color_params = [None if c == "posterior" and self.chains[i].posterior is None else c for i, c in
                                enumerate(color_params)]
            elif isinstance(color_params, list) or isinstance(color_params, tuple):
                for c, chain in zip(color_params, self.chains):
                    p = chain.parameters
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
                colors = self.color_finder.get_colormap(num_chains)
            else:
                if num_chains > len(self._all_colours):
                    num_needed_colours = np.sum([c is None for c in color_params])
                    colour_list = self.color_finder.get_colormap(num_needed_colours)
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
            colors = [colors] * len(self.chains)
        colors = self.color_finder.get_formatted(colors)

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
            linestyles = [linestyles] * len(self.chains)

        # Determine linewidths
        if linewidths is None:
            linewidths = [1.0] * len(self.chains)
        elif isinstance(linewidths, float) or isinstance(linewidths, int):
            linewidths = [linewidths] * len(self.chains)

        # Determine clouds
        if cloud is None:
            cloud = False
        cloud = [cloud or c is not None for c in color_params]

        # Determine cloud points
        if num_cloud is None:
            num_cloud = 30000
        if isinstance(num_cloud, int) or isinstance(num_cloud, float):
            num_cloud = [int(num_cloud)] * num_chains

        # Should we shade the contours
        if shade is None:
            if shade_alpha is None:
                shade = num_chains <= 3
            else:
                shade = True
        if isinstance(shade, bool):
            # If not overridden, do not shade chains with colour scatter points
            shade = [shade and c is None for c in color_params]

        # Modify shade alpha based on how many chains we have
        if shade_alpha is None:
            if num_chains == 1:
                if contour_labels is not None:
                    shade_alpha = 0.75
                else:
                    shade_alpha = 1.0
            else:
                shade_alpha = 1.0 / np.sqrt(num_chains)
        # Decrease the shading amount if there are colour scatter points
        if isinstance(shade_alpha, float) or isinstance(shade_alpha, int):
            shade_alpha = [shade_alpha if c is None else 0.25 * shade_alpha for c in color_params]

        if shade_gradient is None:
            shade_gradient = 1.0
        if isinstance(shade_gradient, float):
            shade_gradient = [shade_gradient] * num_chains
        elif isinstance(shade_gradient, list):
            assert len(shade_gradient) == num_chains, \
                "Have %d shade_gradient but % chains" % (len(shade_gradient), num_chains)

        contour_over_points = num_chains < 20

        if plot_contour is None:
            plot_contour = [contour_over_points if chain.posterior is not None else True for chain in self.chains]
        elif isinstance(plot_contour, bool):
            plot_contour = [plot_contour] * num_chains

        if plot_point is None:
            plot_point = [not contour_over_points] * num_chains
        elif isinstance(plot_point, bool):
            plot_point = [plot_point] * num_chains

        if marker_style is None:
            marker_style = ['.'] * num_chains
        elif isinstance(marker_style, str):
            marker_style = [marker_style] * num_chains

        if marker_size is None:
            marker_size = [20] * num_chains
        elif isinstance(marker_style, (int, float)):
            marker_size = [marker_size] * num_chains

        if marker_alpha is None:
            marker_alpha = [1.0] * num_chains
        elif isinstance(marker_alpha, (int, float)):
            marker_alpha = [marker_alpha] * num_chains

        # Figure out if we should display parameter summaries
        if summary is not None:
            summary = summary and num_chains == 1

        # Figure out bar shading
        if bar_shade is None:
            bar_shade = num_chains <= 3
        if isinstance(bar_shade, bool):
            bar_shade = [bar_shade] * num_chains

        # Figure out how many sigmas to plot
        if sigmas is None:
            if num_chains == 1:
                sigmas = np.array([0, 1, 2])
            else:
                sigmas = np.array([0, 1, 2])
        if sigmas[0] != 0:
            sigmas = np.concatenate(([0], sigmas))
        sigmas = np.sort(sigmas)

        if contour_labels is not None:
            assert isinstance(contour_labels, str), "contour_labels parameter should be a string"
            contour_labels = contour_labels.lower()
            assert contour_labels in ["sigma", "confidence"], "contour_labels should be either sigma or confidence"
        assert isinstance(contour_label_font_size, int) or isinstance(contour_label_font_size, float), \
            "contour_label_font_size needs to be numeric"

        if legend_artists is None:
            legend_artists = len(set(linestyles)) > 1 or len(set(linewidths)) > 1

        if legend_kwargs is not None:
            assert isinstance(legend_kwargs, dict), "legend_kwargs should be a dict"
        else:
            legend_kwargs = {}

        if num_chains < 3:
            labelspacing = 0.5
        elif num_chains == 3:
            labelspacing = 0.2
        else:
            labelspacing = 0.15
        legend_kwargs_default = {
            "labelspacing": labelspacing,
            "loc": "upper right",
            "frameon": False,
            "fontsize": label_font_size,
            "handlelength": 1,
            "handletextpad": 0.2,
            "borderaxespad": 0.0
        }
        legend_kwargs_default.update(legend_kwargs)

        watermark_text_kwargs_default = {
            "color": "#333333",
            "alpha": 0.7,
            "verticalalignment": "center",
            "horizontalalignment": "center"
        }
        if watermark_text_kwargs is None:
            watermark_text_kwargs = {}
        watermark_text_kwargs_default.update(watermark_text_kwargs)

        assert isinstance(summary_area, float), "summary_area needs to be a float, not %s!" % type(summary_area)
        assert summary_area > 0, "summary_area should be a positive number, instead is %s!" % summary_area
        assert summary_area < 1, "summary_area must be less than unity, instead is %s!" % summary_area
        assert isinstance(global_point, bool), "global_point should be a bool"

        # List options
        for i, c in enumerate(self.chains):
            try:
                c.update_unset_config("statistics", statistics[i], override=explicit)
                c.update_unset_config("color", colors[i], override=explicit)
                c.update_unset_config("linestyle", linestyles[i], override=explicit)
                c.update_unset_config("linewidth", linewidths[i], override=explicit)
                c.update_unset_config("cloud", cloud[i], override=explicit)
                c.update_unset_config("shade", shade[i], override=explicit)
                c.update_unset_config("shade_alpha", shade_alpha[i], override=explicit)
                c.update_unset_config("shade_gradient", shade_gradient[i], override=explicit)
                c.update_unset_config("bar_shade", bar_shade[i], override=explicit)
                c.update_unset_config("bins", bins[i], override=explicit)
                c.update_unset_config("kde", kde[i], override=explicit)
                c.update_unset_config("smooth", smooth[i], override=explicit)
                c.update_unset_config("color_params", color_params[i], override=explicit)
                c.update_unset_config("plot_color_params", plot_color_params[i], override=explicit)
                c.update_unset_config("cmap", cmaps[i], override=explicit)
                c.update_unset_config("num_cloud", num_cloud[i], override=explicit)
                c.update_unset_config("marker_style", marker_style[i], override=explicit)
                c.update_unset_config("marker_size", marker_size[i], override=explicit)
                c.update_unset_config("marker_alpha", marker_alpha[i], override=explicit)
                c.update_unset_config("plot_contour", plot_contour[i], override=explicit)
                c.update_unset_config("plot_point", plot_point[i], override=explicit)
                c.config["summary_area"] = summary_area

            except IndentationError as e:
                print("Index error when assigning chain properties, make sure you "
                      "have enough properties set for the number of chains you have loaded! "
                      "See the stack trace for which config item has the wrong number of entries.")
                raise e

        # Non list options
        self.config["sigma2d"] = sigma2d
        self.config["sigmas"] = sigmas
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
        self.config["contour_labels"] = contour_labels
        self.config["contour_label_font_size"] = contour_label_font_size
        self.config["legend_location"] = legend_location
        self.config["legend_kwargs"] = legend_kwargs_default
        self.config["legend_artists"] = legend_artists
        self.config["legend_color_text"] = legend_color_text
        self.config["watermark_text_kwargs"] = watermark_text_kwargs_default
        self.config["global_point"] = global_point

        self._configured = True
        return self

    def configure_truth(self, **kwargs):  # pragma: no cover
        r""" Configure the arguments passed to the ``axvline`` and ``axhline``
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
        r"""
        Returns a ChainConsumer instance containing all the walks of a given chain
        as individual chains themselves.

        This method might be useful if, for example, your chain was made using
        MCMC with 4 walkers. To check the sampling of all 4 walkers agree, you could
        call this to get a ChainConsumer instance with one chain for ech of the
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
        indexes = self._get_chain(chain)
        con = ChainConsumer()

        for index in indexes:
            chain = self.chains[index]
            assert chain.walkers is not None, "The chain you have selected was not added with any walkers!"
            num_walkers = chain.walkers
            data = np.split(chain.chain, num_walkers)
            ws = np.split(chain.weights, num_walkers)
            for j, (c, w) in enumerate(zip(data, ws)):
                con.add_chain(c, weights=w, name="Chain %d" % j, parameters=chain.parameters)
        return con

    def _get_chain(self, chain):
        if isinstance(chain, Chain):
            return [self.chains.index(chain)]
        if isinstance(chain, str):
            names = [c.name for c in self.chains]
            assert chain in names, "Chain %s not found!" % chain
            index = [i for i, n in enumerate(names) if chain == n]
        elif isinstance(chain, int):
            assert chain < len(self.chains), "Chain index %d not found!" % chain
            index = [chain]
        else:
            raise ValueError("Type %s not recognised for chain" % type(chain))
        return index

    def _get_chain_name(self, index):
        return self.chains[index].name

    def _all_names(self):
        return [c.name for c in self.chains]

    # Deprecated methods
    def plot(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.plotter.plot instead")
        return self.plotter.plot(*args, **kwargs)

    def plot_walks(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.plotter.plot_walks instead")
        return self.plotter.plot_walks(*args, **kwargs)

    def get_latex_table(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_latex_table instead")
        return self.analysis.get_latex_table(*args, **kwargs)

    def get_parameter_text(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_parameter_text instead")
        return self.analysis.get_parameter_text(*args, **kwargs)

    def get_summary(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_summary instead")
        return self.analysis.get_summary(*args, **kwargs)

    def get_correlations(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_correlations instead")
        return self.analysis.get_correlations(*args, **kwargs)

    def get_correlation_table(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_correlation_table instead")
        return self.analysis.get_correlation_table(*args, **kwargs)

    def get_covariance(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_covariance instead")
        return self.analysis.get_covariance(*args, **kwargs)

    def get_covariance_table(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.analysis.get_covariance_table instead")
        return self.analysis.get_covariance_table(*args, **kwargs)

    def diagnostic_gelman_rubin(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.diagnostic.gelman_rubin instead")
        return self.diagnostic.gelman_rubin(*args, **kwargs)

    def diagnostic_geweke(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.diagnostic.geweke instead")
        return self.diagnostic.geweke(*args, **kwargs)

    def comparison_aic(self):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.comparison.aic instead")
        return self.comparison.aic()

    def comparison_bic(self):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.comparison.bic instead")
        return self.comparison.bic()

    def comparison_dic(self):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.comparison.dic instead")
        return self.comparison.dic()

    def comparison_table(self, *args, **kwargs):  # pragma: no cover
        print("This method is deprecated. Please use chainConsumer.comparison.comparison_table instead")
        return self.comparison.comparison_table(*args, **kwargs)
