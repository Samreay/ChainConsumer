import numpy as np
import pandas as pd

from .analysis import Analysis
from .chain import Chain, ChainConfig, ChainName, ColumnName
from .colors import ColorInput, colors
from .comparisons import Comparison
from .diagnostic import Diagnostic
from .helpers import get_bins
from .plotter import PlotConfig, Plotter
from .truth import Truth

__all__ = ["ChainConsumer"]


class ChainConsumer:
    """A class for consuming chains produced by an MCMC walk. Or grid searches. To make plots,
    figures, tables, diagnostics, you name it."""

    def __init__(self):
        self.chains: dict[ChainName, Chain] = {}
        self.truths: list[Truth] = []
        self.labels = {}
        self.global_chain_override: ChainConfig | None = None

        self.plotter = Plotter(self)
        self.diagnostic = Diagnostic(self)
        self.comparison = Comparison(self)
        self.analysis = Analysis(self)

    @property
    def all_columns(self) -> list[str]:
        return list(set([c for chain in self.chains.values() for c in chain.samples.columns]))

    def get_label(self, str: ColumnName) -> str:
        return self.labels.get(str, str)

    def set_labels(self, labels: dict[str, str]) -> "ChainConsumer":
        """Set the labels for the chains.

        Args:
            labels (dict[str, str]): A dictionary mapping column names to labels.

        Returns:
            ChainConsumer: Itself, to allow chaining calls.
        """
        self.labels = labels
        return self

    def add_chain(self, chain: Chain):
        """Add a chain to ChainConsumer.

        Args:
            chain (Chain): The chain to add.

        Returns:
            ChainConsumer: Itself, to allow chaining calls.
        """
        key = chain.name
        assert key not in self.chains, f"Chain with name {key} already exists!"
        self.chains[key] = chain
        return self

    def set_plot_config(self, plot_config: PlotConfig) -> "ChainConsumer":
        """Set the plot config for ChainConsumer.

        Args:
            plot_config (PlotConfig): The plot config to use.

        Returns:
            ChainConsumer: Itself, to allow chaining calls.
        """
        self.plotter.set_config(plot_config)
        return self

    def add_marker(
        self,
        location: np.ndarray,
        columns: list[str],
        name: str,
        color: ColorInput | None = None,
        marker_size: float = 20.0,
        marker_style: str = ".",
        marker_alpha: float = 1.0,
    ):
        r"""Add a marker to the plot at the given location.

        Args:
            location (np.ndarray): The location of the marker.
            columns (list[str]): The names of the columns in the chain that correspond to the location.
            name (str): The name of the marker.
            color (ColourInput, optional): The colour of the marker. Defaults to None.
            marker_size (float, optional): The size of the marker. Defaults to 20.0.
            marker_style (str, optional): The style of the marker. Defaults to ".".
            marker_alpha (float, optional): The alpha of the marker. Defaults to 1.0.


        Returns:
            ChainConsumer: Itself, to allow chaining calls.
        """
        assert len(location.shape) == 1, "Location should be a 1D array"
        assert len(location) == len(columns), "Location and columns should be the same length"

        samples = pd.DataFrame(np.atleast_2d(location), columns=columns)
        samples["weight"] = 1.0
        samples["posterior"] = 1.0
        chain = Chain(
            samples=samples,
            name=name,
            color=color,  # type: ignore # ignoring the None override as this means we figure out colour later
            marker_size=marker_size,
            marker_style=marker_style,
            marker_alpha=marker_alpha,
            plot_contour=False,
            plot_point=True,
        )
        self.add_chain(chain)
        return self

    def remove_chain(self, remove: str | Chain) -> "ChainConsumer":
        r"""Removes a chain from ChainConsumer.

        Args:
            remove (str|Chain): The name of the chain to remove, or the chain itself.

        Returns:
            ChainConsumer: Itself, to allow chaining calls.
        """
        if isinstance(remove, Chain):
            remove = remove.name

        assert remove in self.chains, f"Chain with name {remove} does not exist!"
        self.chains.pop(remove)
        return self

    def add_override(
        self,
        override: ChainConfig,
    ) -> "ChainConsumer":
        """Apply a custom override config

        Args:
            override (Config, optional): The override config. Defaults to None.

        Returns:
            ChainConsumer: Itself, to allow chaining calls.
        """
        self.global_chain_override = override
        return self

    def _get_final_chains(self) -> dict[ChainName, Chain]:
        # Copy the original chain list
        final_chains = {k: v.model_copy() for k, v in self.chains.items()}
        chain_list = list(final_chains.values())
        num_chains = len(self.chains)

        # Note we only have to override things without a default
        # and things which should change as the number of chains change
        global_config = {}
        global_config["bar_shade"] = num_chains < 5
        global_config["sigmas"] = [0, 1, 2]
        global_config["shade"] = num_chains < 5
        global_config["shade_alpha"] = 1.0 / np.sqrt(num_chains)

        for _, chain in final_chains.items():
            # copy global config into local config
            local_config = global_config.copy()

            if isinstance(chain.bins, float):
                chain.bins = int(chain.bins * get_bins(chain))

            # Reduce shade alpha if we're showing contour labels
            if chain.show_contour_labels:
                local_config["shade_alpha"] *= 0.5

            # Check to see if the color is set
            if chain.color is None:
                local_config["color"] = next(colors.next_colour())

            chain.apply_if_none(**local_config)

            # Apply user overrides
            if self.global_chain_override is not None:
                chain.apply(**self.global_chain_override.model_dump())

        return final_chains

    # def configure_overrides(
    #     self,
    #     statistics="max",
    #     max_ticks=5,
    #     plot_hists=True,
    #     flip=True,
    #     serif=False,
    #     sigma2d=False,
    #     sigmas=None,
    #     summary=None,
    #     bins=None,
    #     cmap=None,
    #     colors=None,
    #     linestyles=None,
    #     linewidths=None,
    #     kde=False,
    #     smooth=None,
    #     cloud=None,
    #     shade=None,
    #     shade_alpha=None,
    #     shade_gradient=None,
    #     bar_shade=None,
    #     num_cloud=None,
    #     color_params=None,
    #     plot_color_params=False,
    #     cmaps=None,
    #     plot_contour=None,
    #     plot_point=None,
    #     show_as_1d_prior=None,
    #     global_point=True,
    #     marker_style=None,
    #     marker_size=None,
    #     marker_alpha=None,
    #     usetex=False,
    #     diagonal_tick_labels=True,
    #     label_font_size=12,
    #     tick_font_size=10,
    #     spacing=None,
    #     contour_labels=None,
    #     contour_label_font_size=10,
    #     legend_kwargs=None,
    #     legend_location=None,
    #     legend_artists=None,
    #     legend_color_text=True,
    #     watermark_text_kwargs=None,
    #     summary_area=0.6827,
    #     zorder=None,
    # ):  # pragma: no cover
    #     r"""Configure the general plotting parameters common across the bar
    #     and contour plots.

    #     If you do not call this explicitly, the :func:`plot`
    #     method will invoke this method automatically.

    #     Please ensure that you call this method *after* adding all the relevant data to the
    #     chain consumer, as the consume changes configuration values depending on
    #     the supplied data.

    #     Parameters
    #     ----------
    #     statistics : string|list[str], optional
    #         Which sort of statistics to use. Defaults to `"max"` for maximum likelihood
    #         statistics. Other available options are `"mean"`, `"cumulative"`, `"max_symmetric"`,
    #         `"max_closest"` and `"max_central"`. In the
    #         very, very rare case you want to enable different statistics for different
    #         chains, you can pass in a list of strings.
    #     max_ticks : int, optional
    #         The maximum number of ticks to use on the plots
    #     plot_hists : bool, optional
    #         Whether to plot marginalised distributions or not
    #     flip : bool, optional
    #         Set to false if, when plotting only two parameters, you do not want it to
    #         rotate the histogram so that it is horizontal.
    #     sigma2d: bool, optional
    #         Defaults to `False`. When `False`, uses :math:`\sigma` levels for 1D Gaussians - ie confidence
    #         levels of 68% and 95%. When `True`, uses the confidence levels for 2D Gaussians, where 1 and 2
    #         :math:`\sigma` represents 39% and 86% confidence levels respectively.
    #     sigmas : np.array, optional
    #         The :math:`\sigma` contour levels to plot. Defaults to [0, 1, 2, 3] for a single chain
    #         and [0, 1, 2] for multiple chains.
    #     serif : bool, optional
    #         Whether to display ticks and labels with serif font.
    #     summary : bool, optional
    #         If overridden, sets whether parameter summaries should be set as axis titles.
    #         Will not work if you have multiple chains
    #     bins : int|float,list[int|float], optional
    #         The number of bins to use. By default uses :math:`\frac{\sqrt{n}}{10}`, where
    #         :math:`n` are the number of data points. Giving an integer will set the number
    #         of bins to the given value. Giving a float will scale the number of bins, such
    #         that giving ``bins=1.5`` will result in using :math:`\frac{1.5\sqrt{n}}{10}` bins.
    #         Note this parameter is most useful if `kde=False` is also passed, so you
    #         can actually see the bins and not a KDE.
    #     cmap : str, optional
    #         Set to the matplotlib colour map you want to use to overwrite the default colours.
    #         Note that this parameter overwrites colours. The `cmaps` parameters is different,
    #         and used when you ask for an extra dimension to be used to colour scatter points.
    #         See the online examples to see the difference.
    #     colors : str(hex)|list[str(hex)], optional
    #         Provide a list of colours to use for each chain. If you provide more chains
    #         than colours, you *will* get the rainbow colour spectrum. If you only pass
    #         one colour, all chains are set to this colour. This probably won't look good.
    #     linestyles : str|list[str], optional
    #         Provide a list of line styles to plot the contours and marginalised
    #         distributions with. By default, this will become a list of solid lines. If a
    #         string is passed instead of a list, this style is used for all chains.
    #     linewidths : float|list[float], optional
    #         Provide a list of line widths to plot the contours and marginalised
    #         distributions with. By default, this is a width of 1. If a float
    #         is passed instead of a list, this width is used for all chains.
    #     kde : bool|float|list[bool|float], optional
    #         Whether to use a Gaussian KDE to smooth marginalised posteriors. If false, uses
    #         bins and linear interpolation, so ensure you have plenty of samples if your
    #         distribution is highly non-gaussian. Due to the slowness of performing a
    #         KDE on all data, it is often useful to disable this before producing final
    #         plots. If float, scales the width of the KDE bandpass manually.
    #     smooth : int|list[int], optional
    #         Defaults to 3. How much to smooth the marginalised distributions using a gaussian filter.
    #         If ``kde`` is set to true, this parameter is ignored. Setting it to either
    #         ``0``, ``False`` disables smoothing. For grid data, smoothing
    #         is set to 0 by default, not 3.
    #     cloud : bool|list[bool], optional
    #         If set, overrides the default behaviour and plots the cloud or not
    #     shade : bool|list[bool] optional
    #         If set, overrides the default behaviour and plots filled contours or not. If a list of
    #         bools is passed, you can turn shading on or off for specific chains.
    #     shade_alpha : float|list[float], optional
    #         Filled contour alpha value override. Default is 1.0. If a list is passed, you can set the
    #         shade opacity for specific chains.
    #     shade_gradient : float|list[float], optional
    #         How much to vary colours in different contour levels.
    #     bar_shade : bool|list[bool], optional
    #         If set to true, shades in confidence regions in under histogram. By default
    #         this happens if you less than 3 chains, but is disabled if you are comparing
    #         more chains. You can pass a list if you wish to shade some chains but not others.
    #     num_cloud : int|list[int], optional
    #         The number of scatter points to show when enabling `cloud` or setting one of the parameters
    #         to colour scatter. Defaults to 15k per chain.
    #     color_params : str|list[str], optional
    #         The name of the parameter to use for the colour scatter. Defaults to none, for no colour. If set
    #         to 'weights', 'log_weights', or 'posterior' (without the quotes), and that is not a parameter in the chain,
    #         it will respectively  use the weights, log weights, or posterior, to colour the points.
    #     plot_color_params : bool|list[bool], optional
    #         Whether or not the colour parameter should also be plotted as a posterior surface.
    #     cmaps : str|list[str], optional
    #         The matplotlib colourmap to use in the `colour_param`. If you have multiple `color_param`s, you can
    #         specific a different cmap for each variable. By default ChainConsumer will cycle between several
    #         cmaps.
    #     plot_contour : bool|list[bool], optional
    #         Whether to plot the whole contour (as opposed to a point). Defaults to true for less than
    #         25 concurrent chains.
    #     plot_point : bool|list[bool], optional
    #         Whether to plot a maximum likelihood point. Defaults to true for more then 24 chains.
    #     show_as_1d_prior : bool|list[bool], optional
    #         Showing as a 1D prior will show the 1D histograms, but won't plot the 2D contours.
    #     global_point : bool, optional
    #         Whether the point which gets plotted is the global posterior maximum, or the marginalised 2D
    #         posterior maximum. Note that when you use marginalised 2D maximums for the points, you do not
    #          get the 1D histograms. Defaults to `True`, for a global maximum value.
    #     marker_style : str|list[str], optional
    #         The marker style to use when plotting points. Defaults to `'.'`
    #     marker_size : numeric|list[numeric], optional
    #         Size of markers, if plotted. Defaults to `20`.
    #     marker_alpha : numeric|list[numeric], optional
    #         The alpha values when plotting markers.
    #     usetex : bool, optional
    #         Whether or not to parse text as LaTeX in plots.
    #     diagonal_tick_labels : bool, optional
    #         Whether to display tick labels on a 45 degree angle.
    #     label_font_size : int|float, optional
    #         The font size for plot axis labels and axis titles if summaries are configured to display.
    #     tick_font_size : int|float, optional
    #         The font size for the tick labels in the plots.
    #     spacing : float, optional
    #         The amount of spacing to add between plots. Defaults to `None`, which equates to 1.0 for less
    #         than 6 dimensions and 0.0 for higher dimensions.
    #     contour_labels : string, optional
    #         If unset do not plot contour labels. If set to "confidence", label the using confidence
    #         intervals. If set to "sigma", labels using sigma.
    #     contour_label_font_size : int|float, optional
    #         The font size for contour labels, if they are enabled.
    #     legend_kwargs : dict, optional
    #         Extra arguments to pass to the legend api.
    #     legend_location : tuple(int,int), optional
    #         Specifies the subplot in which to locate the legend. By default, this will be (0, -1),
    #         corresponding to the top right subplot if there are more than two parameters,
    #         and the bottom left plot for only two parameters with flip on.
    #         For having the legend in the primary subplot
    #         in the bottom left, set to (-1,0).
    #     legend_artists : bool, optional
    #         Whether to include hide artists in the legend. If all linestyles and line widths are identical,
    #         this will default to false (as only the colours change). Otherwise it will be true.
    #     legend_color_text : bool, optional
    #         Whether to colour the legend text.
    #     watermark_text_kwargs : dict, optional
    #         Options to pass to the fontdict property when generating text for the watermark.
    #     summary_area : float, optional
    #         The confidence interval used when generating parameter summaries. Defaults to 1 sigma, aka 0.6827
    #     zorder : int, optional
    #         The zorder to pass to `matplotlib` to determine visual ordering when plotting.

    #     Returns
    #     -------
    #     ChainConsumer
    #         Itself, to allow chaining calls.
    #     """
    #     # Warn the user if configure has been invoked multiple times
    #     self._num_configure_calls += 1
    #     if self._num_configure_calls > 1:
    #         logger.warning(
    #             "Configure has been called %d times - this is not good - it should be once!" % self._num_configure_calls
    #         )
    #         logger.warning("To avoid this, load your chains in first, then call analysis/plotting methods")

    #     # Dirty way of ensuring overrides happen when requested
    #     l = locals()
    #     explicit = []
    #     for k in l:
    #         if l[k] is not None:
    #             explicit.append(k)
    #             if k.endswith("s"):
    #                 explicit.append(k[:-1])
    #     self._init_params()

    #     num_chains = len(self.chains)

    #     assert cmap is None or colors is None, "You cannot both ask for cmap colours and then give explicit colours"

    #     # Determine statistics
    #     assert statistics is not None, "statistics should be a string or list of strings!"
    #     if isinstance(statistics, str):
    #         assert statistics in list(Analysis.summaries), "statistics {} not recognised. Should be in {}".format(
    #             statistics,
    #             Analysis.summaries,
    #         )
    #         statistics = [statistics.lower()] * len(self.chains)
    #     elif isinstance(statistics, list):
    #         for i, l in enumerate(statistics):
    #             statistics[i] = l.lower()
    #     else:
    #         raise ValueError("statistics is not a string or a list!")

    #     # Determine KDEs
    #     if isinstance(kde, bool | float):
    #         kde = [False if c.grid else kde for c in self.chains]

    #     kde_override = [c.kde for c in self.chains]
    #     kde = [c2 if c2 is not None else c1 for c1, c2 in zip(kde, kde_override)]

    #     # Determine bins
    #     if bins is None:
    #         bins = get_bins(self.chains)
    #     elif isinstance(bins, list):
    #         bins = [b2 if isinstance(b2, int) else np.floor(b2 * b1) for b1, b2 in zip(get_bins(self.chains), bins)]
    #     elif isinstance(bins, float):
    #         bins = [np.floor(b * bins) for b in get_bins(self.chains)]
    #     elif isinstance(bins, int):
    #         bins = [bins] * len(self.chains)
    #     else:
    #         raise ValueError("bins value is not a recognised class (float or int)")

    #     # Determine smoothing
    #     if smooth is None:
    #         smooth = [0 if c.grid or k else 3 for c, k in zip(self.chains, kde)]
    #     else:
    #         if smooth is not None and not smooth:
    #             smooth = 0
    #         if isinstance(smooth, list):
    #             smooth = [0 if k else s for s, k in zip(smooth, kde)]
    #         else:
    #             smooth = [0 if k else smooth for k in kde]

    #     # Determine color parameters
    #     if color_params is None:
    #         color_params = [None] * num_chains
    #     else:
    #         if isinstance(color_params, str):
    #             color_params = [
    #                 color_params if color_params in [*cs.parameters, "log_weights", "weights", "posterior"] else None
    #                 for cs in self.chains
    #             ]
    #             color_params = [
    #                 None if c == "posterior" and self.chains[i].posterior is None else c
    #                 for i, c in enumerate(color_params)
    #             ]
    #         elif isinstance(color_params, list | tuple):
    #             for c, chain in zip(color_params, self.chains):
    #                 p = chain.parameters
    #                 if c is not None:
    #                     assert c in p, f"Color parameter {c} not in parameters {p}"
    #     # Determine if we should plot color parameters
    #     if isinstance(plot_color_params, bool):
    #         plot_color_params = [plot_color_params] * len(color_params)

    #     # Determine cmaps
    #     if cmaps is None:
    #         param_cmaps = {}
    #         cmaps = []
    #         i = 0
    #         for cp in color_params:
    #             if cp is None:
    #                 cmaps.append(None)
    #             elif cp in param_cmaps:
    #                 cmaps.append(param_cmaps[cp])
    #             else:
    #                 param_cmaps[cp] = self._cmaps[i]
    #                 cmaps.append(self._cmaps[i])
    #                 i = (i + 1) % len(self._cmaps)

    #     # Determine colours
    #     if colors is None:
    #         if cmap:
    #             colors = colors.get_colormap(num_chains, cmap)
    #         else:
    #             if num_chains > len(self._all_colours):
    #                 num_needed_colours = np.sum([c is None for c in color_params])
    #                 colour_list = colors.get_colormap(num_needed_colours, "inferno")
    #             else:
    #                 colour_list = self._all_colours
    #             colors = []
    #             ci = 0
    #             for c in color_params:
    #                 if c:
    #                     colors.append("#000000")
    #                 else:
    #                     colors.append(colour_list[ci])
    #                     ci += 1
    #     elif isinstance(colors, str):
    #         colors = [colors] * len(self.chains)
    #     colors = colors.get_formatted(colors)

    #     # Determine linestyles
    #     if linestyles is None:
    #         i = 0
    #         linestyles = []
    #         for c in color_params:
    #             if c is None:
    #                 linestyles.append(self._linestyles[0])
    #             else:
    #                 linestyles.append(self._linestyles[i])
    #                 i = (i + 1) % len(self._linestyles)
    #     elif isinstance(linestyles, str):
    #         linestyles = [linestyles] * len(self.chains)

    #     # Determine linewidths
    #     if linewidths is None:
    #         linewidths = [1.0] * len(self.chains)
    #     elif isinstance(linewidths, float | int):
    #         linewidths = [linewidths] * len(self.chains)

    #     # Determine clouds
    #     if cloud is None:
    #         cloud = False
    #     cloud = [cloud or c is not None for c in color_params]

    #     # Determine cloud points
    #     if num_cloud is None:
    #         num_cloud = 30000
    #     if isinstance(num_cloud, float | int):
    #         num_cloud = [int(num_cloud)] * num_chains

    #     # Should we shade the contours
    #     if shade is None:
    #         shade = num_chains <= 3 if shade_alpha is None else True
    #     if isinstance(shade, bool):
    #         # If not overridden, do not shade chains with colour scatter points
    #         shade = [shade and c is None for c in color_params]

    #     # Modify shade alpha based on how many chains we have
    #     if shade_alpha is None:
    #         if num_chains == 1:
    #             shade_alpha = 0.75 if contour_labels is not None else 1.0
    #         else:
    #             shade_alpha = 1.0 / np.sqrt(num_chains)
    #     # Decrease the shading amount if there are colour scatter points
    #     if isinstance(shade_alpha, float | int):
    #         shade_alpha = [shade_alpha if c is None else 0.25 * shade_alpha for c in color_params]

    #     if shade_gradient is None:
    #         shade_gradient = 1.0
    #     if isinstance(shade_gradient, float):
    #         shade_gradient = [shade_gradient] * num_chains
    #     elif isinstance(shade_gradient, list):
    #         assert len(shade_gradient) == num_chains, "Have %d shade_gradient but % chains" % (
    #             len(shade_gradient),
    #             num_chains,
    #         )

    #     contour_over_points = num_chains < 20

    #     if plot_contour is None:
    #         plot_contour = [contour_over_points if chain.log_posterior is not None else True for chain in self.chains]
    #     elif isinstance(plot_contour, bool):
    #         plot_contour = [plot_contour] * num_chains

    #     if plot_point is None:
    #         plot_point = [not contour_over_points] * num_chains
    #     elif isinstance(plot_point, bool):
    #         plot_point = [plot_point] * num_chains

    #     if show_as_1d_prior is None:
    #         show_as_1d_prior = [not contour_over_points] * num_chains
    #     elif isinstance(show_as_1d_prior, bool):
    #         show_as_1d_prior = [show_as_1d_prior] * num_chains

    #     if marker_style is None:
    #         marker_style = ["."] * num_chains
    #     elif isinstance(marker_style, str):
    #         marker_style = [marker_style] * num_chains

    #     if marker_size is None:
    #         marker_size = [20] * num_chains
    #     elif isinstance(marker_style, int | float):
    #         marker_size = [marker_size] * num_chains

    #     if marker_alpha is None:
    #         marker_alpha = [1.0] * num_chains
    #     elif isinstance(marker_alpha, int | float):
    #         marker_alpha = [marker_alpha] * num_chains

    #     # Figure out if we should display parameter summaries
    #     if summary is not None:
    #         summary = summary and num_chains == 1

    #     # Figure out bar shading
    #     if bar_shade is None:
    #         bar_shade = num_chains <= 3
    #     if isinstance(bar_shade, bool):
    #         bar_shade = [bar_shade] * num_chains

    #     if zorder is None:
    #         zorder = [1] * num_chains

    #     # Figure out how many sigmas to plot
    #     if sigmas is None:
    #         sigmas = np.array([0, 1, 2]) if num_chains == 1 else np.array([0, 1, 2])
    #     if sigmas[0] != 0:
    #         sigmas = np.concatenate(([0], sigmas))
    #     sigmas = np.sort(sigmas)

    #     if contour_labels is not None:
    #         assert isinstance(contour_labels, str), "contour_labels parameter should be a string"
    #         contour_labels = contour_labels.lower()
    #         assert contour_labels in [
    #             "sigma",
    #             "confidence",
    #         ], "contour_labels should be either sigma or confidence"
    #     assert isinstance(contour_label_font_size, float | int), "contour_label_font_size needs to be numeric"

    #     if legend_artists is None:
    #         legend_artists = len(set(linestyles)) > 1 or len(set(linewidths)) > 1

    #     if legend_kwargs is not None:
    #         assert isinstance(legend_kwargs, dict), "legend_kwargs should be a dict"
    #     else:
    #         legend_kwargs = {}

    #     if num_chains < 3:
    #         labelspacing = 0.5
    #     elif num_chains == 3:
    #         labelspacing = 0.2
    #     else:
    #         labelspacing = 0.15
    #     legend_kwargs_default = {
    #         "labelspacing": labelspacing,
    #         "loc": "upper right",
    #         "frameon": False,
    #         "fontsize": label_font_size,
    #         "handlelength": 1,
    #         "handletextpad": 0.2,
    #         "borderaxespad": 0.0,
    #     }
    #     legend_kwargs_default.update(legend_kwargs)

    #     watermark_text_kwargs_default = {
    #         "color": "#333333",
    #         "alpha": 0.7,
    #         "verticalalignment": "center",
    #         "horizontalalignment": "center",
    #     }
    #     if watermark_text_kwargs is None:
    #         watermark_text_kwargs = {}
    #     watermark_text_kwargs_default.update(watermark_text_kwargs)

    #     assert isinstance(summary_area, float), "summary_area needs to be a float, not %s!" % type(summary_area)
    #     assert summary_area > 0, "summary_area should be a positive number, instead is %s!" % summary_area
    #     assert summary_area < 1, "summary_area must be less than unity, instead is %s!" % summary_area
    #     assert isinstance(global_point, bool), "global_point should be a bool"

    #     # List options
    #     for i, c in enumerate(self.chains):
    #         try:
    #             c.update_unset_config("statistics", statistics[i], override=explicit)
    #             c.update_unset_config("color", colors[i], override=explicit)
    #             c.update_unset_config("linestyle", linestyles[i], override=explicit)
    #             c.update_unset_config("linewidth", linewidths[i], override=explicit)
    #             c.update_unset_config("cloud", cloud[i], override=explicit)
    #             c.update_unset_config("shade", shade[i], override=explicit)
    #             c.update_unset_config("shade_alpha", shade_alpha[i], override=explicit)
    #             c.update_unset_config("shade_gradient", shade_gradient[i], override=explicit)
    #             c.update_unset_config("bar_shade", bar_shade[i], override=explicit)
    #             c.update_unset_config("bins", bins[i], override=explicit)
    #             c.update_unset_config("kde", kde[i], override=explicit)
    #             c.update_unset_config("smooth", smooth[i], override=explicit)
    #             c.update_unset_config("color_params", color_params[i], override=explicit)
    #             c.update_unset_config("plot_color_params", plot_color_params[i], override=explicit)
    #             c.update_unset_config("cmap", cmaps[i], override=explicit)
    #             c.update_unset_config("num_cloud", num_cloud[i], override=explicit)
    #             c.update_unset_config("marker_style", marker_style[i], override=explicit)
    #             c.update_unset_config("marker_size", marker_size[i], override=explicit)
    #             c.update_unset_config("marker_alpha", marker_alpha[i], override=explicit)
    #             c.update_unset_config("plot_contour", plot_contour[i], override=explicit)
    #             c.update_unset_config("plot_point", plot_point[i], override=explicit)
    #             c.update_unset_config("show_as_1d_prior", show_as_1d_prior[i], override=explicit)
    #             c.update_unset_config("zorder", zorder[i], override=explicit)
    #             c.config["summary_area"] = summary_area

    #         except IndentationError:
    #             print(
    #                 "Index error when assigning chain properties, make sure you "
    #                 "have enough properties set for the number of chains you have loaded! "
    #                 "See the stack trace for which config item has the wrong number of entries."
    #             )
    #             raise

    #     # Non list options
    #     self.config["sigma2d"] = sigma2d
    #     self.config["sigmas"] = sigmas
    #     self.config["summary"] = summary
    #     self.config["flip"] = flip
    #     self.config["serif"] = serif
    #     self.config["plot_hists"] = plot_hists
    #     self.config["max_ticks"] = max_ticks
    #     self.config["usetex"] = usetex
    #     self.config["diagonal_tick_labels"] = diagonal_tick_labels
    #     self.config["label_font_size"] = label_font_size
    #     self.config["tick_font_size"] = tick_font_size
    #     self.config["spacing"] = spacing
    #     self.config["contour_labels"] = contour_labels
    #     self.config["contour_label_font_size"] = contour_label_font_size
    #     self.config["legend_location"] = legend_location
    #     self.config["legend_kwargs"] = legend_kwargs_default
    #     self.config["legend_artists"] = legend_artists
    #     self.config["legend_color_text"] = legend_color_text
    #     self.config["watermark_text_kwargs"] = watermark_text_kwargs_default
    #     self.config["global_point"] = global_point

    #     self._configured = True
    #     return self

    def get_chain(self, name: str) -> Chain:
        assert name in self.chains, f"Chain with name {name} does not exist!"
        return self.chains[name]

    def get_names(self) -> list[str]:
        return list(self.chains.keys())
