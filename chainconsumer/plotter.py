import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.textpath import TextPath
from numpy import meshgrid
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from chainconsumer.helpers import get_extents, get_smoothed_bins, get_grid_bins
from chainconsumer.kde import MegKDE


class Plotter(object):
    def __init__(self, parent):
        self.parent = parent
        self._logger = logging.getLogger(__name__)

    def plot(self, figsize="GROW", parameters=None, chains=None, extents=None, filename=None,
             display=False, truth=None, legend=None, blind=None, watermark=None):  # pragma: no cover
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
        chains : int|str, list[str|int], optional
            Used to specify which chain to show if more than one chain is loaded in.
            Can be an integer, specifying the
            chain index, or a str, specifying the chain name.
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
        blind : bool|string|list[string], optional
            Whether to blind axes values. Can be set to `True` to blind all parameters,
            or can pass in a string (or list of strings) which specify the parameters to blind.
        watermark : str, optional
            A watermark to add to the figure

        Returns
        -------
        figure
            the matplotlib figure

        """

        chains, parameters, truth, extents, blind = self._sanitise(chains, parameters, truth,
                                                                   extents, color_p=True, blind=blind)
        names = [self.parent._names[i] for i in chains]

        if legend is None:
            legend = len(chains) > 1

        # If no chains have names, don't plot the legend
        legend = legend and len([n for n in names if n]) > 0

        # Calculate cmap extents
        color_params = self.parent.config["color_params"]
        unique_color_params = list(set(color_params))
        num_cax = len(unique_color_params)
        if None in unique_color_params:
            num_cax -= 1
        color_param_extents = {}
        for u in unique_color_params:
            umin, umax = np.inf, -np.inf
            for i, cp in enumerate(color_params):
                if i not in chains:
                    continue
                if cp is not None and u == cp:
                    try:
                        data = self.parent._chains[i][:, self.parent._parameters[i].index(cp)]
                    except ValueError:
                        if cp == "weights":
                            data = self.parent._weights[i]
                        elif cp == "log_weights":
                            data = np.log(self.parent._weights[i])
                        elif cp == "posterior":
                            data = self.parent._posteriors[i]
                    if data is not None:
                        umin = min(umin, data.min())
                        umax = max(umax, data.max())
            color_param_extents[u] = (umin, umax)

        grow_size = 1.5
        if isinstance(figsize, float):
            grow_size *= figsize
            figsize = "GROW"

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

        plot_hists = self.parent.config["plot_hists"]
        flip = (len(parameters) == 2 and plot_hists and self.parent.config["flip"])

        fig, axes, params1, params2, extents = self._get_figure(parameters, chains=chains, figsize=figsize, flip=flip,
                                                                external_extents=extents, blind=blind)
        axl = axes.ravel().tolist()
        summary = self.parent.config["summary"]
        fit_values = self.parent.analysis.get_summary(squeeze=False, parameters=parameters)

        if summary is None:
            summary = len(parameters) < 5 and len(self.parent._chains) == 1
        if len(chains) == 1:
            self._logger.debug("Plotting surfaces for chain of dimension %s" %
                               (self.parent._chains[chains[0]].shape,))
        else:
            self._logger.debug("Plotting surfaces for %d chains" % len(chains))
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
                            enumerate(zip(self.parent._chains, self.parent._weights, self.parent._parameters,
                                          fit_values, self.parent._grids)):
                        if ii not in chains:
                            continue
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        param_summary = summary and p1 not in blind
                        m = self._plot_bars(ii, ax, p1, chain[:, index], weights, grid=grid, fit_values=fit[p1],
                                            flip=do_flip, summary=param_summary, truth=truth)
                        if max_val is None or m > max_val:
                            max_val = m
                    if do_flip:
                        ax.set_xlim(0, 1.1 * max_val)
                    else:
                        ax.set_ylim(0, 1.1 * max_val)

                else:
                    for ii, (chain, parameters, fit, weights, grid, posterior) in \
                            enumerate(zip(self.parent._chains, self.parent._parameters, fit_values,
                                          self.parent._weights, self.parent._grids, self.parent._posteriors)):
                        if ii not in chains:
                            continue
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        color_data = None
                        extent = None
                        if color_params[ii] is not None:
                            try:
                                color_index = parameters.index(color_params[ii])
                                color_data = chain[:, color_index]
                            except ValueError:
                                if color_params[ii] == "weights":
                                    color_data = weights
                                elif color_params[ii] == "log_weights":
                                    color_data = np.log(weights)
                                elif color_params[ii] == "posterior":
                                    color_data = posterior
                            extent = color_param_extents[color_params[ii]]
                        h = self._plot_contour(ii, ax, chain[:, i2], chain[:, i1], weights, p1, p2,
                                               grid, truth=truth, color_data=color_data, color_extent=extent)
                        if h is not None and color_params[ii] not in cbar_done:
                            cbar_done.append(color_params[ii])
                            aspect = figsize[1] / 0.15
                            fraction = 0.85 / figsize[0]
                            cbar = fig.colorbar(h, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                            label = color_params[ii]
                            if label == "weights":
                                label = "Weights"
                            elif label == "log_weights":
                                label = "log(Weights)"
                            elif label == "posterior":
                                label = "log(Posterior)"
                            cbar.set_label(label, fontsize=14)
                            cbar.solids.set(alpha=1)

        colors = self.parent.config["colors"]
        linestyles = self.parent.config["linestyles"]
        linewidths = self.parent.config["linewidths"]
        legend_kwargs = self.parent.config["legend_kwargs"]
        legend_artists = self.parent.config["legend_artists"]
        legend_color_text = self.parent.config["legend_color_text"]
        legend_location = self.parent.config["legend_location"]
        if legend_location is None:
            if not flip or len(parameters) > 2:
                legend_location = (0, -1)
            else:
                legend_location = (-1, 0)
        outside = (legend_location[0] >= legend_location[1])
        if names is not None and legend:
            ax = axes[legend_location[0], legend_location[1]]
            if "markerfirst" not in legend_kwargs:
                # If we have legend inside a used subplot, switch marker order
                legend_kwargs["markerfirst"] = outside or not legend_artists
            linewidths2 = linewidths if legend_artists else [0]*len(linewidths)
            linestyles2 = linestyles if legend_artists else ["-"]*len(linestyles)

            artists = [plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw)
                       for i, (n, c, ls, lw) in enumerate(zip(self.parent._names, colors, linestyles2, linewidths2)) if n is not None and i in chains]
            leg = ax.legend(artists, names, **legend_kwargs)
            if legend_color_text:
                cs = [c for i, c in enumerate(colors) if i in chains]
                for text, c in zip(leg.get_texts(), cs):
                    text.set_weight("medium")
                    text.set_color(c)
            if not outside:
                loc = legend_kwargs.get("loc") or ""
                if "right" in loc.lower():
                    vp = leg._legend_box._children[-1]._children[0]
                    vp.align = "right"

        fig.canvas.draw()
        for ax in axes[-1, :]:
            offset = ax.get_xaxis().get_offset_text()
            ax.set_xlabel('{0} {1}'.format(ax.get_xlabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
            offset.set_visible(False)
        for ax in axes[:, 0]:
            offset = ax.get_yaxis().get_offset_text()
            ax.set_ylabel('{0} {1}'.format(ax.get_ylabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
            offset.set_visible(False)

        dpi = 300
        if watermark:
            if flip and len(parameters) == 2:
                ax = axes[-1, 0]
            else:
                ax = None
            self._add_watermark(fig, ax, figsize, watermark, dpi=dpi)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True, pad_inches=0.05)
        if display:
            plt.show()

        return fig

    def _add_watermark(self, fig, axes, figsize, text, dpi=300):  # pragma: no cover
        # Code based off github repository https://github.com/cpadavis/preliminize
        dx, dy = figsize
        dy, dx = dy * dpi, dx * dpi
        rotation = 180 / np.pi * np.arctan2(-dy, dx)
        fontdict = self.parent.config["watermark_text_kwargs"]
        if "usetex" in fontdict:
            usetex = fontdict["usetex"]
        else:
            usetex = self.parent.config["usetex"]
            fontdict["usetex"] = usetex
        if fontdict["usetex"]:
            px, py, scale = 0.5, 0.5, 1.0
        else:
            px, py, scale = 0.45, 0.55, 0.8
        bb0 = TextPath((0, 0), text, size=50, props=fontdict, usetex=usetex).get_extents()
        bb1 = TextPath((0, 0), text, size=51, props=fontdict, usetex=usetex).get_extents()
        dw = (bb1.width - bb0.width) * (dpi / 100)
        dh = (bb1.height - bb0.height) * (dpi / 100)
        size = np.sqrt(dy ** 2 + dx ** 2) / (dh * abs(dy / dx) + dw) * 0.6 * scale
        if axes is not None:
            if fontdict["usetex"]:
                size *= 0.7
            else:
                size *= 0.85
        fontdict['size'] = int(size)
        if axes is None:
            fig.text(px, py, text, fontdict=fontdict, rotation=rotation)
        else:
            axes.text(px, py, text, transform=axes.transAxes, fontdict=fontdict, rotation=rotation)

    def plot_walks(self, parameters=None, truth=None, extents=None, display=False,
                   filename=None, chains=None, convolve=None, figsize=None,
                   plot_weights=True, plot_posterior=True, log_weight=None):  # pragma: no cover
        """ Plots the chain walk; the parameter values as a function of step index.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains are well behaved, or if certain parameters are suspect
        or require a greater burn in period.

        The desired outcome is to see an unchanging distribution along the x-axis of the plot.
        If there are obvious tails or features in the parameters, you probably want
        to investigate.

        Parameters
        ----------
        parameters : list[str]|int, optional
            Specify a subset of parameters to plot. If not set, all parameters are plotted.
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

        chains, parameters, truth, extents, _ = self._sanitise(chains, parameters, truth, extents)

        n = len(parameters)
        extra = 0
        if plot_weights:
            plot_weights = plot_weights and np.any([np.any(self.parent._weights[c] != 1.0) for c in chains])

        plot_posterior = plot_posterior and np.any([self.parent._posteriors[c] is not None for c in chains])

        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            figsize = (8, 0.75 + (n + extra))

        colors = self.parent.config["colors"]

        fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)

        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = parameters[i - n]
                for index in chains:
                    if p in self.parent._parameters[index]:
                        chain_row = self.parent._chains[index][:, self.parent._parameters[index].index(p)]
                        self._plot_walk(ax, p, chain_row, truth=truth.get(p),
                                        extents=extents.get(p), convolve=convolve, color=colors[index])
                        truth[p] = None
            else:
                if i == 0 and plot_posterior:
                    for index in chains:
                        if self.parent._posteriors[index] is not None:
                            self._plot_walk(ax, "$\log(P)$", self.parent._posteriors[index] - self.parent._posteriors[index].max(),
                                            convolve=convolve, color=colors[index])
                else:
                    if log_weight is None:
                        log_weight = np.any([self.parent._weights[index].mean() < 0.1 for index in chains])
                    if log_weight:
                        for index in chains:
                            self._plot_walk(ax, r"$\log_{10}(w)$", np.log10(self.parent._weights[index]),
                                            convolve=convolve, color=colors[index])
                    else:
                        for index in chains:
                            self._plot_walk(ax, "$w$", self.parent._weights[index],
                                            convolve=convolve, color=colors[index])

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def plot_distributions(self, parameters=None, truth=None, extents=None, display=False,
                   filename=None, chains=None, col_wrap=4, figsize=None, blind=None):  # pragma: no cover
        """ Plots the 1D parameter distributions for verification purposes.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains give well behaved distributions, or if certain parameters are suspect
        or require a greater burn in period.


        Parameters
        ----------
        parameters : list[str]|int, optional
            Specify a subset of parameters to plot. If not set, all parameters are plotted.
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
        col_wrap : int, optional
            How many columns to plot before wrapping.
        figsize : tuple(float)|float, optional
            Either a tuple specifying the figure size or a float scaling factor.
        blind : bool|string|list[string], optional
            Whether to blind axes values. Can be set to `True` to blind all parameters,
            or can pass in a string (or list of strings) which specify the parameters to blind.

        Returns
        -------
        figure
            the matplotlib figure created

        """
        chains, parameters, truth, extents, blind = self._sanitise(chains, parameters, truth, extents, blind=blind)

        n = len(parameters)
        num_cols = min(n, col_wrap)
        num_rows = int(np.ceil(1.0 * n / col_wrap))

        if figsize is None:
            figsize = 1.0
        if isinstance(figsize, float):
            figsize_float = figsize
            figsize = (num_cols * 2 * figsize, num_rows * 2 * figsize)
        else:
            figsize_float = 1.0

        fit_values = self.parent.analysis.get_summary(squeeze=False, parameters=parameters)
        summary = self.parent.config["summary"]
        label_font_size = self.parent.config["label_font_size"]
        tick_font_size = self.parent.config["tick_font_size"]
        max_ticks = self.parent.config["max_ticks"]
        diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]

        if summary is None:
            summary = len(self.parent._chains) == 1

        hspace = (0.8 if summary else 0.5) / figsize_float
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=hspace)

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))

        for i, ax in enumerate(axes.flatten()):
            if i >= len(parameters):
                ax.set_axis_off()
                continue
            p = parameters[i]

            ax.set_yticks([])
            if p in blind:
                ax.set_xticks([])
            else:
                if diagonal_tick_labels:
                    _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
                _ = [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]
                ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                ax.xaxis.set_major_formatter(formatter)
            ax.set_xlim(extents.get(p) or self._get_parameter_extents(p, chains))
            max_val = None
            for index in chains:
                if p in self.parent._parameters[index]:
                    chain_row = self.parent._chains[index][:, self.parent._parameters[index].index(p)]
                    weights = self.parent._weights[index]
                    fit = fit_values[index][p]
                    param_summary = summary and p not in blind
                    m = self._plot_bars(index, ax, p, chain_row, weights, grid=self.parent._grids[index],
                                        fit_values=fit, summary=param_summary, truth=truth)
                    if max_val is None or m > max_val:
                        max_val = m
            ax.set_ylim(0, 1.1 * max_val)
            ax.set_xlabel(p, fontsize=label_font_size)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def _sanitise(self, chains, parameters, truth, extents, color_p=False, blind=None):  # pragma: no cover
        if not self.parent._configured:
            self.parent.configure()
        if not self.parent._configured_truth:
            self.parent.configure_truth()

        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()

        if chains is None:
            chains = list(range(len(self.parent._chains)))
        else:
            if isinstance(chains, str) or isinstance(chains, int):
                chains = [chains]
            chains = [self.parent._get_chain(c) for c in chains]

        if color_p:
            # Get all parameters to plot, taking into account some of them
            # might be excluded colour parameters
            color_params = self.parent.config["color_params"]
            plot_color_params = self.parent.config["plot_color_params"]
            all_parameters = []
            for i, (cp, ps, pc) in enumerate(zip(color_params, self.parent._parameters, plot_color_params)):
                if i not in chains:
                    continue
                for p in ps:
                    if (p != cp or pc) and p not in all_parameters:
                        all_parameters.append(p)
        else:
            all_parameters = list(set([p for i in chains for p in self.parent._parameters[i]]))

        if parameters is None:
            parameters = all_parameters
        elif isinstance(parameters, int):
            parameters = self.parent.all_parameters[:parameters]

        if truth is None:
            truth = {}
        else:
            if isinstance(truth, np.ndarray):
                truth = truth.tolist()
            if isinstance(truth, list):
                truth = dict((p, t) for p, t in zip(parameters, truth))

        if extents is None:
            extents = {}
        elif isinstance(extents, list):
            extents = dict((p, e) for p, e in zip(parameters, extents))

        if blind is None:
            blind = []
        elif isinstance(blind, str):
            blind = [blind]
        elif isinstance(blind, bool) and blind:
            blind = parameters

        if self.parent.config["usetex"]:
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        if self.parent.config["serif"]:
            plt.rc('font', family='serif')
        else:
            plt.rc('font', family='sans-serif')

        return chains, parameters, truth, extents, blind

    def _get_figure(self, all_parameters, flip, figsize=(5, 5), external_extents=None,
                    chains=None, blind=None):  # pragma: no cover
        n = len(all_parameters)
        max_ticks = self.parent.config["max_ticks"]
        spacing = self.parent.config["spacing"]
        plot_hists = self.parent.config["plot_hists"]
        label_font_size = self.parent.config["label_font_size"]
        tick_font_size = self.parent.config["tick_font_size"]
        diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]
        if blind is None:
            blind = []

        if chains is None:
            chains = list(range(len(self.parent._chains)))

        if not plot_hists:
            n -= 1

        if spacing is None:
            spacing = 1.0 if n < 6 else 0.0

        if n == 2 and plot_hists and flip:
            gridspec_kw = {'width_ratios': [3, 1], 'height_ratios': [1, 3]}
        else:
            gridspec_kw = {}

        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05 * spacing, hspace=0.05 * spacing)

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))

        extents = {}
        for p in all_parameters:
            if external_extents is not None and p in external_extents:
                extents[p] = external_extents[p]
            else:
                extents[p] = self._get_parameter_extents(p, chains)

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
                        if p2 in blind:
                            ax.set_xticks([])
                        else:
                            display_x_ticks = True
                        if isinstance(p2, str):
                            ax.set_xlabel(p2, fontsize=label_font_size)
                    if j != 0 or (plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        if p1 in blind:
                            ax.set_yticks([])
                        else:
                            display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(p1, fontsize=label_font_size)
                    if display_x_ticks:
                        if diagonal_tick_labels:
                            _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
                        _ = [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]
                        ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                        ax.xaxis.set_major_formatter(formatter)
                    if display_y_ticks:
                        if diagonal_tick_labels:
                            _ = [l.set_rotation(45) for l in ax.get_yticklabels()]
                        _ = [l.set_fontsize(tick_font_size) for l in ax.get_yticklabels()]
                        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                        ax.yaxis.set_major_formatter(formatter)
                    if i != j or not plot_hists:
                        ax.set_ylim(extents[p1])
                    elif flip and i == 1:
                        ax.set_ylim(extents[p1])
                    ax.set_xlim(extents[p2])

        return fig, axes, params1, params2, extents

    def _get_parameter_extents(self, parameter, chain_indexes):
        min_val, max_val = None, None
        for i, (chain, parameters, w) in enumerate(
                zip(self.parent._chains, self.parent._parameters, self.parent._weights)):
            if parameter not in parameters or i not in chain_indexes:
                continue  # pragma: no cover
            index = parameters.index(parameter)
            if self.parent._grids[i]:
                min_prop = chain[:, index].min()
                max_prop = chain[:, index].max()
            else:
                min_prop, max_prop = get_extents(chain[:, index], w, plot=True)
            if min_val is None or min_prop < min_val:
                min_val = min_prop
            if max_val is None or max_prop > max_val:
                max_val = max_prop
        return min_val, max_val

    def _get_levels(self):
        sigma2d = self.parent.config["sigma2d"]
        if sigma2d:
            levels = 1.0 - np.exp(-0.5 * self.parent.config["sigmas"] ** 2)
        else:
            levels = 2 * norm.cdf(self.parent.config["sigmas"]) - 1.0
        return levels

    def _plot_contour(self, iindex, ax, x, y, w, px, py, grid, truth=None, color_data=None, color_extent=None):  # pragma: no cover
        levels = self._get_levels()
        cloud = self.parent.config["cloud"][iindex]
        smooth = self.parent.config["smooth"][iindex]
        colour = self.parent.config["colors"][iindex]
        bins = self.parent.config["bins"][iindex]
        shade = self.parent.config["shade"][iindex]
        shade_alpha = self.parent.config["shade_alpha"][iindex]
        shade_gradient = self.parent.config["shade_gradient"][iindex]
        linestyle = self.parent.config["linestyles"][iindex]
        linewidth = self.parent.config["linewidths"][iindex]
        cmap = self.parent.config["cmaps"][iindex]
        kde = self.parent.config["kde"][iindex]
        contour_labels = self.parent.config["contour_labels"]

        h = None

        if grid:
            binsx = get_grid_bins(x)
            binsy = get_grid_bins(y)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)
        else:
            binsx, smooth = get_smoothed_bins(smooth, bins, x, w, marginalised=False)
            binsy, _ = get_smoothed_bins(smooth, bins, y, w, marginalised=False)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)

        colours = self._scale_colours(colour, len(levels), shade_gradient)
        colours2 = [self._scale_colour(colours[0], 0.7)] + \
                   [self._scale_colour(c, 0.8) for c in colours[:-1]]

        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        if kde:
            nn = x_centers.size * 2  # Double samples for KDE because smooth
            x_centers = np.linspace(x_bins.min(), x_bins.max(), nn)
            y_centers = np.linspace(y_bins.min(), y_bins.max(), nn)
            xx, yy = meshgrid(x_centers, y_centers, indexing='ij')
            coords = np.vstack((xx.flatten(), yy.flatten())).T
            data = np.vstack((x, y)).T
            hist = MegKDE(data, w, kde).evaluate(coords).reshape((nn, nn))
        elif smooth:
            hist = gaussian_filter(hist, smooth, mode=self.parent._gauss_mode)
        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if cloud:
            n = self.parent.config["num_cloud"][iindex]
            skip = max(1, int(x.size / n))
            kwargs = {"c": colours[1], "alpha": 0.3}
            if color_data is not None:
                kwargs["c"] = color_data[::skip]
                kwargs["cmap"] = cmap
                if color_extent is not None:
                    kwargs["vmin"] = color_extent[0]
                    kwargs["vmax"] = color_extent[1]
                    if color_extent[0] == color_extent[1]:
                        kwargs["vmax"] = kwargs["vmin"] + 1.0

            h = ax.scatter(x[::skip], y[::skip], s=10, marker=".", edgecolors="none", **kwargs)
            if color_data is None:
                h = None

        if shade:
            ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours,
                        alpha=shade_alpha)
        con = ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2,
                   linestyles=linestyle, linewidths=linewidth)

        if contour_labels is not None:
            if contour_labels == "sigma":
                sigmas = self.parent.config["sigmas"]
                fmt = dict([(l, ("$%.1f \\sigma$" % s).replace(".0", "")) for l, s in zip(con.levels, sigmas)])
            else:
                fmt = dict([(l, '%d\\%%' % (100 * l)) for l in con.levels])
            ax.clabel(con, con.levels, inline=True, fmt=fmt, fontsize=self.parent.config["contour_label_font_size"])
        if truth is not None:
            truth_value = truth.get(px)
            if truth_value is not None:
                ax.axhline(truth_value, **self.parent.config_truth)
            truth_value = truth.get(py)
            if truth_value is not None:
                ax.axvline(truth_value, **self.parent.config_truth)
        return h

    def _plot_bars(self, iindex, ax, parameter, chain_row, weights, flip=False, summary=False, fit_values=None,
                   truth=None, grid=False):  # pragma: no cover

        # Get values from config
        colour = self.parent.config["colors"][iindex]
        linestyle = self.parent.config["linestyles"][iindex]
        bar_shade = self.parent.config["bar_shade"][iindex]
        linewidth = self.parent.config["linewidths"][iindex]
        bins = self.parent.config["bins"][iindex]
        smooth = self.parent.config["smooth"][iindex]
        kde = self.parent.config["kde"][iindex]
        title_size = self.parent.config["label_font_size"]

        if smooth or kde:
            xs, ys, _ = self.parent.analysis._get_smoothed_histogram(chain_row, weights, iindex, grid)
            if flip:
                ax.plot(ys, xs, color=colour, ls=linestyle, lw=linewidth)
            else:
                ax.plot(xs, ys, color=colour, ls=linestyle, lw=linewidth)
        else:
            if flip:
                orientation = "horizontal"
            else:
                orientation = "vertical"
            if grid:
                bins = get_grid_bins(chain_row)
            else:
                bins, smooth = get_smoothed_bins(smooth, bins, chain_row, weights)
            hist, edges = np.histogram(chain_row, bins=bins, normed=True, weights=weights)
            edge_center = 0.5 * (edges[:-1] + edges[1:])
            xs, ys = edge_center, hist
            ax.hist(xs, weights=ys, bins=bins, histtype="step",
                        color=colour, orientation=orientation, ls=linestyle, lw=linewidth)
        interp_type = "linear" if smooth else "nearest"
        interpolator = interp1d(xs, ys, kind=interp_type)

        if bar_shade and fit_values is not None:
            lower = fit_values[0]
            upper = fit_values[2]
            if lower is not None and upper is not None:
                if lower < xs.min():
                    lower = xs.min()
                if upper > xs.max():
                    upper = xs.max()
                x = np.linspace(lower, upper, 1000)
                if flip:
                    ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x),
                                     color=colour, alpha=0.2)
                else:
                    ax.fill_between(x, np.zeros(x.shape), interpolator(x),
                                    color=colour, alpha=0.2)
                if summary:
                    t = self.parent.analysis.get_parameter_text(*fit_values)
                    if isinstance(parameter, str):
                        ax.set_title(r"$%s = %s$" % (parameter.strip("$"), t), fontsize=title_size)
                    else:
                        ax.set_title(r"$%s$" % t, fontsize=title_size)
        if truth is not None:
            truth_value = truth.get(parameter)
            if truth_value is not None:
                if flip:
                    ax.axhline(truth_value, **self.parent.config_truth)
                else:
                    ax.axvline(truth_value, **self.parent.config_truth)
        return ys.max()

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
        max_ticks = self.parent.config["max_ticks"]
        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))

        if convolve is not None:
            color2 = self._scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=':', color=color2, alpha=1)
        if truth is not None:
            ax.axhline(truth, **self.parent.config_truth)

    def _convert_to_stdev(self, sigma):  # pragma: no cover
        # From astroML
        shape = sigma.shape
        sigma = sigma.ravel()
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)

        sigma_cumsum = 1.0 * sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]

        return sigma_cumsum[i_unsort].reshape(shape)

    def _clamp(self, val, minimum=0, maximum=255):  # pragma: no cover
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val

    def _scale_colours(self, colour, num, shade_gradient):  # pragma: no cover
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        minv, maxv = 1 - 0.1 * shade_gradient, 1 + 0.3 * shade_gradient
        scales = np.logspace(np.log(minv), np.log(maxv), num)
        colours = [self._scale_colour(colour, scale) for scale in scales]
        return colours

    def _scale_colour(self, colour, scalefactor):  # pragma: no cover
        if isinstance(colour, np.ndarray):
            r, g, b = colour[:3] * 255.0
        else:
            hexx = colour.strip('#')
            if scalefactor < 0 or len(hexx) != 6:
                return hexx
            r, g, b = int(hexx[:2], 16), int(hexx[2:4], 16), int(hexx[4:], 16)
        r = self._clamp(int(r * scalefactor))
        g = self._clamp(int(g * scalefactor))
        b = self._clamp(int(b * scalefactor))
        return "#%02x%02x%02x" % (r, g, b)
