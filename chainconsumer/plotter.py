# -*- coding: utf-8 -*-
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.textpath import TextPath
from numpy import meshgrid
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from .helpers import get_extents, get_smoothed_bins, get_grid_bins
from .kde import MegKDE


class Plotter(object):
    def __init__(self, parent):
        self.parent = parent
        self._logger = logging.getLogger("chainconsumer")

        self.usetex_old = matplotlib.rcParams["text.usetex"]
        self.serif_old = matplotlib.rcParams["font.family"]

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
        names = [chain.name for chain in chains]

        if legend is None:
            legend = len(chains) > 1

        # If no chains have names, don't plot the legend
        legend = legend and len([n for n in names if n]) > 0

        # Calculate cmap extents
        unique_color_params = list(set([c.config["color_params"] for c in chains if c.config["color_params"] is not None]))
        num_cax = len(unique_color_params)
        color_param_extents = {}
        for u in unique_color_params:
            umin, umax = np.inf, -np.inf
            for chain in chains:
                if chain.config["color_params"] == u:
                    data = chain.get_color_data()
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

        if summary is None:
            summary = len(parameters) < 5 and len(self.parent.chains) == 1
        if len(chains) == 1:
            self._logger.debug("Plotting surfaces for chain of dimension %s" %
                               (chains[0].chain.shape,))
        else:
            self._logger.debug("Plotting surfaces for %d chains" % len(chains))
        cbar_done = []

        chain_points = [c for c in chains if c.config["plot_point"]]
        num_chain_points = len(chain_points)
        if num_chain_points:
            subgroup_names = list(set([c.name for c in chain_points]))
            subgroups = [[c for c in chain_points if c.name == n] for n in subgroup_names]
            markers = [group[0].config["marker_style"] for group in subgroups]  # Only one marker per group
            marker_sizes = [[g.config["marker_size"] for g in group] for group in subgroups]  # But size can diff
            marker_alphas = [group[0].config["marker_alpha"] for group in subgroups]  # Only one marker per group
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j:
                    continue
                ax = axes[i, j]
                do_flip = (flip and i == len(params1) - 1)

                # Plot the histograms
                if plot_hists and i == j:
                    if do_flip:
                        self._add_truth(ax, truth, p1)
                    else:
                        self._add_truth(ax, truth,  None, py=p2)
                    max_val = None

                    # Plot each chain
                    for chain in chains:
                        if p1 not in chain.parameters:
                            continue
                        if not chain.config["plot_contour"]:
                            continue

                        param_summary = summary and p1 not in blind
                        m = self._plot_bars(ax, p1, chain, flip=do_flip, summary=param_summary)

                        if max_val is None or m > max_val:
                            max_val = m

                    if num_chain_points and self.parent.config["global_point"]:
                        m = self._plot_point_histogram(ax, subgroups, p1, flip=do_flip)
                        if max_val is None or m > max_val:
                            max_val = m

                    if max_val is not None:
                        if do_flip:
                            ax.set_xlim(0, 1.1 * max_val)
                        else:
                            ax.set_ylim(0, 1.1 * max_val)

                else:
                    for chain in chains:
                        if p1 not in chain.parameters or p2 not in chain.parameters:
                            continue
                        if not chain.config["plot_contour"]:
                            continue
                        h = None
                        if p1 in chain.parameters and p2 in chain.parameters:
                            h = self._plot_contour(ax, chain, p1, p2, color_extents=color_param_extents)
                        cp = chain.config["color_params"]
                        if h is not None and cp is not None and cp not in cbar_done:
                            cbar_done.append(cp)
                            aspect = figsize[1] / 0.15
                            fraction = 0.85 / figsize[0]
                            cbar = fig.colorbar(h, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                            label = cp
                            if label == "weights":
                                label = "Weights"
                            elif label == "log_weights":
                                label = "log(Weights)"
                            elif label == "posterior":
                                label = "log(Posterior)"
                            cbar.set_label(label, fontsize=14)
                            cbar.solids.set(alpha=1)

                    if num_chain_points:
                        self._plot_points(ax, subgroups, markers, marker_sizes, marker_alphas, p1, p2)

                    self._add_truth(ax, truth, p1, py=p2)

        colors = [c.config["color"] for c in chains]
        plot_points = [c.config["plot_point"] for c in chains]
        plot_contours = [c.config["plot_contour"] for c in chains]
        linestyles = [c.config["linestyle"] for c in chains]
        linewidths = [c.config["linewidth"] for c in chains]
        marker_styles = [c.config["marker_style"] for c in chains]
        marker_sizes = [c.config["marker_size"] for c in chains]
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
            marker_sizes2 = marker_sizes if legend_artists else [0]*len(linestyles)

            artists = []
            done_names = []
            final_colors = []
            for i, (n, c, ls, lw, marker, size, pp, pc) in enumerate(zip(names, colors, linestyles2, linewidths2,
                                                                         marker_styles, marker_sizes2, plot_points, plot_contours)):
                if n is None or n in done_names:
                    continue
                done_names.append(n)
                final_colors.append(c)
                size = np.sqrt(size)  # plot vs scatter use size differently, hence the sqrt
                if pc and not pp:
                    artists.append(plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw))
                elif not pc and pp:
                    artists.append(plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=0, marker=marker, markersize=size))
                else:
                    artists.append(plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw, marker=marker, markersize=size))

            leg = ax.legend(artists, done_names, **legend_kwargs)
            if legend_color_text:
                for text, c in zip(leg.get_texts(), final_colors):
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
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, dpi)
        if display:
            plt.show()

        return fig

    def _save_fig(self, fig, filename, dpi):  # pragma: no cover
        fig.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True, pad_inches=0.05)

    def _add_watermark(self, fig, axes, figsize, text, dpi=300, size_scale=1.0):  # pragma: no cover
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
        size = np.sqrt(dy ** 2 + dx ** 2) / (dh * abs(dy / dx) + dw) * 0.6 * scale * size_scale
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
        chains = [c for c in chains if c.mcmc_chain]
        n = len(parameters)
        extra = 0
        if plot_weights:
            plot_weights = plot_weights and np.any([np.any(c.weights != 1.0) for c in chains])

        plot_posterior = plot_posterior and np.any([c.posterior is not None for c in chains])

        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            figsize = (8, 0.75 + (n + extra))

        fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)

        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = parameters[i - n]
                for chain in chains:
                    if p in chain.parameters:
                        chain_row = chain.get_data(p)
                        self._plot_walk(ax, p, chain_row, extents=extents.get(p), convolve=convolve, color=chain.config["color"])
                if truth.get(p) is not None:
                    self._plot_walk_truth(ax, truth.get(p))
            else:
                if i == 0 and plot_posterior:
                    for chain in chains:
                        if chain.posterior is not None:
                            self._plot_walk(ax, "$\log(P)$", chain.posterior - chain.posterior.max(),
                                            convolve=convolve, color=chain.config["color"])
                else:
                    if log_weight is None:
                        log_weight = np.any([chain.weights.mean() < 0.1 for chain in chains])
                    if log_weight:
                        for chain in chains:
                            self._plot_walk(ax, r"$\log_{10}(w)$", np.log10(chain.weights),
                                            convolve=convolve, color=chain.config["color"])
                    else:
                        for chain in chains:
                            self._plot_walk(ax, "$w$", chain.weights,
                                            convolve=convolve, color=chain.config["color"])

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, 300)
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

        summary = self.parent.config["summary"]
        label_font_size = self.parent.config["label_font_size"]
        tick_font_size = self.parent.config["tick_font_size"]
        max_ticks = self.parent.config["max_ticks"]
        diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]

        if summary is None:
            summary = len(self.parent.chains) == 1

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
            for chain in chains:
                if not chain.config["plot_contour"]:
                    continue
                if p in chain.parameters:
                    param_summary = summary and p not in blind
                    m = self._plot_bars(ax, p, chain, summary=param_summary)
                    if max_val is None or m > max_val:
                        max_val = m

            self._add_truth(ax, truth, None, py=p)
            ax.set_ylim(0, 1.1 * max_val)
            ax.set_xlabel(p, fontsize=label_font_size)

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, 300)
        if display:
            plt.show()

        return fig

    def plot_summary(self, parameters=None, truth=None, extents=None, display=False,
                     filename=None, chains=None, figsize=1.0, errorbar=False, include_truth_chain=True,
                     blind=None, watermark=None, extra_parameter_spacing=0.5,
                     vertical_spacing_ratio=1.0, show_names=True):  # pragma: no cover
        """ Plots parameter summaries

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains give well behaved distributions, or if certain parameters are suspect
        or require a greater burn in period.


        Parameters
        ----------
        parameters : list[str]|int, optional
            Specify a subset of parameters to plot. If not set, all parameters are plotted.
            If an integer is given, only the first so many parameters are plotted.
        truth : list[float]|list|list[float]|dict[str]|str, optional
            A list of truth values corresponding to parameters, or a dictionary of
            truth values keyed by the parameter. Each "truth value" can be either a float (will
            draw a vertical line), two floats (a shaded interval) or three floats (min, mean, max),
            which renders as a shaded interval with a line for the mean. Or, supply a string
            which matches a chain name, and the results for that chain will be used as the 'truth'
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
        figsize : float, optional
            Scale horizontal and vertical figure size.
        errorbar : bool, optional
            Whether to onle plot an error bar, instead of the marginalised distribution.
        include_truth_chain : bool, optional
            If you specify another chain as the truth chain, determine if it should still
            be plotted.
        blind : bool|string|list[string], optional
            Whether to blind axes values. Can be set to `True` to blind all parameters,
            or can pass in a string (or list of strings) which specify the parameters to blind.
        watermark : str, optional
            A watermark to add to the figure
        extra_parameter_spacing : float, optional
            Increase horizontal space for parameter values
        vertical_spacing_ratio : float, optional
            Increase vertical space for each model
        show_names : bool, optional
            Whether to show chain names or not. Defaults to `True`.

        Returns
        -------
        figure
            the matplotlib figure created

        """
        wide_extents = not errorbar
        chains, parameters, truth, extents, blind = self._sanitise(chains, parameters, truth, extents, blind=blind, wide_extents=wide_extents)

        all_names = [c.name for c in self.parent.chains]

        # Check if we're using a chain for truth values
        if isinstance(truth, str):
            assert truth in all_names, "Truth chain %s is not in the list of added chains %s" % (truth, all_names)
            if not include_truth_chain:
                chains = [c for c in chains if c.name != truth]
            truth = self.parent.analysis.get_summary(chains=truth, parameters=parameters)

        max_param = self._get_size_of_texts(parameters)
        fid_dpi = 65  # Seriously I have no idea what value this should be
        param_width = extra_parameter_spacing + max(0.5, max_param / fid_dpi)

        if show_names:
            max_model_name = self._get_size_of_texts([chain.name for chain in chains])
            model_width = 0.25 + (max_model_name / fid_dpi)
            gridspec_kw = {'width_ratios': [model_width] + [param_width] * len(parameters), 'height_ratios': [1] * len(chains)}
            ncols = 1 + len(parameters)
        else:
            model_width = 0
            gridspec_kw = {'width_ratios': [param_width] * len(parameters), 'height_ratios': [1] * len(chains)}
            ncols = len(parameters)

        top_spacing = 0.3
        bottom_spacing = 0.2
        row_height = (0.5 if not errorbar else 0.3) * vertical_spacing_ratio
        width = param_width * len(parameters) + model_width
        height = top_spacing + bottom_spacing + row_height * len(chains)
        top_ratio = 1 - (top_spacing / height)
        bottom_ratio = bottom_spacing / height

        figsize = (width * figsize, height * figsize)
        fig, axes = plt.subplots(nrows=len(chains), ncols=ncols, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0.05, right=0.95, top=top_ratio, bottom=bottom_ratio, wspace=0.0, hspace=0.0)
        label_font_size = self.parent.config["label_font_size"]
        legend_color_text = self.parent.config["legend_color_text"]

        max_vals = {}
        for i, row in enumerate(axes):
            chain = chains[i]

            cs, ws, ps, = chain.chain, chain.weights, chain.parameters
            gs, ns = chain.grid, chain.name

            colour = chain.config["color"]

            # First one put name of model
            if show_names:
                ax_first = row[0]
                ax_first.set_axis_off()
                text_colour = "k" if not legend_color_text else colour
                ax_first.text(0, 0.5, ns, transform=ax_first.transAxes, fontsize=label_font_size, verticalalignment="center", color=text_colour, weight="medium")
                cols = row[1:]
            else:
                cols = row

            for ax, p in zip(cols, parameters):
                # Set up the frames
                if i > 0:
                    ax.spines['top'].set_visible(False)
                if i < (len(chains) - 1):
                    ax.spines['bottom'].set_visible(False)
                if i < (len(chains) - 1) or p in blind:
                    ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(extents[p])

                # Put title in
                if i == 0:
                    ax.set_title(r"$%s$" % p, fontsize=label_font_size)

                # Add truth values
                truth_value = truth.get(p)
                if truth_value is not None:
                    if isinstance(truth_value, float) or isinstance(truth_value, int):
                        truth_mean = truth_value
                        truth_min, truth_max = None, None
                    else:
                        if len(truth_value) == 1:
                            truth_mean = truth_value
                            truth_min, truth_max = None, None
                        elif len(truth_value) == 2:
                            truth_min, truth_max = truth_value
                            truth_mean = None
                        else:
                            truth_min, truth_mean, truth_max = truth_value
                    if truth_mean is not None:
                        ax.axvline(truth_mean, **self.parent.config_truth)
                    if truth_min is not None and truth_max is not None:
                        ax.axvspan(truth_min, truth_max, color=self.parent.config_truth["color"], alpha=0.15, lw=0)
                # Skip if this chain doesnt have the parameter
                if p not in ps:
                    continue

                # Plot the good stuff
                if errorbar:
                    fv = self.parent.analysis.get_parameter_summary(chain, p)
                    if fv[0] is not None and fv[2] is not None:
                        diff = np.abs(np.diff(fv))
                        ax.errorbar([fv[1]], 0, xerr=[[diff[0]], [diff[1]]], fmt='o', color=colour)
                else:
                    m = self._plot_bars(ax, p, chain)
                    if max_vals.get(p) is None or m > max_vals.get(p):
                        max_vals[p] = m

        for i, row in enumerate(axes):
            index = 1 if show_names else 0
            for ax, p in zip(row[index:], parameters):
                if not errorbar:
                    ax.set_ylim(0, 1.1 * max_vals[p])

        dpi = 300
        if watermark:
            ax = None
            self._add_watermark(fig, ax, figsize, watermark, dpi=dpi, size_scale=0.8)

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, dpi)
        if display:
            plt.show()

        return fig

    def _get_size_of_texts(self, texts):  # pragma: no cover
        usetex = self.parent.config["usetex"]
        size = self.parent.config["label_font_size"]
        widths = [TextPath((0, 0), text, usetex=usetex, size=size).get_extents().width for text in texts]
        return max(widths)

    def _sanitise(self, chains, parameters, truth, extents, color_p=False, blind=None, wide_extents=True):  # pragma: no cover
        if not self.parent._configured:
            self.parent.configure()
        if not self.parent._configured_truth:
            self.parent.configure_truth()

        if chains is None:
            chains = list(range(len(self.parent.chains)))
        else:
            if isinstance(chains, str) or isinstance(chains, int):
                chains = [chains]
            chains = [i for c in chains for i in self.parent._get_chain(c)]

        chains = [self.parent.chains[i] for i in chains]

        if color_p:
            # Get all parameters to plot, taking into account some of them
            # might be excluded colour parameters
            all_parameters = []
            for chain in chains:
                pc = chain.config["plot_color_params"]
                cp = chain.config["color_params"]
                ps = chain.parameters
                for p in ps:
                    if (p != cp or pc) and p not in all_parameters:
                        all_parameters.append(p)
        else:
            all_parameters = []
            for chain in chains:
                for p in chain.parameters:
                    if p not in all_parameters:
                        all_parameters.append(p)

        if parameters is None:
            parameters = all_parameters
        elif isinstance(parameters, int):
            parameters = self.parent._all_parameters[:parameters]

        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()
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

        extents = self._get_custom_extents(parameters, chains, extents, wide_extents=wide_extents)

        if blind is None:
            blind = []
        elif isinstance(blind, str):
            blind = [blind]
        elif isinstance(blind, bool) and blind:
            blind = parameters

        self.set_rc_params()

        return chains, parameters, truth, extents, blind

    def set_rc_params(self):
        if self.parent.config["usetex"]:
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        if self.parent.config["serif"]:
            plt.rc('font', family='serif')
        else:
            plt.rc('font', family='sans-serif')

    def restore_rc_params(self):
        """ Restores the matplotlib rc parameters modified by usetex and serif.

        Unfortunately this cannot be automated because you cannot invoke it whilst you have
        an active figure object or matplotlib will destroy you. So do all your plotting, close
        the plots, and then you can call this.
        """
        plt.rc('text', usetex=self.usetex_old)
        plt.rc('font', family=self.serif_old)

    def _get_custom_extents(self, parameters, chains, external_extents, wide_extents=True):  # pragma: no cover
        extents = {}
        for p in parameters:
            if external_extents is not None and p in external_extents:
                extents[p] = external_extents[p]
            else:
                extents[p] = self._get_parameter_extents(p, chains, wide_extents=wide_extents)
        return extents

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
            chains = self.parent.chains

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

        extents = self._get_custom_extents(all_parameters, chains, external_extents)

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

    def _get_parameter_extents(self, parameter, chains, wide_extents=True):
        min_val, max_val = None, None
        for chain in chains:
            if parameter not in chain.parameters:
                continue  # pragma: no cover
            if not chain.config["plot_contour"]:
                data = chain.get_data(parameter)
                if data.size < 10:
                    min_prop, max_prop = np.min(data), np.max(data)
                else:
                    if self.parent.config["global_point"]:
                        min_prop = chain.posterior_max_params.get(parameter)
                        max_prop = min_prop
                    else:
                        data = chain.get_data(parameter)
                        min_prop, max_prop = get_extents(data, chain.weights, tiny=True)
            else:
                data = chain.get_data(parameter)
                if chain.grid:
                    min_prop = data.min()
                    max_prop = data.max()
                else:
                    min_prop, max_prop = get_extents(data, chain.weights, plot=True, wide_extents=wide_extents)
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

    def _plot_points(self, ax, chains_groups, markers, sizes, alphas, py, px):  # pragma: no cover
        global_point = self.parent.config["global_point"]
        for marker, chains, size, alpha in zip(markers, chains_groups, sizes, alphas):
            if global_point:
                res = self.parent.analysis.get_max_posteriors(parameters=[px, py], chains=chains, squeeze=False)
                xs = [r[px] for r in res if r is not None]
                ys = [r[py] for r in res if r is not None]
            else:
                xs, ys, res = [], [], []
                for chain in chains:
                    if px in chain.parameters and py in chain.parameters:
                        x = chain.get_data(px)
                        y = chain.get_data(py)
                        if x.size <= 2:  # Marker only
                            xs.append(x[0])
                            ys.append(y[0])
                            res.append({"px": x[0], "py": y[0]})
                        else:
                            hist, x_centers, y_centers = self._get_smoothed_histogram2d(chain, py, px)
                            index = np.unravel_index(hist.argmax(), hist.shape)
                            ys.append(x_centers[index[0]])
                            xs.append(y_centers[index[1]])
                            res.append({"px": xs[-1], "py": ys[-1]})
                    else:
                        res.append(None)
            cs = [c.config["color"] for c, r in zip(chains, res) if r is not None]
            h = ax.scatter(xs, ys, marker=marker, c=cs, s=size, linewidth=0.7, alpha=alpha)
        return h

    def _plot_contour(self, ax, chain, px, py, color_extents=None):  # pragma: no cover

        levels = self._get_levels()
        cloud = chain.config["cloud"]
        colour = chain.config["color"]
        shade = chain.config["shade"]
        shade_alpha = chain.config["shade_alpha"]
        shade_gradient = chain.config["shade_gradient"]
        linestyle = chain.config["linestyle"]
        linewidth = chain.config["linewidth"]
        cmap = chain.config["cmap"]
        contour_labels = self.parent.config["contour_labels"]

        h = None
        color_data = chain.get_color_data()
        x = chain.get_data(py)
        y = chain.get_data(px)
        color_extent = color_extents.get(chain.config["color_params"])

        cf = self.parent.color_finder
        colours = self._scale_colours(colour, len(levels), shade_gradient)
        sub = max(0.1, 1 - 0.2 * shade_gradient)
        if shade:
            sub *= 0.9
        colours2 = [cf.scale_colour(colours[0], sub)] + \
                   [cf.scale_colour(c, sub) for c in colours[:-1]]

        hist, x_centers, y_centers = self._get_smoothed_histogram2d(chain, py, px)

        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if cloud:
            n = chain.config["num_cloud"]
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
            lvls = [l for l in con.levels if l != 0.0]
            if contour_labels == "sigma":
                sigmas = self.parent.config["sigmas"]
                fmt = dict([(l, ("$%.1f \\sigma$" % s).replace(".0", "")) for l, s in zip(lvls, sigmas[1:])])
            else:
                fmt = dict([(l, '%d\\%%' % (100 * l)) for l in lvls])
            ax.clabel(con, lvls, inline=True, fmt=fmt, fontsize=self.parent.config["contour_label_font_size"])
        return h

    def _add_truth(self, ax, truth, px, py=None):  # pragma: no cover
        if truth is not None:
            if px is not None:
                truth_value = truth.get(px)
                if truth_value is not None:
                    ax.axhline(truth_value, **self.parent.config_truth)
            if py is not None:
                truth_value = truth.get(py)
                if truth_value is not None:
                    ax.axvline(truth_value, **self.parent.config_truth)

    def _plot_bars(self, ax, parameter, chain, flip=False, summary=False):  # pragma: no cover

        # Get values from config
        colour = chain.config["color"]
        linestyle = chain.config["linestyle"]
        bar_shade = chain.config["bar_shade"]
        linewidth = chain.config["linewidth"]
        bins = chain.config["bins"]
        smooth = chain.config["smooth"]
        kde = chain.config["kde"]
        title_size = self.parent.config["label_font_size"]

        chain_row = chain.get_data(parameter)
        weights = chain.weights
        if smooth or kde:
            xs, ys, _ = self.parent.analysis._get_smoothed_histogram(chain, parameter)
            if flip:
                ax.plot(ys, xs, color=colour, ls=linestyle, lw=linewidth)
            else:
                ax.plot(xs, ys, color=colour, ls=linestyle, lw=linewidth)
        else:
            if flip:
                orientation = "horizontal"
            else:
                orientation = "vertical"
            if chain.grid:
                bins = get_grid_bins(chain_row)
            else:
                bins, smooth = get_smoothed_bins(smooth, bins, chain_row, weights)
            hist, edges = np.histogram(chain_row, bins=bins, density=True, weights=weights)
            if chain.power is not None:
                hist = hist ** chain.power
            edge_center = 0.5 * (edges[:-1] + edges[1:])
            xs, ys = edge_center, hist
            ax.hist(xs, weights=ys, bins=bins, histtype="step",
                        color=colour, orientation=orientation, ls=linestyle, lw=linewidth)
        interp_type = "linear" if smooth else "nearest"
        interpolator = interp1d(xs, ys, kind=interp_type)

        if bar_shade:
            fit_values = self.parent.analysis.get_parameter_summary(chain, parameter)
            if fit_values is not None:
                lower = fit_values[0]
                upper = fit_values[2]
                if lower is not None and upper is not None:
                    if lower < xs.min():
                        lower = xs.min()
                    if upper > xs.max():
                        upper = xs.max()
                    x = np.linspace(lower, upper, 1000)
                    if flip:
                        ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2)
                    else:
                        ax.fill_between(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2)
                    if summary:
                        t = self.parent.analysis.get_parameter_text(*fit_values)
                        if isinstance(parameter, str):
                            ax.set_title(r"$%s = %s$" % (parameter.strip("$"), t), fontsize=title_size)
                        else:
                            ax.set_title(r"$%s$" % t, fontsize=title_size)
        return ys.max()

    def _plot_point_histogram(self, ax, chains_groups, parameter, flip=False):  # pragma: no cover
        max_val = 0
        for chains in chains_groups:
            if len(chains) < 10:  # You probably dont want a contour if you only have a small group
                continue  # And even if you do, it'll be so inaccurate...
            res = self.parent.analysis.get_max_posteriors(parameters=parameter, chains=chains, squeeze=False)
            xs = [r[parameter] for r in res if r is not None]
            colour = chains[0].config["color"]
            num_bins = int(max(5, np.power(len(xs), 0.4)))
            hist, bin_edges = np.histogram(xs, bins=num_bins, density=True)
            if hist.max() > max_val:
                max_val = hist.max()
            if flip:
                orientation = "horizontal"
            else:
                orientation = "vertical"

            bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            xs, ys = bin_center, hist
            ax.hist(xs, weights=ys, bins=bin_edges, histtype="step", color=colour, orientation=orientation)
        return max_val

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
            color2 = self.parent.color_finder.scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=':', color=color2, alpha=1)

    def _plot_walk_truth(self, ax, truth):
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

    def _scale_colours(self, colour, num, shade_gradient):  # pragma: no cover
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        minv, maxv = 1 - 0.1 * shade_gradient, 1 + 0.5 * shade_gradient
        scales = np.logspace(np.log(minv), np.log(maxv), num)
        colours = [self.parent.color_finder.scale_colour(colour, scale) for scale in scales]
        return colours

    def _get_smoothed_histogram2d(self, chain, param1, param2):  # pragma: no cover
        # No test coverage here because
        smooth = chain.config["smooth"]
        bins = chain.config["bins"]
        kde = chain.config["kde"]

        x = chain.get_data(param1)
        y = chain.get_data(param2)
        w = chain.weights

        if chain.grid:
            binsx = get_grid_bins(x)
            binsy = get_grid_bins(y)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)
        else:
            binsx, smooth = get_smoothed_bins(smooth, bins, x, w, marginalised=False)
            binsy, _ = get_smoothed_bins(smooth, bins, y, w, marginalised=False)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)

        if chain.power is not None:
            hist = hist ** chain.power

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
            if chain.power is not None:
                hist = hist ** chain.power
        elif smooth:
            hist = gaussian_filter(hist, smooth, mode=self.parent._gauss_mode)

        return hist, x_centers, y_centers