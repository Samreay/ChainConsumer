from enum import Enum
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
from scipy.interpolate import interp1d  # type: ignore

from chainconsumer.plotting.truth import plot_truths  # type: ignore
from chainconsumer.truth import Truth

from .base import BetterBase
from .chain import Chain, ChainName, ColumnName
from .color_finder import colors
from .helpers import get_bins, get_extents, get_grid_bins, get_smoothed_bins
from .plotting import add_watermark, plot_surface
from .plotting.config import PlotConfig


class PlottingBase(BetterBase):
    chains: list[Chain]
    columns: list[ColumnName]
    extents: dict[ColumnName, tuple[float, float]]
    blind: list[ColumnName]
    log_scales: list[ColumnName]


class FigSize(Enum):
    """Enum for figure size options"""

    COLUMN = "COLUMN"
    PAGE = "PAGE"
    GROW = "GROW"

    @classmethod
    def get_size(
        cls, input: "FigSize | float | int | tuple[float, float]", num_columns: int, has_cax: bool
    ) -> tuple[float, float]:
        if input == FigSize.PAGE:
            return 10, 10
        if input == FigSize.COLUMN:
            return 5 + (1 if has_cax else 0), 5
        grow_factor = 1.0
        if isinstance(input, float):
            grow_factor = input
        elif isinstance(input, int):
            return float(input), float(input)
        elif isinstance(input, tuple):
            return input

        # Otherwise it must be grow, which is the default
        return 3 + grow_factor * 2 * num_columns + (1 if has_cax else 0), 3 + grow_factor * 2 * num_columns


def get_artists_from_chains(chains: list[Chain]) -> list[Artist]:
    artists: list[Artist] = []
    for chain in chains:
        if not chain.show_label_in_legend:
            continue
        if chain.plot_contour and not chain.plot_point:
            artists.append(
                Line2D(
                    (0, 1),
                    (0, 0),
                    color=colors.format(chain.color),
                    ls=chain.linestyle,
                    lw=chain.linewidth,
                    label="  " + chain.name,
                )
            )
        elif not chain.plot_contour and chain.plot_point:
            artists.append(
                Line2D(
                    (0, 1),
                    (0, 0),
                    color=colors.format(chain.color),
                    ls=chain.linestyle,
                    lw=0,
                    marker=chain.marker_style,
                    markersize=np.sqrt(chain.marker_size),
                    label="  " + chain.name,
                )
            )
        else:
            artists.append(
                Line2D(
                    (0, 1),
                    (0, 0),
                    color=colors.format(chain.color),
                    ls=chain.linestyle,
                    lw=chain.linewidth,
                    marker=chain.marker_style,
                    markersize=np.sqrt(chain.marker_size),
                    label="  " + chain.name,
                )
            )
    return artists


class Plotter:
    def __init__(self, parent: "ChainConsumer") -> None:
        self.parent: "ChainConsumer" = parent
        self._config: PlotConfig | None = None
        self._default_config = PlotConfig()

        self.usetex_old = matplotlib.rcParams["text.usetex"]
        self.serif_old = matplotlib.rcParams["font.family"]

    def set_config(self, config: PlotConfig) -> None:
        """Sets the plot config to the chosen `PlotConfig` model.

        Args:
            config: The config to use

        """
        self._config = config

    @property
    def config(self) -> PlotConfig:
        if self._config is None:
            return self._default_config
        return self._config

    def plot(
        self,
        chains: list[ChainName | Chain] | None = None,
        columns: list[ColumnName] | None = None,
        filename: list[str | Path] | str | Path | None = None,
        figsize: FigSize | float | int | tuple[float, float] = FigSize.GROW,
    ) -> Figure:  # pragma: no cover
        """Plot the chain!

        Args:
            chains:
                Used to specify which chain to show if more than one chain is loaded in.
                Can be an integer, specifying the
                chain index, or a str, specifying the chain name.
            columns:
                If set, only creates a plot for those specific parameters (if list). If an
                integer is given, only plots the fist so many parameters.
            filename:
                If set, saves the figure to this location
            figsize:
                The figure size to generate. Accepts a regular two tuple of size in inches,
                or one of several key words. The default value of ``COLUMN`` creates a figure
                of appropriate size of insertion into an A4 LaTeX document in two-column mode.
                ``PAGE`` creates a full page width figure. ``GROW`` creates an image that
                scales with parameters (1.5 inches per parameter). String arguments are not
                case sensitive. If you pass a float, it will scale the default ``GROW`` by
                that amount, so ``2.0`` would result in a plot 3 inches per parameter.

        Returns:
            the matplotlib figure

        """
        base = self._sanitise(
            chains, columns, self.config.extents, blind=self.config.blind, log_scales=self.config.log_scales
        )

        show_legend = self.config.show_legend
        if show_legend is None:
            show_legend = len(base.chains) > 1

        num_cax = len(set([chain.color_param for chain in base.chains if chain.color_param is not None]))
        fig_size = FigSize.get_size(figsize, len(base.columns), num_cax > 0)
        plot_hists = self.config.plot_hists
        flip = len(base.columns) == 2 and plot_hists and self.config.flip
        fig, axes, params_x, params_y = self._get_triangle_figure(base, figsize=fig_size)

        axl = axes.ravel().tolist()
        summarise = self.config.summarise and len(base.chains) == 1

        paths_for_cbar: dict[ColumnName, PathCollection] = {}
        for i, p1 in enumerate(params_x):
            for j, p2 in enumerate(params_y):
                if i < j:
                    continue
                ax: Axes = axes[i, j]
                do_flip = flip and i == len(params_x) - 1

                # Plot the histograms
                if plot_hists and i == j:
                    if do_flip:
                        plot_truths(ax, self.parent._truths, py=p1)
                    else:
                        plot_truths(ax, self.parent._truths, px=p1)
                    max_val = None

                    # Plot each chain
                    for chain in base.chains:
                        if not chain.plot_contour or p1 not in chain.samples:
                            continue

                        do_summary = summarise and p1 not in base.blind
                        max_hist_val = self._plot_bars(ax, p1, chain, flip=do_flip, summary=do_summary)

                        if max_val is None or max_hist_val > max_val:
                            max_val = max_hist_val

                    if max_val is not None:
                        if do_flip:
                            ax.set_xlim(0, 1.1 * max_val)
                        else:
                            ax.set_ylim(0, 1.1 * max_val)

                else:
                    paths_for_cbar |= plot_surface(ax, base.chains, p2, p1, self.config)
                    plot_truths(ax, self.parent._truths, px=p2, py=p1)

        # Create all the colorbars we need
        if paths_for_cbar:
            aspect = fig_size[1] / 0.15
            fraction = 0.85 / fig_size[0]
            for column, path in paths_for_cbar.items():
                cbar = fig.colorbar(path, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                label = self.config.get_label(column)
                if label == "weight":
                    label = "Weights"
                elif label == "log_weight":
                    label = "log(Weights)"
                elif label == "posterior":
                    label = "log(Posterior)"
                cbar.set_label(label, fontsize=self.config.label_font_size)
                if cbar.solids is not None:
                    cbar.solids.set(alpha=1)

        legend_location = self.config.legend_location
        if legend_location is None:
            legend_location = (0, -1) if not flip or len(base.columns) > 2 else (-1, 0)
        legend_outside = legend_location[0] >= legend_location[1]

        if show_legend:
            ax = axes[legend_location[0], legend_location[1]]
            legend_kwargs = self.config.legend_kwargs_final.copy()
            if "markerfirst" not in legend_kwargs:
                legend_kwargs["markerfirst"] = legend_outside or not self.config.legend_artists

            chains_to_show_on_legend = [c for c in base.chains if c.show_label_in_legend]
            artists = get_artists_from_chains(chains_to_show_on_legend)
            leg = ax.legend(handles=artists, **legend_kwargs)
            if self.config.legend_color_text:
                for text, chain in zip(leg.get_texts(), chains_to_show_on_legend):
                    text.set_fontweight("medium")
                    text.set_color(colors.format(chain.color))
        fig.canvas.draw()
        for ax in axes[-1, :]:
            offset = ax.get_xaxis().get_offset_text()
            ax.set_xlabel("{} {}".format(ax.get_xlabel(), f"[{offset.get_text()}]" if offset.get_text() else ""))
            offset.set_visible(False)
        for ax in axes[:, 0]:
            offset = ax.get_yaxis().get_offset_text()
            ax.set_ylabel("{} {}".format(ax.get_ylabel(), f"[{offset.get_text()}]" if offset.get_text() else ""))
            offset.set_visible(False)

        if self.config.watermark is not None:
            ax_watermark = axes[-1, 0] if flip and len(base.columns) == 2 else None
            add_watermark(fig, ax_watermark, fig_size, self.config)

        self._save_fig(fig, filename, dpi=self.config.dpi)

        return fig

    def _save_fig(self, fig: Figure, filename: list[str | Path] | str | Path | None = None, dpi: int = 300) -> None:
        if filename is not None:
            if not isinstance(filename, list):
                filename = [filename]
            for f in filename:
                fig.savefig(f, bbox_inches="tight", dpi=dpi, transparent=True, pad_inches=0.05)

    def plot_walks(
        self,
        chains: list[ChainName | Chain] | None = None,
        columns: list[ColumnName] | None = None,
        filename: list[str | Path] | str | Path | None = None,
        figsize: float | tuple[float, float] | None = None,
        convolve: int | None = None,
        plot_weights: bool = True,
        plot_posterior: bool = True,
        log_weight: bool = False,
    ) -> Figure:  # pragma: no cover
        """Plots the chain walk; the parameter values as a function of step index.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains are well behaved, or if certain parameters are suspect
        or require a greater burn in period.

        The desired outcome is to see an unchanging distribution along the x-axis of the plot.
        If there are obvious tails or features in the parameters, you probably want
        to investigate.

        Args:
            chains:
                Used to specify which chain to show if more than one chain is loaded in.
                Can be an integer, specifying the
                chain index, or a str, specifying the chain name.
            columns:
                If set, only creates a plot for those specific parameters (if list). If an
                integer is given, only plots the fist so many parameters.
            filename:
                If set, saves the figure to this location
            figsize:
                Scale horizontal and vertical figure size.
            col_wrap:
                How many columns to plot before wrapping.
            convolve:
                If set, overplots a smoothed version of the steps using ``convolve`` as
                the width of the smoothing filter.
            plot_weights:
                If true, plots the weight if they are available
            plot_posterior:
                If true, plots the log posterior if they are available
            log_weight:
                Whether to display weights in log space or not. If None, the value is
                inferred by the mean weights of the plotted chains.

        Returns:
            the matplotlib figure created

        """

        base = self._sanitise(
            chains,
            columns,
            self.config.extents,
            blind=self.config.blind,
            log_scales=self.config.log_scales,
        )

        n = len(base.columns)
        extra = 0

        plot_posterior = plot_posterior and any([c.log_posterior is not None for c in base.chains])
        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            fig_size = (8, 0.75 + (n + extra))
        elif isinstance(figsize, float | int):
            fig_size = (figsize, figsize)
        else:
            fig_size = figsize

        fig, axes = plt.subplots(figsize=fig_size, nrows=n + extra, squeeze=False, sharex=True)
        max_points = 100000
        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = base.columns[i - extra]
                for chain in base.chains:
                    if p in chain.data_columns:
                        chain_row = chain.get_data(p)
                        if len(chain_row) > max_points:
                            chain_row = chain_row[:: int(len(chain_row) / max_points)]
                        log = p in base.log_scales
                        self._plot_walk(
                            ax,
                            p,
                            chain_row,
                            extents=base.extents.get(p),
                            convolve=convolve,
                            color=colors.format(chain.color),
                            log_scale=log,
                        )
                for truth in self.parent._truths:
                    if p in truth.location:
                        self._plot_walk_truth(ax, truth, p)

                if p in base.blind:
                    ax.set_yticks([])
            else:  # noqa: PLR5501
                if i == 0 and plot_posterior:
                    for chain in base.chains:
                        if chain.log_posterior is not None:
                            posterior = chain.log_posterior - chain.log_posterior.max()
                            if len(posterior) > max_points:
                                posterior = posterior[:: int(len(posterior) / max_points)]

                            self._plot_walk(
                                ax,
                                r"$\log(P)$",
                                posterior,
                                convolve=convolve,
                                color=colors.format(chain.color),
                            )
                else:
                    label = r"$\log_{10}$Weight" if log_weight else "Weight"

                    for chain in base.chains:
                        if chain.weights is not None:
                            weights = chain.weights
                            if len(weights) > max_points:
                                weights = weights[:: int(len(weights) / max_points)]
                            self._plot_walk(
                                ax,
                                label,
                                np.log10(weights) if log_weight else weights,  # type: ignore
                                convolve=convolve,
                                color=colors.format(chain.color),
                            )

        add_watermark(fig, None, fig_size, self.config, size_scale=0.8)
        self._save_fig(fig, filename, dpi=self.config.dpi)

        return fig

    def plot_distributions(
        self,
        chains: list[ChainName | Chain] | None = None,
        columns: list[ColumnName] | None = None,
        filename: list[str | Path] | str | Path | None = None,
        col_wrap: int = 4,
        figsize: float | tuple[float, float] | None = None,
    ) -> Figure:  # pragma: no cover
        """Plots the 1D parameter distributions for verification purposes.

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains give well behaved distributions, or if certain parameters are suspect
        or require a greater burn in period.

        Args:
            chains:
                Used to specify which chain to show if more than one chain is loaded in.
                Can be an integer, specifying the
                chain index, or a str, specifying the chain name.
            columns:
                If set, only creates a plot for those specific parameters (if list). If an
                integer is given, only plots the fist so many parameters.
            filename:
                If set, saves the figure to this location
            figsize:
                Scale horizontal and vertical figure size.
            col_wrap:
                How many columns to plot before wrapping.

        Returns:
            the matplotlib figure created

        """
        base = self._sanitise(
            chains,
            columns,
            self.config.extents,
            blind=self.config.blind,
            log_scales=self.config.log_scales,
        )

        n = len(base.columns)
        num_cols = min(n, col_wrap)
        num_rows = int(np.ceil(1.0 * n / col_wrap))

        if figsize is None:
            figsize = 1.0
        if isinstance(figsize, float | int):
            figsize_float = figsize
            figsize = (num_cols * 2.5 * figsize, num_rows * 2.5 * figsize)
        else:
            figsize_float = 1.0

        summary = self.config.summarise and len(base.chains) == 1
        hspace = (0.8 if summary else 0.5) / figsize_float
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=hspace)

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))

        for i, ax in enumerate(axes.flatten()):
            if i >= n:
                ax.set_axis_off()
                continue
            p = base.columns[i]

            ax.set_yticks([])
            if p in base.log_scales:
                ax.set_xscale("log")
            if p in base.blind:
                ax.set_xticks([])
            else:
                if self.config.diagonal_tick_labels:
                    _ = [label.set_rotation(45) for label in ax.get_xticklabels()]
                _ = [label.set_fontsize(self.config.tick_font_size) for label in ax.get_xticklabels()]

                if p in base.log_scales:
                    ax.xaxis.set_major_locator(LogLocator(numticks=self.config.max_ticks))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(self.config.max_ticks, prune="lower"))
                    ax.xaxis.set_major_formatter(formatter)
            ax.set_xlim(base.extents.get(p) or self._get_parameter_extents(p, base.chains))

            max_val = -np.inf
            for chain in base.chains:
                if not chain.plot_contour:
                    continue
                if p in chain.plotting_columns:
                    param_summary = summary and p not in base.blind
                    m = self._plot_bars(ax, p, chain, summary=param_summary)
                    if max_val is None or m > max_val:
                        max_val = m
            plot_truths(ax, self.parent._truths, py=p)
            ax.set_ylim(0, 1.1 * max_val)
            ax.set_xlabel(p, fontsize=self.config.label_font_size)

        add_watermark(fig, None, figsize, self.config, size_scale=0.8)
        self._save_fig(fig, filename, dpi=self.config.dpi)
        return fig

    def plot_summary(
        self,
        chains: list[ChainName | Chain] | None = None,
        columns: list[ColumnName] | None = None,
        filename: list[str | Path] | str | Path | None = None,
        figsize: float = 1.0,
        errorbar: bool = False,
        extra_parameter_spacing: float = 1.0,
        vertical_spacing_ratio: float = 1.0,
    ) -> Figure:  # pragma: no cover
        """Plots parameter summaries

        This plot is more for a sanity or consistency check than for use with final results.
        Plotting this before plotting with :func:`plot` allows you to quickly see if the
        chains give well behaved distributions, or if certain parameters are suspect
        or require a greater burn in period.

        Args:
            chains:
                Used to specify which chain to show if more than one chain is loaded in.
                Can be an integer, specifying the
                chain index, or a str, specifying the chain name.
            columns:
                If set, only creates a plot for those specific parameters (if list). If an
                integer is given, only plots the fist so many parameters.
            filename:
                If set, saves the figure to this location
            figsize:
                Scale horizontal and vertical figure size.
            errorbar:
                Whether to onle plot an error bar, instead of the marginalised distribution.
            include_truth_chain:
                If you specify another chain as the truth chain, determine if it should still
                be plotted.
            extra_parameter_spacing:
                Increase horizontal space for parameter values
            vertical_spacing_ratio:
                Increase vertical space for each model
        Returns:
            the matplotlib figure created

        """
        wide_extents = not errorbar
        base = self._sanitise(
            chains,
            columns,
            self.config.extents,
            blind=self.config.blind,
            log_scales=self.config.log_scales,
            wide_extents=wide_extents,
        )

        # We have a bit of fun to go from chain names to the width of the
        # subplot used to display said names
        max_param = self._get_size_of_texts(base.columns)
        fid_dpi = 65  # Seriously I have no idea what value this should be
        param_width = extra_parameter_spacing + max(0.5, max_param / fid_dpi)
        max_model_name = self._get_size_of_texts([chain.name for chain in base.chains])
        model_width = 0.25 + (max_model_name / fid_dpi)
        gridspec_kw = {
            "width_ratios": [model_width] + [param_width] * len(base.columns),
            "height_ratios": [1] * len(base.chains),
        }
        ncols = 1 + len(base.columns)
        top_spacing = 0.3
        bottom_spacing = 0.2
        row_height = (0.5 if errorbar else 0.8) * vertical_spacing_ratio
        width = param_width * len(base.columns) + model_width
        height = top_spacing + bottom_spacing + row_height * len(base.chains)
        top_ratio = 1 - (top_spacing / height)
        bottom_ratio = bottom_spacing / height

        fig_size = (width * figsize, height * figsize)
        fig, axes = plt.subplots(
            nrows=len(base.chains), ncols=ncols, figsize=fig_size, squeeze=False, gridspec_kw=gridspec_kw
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=top_ratio, bottom=bottom_ratio, wspace=0.0, hspace=0.0)
        label_font_size = self.config.label_font_size
        legend_color_text = self.config.legend_color_text

        max_vals: dict[ColumnName, float] = {}
        num_chains = len(base.chains)
        for i, axes_row in enumerate(axes):
            chain = base.chains[i]
            colour = colors.format(chain.color)

            # First one put name of model
            ax_first = axes_row[0]
            ax_first.set_axis_off()
            text_colour = "k" if not legend_color_text else colour
            ax_first.text(
                0,
                0.5,
                chain.name,
                transform=ax_first.transAxes,
                fontsize=label_font_size,
                verticalalignment="center",
                color=text_colour,
                weight="medium",
            )
            axes_for_summaries = axes_row[1:]

            for ax, p in zip(axes_for_summaries, base.columns):
                # Set up the frames
                if i > 0:
                    ax.spines["top"].set_visible(False)
                if i < (num_chains - 1):
                    ax.spines["bottom"].set_visible(False)
                if i < (num_chains - 1) or p in base.blind:
                    ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(base.extents[p])
                if p in base.log_scales:
                    ax.set_xscale("log")

                # Put title in
                if i == 0:
                    ax.set_title(self.config.get_label(p), fontsize=label_font_size)

                # Add truth values
                for truth in self.parent._truths:
                    truth_value = truth.location.get(p)
                    if truth_value is not None:
                        ax.axvline(truth_value, **truth._kwargs)

                # Skip if this chain doesnt have the parameter
                if p not in chain.data_columns:
                    continue

                # Plot the good stuff
                if errorbar:
                    fv = self.parent.analysis.get_parameter_summary(chain, p)
                    if fv is None or fv.all_none:
                        continue
                    if fv.lower is not None and fv.upper is not None:
                        diff = np.abs(np.diff(fv.array))
                        ax.errorbar([fv.center], 0, xerr=[[diff[0]], [diff[1]]], fmt="o", color=colour)
                else:
                    m = self._plot_bars(ax, p, chain)
                    if max_vals.get(p) is None or m > max_vals[p]:
                        max_vals[p] = m

        for i, axes_row in enumerate(axes):
            for ax, p in zip(axes_row, base.columns):
                if not errorbar:
                    ax.set_ylim(0, 1.1 * max_vals[p])

        add_watermark(fig, None, fig_size, self.config, size_scale=0.8)
        self._save_fig(fig, filename, dpi=self.config.dpi)

        return fig

    def _get_size_of_texts(self, texts: list[str]) -> float:  # pragma: no cover
        usetex = self.config.usetex
        size = self.config.label_font_size
        widths = [TextPath((0, 0), text, usetex=usetex, size=size).get_extents().width for text in texts]
        return max(widths)

    def _sanitise_columns(self, columns: list[ColumnName] | None, chains: list[Chain]) -> list[ColumnName]:
        if columns is None:
            res = []  # Doing it without set to preserve order
            for chain in chains:
                for column in chain.plotting_columns:
                    if column not in res:
                        res.append(column)
            return res
        return columns

    def _sanitise_logscale(self, log_scales: list[ColumnName] | None) -> list[ColumnName]:
        # We could at some point determine if something should be a log scale by analyising
        # its distribution, but for now assume its all linear
        if log_scales is None:
            return []
        return log_scales

    def _sanitise_blinds(self, blind: bool | list[ColumnName] | None, columns: list[ColumnName]) -> list[ColumnName]:
        if blind is None or blind is False:
            return []
        elif blind is True:
            return columns
        return blind

    def _sanitise(
        self,
        chains: list[ChainName | Chain] | None,
        columns: list[ColumnName] | None,
        extents: dict[str, tuple[float, float]] | None,
        blind: bool | list[ColumnName] | None = None,
        log_scales: list[ColumnName] | None = None,
        wide_extents: bool = True,
    ) -> PlottingBase:
        final_chains = self._sanitise_chains(chains)
        final_columns = self._sanitise_columns(columns, final_chains)
        extents = self._get_custom_extents(final_columns, final_chains, extents, wide_extents=wide_extents)
        self.set_rc_params()

        return PlottingBase(
            chains=final_chains,
            columns=final_columns,
            extents=extents,
            log_scales=self._sanitise_logscale(log_scales),
            blind=self._sanitise_blinds(blind, final_columns),
        )

    def set_rc_params(self) -> None:
        if self.config.usetex:
            plt.rc("text", usetex=True)
        else:
            plt.rc("text", usetex=False)
        if self.config.serif:
            plt.rc("font", family="serif")
        else:
            plt.rc("font", family="sans-serif")

    def restore_rc_params(self):
        """Restores the matplotlib rc parameters modified by usetex and serif.

        Unfortunately this cannot be automated because you cannot invoke it whilst you have
        an active figure object or matplotlib will destroy you. So do all your plotting, close
        the plots, and then you can call this.
        """
        plt.rc("text", usetex=self.usetex_old)
        plt.rc("font", family=self.serif_old)

    def _get_custom_extents(
        self,
        columns: list[ColumnName],
        chains: list[Chain],
        initial_extents: dict[ColumnName, tuple[float, float]] | None,
        wide_extents: bool = True,
    ) -> dict[ColumnName, tuple[float, float]]:  # pragma: no cover
        if initial_extents is None:
            initial_extents = {}
        extents = {} | initial_extents
        for p in columns:
            if p not in initial_extents:
                extents[p] = self._get_parameter_extents(p, chains, wide_extents=wide_extents)
        return extents

    def _get_triangle_figure(
        self, base: PlottingBase, figsize: tuple[float, float]
    ) -> tuple[Figure, np.ndarray, list[ColumnName], list[ColumnName]]:
        n = len(base.columns)
        if not self.config.plot_hists:
            n -= 1

        spacing = self.config.spacing
        if spacing is None:
            spacing = 1.0 if n < 6 else 0.0

        gridspec_kw = {}
        if n == 2 and self.config.plot_hists and self.config.flip:
            gridspec_kw = {"width_ratios": [3, 1], "height_ratios": [1, 3]}

        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
        min_left_for_axes = min(max(0.85 / figsize[0], 0.1), 0.3)
        min_bottom_for_axes = min(max(0.85 / figsize[1], 0.1), 0.3)
        fig.subplots_adjust(
            left=min_left_for_axes,
            right=0.95,
            top=0.9,
            bottom=min_bottom_for_axes,
            wspace=0.05 * spacing,
            hspace=0.05 * spacing,
        )

        if self.config.plot_hists:
            params_x = base.columns
            params_y = base.columns
        else:
            params_x = base.columns[1:]
            params_y = base.columns[:-1]
        for i, p1 in enumerate(params_x):
            for j, p2 in enumerate(params_y):
                ax = axes[i, j]
                formatter_x = ScalarFormatter(useOffset=True)
                formatter_x.set_powerlimits((-3, 4))
                formatter_y = ScalarFormatter(useOffset=True)
                formatter_y.set_powerlimits((-3, 4))

                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    logx = False
                    logy = False
                    if p1 == p2:
                        if p1 in base.log_scales:
                            if self.config.flip and j == n - 1:
                                ax.set_yscale("log")
                                logy = True
                            else:
                                ax.set_xscale("log")
                                logx = True
                    else:
                        if p1 in base.log_scales:
                            ax.set_yscale("log")
                            logy = True
                        if p2 in base.log_scales:
                            ax.set_xscale("log")
                            logx = True
                    if i != n - 1 or (self.config.flip and j == n - 1):
                        ax.set_xticks([])
                    else:
                        if p2 in base.blind:
                            ax.set_xticks([])
                        else:
                            display_x_ticks = True
                        if isinstance(p2, str):
                            ax.set_xlabel(self.config.get_label(p2), fontsize=self.config.label_font_size)
                    if j != 0 or (self.config.plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        if p1 in base.blind:
                            ax.set_yticks([])
                        else:
                            display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(self.config.get_label(p1), fontsize=self.config.label_font_size)
                    if display_x_ticks:
                        if self.config.diagonal_tick_labels:
                            _ = [label.set_rotation(45) for label in ax.get_xticklabels()]
                        _ = [label.set_fontsize(self.config.tick_font_size) for label in ax.get_xticklabels()]
                        if not logx:
                            ax.xaxis.set_major_locator(MaxNLocator(self.config.max_ticks, prune="lower"))
                            ax.xaxis.set_major_formatter(formatter_x)
                        else:
                            ax.xaxis.set_major_locator(LogLocator(numticks=self.config.max_ticks))
                    else:
                        ax.set_xticks([])
                    if display_y_ticks:
                        if self.config.diagonal_tick_labels:
                            _ = [label.set_rotation(45) for label in ax.get_yticklabels()]
                        _ = [label.set_fontsize(self.config.tick_font_size) for label in ax.get_yticklabels()]
                        if not logy:
                            ax.yaxis.set_major_locator(MaxNLocator(self.config.max_ticks, prune="lower"))
                            ax.yaxis.set_major_formatter(formatter_y)
                        else:
                            ax.yaxis.set_major_locator(LogLocator(numticks=self.config.max_ticks))
                    else:
                        ax.set_yticks([])
                    if (i != j or not self.config.plot_hists) or (self.config.flip and i == 1):
                        ax.set_ylim(base.extents[p1])
                    ax.set_xlim(base.extents[p2])

        return fig, axes, params_x, params_y

    def _get_parameter_extents(
        self, column: ColumnName, chains: list[Chain], wide_extents: bool = True
    ) -> tuple[float, float]:
        min_val, max_val = np.inf, -np.inf
        for chain in chains:
            if column not in chain.samples:
                continue  # pragma: no cover

            data = chain.get_data(column)
            min_prop, max_prop = np.inf, -np.inf
            if chain.plot_contour or chain.plot_cloud:
                if chain.grid:
                    min_prop = data.min()
                    max_prop = data.max()
                else:
                    min_prop, max_prop = get_extents(data, chain.weights, plot=True, wide_extents=wide_extents)

            else:
                point = chain.get_max_posterior_point()
                if point is not None and column in point.coordinate:
                    min_prop = point.coordinate[column]
                    max_prop = min_prop

            if min_prop < min_val:
                min_val = min_prop
            if max_prop > max_val:
                max_val = max_prop

        return min_val, max_val

    def _sanitise_chains(
        self, chains: list[Chain | ChainName] | dict[ChainName, Chain] | None, include_skip: bool = False
    ) -> list[Chain]:
        overriden_chains = self.parent._get_final_chains()
        final_chains = []
        if isinstance(chains, list):
            final_chains = [overriden_chains[c if isinstance(c, ChainName) else c.name] for c in chains]
        elif isinstance(chains, dict):
            final_chains = [overriden_chains[c.name] for c in chains.values()]
        else:
            final_chains = list(overriden_chains.values())
        return [c for c in final_chains if include_skip or not c.skip]

    def _plot_bars(
        self, ax: Axes, column: str, chain: Chain, flip: bool = False, summary: bool = False
    ) -> float:  # pragma: no cover
        # Get values from config
        data = chain.get_data(column)
        if chain.smooth or chain.kde:
            xs, ys, _ = self.parent.analysis._get_smoothed_histogram(chain, column, pad=True)
            if flip:
                ax.plot(ys, xs, color=chain.color, ls=chain.linestyle, lw=chain.linewidth, zorder=chain.zorder)
            else:
                ax.plot(xs, ys, color=chain.color, ls=chain.linestyle, lw=chain.linewidth, zorder=chain.zorder)
        else:
            if chain.grid:
                bins = get_grid_bins(data)
            else:
                bins, _ = get_smoothed_bins(chain.smooth, get_bins(chain), data, chain.weights)
            hist, edges = np.histogram(data, bins=bins, density=True, weights=chain.weights)
            if chain.power is not None:
                hist = hist**chain.power
            edge_center = 0.5 * (edges[:-1] + edges[1:])
            xs, ys = edge_center, hist
            ax.hist(
                xs,
                weights=ys,
                bins=bins,  # type: ignore
                histtype="step",
                color=chain.color,  # type: ignore
                orientation="horizontal" if flip else "vertical",
                ls=chain.linestyle,
                lw=chain.linewidth,
                zorder=chain.zorder,
            )
        interp_type = "linear" if chain.smooth else "nearest"
        interpolator = interp1d(xs, ys, kind=interp_type)

        if chain.bar_shade:
            fit_values = self.parent.analysis.get_parameter_summary(chain, column)
            if fit_values is not None:
                lower = fit_values.lower
                upper = fit_values.upper
                if lower is not None and upper is not None:
                    if lower < xs.min():
                        lower = xs.min()
                    if upper > xs.max():
                        upper = xs.max()
                    x = np.linspace(lower, upper, 1000)  # type: ignore
                    if flip:
                        ax.fill_betweenx(
                            x,
                            np.zeros(x.shape),
                            interpolator(x),
                            color=chain.color,
                            alpha=0.2,
                            zorder=chain.zorder,
                        )
                    else:
                        ax.fill_between(
                            x,
                            np.zeros(x.shape),
                            interpolator(x),
                            color=chain.color,
                            alpha=0.2,
                            zorder=chain.zorder,
                        )
                    if summary:
                        t = self.parent.analysis.get_parameter_text(fit_values)
                        label = self.config.get_label(column)
                        if isinstance(column, str):
                            ax.set_title(
                                r"${} = {}$".format(label.strip("$"), t), fontsize=self.config.summary_font_size
                            )
                        else:
                            ax.set_title(r"$%s$" % t, fontsize=self.config.summary_font_size)
        return float(ys.max())

    def _plot_walk(
        self,
        ax: Axes,
        column: ColumnName,
        data: pd.Series,
        extents: tuple[float, float] | None = None,
        convolve: int | None = None,
        color: str | None = None,
        log_scale: bool = False,
    ) -> None:  # pragma: no cover
        if extents is not None:
            ax.set_ylim(extents)
        assert convolve is None or isinstance(convolve, int), "Convolve must be an integer pixel window width"
        x = np.arange(data.size)
        ax.set_xlim(0, x[-1])
        ax.set_ylabel(self.config.get_label(column))
        if color is None:
            color = "#0345A1"
        ax.scatter(x, data, c=color, s=2, marker=".", edgecolors="none", alpha=0.5)
        max_ticks = self.config.max_ticks
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(numticks=max_ticks))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))

        if convolve is not None:
            trim = int(0.5 * convolve)
            color2 = colors.scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[trim:-trim], filtered[trim:-trim], color=color2, alpha=1)

    def _plot_walk_truth(self, ax: Axes, truth: Truth, col: str) -> None:
        ax.axhline(truth.location[col], **truth._kwargs)


if __name__ == "__main__":
    from .chainconsumer import ChainConsumer
