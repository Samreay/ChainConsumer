from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
from numpy import meshgrid
from pydantic import Field
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from chainconsumer.truth import Truth

from .base import BetterBase
from .chain import Chain, ChainName, ColumnName
from .colors import ColorInput, colors
from .helpers import get_extents, get_grid_bins, get_smoothed_bins
from .kde import MegKDE
from .log import logger


class PlottingBase(BetterBase):
    chains: list[Chain]
    columns: list[ColumnName]
    extents: dict[ColumnName, tuple[float, float]]
    blind: list[ColumnName]
    log_scales: list[ColumnName]


class PlotConfig(BetterBase):
    labels: dict[ColumnName, str] = Field(default={}, description="Labels for parameters")
    sigma2d: bool | None = Field(
        default=None,
        description=(
            "Whether to use 2D sigmas for summary statistics. Ie in 2D a 1sigma contour"
            r" does *not* encapsulate 68% of the volume, it covers 39.3% of the volume."
        ),
    )
    max_ticks: int = Field(default=5, ge=0, description="Maximum number of ticks to use on axes")
    plot_hists: bool = Field(default=True, description="Whether to plot the 1D histograms")
    flip: bool = Field(default=False, description="Whether to flip the 1D histograms")
    serif: bool = Field(default=False, description="Whether to use a serif font")
    usetex: bool = Field(default=False, description="Whether to use LaTeX for text rendering")
    diagonal_tick_labels: bool = Field(default=True, description="Whether to show tick labels on the diagonal")
    label_font_size: int = Field(default=12, ge=0, description="Font size for axis labels")
    tick_font_size: int = Field(default=10, ge=0, description="Font size for axis ticks")
    spacing: float = Field(default=None, ge=0, description="Spacing between subplots")
    contour_label_font_size: int = Field(default=10, ge=0, description="Font size for contour labels")
    legend_kwargs: dict[str, Any] = Field(default={}, description="Kwargs to pass to the legend")
    legend_location: tuple[int, int] | None = Field(default=None, description="Which subplot to put the legend in")
    legend_artists: bool | None = Field(default=None, description="Whether to show artists in the legend")
    legend_color_text: bool = Field(default=True, description="Whether to color the legend text")
    watermark_text_kwargs: dict[str, Any] = Field(default={}, description="Kwargs to pass to the watermark text")
    title_size: int = Field(default=14, ge=0, description="Font size for titles")
    summarise: bool = Field(default=True, description="Whether to annotate the plot with summary statistics")

    @property
    def legend_kwargs_final(self) -> dict[str, Any]:
        default = {
            "labelspacing": 0.3,
            "loc": "upper right",
            "frameon": False,
            "fontsize": self.label_font_size,
            "handlelength": 1,
            "handletextpad": 0.2,
            "borderaxespad": 0.0,
        }
        return default | self.legend_kwargs

    @property
    def watermark_text_kwargs_final(self) -> dict[str, Any]:
        default = {
            "color": "#333333",
            "alpha": 0.7,
            "verticalalignment": "center",
            "horizontalalignment": "center",
        }
        return default | self.watermark_text_kwargs

    def get_label(self, column: ColumnName) -> str:
        return self.labels.get(column, column)


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
        return grow_factor * 1.5 * num_columns + (1 if has_cax else 0), grow_factor * 1.5 * num_columns


def get_artists_from_chains(chains: list[Chain]):
    artists = []
    for chain in chains:
        if chain.plot_contour and not chain.plot_point:
            artists.append(
                Line2D((0, 1), (0, 0), color=colors.format(chain.color), ls=chain.linestyle, lw=chain.linewidth)
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
                    markersize=chain.marker_size,
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
                    markersize=chain.marker_size,
                )
            )
    return artists


class Plotter:
    def __init__(self, parent):
        self.parent: "ChainConsumer" = parent
        self._config: PlotConfig | None = None
        self._default_config = PlotConfig()

        self.usetex_old = matplotlib.rcParams["text.usetex"]
        self.serif_old = matplotlib.rcParams["font.family"]

    def set_config(self, config: PlotConfig):
        self._config = config

    @property
    def config(self) -> PlotConfig:
        if self._config is None:
            return self._default_config
        return self._config

    def plot(
        self,
        figsize: FigSize | float | int | tuple[float, float] = FigSize.GROW,
        columns: list[ColumnName] | None = None,
        chains: list[ChainName | Chain] | None = None,
        extents: dict[ColumnName, tuple[float, float]] | None = None,
        filename: list[str | Path] | str | Path | None = None,
        display: bool = False,
        show_legend: bool | None = None,
        blind: bool | list[str] | None = None,
        watermark: str | None = None,
        log_scales: list[ColumnName] | None = None,
    ):  # pragma: no cover
        """Plot the chain!

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
        columns : list[str], optional
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
        log_scales : bool, list[ColumnName], optional
            Whether or not to use a log scale on any given axis. Can be a list of True/False, a list of param
            names to set to true, a dictionary of param names with true/false
            or just a bool (just `True` would set everything to log scales).

        Returns
        -------
        figure
            the matplotlib figure

        """

        base = self._sanitise(chains, columns, extents, blind=blind, log_scales=log_scales)

        if show_legend is None:
            show_legend = len(base.chains) > 1

        num_cax = len(set([chain.color_param for chain in base.chains if chain.color_param is not None]))
        fig_size = FigSize.get_size(figsize, len(base.columns), num_cax > 0)
        plot_hists = self.config.plot_hists
        flip = len(base.columns) == 2 and plot_hists and self.config.flip
        fig, axes, params_x, params_y = self._get_figure(base, figsize=fig_size)

        axl = axes.ravel().tolist()
        summarise = self.config.summarise and len(base.chains) == 1

        cbar_done = []
        for i, p1 in enumerate(params_x):
            for j, p2 in enumerate(params_y):
                if i < j:
                    continue
                ax: Axes = axes[i, j]
                do_flip = flip and i == len(params_x) - 1

                # Plot the histograms
                if plot_hists and i == j:
                    for truth in self.parent.truths:
                        if do_flip:
                            self._add_truth(ax, truth, px=p1)
                        else:
                            self._add_truth(ax, truth, py=p2)
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
                    for chain in base.chains:
                        if p1 not in chain.samples or p2 not in chain.samples:
                            continue
                        if not chain.plot_contour or chain.show_as_1d_prior:
                            continue

                        h = self._plot_contour(ax, chain, p1, p2)
                        cp = chain.color_param
                        if h is not None and cp is not None and cp not in cbar_done:
                            cbar_done.append(cp)
                            aspect = fig_size[1] / 0.15
                            fraction = 0.85 / fig_size[0]
                            cbar = fig.colorbar(h, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                            label = self.config.get_label(cp)
                            if label == "weights":
                                label = "Weights"
                            elif label == "log_weights":
                                label = "log(Weights)"
                            elif label == "posterior":
                                label = "log(Posterior)"
                            cbar.set_label(label, fontsize=self.config.label_font_size)
                            if cbar.solids is not None:
                                cbar.solids.set(alpha=1)

                    for truth in self.parent.truths:
                        self._add_truth(ax, truth, px=p1, py=p2)

        legend_location = self.config.legend_location
        if legend_location is None:
            legend_location = (0, -1) if not flip or len(base.columns) > 2 else (-1, 0)
        legend_outside = legend_location[0] >= legend_location[1]

        if show_legend:
            ax = axes[legend_location[0], legend_location[1]]
            legend_kwargs = self.config.legend_kwargs_final.copy()
            if "markerfirst" not in legend_kwargs:
                legend_kwargs["markerfirst"] = legend_outside or not self.config.legend_artists

            artists = get_artists_from_chains(base.chains)
            leg = ax.legend(artists, **legend_kwargs)
            if self.config.legend_color_text:
                for text, chain in zip(leg.get_texts(), base.chains):
                    text.set_fontweight("medium")
                    text.set_color(colors.format(chain.color))

            # TODO: This seems like behaviour which no longer works
            # if not legend_outside:
            #     loc = legend_kwargs.get("loc") or ""
            #     if isinstance(loc, str) and "right" in loc.lower():
            #         vp = leg._legend_box._children[-1]._children[0]
            #         vp.align = "right"

        fig.canvas.draw()
        for ax in axes[-1, :]:
            offset = ax.get_xaxis().get_offset_text()
            ax.set_xlabel("{} {}".format(ax.get_xlabel(), f"[{offset.get_text()}]" if offset.get_text() else ""))
            offset.set_visible(False)
        for ax in axes[:, 0]:
            offset = ax.get_yaxis().get_offset_text()
            ax.set_ylabel("{} {}".format(ax.get_ylabel(), f"[{offset.get_text()}]" if offset.get_text() else ""))
            offset.set_visible(False)

        dpi = 300
        if watermark:
            ax_watermark = axes[-1, 0] if flip and len(base.columns) == 2 else None
            self._add_watermark(fig, ax_watermark, fig_size, watermark, dpi=dpi)

        if filename is not None:
            if not isinstance(filename, list):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, dpi)
        if display:
            plt.show()

        return fig

    def _save_fig(self, fig: Figure, filename: str | Path, dpi: int) -> None:  # pragma: no cover
        fig.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True, pad_inches=0.05)

    def _add_watermark(
        self, fig: Figure, axes: Axes | None, fig_size: tuple[float, float], text: str, dpi=300, size_scale: float = 1.0
    ) -> None:  # pragma: no cover
        # Code based off github repository https://github.com/cpadavis/preliminize
        dx, dy = fig_size
        dy, dx = dy * dpi, dx * dpi
        rotation = 180 / np.pi * np.arctan2(-dy, dx)
        property_dict = self.config.watermark_text_kwargs

        keys_in_font_dict = ["family", "style", "variant", "weight", "stretch", "size"]
        fontdict = {k: property_dict[k] for k in keys_in_font_dict if k in property_dict}
        font_prop = FontProperties(**fontdict)
        usetex = property_dict.get("usetex", self.config.usetex)
        if usetex:
            px, py, scale = 0.5, 0.5, 1.0
        else:
            px, py, scale = 0.5, 0.5, 0.8

        bb0 = TextPath((0, 0), text, size=50, prop=font_prop, usetex=usetex).get_extents()
        bb1 = TextPath((0, 0), text, size=51, prop=font_prop, usetex=usetex).get_extents()
        dw = (bb1.width - bb0.width) * (dpi / 100)
        dh = (bb1.height - bb0.height) * (dpi / 100)
        size = np.sqrt(dy**2 + dx**2) / (dh * abs(dy / dx) + dw) * 0.6 * scale * size_scale
        if axes is not None:
            if usetex:
                size *= 0.7
            else:
                size *= 0.8
        size = int(size)
        if axes is None:
            fig.text(px, py, text, fontdict=property_dict, rotation=rotation, fontsize=size)
        else:
            axes.text(px, py, text, transform=axes.transAxes, fontdict=property_dict, rotation=rotation, fontsize=size)

    # def plot_walks(
    #     self,
    #     parameters=None,
    #     truth=None,
    #     extents=None,
    #     display=False,
    #     filename=None,
    #     chains=None,
    #     convolve=None,
    #     figsize=None,
    #     plot_weights=True,
    #     plot_posterior=True,
    #     log_weight=None,
    #     log_scales=None,
    # ):  # pragma: no cover
    #     """Plots the chain walk; the parameter values as a function of step index.

    #     This plot is more for a sanity or consistency check than for use with final results.
    #     Plotting this before plotting with :func:`plot` allows you to quickly see if the
    #     chains are well behaved, or if certain parameters are suspect
    #     or require a greater burn in period.

    #     The desired outcome is to see an unchanging distribution along the x-axis of the plot.
    #     If there are obvious tails or features in the parameters, you probably want
    #     to investigate.

    #     Parameters
    #     ----------
    #     parameters : list[str]|int, optional
    #         Specify a subset of parameters to plot. If not set, all parameters are plotted.
    #         If an integer is given, only the first so many parameters are plotted.
    #     truth : list[float]|dict[str], optional
    #         A list of truth values corresponding to parameters, or a dictionary of
    #         truth values keyed by the parameter.
    #     extents : list[tuple]|dict[str], optional
    #         A list of two-tuples for plot extents per parameter, or a dictionary of
    #         extents keyed by the parameter.
    #     display : bool, optional
    #         If set, shows the plot using ``plt.show()``
    #     filename : str, optional
    #         If set, saves the figure to the filename
    #     chains : int|str, list[str|int], optional
    #         Used to specify which chain to show if more than one chain is loaded in.
    #         Can be an integer, specifying the
    #         chain index, or a str, specifying the chain name.
    #     convolve : int, optional
    #         If set, overplots a smoothed version of the steps using ``convolve`` as
    #         the width of the smoothing filter.
    #     figsize : tuple, optional
    #         If set, sets the created figure size.
    #     plot_weights : bool, optional
    #         If true, plots the weight if they are available
    #     plot_posterior : bool, optional
    #         If true, plots the log posterior if they are available
    #     log_weight : bool, optional
    #         Whether to display weights in log space or not. If None, the value is
    #         inferred by the mean weights of the plotted chains.
    #     log_scales : bool, list[bool] or dict[bool], optional
    #         Whether or not to use a log scale on any given axis. Can be a list of True/False, a list of param
    #         names to set to true, a dictionary of param names with true/false
    #         or just a bool (just `True` would set everything to log scales).

    #     Returns
    #     -------
    #     figure
    #         the matplotlib figure created

    #     """

    #     chains, parameters, truth, extents, _, log_scales = self._sanitise(
    #         chains, parameters, truth, extents, log_scales=log_scales
    #     )

    #     chains = [c for c in chains if c.mcmc_chain]
    #     n = len(parameters)
    #     extra = 0
    #     if plot_weights:
    #         plot_weights = plot_weights and np.any([np.any(c.weights != 1.0) for c in chains])

    #     plot_posterior = plot_posterior and np.any([c.posterior is not None for c in chains])

    #     if plot_weights:
    #         extra += 1
    #     if plot_posterior:
    #         extra += 1

    #     if figsize is None:
    #         figsize = (8, 0.75 + (n + extra))

    #     fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)

    #     for i, axes_row in enumerate(axes):
    #         ax = axes_row[0]
    #         if i >= extra:
    #             p = parameters[i - n]
    #             for chain in chains:
    #                 if p in chain.parameters:
    #                     chain_row = chain.get_data(p)
    #                     log = log_scales.get(p, False)
    #                     self._plot_walk(
    #                         ax,
    #                         p,
    #                         chain_row,
    #                         extents=extents.get(p),
    #                         convolve=convolve,
    #                         color=chain.config["color"],
    #                         log_scale=log,
    #                     )
    #             if truth.get(p) is not None:
    #                 self._plot_walk_truth(ax, truth.get(p))
    #         else:
    #             if i == 0 and plot_posterior:
    #                 for chain in chains:
    #                     if chain.log_posterior is not None:
    #                         self._plot_walk(
    #                             ax,
    #                             r"$\log(P)$",
    #                             chain.log_posterior - chain.log_posterior.max(),
    #                             convolve=convolve,
    #                             color=chain.config["color"],
    #                         )
    #             else:
    #                 if log_weight is None:
    #                     log_weight = np.any([chain.weights.mean() < 0.1 for chain in chains])
    #                 if log_weight:
    #                     for chain in chains:
    #                         self._plot_walk(
    #                             ax,
    #                             r"$\log_{10}(w)$",
    #                             np.log10(chain.weights),
    #                             convolve=convolve,
    #                             color=chain.config["color"],
    #                         )
    #                 else:
    #                     for chain in chains:
    #                         self._plot_walk(ax, "$w$", chain.weights, convolve=convolve, color=chain.config["color"])

    #     if filename is not None:
    #         if isinstance(filename, str):
    #             filename = [filename]
    #         for f in filename:
    #             self._save_fig(fig, f, 300)
    #     if display:
    #         plt.show()

    #     return fig

    # def plot_distributions(
    #     self,
    #     parameters=None,
    #     truth=None,
    #     extents=None,
    #     display=False,
    #     filename=None,
    #     chains=None,
    #     col_wrap=4,
    #     figsize=None,
    #     blind=None,
    #     log_scales=None,
    # ):  # pragma: no cover
    #     """Plots the 1D parameter distributions for verification purposes.

    #     This plot is more for a sanity or consistency check than for use with final results.
    #     Plotting this before plotting with :func:`plot` allows you to quickly see if the
    #     chains give well behaved distributions, or if certain parameters are suspect
    #     or require a greater burn in period.

    #     Parameters
    #     ----------
    #     parameters : list[str]|int, optional
    #         Specify a subset of parameters to plot. If not set, all parameters are plotted.
    #         If an integer is given, only the first so many parameters are plotted.
    #     truth : list[float]|dict[str], optional
    #         A list of truth values corresponding to parameters, or a dictionary of
    #         truth values keyed by the parameter.
    #     extents : list[tuple]|dict[str], optional
    #         A list of two-tuples for plot extents per parameter, or a dictionary of
    #         extents keyed by the parameter.
    #     display : bool, optional
    #         If set, shows the plot using ``plt.show()``
    #     filename : str, optional
    #         If set, saves the figure to the filename
    #     chains : int|str, list[str|int], optional
    #         Used to specify which chain to show if more than one chain is loaded in.
    #         Can be an integer, specifying the
    #         chain index, or a str, specifying the chain name.
    #     col_wrap : int, optional
    #         How many columns to plot before wrapping.
    #     figsize : tuple(float)|float, optional
    #         Either a tuple specifying the figure size or a float scaling factor.
    #     blind : bool|string|list[string], optional
    #         Whether to blind axes values. Can be set to `True` to blind all parameters,
    #         or can pass in a string (or list of strings) which specify the parameters to blind.
    #     log_scales : bool, list[bool] or dict[bool], optional
    #         Whether or not to use a log scale on any given axis. Can be a list of True/False, a list of param
    #         names to set to true, a dictionary of param names with true/false
    #         or just a bool (just `True` would set everything to log scales).

    #     Returns
    #     -------
    #     figure
    #         the matplotlib figure created

    #     """
    #     chains, parameters, truth, extents, blind, log_scales = self._sanitise(
    #         chains, parameters, truth, extents, blind=blind, log_scales=log_scales
    #     )

    #     n = len(parameters)
    #     num_cols = min(n, col_wrap)
    #     num_rows = int(np.ceil(1.0 * n / col_wrap))

    #     if figsize is None:
    #         figsize = 1.0
    #     if isinstance(figsize, float):
    #         figsize_float = figsize
    #         figsize = (num_cols * 2 * figsize, num_rows * 2 * figsize)
    #     else:
    #         figsize_float = 1.0

    #     summary = self.parent.config["summary"]
    #     label_font_size = self.parent.config["label_font_size"]
    #     tick_font_size = self.parent.config["tick_font_size"]
    #     max_ticks = self.parent.config["max_ticks"]
    #     diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]

    #     if summary is None:
    #         summary = len(self.parent.chains) == 1

    #     hspace = (0.8 if summary else 0.5) / figsize_float
    #     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, squeeze=False)
    #     fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=hspace)

    #     formatter = ScalarFormatter(useOffset=False)
    #     formatter.set_powerlimits((-3, 4))

    #     for i, ax in enumerate(axes.flatten()):
    #         if i >= len(parameters):
    #             ax.set_axis_off()
    #             continue
    #         p = parameters[i]

    #         ax.set_yticks([])
    #         if log_scales.get(p, False):
    #             ax.set_xscale("log")
    #         if p in blind:
    #             ax.set_xticks([])
    #         else:
    #             if diagonal_tick_labels:
    #                 _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
    #             _ = [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]

    #             if log_scales.get(p, False):
    #                 ax.xaxis.set_major_locator(LogLocator(numticks=max_ticks))
    #             else:
    #                 ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
    #                 ax.xaxis.set_major_formatter(formatter)
    #         ax.set_xlim(extents.get(p) or self._get_parameter_extents(p, chains))

    #         max_val = -np.inf
    #         for chain in chains:
    #             if not chain.config["plot_contour"]:
    #                 continue
    #             if p in chain.parameters:
    #                 param_summary = summary and p not in blind
    #                 m = self._plot_bars(ax, p, chain, summary=param_summary)
    #                 if max_val is None or m > max_val:
    #                     max_val = m

    #         self._add_truth(ax, truth, None, py=p)
    #         ax.set_ylim(0, 1.1 * max_val)
    #         ax.set_xlabel(p, fontsize=label_font_size)

    #     if filename is not None:
    #         if isinstance(filename, str):
    #             filename = [filename]
    #         for f in filename:
    #             self._save_fig(fig, f, 300)
    #     if display:
    #         plt.show()

    #     return fig

    # def plot_summary(
    #     self,
    #     parameters=None,
    #     truth=None,
    #     extents=None,
    #     display=False,
    #     filename=None,
    #     chains=None,
    #     figsize=1.0,
    #     errorbar=False,
    #     include_truth_chain=True,
    #     blind=None,
    #     watermark=None,
    #     extra_parameter_spacing=0.5,
    #     vertical_spacing_ratio=1.0,
    #     show_names=True,
    #     log_scales=None,
    # ):  # pragma: no cover
    #     """Plots parameter summaries

    #     This plot is more for a sanity or consistency check than for use with final results.
    #     Plotting this before plotting with :func:`plot` allows you to quickly see if the
    #     chains give well behaved distributions, or if certain parameters are suspect
    #     or require a greater burn in period.

    #     Parameters
    #     ----------
    #     parameters : list[str]|int, optional
    #         Specify a subset of parameters to plot. If not set, all parameters are plotted.
    #         If an integer is given, only the first so many parameters are plotted.
    #     truth : list[float]|list|list[float]|dict[str]|str, optional
    #         A list of truth values corresponding to parameters, or a dictionary of
    #         truth values keyed by the parameter. Each "truth value" can be either a float (will
    #         draw a vertical line), two floats (a shaded interval) or three floats (min, mean, max),
    #         which renders as a shaded interval with a line for the mean. Or, supply a string
    #         which matches a chain name, and the results for that chain will be used as the 'truth'
    #     extents : list[tuple]|dict[str], optional
    #         A list of two-tuples for plot extents per parameter, or a dictionary of
    #         extents keyed by the parameter.
    #     display : bool, optional
    #         If set, shows the plot using ``plt.show()``
    #     filename : str, optional
    #         If set, saves the figure to the filename
    #     chains : int|str, list[str|int], optional
    #         Used to specify which chain to show if more than one chain is loaded in.
    #         Can be an integer, specifying the
    #         chain index, or a str, specifying the chain name.
    #     figsize : float, optional
    #         Scale horizontal and vertical figure size.
    #     errorbar : bool, optional
    #         Whether to onle plot an error bar, instead of the marginalised distribution.
    #     include_truth_chain : bool, optional
    #         If you specify another chain as the truth chain, determine if it should still
    #         be plotted.
    #     blind : bool|string|list[string], optional
    #         Whether to blind axes values. Can be set to `True` to blind all parameters,
    #         or can pass in a string (or list of strings) which specify the parameters to blind.
    #     watermark : str, optional
    #         A watermark to add to the figure
    #     extra_parameter_spacing : float, optional
    #         Increase horizontal space for parameter values
    #     vertical_spacing_ratio : float, optional
    #         Increase vertical space for each model
    #     show_names : bool, optional
    #         Whether to show chain names or not. Defaults to `True`.
    #     log_scales : bool, list[bool] or dict[bool], optional
    #         Whether or not to use a log scale on any given axis. Can be a list of True/False, a list of param
    #         names to set to true, a dictionary of param names with true/false
    #         or just a bool (just `True` would set everything to log scales).

    #     Returns
    #     -------
    #     figure
    #         the matplotlib figure created

    #     """
    #     wide_extents = not errorbar
    #     chains, parameters, truth, extents, blind, log_scales = self._sanitise(
    #         chains, parameters, truth, extents, blind=blind, wide_extents=wide_extents, log_scales=log_scales
    #     )

    #     all_names = [c.name for c in self.parent.chains]

    #     # Check if we're using a chain for truth values
    #     if isinstance(truth, str):
    #         assert truth in all_names, f"Truth chain {truth} is not in the list of added chains {all_names}"
    #         if not include_truth_chain:
    #             chains = [c for c in chains if c.name != truth]
    #         truth = self.parent.analysis.get_summary(chains=truth, parameters=parameters)

    #     max_param = self._get_size_of_texts(parameters)
    #     fid_dpi = 65  # Seriously I have no idea what value this should be
    #     param_width = extra_parameter_spacing + max(0.5, max_param / fid_dpi)

    #     if show_names:
    #         max_model_name = self._get_size_of_texts([chain.name for chain in chains])
    #         model_width = 0.25 + (max_model_name / fid_dpi)
    #         gridspec_kw = {
    #             "width_ratios": [model_width] + [param_width] * len(parameters),
    #             "height_ratios": [1] * len(chains),
    #         }
    #         ncols = 1 + len(parameters)
    #     else:
    #         model_width = 0
    #         gridspec_kw = {"width_ratios": [param_width] * len(parameters), "height_ratios": [1] * len(chains)}
    #         ncols = len(parameters)

    #     top_spacing = 0.3
    #     bottom_spacing = 0.2
    #     row_height = (0.5 if not errorbar else 0.3) * vertical_spacing_ratio
    #     width = param_width * len(parameters) + model_width
    #     height = top_spacing + bottom_spacing + row_height * len(chains)
    #     top_ratio = 1 - (top_spacing / height)
    #     bottom_ratio = bottom_spacing / height

    #     figsize = (width * figsize, height * figsize)
    #     fig, axes = plt.subplots(
    #         nrows=len(chains), ncols=ncols, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw
    #     )
    #     fig.subplots_adjust(left=0.05, right=0.95, top=top_ratio, bottom=bottom_ratio, wspace=0.0, hspace=0.0)
    #     label_font_size = self.parent.config["label_font_size"]
    #     legend_color_text = self.parent.config["legend_color_text"]

    #     max_vals = {}
    #     for i, row in enumerate(axes):
    #         chain = chains[i]

    #         (
    #             cs,
    #             ws,
    #             ps,
    #         ) = (
    #             chain.chain,
    #             chain.weights,
    #             chain.parameters,
    #         )
    #         gs, ns = chain.grid, chain.name

    #         colour = chain.config["color"]

    #         # First one put name of model
    #         if show_names:
    #             ax_first = row[0]
    #             ax_first.set_axis_off()
    #             text_colour = "k" if not legend_color_text else colour
    #             ax_first.text(
    #                 0,
    #                 0.5,
    #                 ns,
    #                 transform=ax_first.transAxes,
    #                 fontsize=label_font_size,
    #                 verticalalignment="center",
    #                 color=text_colour,
    #                 weight="medium",
    #             )
    #             cols = row[1:]
    #         else:
    #             cols = row

    #         for ax, p in zip(cols, parameters):
    #             # Set up the frames
    #             if i > 0:
    #                 ax.spines["top"].set_visible(False)
    #             if i < (len(chains) - 1):
    #                 ax.spines["bottom"].set_visible(False)
    #             if i < (len(chains) - 1) or p in blind:
    #                 ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.set_xlim(extents[p])
    #             if log_scales.get(p):
    #                 ax.set_xscale("log")

    #             # Put title in
    #             if i == 0:
    #                 ax.set_title(r"$%s$" % p, fontsize=label_font_size)

    #             # Add truth values
    #             truth_value = truth.get(p)
    #             if truth_value is not None:
    #                 if isinstance(truth_value, float | int):
    #                     truth_mean = truth_value
    #                     truth_min, truth_max = None, None
    #                 else:
    #                     if len(truth_value) == 1:
    #                         truth_mean = truth_value
    #                         truth_min, truth_max = None, None
    #                     elif len(truth_value) == 2:
    #                         truth_min, truth_max = truth_value
    #                         truth_mean = None
    #                     else:
    #                         truth_min, truth_mean, truth_max = truth_value
    #                 if truth_mean is not None:
    #                     ax.axvline(truth_mean, **self.parent.config_truth)
    #                 if truth_min is not None and truth_max is not None:
    #                     ax.axvspan(truth_min, truth_max, color=self.parent.config_truth["color"], alpha=0.15, lw=0)
    #             # Skip if this chain doesnt have the parameter
    #             if p not in ps:
    #                 continue

    #             # Plot the good stuff
    #             if errorbar:
    #                 fv = self.parent.analysis.get_parameter_summary(chain, p)
    #                 if fv[0] is not None and fv[2] is not None:
    #                     diff = np.abs(np.diff(fv))
    #                     ax.errorbar([fv[1]], 0, xerr=[[diff[0]], [diff[1]]], fmt="o", color=colour)
    #             else:
    #                 m = self._plot_bars(ax, p, chain)
    #                 if max_vals.get(p) is None or m > max_vals.get(p):
    #                     max_vals[p] = m

    #     for i, row in enumerate(axes):
    #         index = 1 if show_names else 0
    #         for ax, p in zip(row[index:], parameters):
    #             if not errorbar:
    #                 ax.set_ylim(0, 1.1 * max_vals[p])

    #     dpi = 300
    #     if watermark:
    #         ax = None
    #         self._add_watermark(fig, ax, figsize, watermark, dpi=dpi, size_scale=0.8)

    #     if filename is not None:
    #         if isinstance(filename, str):
    #             filename = [filename]
    #         for f in filename:
    #             self._save_fig(fig, f, dpi)
    #     if display:
    #         plt.show()

    #     return fig

    def _get_size_of_texts(self, texts: list[str]) -> float:  # pragma: no cover
        usetex = self.config.usetex
        size = self.config.label_font_size
        widths = [TextPath((0, 0), text, usetex=usetex, size=size).get_extents().width for text in texts]
        return max(widths)

    def _sanitise_columns(self, columns: list[ColumnName] | None, chains: list[Chain]) -> list[ColumnName]:
        if columns is None:
            return list(set([c for chain in chains for c in chain.samples.columns]))
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

    def set_rc_params(self):
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
        extents: dict[ColumnName, tuple[float, float]] | None,
        wide_extents: bool = True,
    ) -> dict[ColumnName, tuple[float, float]]:  # pragma: no cover
        if extents is None:
            extents = {}
        for p in columns:
            if p not in extents:
                extents[p] = self._get_parameter_extents(p, chains, wide_extents=wide_extents)
        return extents

    def _get_figure(
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
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05 * spacing, hspace=0.05 * spacing)

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
                            ax.set_xlabel(p2, fontsize=self.config.label_font_size)
                    if j != 0 or (self.config.plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        if p1 in base.blind:
                            ax.set_yticks([])
                        else:
                            display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(p1, fontsize=self.config.label_font_size)
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

    def _get_levels(self, sigmas: list[float]) -> np.ndarray:
        sigma2d = self.config.sigma2d
        if sigma2d:
            levels: np.ndarray = 1.0 - np.exp(-0.5 * np.array(sigmas) ** 2)
        else:
            levels: np.ndarray = 2 * norm.cdf(sigmas) - 1.0
        return levels

    def _plot_point(self, ax: Axes, chain: Chain, px: str, py: str) -> PathCollection:  # pragma: no cover
        point = chain.get_max_posterior_point()
        if point is None or px not in point.coordinate or py not in point.coordinate:
            return
        h = ax.scatter(
            [point.coordinate[px]],
            point.coordinate[py],
            marker=chain.marker_style,
            c=colors.format(chain.color),
            s=chain.marker_size,
            alpha=chain.marker_alpha,
        )
        return h

    def _sanitise_chains(self, chains: list[Chain | ChainName] | dict[ChainName, Chain] | None) -> list[Chain]:
        if isinstance(chains, list):
            final_chains = [self.parent.chains[n] if isinstance(n, ChainName) else n for n in chains]
        elif isinstance(chains, dict):
            final_chains = list(chains.values())
        else:
            final_chains = list(self.parent.chains.values())
        return [c for c in final_chains if not c.skip]

    def plot_contour(
        self,
        ax: Axes,
        column_x: str,
        column_y: str,
        chains: list[Chain | ChainName] | dict[ChainName, Chain] | None = None,
    ) -> None:
        """A lightweight method to plot contours in an external axis given two specified parameters

        Args:
            ax (Axes): The axis to plot on
            column_x (str): The parameter to plot on the x axis
            column_y (str): The parameter to plot on the y axis
            chains (list[Chain | ChainName] | dict[ChainName, str], optional): The chains to plot. Defaults to None.
        """

        final_chains = self._sanitise_chains(chains)
        for chain in final_chains:
            self._plot_contour(ax, chain, column_y, column_x)

    def _plot_scatter(self, ax: Axes, chain: Chain, color: str, x: pd.Series, y: pd.Series) -> PathCollection | None:
        skip = max(1, int(x.size / chain.num_cloud))
        if chain.color_data is not None:
            kwargs = {"c": chain.color_data[::skip], "cmap": chain.cmap}
        else:
            kwargs = {"c": color, "alpha": 0.3}

        h = ax.scatter(x[::skip], y[::skip], s=10, marker=".", edgecolors="none", **kwargs)
        if chain.color_data is not None:
            return h
        else:
            return None

    def _plot_contour(self, ax: Axes, chain: Chain, px: str, py: str) -> PathCollection | None:  # pragma: no cover
        levels = self._get_levels(chain.sigmas)
        x = chain.get_data(py)
        y = chain.get_data(px)

        contour_colours = self._scale_colours(chain.color, len(levels), chain.shade_gradient)
        sub = max(0.1, 1 - 0.2 * chain.shade_gradient)
        paths = None

        if chain.cloud:
            paths = self._plot_scatter(ax, chain, contour_colours[1], x, y)

        # TODO: Figure out whats going on here
        if chain.shade:
            sub *= 0.9
        colours2 = [colors.scale_colour(contour_colours[0], sub)] + [
            colors.scale_colour(c, sub) for c in contour_colours[:-1]
        ]

        hist, x_centers, y_centers = self._get_smoothed_histogram2d(chain, py, px)
        hist[hist == 0] = 1e-16
        vals = self._convert_to_stdev(hist.T)

        if chain.shade and chain.shade_alpha > 0:
            ax.contourf(
                x_centers,
                y_centers,
                vals,
                levels=levels,
                colors=contour_colours,
                alpha=chain.shade_alpha,
                zorder=chain.zorder,
            )
        con = ax.contour(
            x_centers,
            y_centers,
            vals,
            levels=levels,
            colors=colours2,
            linestyles=chain.linestyle,
            linewidths=chain.linewidth,
            zorder=chain.zorder,
        )

        if chain.show_contour_labels:
            lvls = [lvl for lvl in con.levels if lvl != 0.0]
            # TODO: see if this can just be "0.0%"
            fmt = {lvl: f"{lvl:0.0%}" for lvl in lvls}
            ax.clabel(con, lvls, inline=True, fmt=fmt, fontsize=self.config.contour_label_font_size)

        if chain.plot_point:
            self._plot_point(ax, chain, px, py)
        return paths

    def _add_truth(
        self, ax: Axes, truth: Truth, px: str | None = None, py: str | None = None
    ) -> None:  # pragma: no cover
        if px is not None:
            val_x = truth.truth_value.get(px)
            if val_x is not None:
                ax.axhline(val_x, **truth.kwargs)
        if py is not None:
            val_y = truth.truth_value.get(py)
            if val_y is not None:
                ax.axvline(val_y, **truth.kwargs)

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
                bins, _ = get_smoothed_bins(chain.smooth, int(chain.bins), data, chain.weights)
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
                lower = fit_values[0]
                upper = fit_values[2]
                if lower is not None and upper is not None:
                    if lower < xs.min():
                        lower = xs.min()
                    if upper > xs.max():
                        upper = xs.max()
                    x = np.linspace(lower, upper, 1000)
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
                        t = self.parent.analysis.get_parameter_text(*fit_values)
                        if isinstance(column, str):
                            ax.set_title(r"${} = {}$".format(column.strip("$"), t), fontsize=self.config.title_size)
                        else:
                            ax.set_title(r"$%s$" % t, fontsize=self.config.title_size)
        return ys.max()

    def _plot_walk(
        self, ax, parameter, data, truth=None, extents=None, convolve=None, color=None, log_scale=False
    ):  # pragma: no cover
        if extents is not None:
            ax.set_ylim(extents)
        assert convolve is None or isinstance(convolve, int), "Convolve must be an integer pixel window width"
        x = np.arange(data.size)
        ax.set_xlim(0, x[-1])
        ax.set_ylabel(parameter)
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
            color2 = colors.scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=":", color=color2, alpha=1)

    def _plot_walk_truth(self, ax: Axes, truth: Truth, col: str) -> None:
        ax.axhline(truth.truth_value[col], **truth.kwargs)

    def _convert_to_stdev(self, sigma: np.ndarray) -> np.ndarray:  # pragma: no cover
        # From astroML
        shape = sigma.shape
        sigma = sigma.ravel()
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)

        sigma_cumsum = 1.0 * sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]

        return sigma_cumsum[i_unsort].reshape(shape)

    def _scale_colours(self, colour: ColorInput, num: int, shade_gradient: float) -> list[str]:  # pragma: no cover
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        minv, maxv = 1 - 0.1 * shade_gradient, 1 + 0.5 * shade_gradient
        scales = np.logspace(np.log(minv), np.log(maxv), num)
        colours = [colors.scale_colour(colour, scale) for scale in scales]
        return colours

    def _get_smoothed_histogram2d(
        self,
        chain: Chain,
        col1: str,
        col2: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
        """Returns a smoothed 2D histogram of two parameters.

        Args:
            chain (Chain): The chain to plot
            col1 (str): The first parameter
            col2 (str): The second parameter

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The histogram, x bin enters, y bin centers
        """
        x = chain.get_data(col1)
        y = chain.get_data(col2)
        w = chain.weights

        if chain.grid:
            binsx = get_grid_bins(x)
            binsy = get_grid_bins(y)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)
        else:
            binsx, smooth = get_smoothed_bins(chain.smooth, int(chain.bins), x, w)
            binsy, _ = get_smoothed_bins(smooth, int(chain.bins), y, w)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)

        if chain.power is not None:
            hist = hist**chain.power

        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

        if chain.kde:
            nn = x_centers.size * 2  # Double samples for KDE because smooth
            x_centers = np.linspace(x_bins.min(), x_bins.max(), nn)
            y_centers = np.linspace(y_bins.min(), y_bins.max(), nn)
            xx, yy = meshgrid(x_centers, y_centers, indexing="ij")
            coords = np.vstack((xx.flatten(), yy.flatten())).T
            data = np.vstack((x, y)).T
            hist = MegKDE(data, w, chain.kde).evaluate(coords).reshape((nn, nn))
            if chain.power is not None:
                hist = hist**chain.power
        elif chain.smooth:
            hist = gaussian_filter(hist, chain.smooth, mode="reflect")

        return hist, x_centers, y_centers


if __name__ == "__main__":
    from .chainconsumer import ChainConsumer
