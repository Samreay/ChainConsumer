import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy.stats import norm

from ..chain import Chain, ColumnName
from ..color_finder import ColorInput, colors
from ..helpers import get_smoothed_histogram2d
from .config import PlotConfig


def plot_surface(
    ax: Axes,
    chains: list[Chain],
    px: ColumnName,
    py: ColumnName,
    config: PlotConfig | None = None,
) -> dict[ColumnName, PathCollection]:
    """Plot the chains onto a 2D surface, using clouds, contours and points.

    Returns:
        A map from column name to paths to be added as colorbars.
    """
    if config is None:
        config = PlotConfig()
    paths: dict[ColumnName, PathCollection] = {}
    for chain in chains:
        if px not in chain.plotting_columns or py not in chain.plotting_columns:
            continue

        if chain.plot_cloud:
            paths |= plot_cloud(ax, chain, px, py)

        if chain.plot_contour:
            plot_contour(ax, chain, px, py, config)

        if chain.plot_point:
            plot_point(ax, chain, px, py)

    return paths


def plot_cloud(ax: Axes, chain: Chain, px: ColumnName, py: ColumnName) -> dict[ColumnName, PathCollection]:
    x = chain.get_data(px)
    y = chain.get_data(py)
    skip = max(1, int(x.size / chain.num_cloud))
    if chain.color_data is not None:
        kwargs = {"c": chain.color_data[::skip], "cmap": chain.cmap}
    else:
        kwargs = {"c": colors.format(chain.color), "alpha": 0.3}

    h = ax.scatter(
        x[::skip],
        y[::skip],
        s=10,
        marker=".",
        edgecolors="none",
        zorder=chain.zorder - 5,
        **kwargs,  # type: ignore
    )
    if chain.color_data is not None and chain.color_param is not None:
        return {chain.color_param: h}
    return {}


def plot_contour(ax: Axes, chain: Chain, px: ColumnName, py: ColumnName, config: PlotConfig | None = None) -> None:
    """A lightweight method to plot contours in an external axis given two specified parameters

    Args:
        ax: The axis to plot on
        chain: The chain to plot
        px: The parameter to plot on the x axis
        py: The parameter to plot on the y axis
    """
    if config is None:
        config = PlotConfig()
    levels = _get_levels(chain.sigmas, config)
    contour_colours = _scale_colours(colors.format(chain.color), len(levels), chain.shade_gradient)
    sub = max(0.1, 1 - 0.2 * chain.shade_gradient)
    paths = None

    # TODO: Figure out what's going on here
    if chain.shade:
        sub *= 0.9
    colours2 = [colors.scale_colour(contour_colours[0], sub)] + [
        colors.scale_colour(c, sub) for c in contour_colours[:-1]
    ]

    hist, x_centers, y_centers = get_smoothed_histogram2d(chain, px, py)
    hist[hist == 0] = 1e-16
    vals = _convert_to_stdev(hist)

    if chain.shade and chain.shade_alpha > 0:
        ax.contourf(
            x_centers,
            y_centers,
            vals.T,
            levels=levels,
            colors=contour_colours,
            alpha=chain.shade_alpha,
            zorder=chain.zorder - 2,
        )
    con = ax.contour(
        x_centers,
        y_centers,
        vals.T,
        levels=levels,
        colors=colours2,
        linestyles=chain.linestyle,
        linewidths=chain.linewidth,
        zorder=chain.zorder,
    )

    if chain.show_contour_labels:
        lvls = [lvl for lvl in con.levels if lvl != 0.0]
        fmt = {lvl: f" {lvl:0.0%} " if lvl < 0.991 else f" {lvl:0.1%} " for lvl in lvls}
        texts = ax.clabel(con, lvls, inline=True, fmt=fmt, fontsize=config.contour_label_font_size)
        for text in texts:
            text.set_fontweight("semibold")

    return paths


def plot_point(ax: Axes, chain: Chain, px: ColumnName, py: ColumnName) -> None:
    point = chain.get_max_posterior_point()
    if point is None or px not in point.coordinate or py not in point.coordinate:
        return

    c = colors.format(chain.color)
    if chain.plot_contour:
        c = colors.scale_colour(colors.format(chain.color), 0.5)
    ax.scatter(
        [point.coordinate[px]],
        [point.coordinate[py]],
        marker=chain.marker_style,
        c=c,
        s=chain.marker_size,
        alpha=chain.marker_alpha,
        zorder=chain.zorder + 1,
    )


def _convert_to_stdev(sigma: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Convert a 2D histogram of samples into the equivalent sigma levels."""
    # From astroML
    shape = sigma.shape
    sigma = sigma.ravel()
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = 1.0 * sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)


def _scale_colours(colour: ColorInput, num: int, shade_gradient: float) -> list[str]:  # pragma: no cover
    """Scale a colour lighter or darker."""
    # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
    minv, maxv = 1 - 0.1 * shade_gradient, 1 + 0.5 * shade_gradient
    scales = np.logspace(np.log(minv), np.log(maxv), num)
    colours = [colors.scale_colour(colour, scale) for scale in scales]
    return colours


def _get_levels(sigmas: list[float], config: PlotConfig) -> np.ndarray:
    """Turn sigmas into percentages."""
    sigma2d = config.sigma2d
    if sigma2d:
        levels: np.ndarray = 1.0 - np.exp(-0.5 * np.array(sigmas) ** 2)
    else:
        levels: np.ndarray = 2 * norm.cdf(sigmas) - 1.0
    return levels
