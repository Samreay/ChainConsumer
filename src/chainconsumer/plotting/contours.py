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

def plot_dist(ax: Axes, chain: Chain, px: ColumnName, config: PlotConfig | None = None, summary: bool = False) -> None:
    """A lightweight method to plot a 1D distribution in an external axis given one specified parameter

    Args:
        ax: The axis to plot on
        chain: The chain to plot
        px: The parameter to plot on the x axis
        config: The plot configuration
        summary: If True, add parameter summary text above the plot
    """
    from ..helpers import get_bins, get_smoothed_bins, get_grid_bins
    from scipy.interpolate import interp1d

    if config is None:
        config = PlotConfig()

    # Get the data for the parameter
    data = chain.get_data(px)

    # Create histogram based on chain settings (following _plot_bars logic from plotter.py)
    if chain.smooth_value or chain.kde:
        # Use KDE or smoothed histogram
        from ..helpers import get_extents, get_smoothed_bins, get_bins

        if chain.kde:
            from ..kde import MegKDE
            bins, _ = get_smoothed_bins(chain.smooth_value, get_bins(chain), data, chain.weights, pad=True)
            kde = MegKDE(data.values.reshape(-1, 1), chain.weights, chain.kde)
            xs = np.linspace(bins.min(), bins.max(), 1000)
            ys = kde.evaluate(xs.reshape(-1, 1)).flatten()
            if chain.power is not None:
                ys = ys ** chain.power
        else:
            # Smoothed histogram
            bins, smooth = get_smoothed_bins(chain.smooth_value, get_bins(chain), data, chain.weights, pad=True)
            hist, edges = np.histogram(data, bins=bins, density=True, weights=chain.weights)
            if chain.power is not None:
                hist = hist ** chain.power
            from scipy.ndimage import gaussian_filter
            hist = gaussian_filter(hist, chain.smooth_value, mode='reflect')
            xs = 0.5 * (edges[:-1] + edges[1:])
            ys = hist

        ys *= chain.histogram_relative_height
        ax.plot(xs, ys, color=chain.color, ls=chain.linestyle, lw=chain.linewidth, zorder=chain.zorder)
    else:
        # Regular histogram
        if chain.grid:
            bins = get_grid_bins(data)
        else:
            bins, _ = get_smoothed_bins(chain.smooth_value, get_bins(chain), data, chain.weights)

        hist, edges = np.histogram(data, bins=bins, density=True, weights=chain.weights)
        if chain.power is not None:
            hist = hist ** chain.power

        edge_center = 0.5 * (edges[:-1] + edges[1:])
        xs, ys = edge_center, hist
        ys *= chain.histogram_relative_height

        ax.hist(
            xs,
            weights=ys,
            bins=bins,
            histtype="step",
            color=chain.color,
            orientation="vertical",
            ls=chain.linestyle,
            lw=chain.linewidth,
            zorder=chain.zorder,
        )

    # Add shading for confidence interval (e.g., 1-sigma) if requested
    fit_values = None
    if chain.shade and chain.shade_alpha > 0:
        from ..analysis import Analysis
        analysis = Analysis(None)  # type: ignore
        fit_values = analysis.get_parameter_summary(chain, px)

        if fit_values is not None:
            lower = fit_values.lower
            upper = fit_values.upper
            if lower is not None and upper is not None:
                interp_type = "linear" if chain.smooth_value else "nearest"
                interpolator = interp1d(xs, ys, kind=interp_type)
                lower = max(lower, xs.min())
                upper = min(upper, xs.max())
                x = np.linspace(lower, upper, 1000)

                # Use shade_gradient to control the opacity of the confidence interval shading
                # shade_gradient scales the alpha value (higher = more opaque)
                effective_alpha = chain.shade_alpha * (0.2 + 0.8 * chain.shade_gradient)

                ax.fill_between(
                    x,
                    np.zeros(x.shape),
                    interpolator(x),
                    color=chain.color,
                    alpha=effective_alpha,
                    zorder=chain.zorder - 1,
                )

    # Add parameter summary text above the plot if requested
    if summary:
        if fit_values is None:
            from ..analysis import Analysis
            analysis = Analysis(None)  # type: ignore
            fit_values = analysis.get_parameter_summary(chain, px)

        if fit_values is not None:
            # Format the parameter text (e.g., "param = value +upper -lower")
            def format_value(v):
                """Format a value with appropriate precision"""
                if abs(v) >= 100:
                    return f"{v:.1f}"
                elif abs(v) >= 10:
                    return f"{v:.2f}"
                elif abs(v) >= 1:
                    return f"{v:.2f}"
                else:
                    return f"{v:.3f}"

            center_str = format_value(fit_values.center)
            if fit_values.lower is not None and fit_values.upper is not None:
                lower_err = fit_values.center - fit_values.lower
                upper_err = fit_values.upper - fit_values.center
                lower_str = format_value(lower_err)
                upper_str = format_value(upper_err)
                text = rf"${px} = {center_str}^{{+{upper_str}}}_{{-{lower_str}}}$"
            else:
                text = rf"${px} = {center_str}$"

            ax.set_title(text, fontsize=config.summary_font_size, pad=10)


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
