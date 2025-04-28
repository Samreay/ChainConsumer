from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .chain import Chain
from .kde import MegKDE


def get_extents(
    data: pd.Series,
    weight: np.ndarray,
    plot: bool = False,
    wide_extents: bool = True,
    tiny: bool = False,
    pad: bool = False,
) -> tuple[float, float]:
    hist, be = np.histogram(data, weights=weight, bins=2000)
    bc = 0.5 * (be[1:] + be[:-1])
    cdf = hist.cumsum()
    cdf = cdf / cdf.max()
    icdf = (1 - cdf)[::-1]
    icdf = icdf / icdf.max()
    cdf = 1 - icdf[::-1]
    threshold = 1e-4 if plot else 1e-5
    if plot and not wide_extents:
        threshold = 0.05
    if tiny:
        threshold = 0.3
    i1 = np.where(cdf > threshold)[0][0]
    i2 = np.where(icdf > threshold)[0][0]
    lower = float(bc[i1])
    upper = float(bc[-i2])
    if pad:
        width = upper - lower
        lower -= 0.2 * width
        upper += 0.2 * width
    return lower, upper


def get_bins(chain: Chain) -> int:
    if chain.bins is not None:
        return chain.bins
    max_v = 35 if chain.smooth_value > 0 else 100
    return max((max_v, int(np.floor(1.0 * np.power(chain.samples.shape[0] / chain.samples.shape[1], 0.25)))))


def get_smoothed_bins(
    smooth: int,
    bins: int,
    data: pd.Series,
    weight: np.ndarray,
    plot: bool = False,
    pad: bool = False,
) -> tuple[np.ndarray, int]:
    """Get the bins for a histogram, with smoothing.

    Args:
        smooth (int): The smoothing factor
        bins (int): The number of bins
        data (pd.Series): The data
        weight (np.ndarray): The weights
        plot (bool, optional): Whether this is used in plotting. Determines how conservative to be on extents
            Defaults to False.
        pad (bool, optional): Whether to pad the histogram.  Determines how conservative to be on extents
            Defaults to False.
    """
    minv, maxv = get_extents(data, weight, plot=plot, pad=pad)
    if smooth == 0:
        return np.linspace(minv, maxv, int(bins)), 0
    else:
        return np.linspace(minv, maxv, 2 * smooth * bins), smooth


def get_smoothed_histogram2d(
    chain: Chain,
    px: str,
    py: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """Returns a smoothed 2D histogram of two parameters.

    Args:
        chain (Chain): The chain to plot
        col1 (str): The first parameter
        col2 (str): The second parameter

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The histogram, x bin enters, y bin centers
    """
    x = chain.get_data(px)
    y = chain.get_data(py)
    w = chain.weights

    if chain.grid:
        binsx = get_grid_bins(x)
        binsy = get_grid_bins(y)
        hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)
    else:
        binsx, smooth = get_smoothed_bins(chain.smooth_value, get_bins(chain), x, w)
        binsy, _ = get_smoothed_bins(smooth, get_bins(chain), y, w)
        hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)

    if chain.power is not None:
        hist = hist**chain.power

    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

    if chain.kde:
        nn = x_centers.size * 2  # Double samples for KDE because smooth
        x_centers = np.linspace(x_bins.min(), x_bins.max(), nn)
        y_centers = np.linspace(y_bins.min(), y_bins.max(), nn)
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        coords = np.vstack((xx.flatten(), yy.flatten())).T
        data = np.vstack((x, y)).T
        hist = MegKDE(data, w, chain.kde).evaluate(coords).reshape((nn, nn))
        if chain.power is not None:
            hist = hist**chain.power
    elif chain.smooth_value:
        hist = gaussian_filter(hist, chain.smooth_value, mode="reflect")

    return hist, x_centers, y_centers


def get_grid_bins(data: pd.Series[float]) -> np.ndarray:
    bin_c = np.sort(np.unique(data))
    delta = 0.5 * (bin_c[1] - bin_c[0])
    bins = np.concatenate((bin_c - delta, [bin_c[-1] + delta]))
    return bins


def get_latex_table_frame(caption: str, label: str) -> str:  # pragma: no cover
    base_string = rf"""\begin{{table}}
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \begin{{tabular}}{{%s}}
        %s    \end{{tabular}}
\end{{table}}"""
    return base_string
