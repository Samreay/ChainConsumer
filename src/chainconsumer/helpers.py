from __future__ import annotations

import numpy as np
import pandas as pd

from .chain import Chain


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
    max_v = 35 if chain.smooth > 0 else 100
    return max((max_v, np.floor(1.0 * np.power(chain.samples.shape[0] / chain.samples.shape[1], 0.25))))


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
