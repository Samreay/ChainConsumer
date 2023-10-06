import numpy as np

from .chain import Chain


def get_extents(data, weight, plot=False, wide_extents=True, tiny=False, pad=False):
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
    lower = bc[i1]
    upper = bc[-i2]
    if pad:
        width = upper - lower
        lower -= 0.2 * width
        upper += 0.2 * width
    return lower, upper


def get_bins(chains: list[Chain]):
    proposal = [
        max(35, np.floor(1.0 * np.power(chain.samples.shape[0] / chain.samples.shape[1], 0.25))) for chain in chains
    ]
    return proposal


def get_smoothed_bins(smooth, bins, data, weight, marginalised=True, plot=False, pad=False):
    minv, maxv = get_extents(data, weight, plot=plot, pad=pad)
    if smooth is None or not smooth or smooth == 0:
        return np.linspace(minv, maxv, int(bins)), 0
    else:
        return np.linspace(minv, maxv, int((2 if marginalised else 2) * smooth * bins)), smooth


def get_grid_bins(data):
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
