# -*- coding: utf-8 -*-
import numpy as np


def get_extents(data, weight, plot=False, wide_extents=True, tiny=False):
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
    return bc[i1], bc[-i2]


def get_bins(chains):
    proposal = [max(35, np.floor(1.0 * np.power(chain.chain.shape[0] / chain.chain.shape[1], 0.25)))
                for chain in chains]
    return proposal


def get_smoothed_bins(smooth, bins, data, weight, marginalised=True, plot=False):
    minv, maxv = get_extents(data, weight, plot=plot)
    if smooth is None or not smooth or smooth == 0:
        return np.linspace(minv, maxv, int(bins)), 0
    else:
        return np.linspace(minv, maxv, int((2 if marginalised else 2) * smooth * bins)), smooth


def get_grid_bins(data):
    bin_c = np.sort(np.unique(data))
    delta = 0.5 * (bin_c[1] - bin_c[0])
    bins = np.concatenate((bin_c - delta, [bin_c[-1] + delta]))
    return bins


def get_latex_table_frame(caption, label):  # pragma: no cover
    base_string = r"""\begin{table}
    \centering
    \caption{%s}
    \label{%s}
    \begin{tabular}{%s}
        %s    \end{tabular}
\end{table}"""
    return base_string % (caption, label, "%s", "%s")

