# -*- coding: utf-8 -*-
"""
=============
External axes
=============

To put contours in external figures.

To help in inserting contours into other plots and figures, you can call the simplified
`plot_contour` method and pass in an axis. Note that this is a minimal routine, and will not
do auto-extents, labels, ticks, truth values, etc. But it will make contours for you.

"""

from chainconsumer import ChainConsumer
from scipy.stats import multivariate_normal as mv
import matplotlib.pyplot as plt

data = mv.rvs(mean=[5, 6], cov=[[1, 0.9], [0.9, 1]], size=10000)

fig, axes = plt.subplots(nrows=2, figsize=(4, 6), sharex=True)
axes[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.1)

c = ChainConsumer()
c.add_chain(data, parameters=["a", "b"])
c.plotter.plot_contour(axes[1], "a", "b")

for ax in axes:
    ax.axvline(5)
    ax.axhline(6)
