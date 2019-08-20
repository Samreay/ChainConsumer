# -*- coding: utf-8 -*-
"""
===================
Colours and Shading
===================

Choose custom colours and plot multiple chains with shading.

Normally when plotting more than two chains, shading is removed so
you can clearly see the outlines. However, you can turn shading back
on and modify the shade opacity if you prefer colourful plots.

Note that the contour shading and marginalised shading are separate
and are configured independently.

Colours should be given as hex colours.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(2)
cov = normal(size=(2, 2)) + np.identity(2)
d1 = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=100000)
cov = normal(size=(2, 2)) + np.identity(2)
d2 = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=100000)
cov = normal(size=(2, 2)) + np.identity(2)
d3 = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(d1, parameters=["$x$", "$y$"]).add_chain(d2).add_chain(d3)
c.configure(colors=["#B32222", "#D1D10D", "#455A64"], shade=True, shade_alpha=0.2, bar_shade=True)
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


