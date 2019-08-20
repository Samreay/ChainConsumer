"""
=================
Confidence Levels
=================

When setting the sigma levels for ChainConsumer, we need to be careful
if we are talking about 1D or 2D Gaussians. For 1D Gaussians, 1 and 2 :math:`\sigma` correspond
to 68% and 95% confidence levels. However, for a a 2D Gaussian, integrating over 1 and 2 :math:`\sigma`
levels gives 39% and 86% confidence levels.

By default ChainConsumer uses the 2D levels, such that the contours will line up and agree with the
marginalised distributions shown above them, however you can also choose to switch to using the 1D
Gaussian method, such that the contour encloses 68% and 95% confidence regions, by switching `sigma2d` to `False` 

"""

import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data = multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$"])
c.configure(flip=False, sigma2d=False, sigmas=[1, 2])  # The default case, so you don't need to specify sigma2d
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Demonstrating the 1D Gaussian confidence levels. Notice the change in contour size
# The contours shown below now show the 68% and 95% confidence regions.

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$"])
c.configure(flip=False, sigma2d=True, sigmas=[1, 2])
fig = c.plotter.plot()# -*- coding: utf-8 -*-



fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
