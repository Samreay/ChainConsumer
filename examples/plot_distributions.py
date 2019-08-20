# -*- coding: utf-8 -*-
"""
==================
Plot Distributions
==================

If you want a fast check of your distributions for high dimensional spaces (such
that you can only generate a surfaces for a subset of parameters), you can
simply plot all of the marginalised distributions using this method.


"""

import numpy as np
from numpy.random import random, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
means, cov = np.arange(8), random(size=(8, 8))
data = multivariate_normal(means, np.dot(cov, cov.T), size=1000000)

params = ["$x$", "$y$", "$z$", "a", "b", "c", "d", "e"]
c = ChainConsumer().add_chain(data, parameters=params)

fig = c.plotter.plot_distributions(truth=means)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
