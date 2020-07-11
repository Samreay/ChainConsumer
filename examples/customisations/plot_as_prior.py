# -*- coding: utf-8 -*-
"""
================
Plotting a prior
================

If you have 1D priors that you don't want to appear in the contour, thats possible too.

"""
import numpy as np
from numpy.random import normal, random, multivariate_normal
from chainconsumer import ChainConsumer


if __name__ == "__main__":
    np.random.seed(0)
    cov = random(size=(2, 2)) + np.identity(2)
    data = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=100000)

    prior = normal(0, 1, size=100000)

    fig = ChainConsumer()\
        .add_chain(data, parameters=["x", "y"], name="Normal")\
        .add_chain(prior, parameters=["y"], name="Prior", show_as_1d_prior=True)\
        .plotter.plot()

    fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
