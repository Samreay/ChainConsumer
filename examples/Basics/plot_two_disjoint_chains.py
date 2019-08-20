# -*- coding: utf-8 -*-
"""
===================
Two Disjoint Chains
===================

You can plot multiple chains. They can even have different parameters!


"""
import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer


np.random.seed(0)
cov = normal(size=(3, 3))
cov2 = normal(size=(4, 4))
data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)
data2 = multivariate_normal(normal(size=4), np.dot(cov2, cov2.T), size=100000)

c = ChainConsumer()
c.add_chain(data, parameters=["$x$", "$y$", r"$\alpha$"])
c.add_chain(data2, parameters=["$x$", "$y$", r"$\alpha$", r"$\gamma$"])
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
