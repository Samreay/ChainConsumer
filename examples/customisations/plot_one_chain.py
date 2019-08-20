# -*- coding: utf-8 -*-
"""
=========
One Chain
=========

Plot one chain with parameter names.

Because we are only plotting one chain, we will get
parameter bounds on the marginalised surfaces by
default.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer


np.random.seed(0)
cov = 1e2 * normal(size=(3, 3))
data = multivariate_normal(1e3 * normal(size=3), np.dot(cov, cov.T), size=100000)

# If you pass in parameter labels and only one chain, you can also get parameter bounds
fig = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"]).plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
