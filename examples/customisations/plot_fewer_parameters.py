# -*- coding: utf-8 -*-
"""
====================
Truncated Parameters
====================

If you have a large model, you don't need to plot all parameters at once.

Here we only plot the first four parameters. You could also simply pass the number four,
which means the *first* four parameters.

For fun, we also plot everything in green. Note you don't need to give multiple colours,
the shading is all computed from the colour hex code.
"""

import numpy as np
from numpy.random import normal, random, multivariate_normal
from chainconsumer import ChainConsumer


np.random.seed(0)
cov = random(size=(6, 6))
data = multivariate_normal(normal(size=6), np.dot(cov, cov.T), size=200000)
parameters = ["$x$", "$y$", "$z$", "$a$", "$b$", "$c$"]
c = ChainConsumer().add_chain(data, parameters=parameters).configure(colors="#388E3C")
fig = c.plotter.plot(parameters=parameters[:4], figsize="page")

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
