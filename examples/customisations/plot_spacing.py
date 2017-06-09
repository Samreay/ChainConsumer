"""
===============
Subplot Spacing
===============

By default ChainConsumer will reduce subplot whitespace when you hit
a certain dimensionality, but you can also customise this yourself.
"""

import numpy as np
from numpy.random import normal, random, multivariate_normal
from chainconsumer import ChainConsumer


np.random.seed(0)
cov = random(size=(3, 3))
data = multivariate_normal(normal(size=3), cov * cov.T, size=200000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$"])
c.configure(spacing=0.0)
fig = c.plotter.plot(figsize="column")

fig.set_size_inches(4.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
