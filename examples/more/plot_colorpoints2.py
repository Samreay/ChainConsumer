# -*- coding: utf-8 -*-
"""
========================
Multiple Colour Scatter!
========================

Why show only one colour, when you can display more!

In the basic colour example, we showed one parameter being used
to give colour information. However, you can pick a different colour, or no colour (`None`),
for each chain.

You can also pick the same parameter in multiple chains, and all the scatter points will be put
on the same colour scale. The underlying contours will still be distinguishable automatically
by adding alternative linestyles, as shown below.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(1)
cov = normal(size=(4, 4))
data = multivariate_normal(normal(size=4), np.dot(cov, cov.T), size=100000)
cov = 1 + 0.5 * normal(size=(4, 4))
data2 = multivariate_normal(4+normal(size=4), np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$", "$g$"], name="a")
c.add_chain(data2, parameters=["$x$", "$y$", "$z$", "$t$"], name="b")
c.configure(color_params=["$g$", "$t$"])
fig = c.plotter.plot(figsize=1.75)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
