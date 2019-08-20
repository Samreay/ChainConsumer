# -*- coding: utf-8 -*-
"""
=============
Colour Points
=============

Add colour scatter to show an extra dimension.

If we have a secondary parameter that might not be best displayed
as a posterior surface and would be useful to instead give
context to other surfaces, we can select that point to give a
colour mapped scatter plot.

We can *also* display this as a posterior surface by setting
`plot_colour_params=True`, if we wanted.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(1)
cov = normal(size=(3, 3))
data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$"])
c.configure(color_params="$z$")
fig = c.plotter.plot(figsize=1.0)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# You can also plot the weights or posterior if they are specified. Showing weights here.

weights = 1 / (1 + data[:, 0]**2 + data[:, 1]**2)
c = ChainConsumer().add_chain(data[:, :2], parameters=["$x$", "$y$"], weights=weights)
c.configure(color_params="weights")
fig = c.plotter.plot(figsize=3.0)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# And showing the posterior color parameter here

weights = 1 / (1 + data[:, 0]**2 + data[:, 1]**2)
posterior = np.log(weights)
c = ChainConsumer().add_chain(data[:, :2], parameters=["$x$", "$y$"], weights=weights, posterior=posterior)
c.configure(color_params="posterior")
fig = c.plotter.plot(figsize=3.0)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
