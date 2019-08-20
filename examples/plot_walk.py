# -*- coding: utf-8 -*-
"""
==========
Plot Walks
==========

You can also plot the walks that your chains have undertaken.

This is a very helpful plot to create when determining if your chains have
converged and mixed. Below is an example walk from a Metropolis-Hastings run,
where we have set the optional parameters for the weights and posteriors,
giving the top two subplots.

.. figure::     ../../examples/resources/exampleWalk.png
    :align:     center

"""


###############################################################################
# To generate your own walk, with a 100 point smoothed walk overplotting,
# you can use the following code:

import numpy as np
from chainconsumer import ChainConsumer

np.random.seed(0)
data1 = np.random.randn(100000, 2)
data2 = np.random.randn(100000, 2) - 2
data1[:, 1] += 1

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"])
c.add_chain(data2, parameters=["$x$", "$z$"])
fig = c.plotter.plot_walks(truth={"$x$": -1, "$y$": 1, "$z$": -2}, convolve=100)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

