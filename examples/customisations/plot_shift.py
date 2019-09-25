# -*- coding: utf-8 -*-
"""
==============
Shifting Plots
==============

Shift all your plots to the same location for blind uncertainty comparison.


Plots will shift to the location you tell them to, in the same format as a truth dictionary.
So you can use truth dict for both! Takes a list or a dict as input for convenience.

"""

import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data1 = multivariate_normal([1, 0], [[3, 2], [2, 3]], size=300000)
data2 = multivariate_normal([0, 0.5], [[1, -0.7], [-0.7, 1]], size=300000)
data3 = multivariate_normal([2, -1], [[0.5, 0], [0, 0.5]], size=300000)

###############################################################################
# And this is how easy it is to shift them. Note the different means for each dataset!

truth = {"$x$": 1, "$y$": 0}
c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Chain A", shift_params=truth)
c.add_chain(data2, name="Chain B", shift_params=truth)
c.add_chain(data3, name="Chain C", shift_params=truth)
fig = c.plotter.plot(truth=truth)

fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
