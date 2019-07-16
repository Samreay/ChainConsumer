# -*- coding: utf-8 -*-
"""
================
Excluding Chains
================

You don't have to plot everything at once!


For the main plotting methods you can specify which chains you want to plot. You can
do this using either the chain index or using the chain names. Like so:

"""

import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data1 = multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000000)
data2 = multivariate_normal([2, 0], [[1, 0], [0, 1]], size=1000000)
data3 = multivariate_normal([4, 0], [[1, 0], [0, 1]], size=1000000)

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Chain A")
c.add_chain(data2, name="Chain B")
c.add_chain(data3, name="Chain C")
fig = c.plotter.plot(chains=["Chain A", "Chain C"])

fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.