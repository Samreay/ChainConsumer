# -*- coding: utf-8 -*-
"""
================
Changing Z-Order
================

Force matplotlib to show the plots we want.

Here is a bad plot because it's hiding what we want.


"""

import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data1 = multivariate_normal([3, 5], [[1, 0], [0, 1]], size=100000)
data2 = multivariate_normal([3, 5], [[0.2, 0.1], [0.1, 0.3]], size=100000)


c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], color="k", shade_alpha=0.7, zorder=1)
c.add_chain(data2, color="o", shade_alpha=0.7, zorder=2)
c.configure(spacing=0)
c.plotter.plot(display=True, figsize=2.0)

###############################################################################
# Reversing the zorder

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], color="k", shade_alpha=0.7, zorder=2)
c.add_chain(data2, color="o", shade_alpha=0.7, zorder=1)
c.configure(spacing=0)
c.plotter.plot(display=True, figsize=2.0)

