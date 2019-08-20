# -*- coding: utf-8 -*-
"""
=========================
Plot Many Things For Fun!
=========================

Lets try a few things to give what might be a usable plot to through into our latest paper.

Or something.

First lets mock some highly correlated data with colour scatter. And then throw a few more
data sets in to get some overlap.
"""

import numpy as np
from numpy.random import normal, multivariate_normal, uniform
from chainconsumer import ChainConsumer

np.random.seed(1)
n = 1000000
data = multivariate_normal([0.4, 1], [[0.01, -0.003], [-0.003, 0.001]], size=n)
data = np.hstack((data, (67 + 10 * data[:, 0] - data[:, 1] ** 2)[:, None]))
data2 = np.vstack((uniform(-0.1, 1.1, n), normal(1.2, 0.1, n))).T
data2[:, 1] -= (data2[:, 0] ** 2)
data3 = multivariate_normal([0.3, 0.7], [[0.02, 0.05], [0.05, 0.1]], size=n)

c = ChainConsumer()
c.add_chain(data2, parameters=["$\Omega_m$", "$-w$"], name="B")
c.add_chain(data3, name="S")
c.add_chain(data, parameters=["$\Omega_m$", "$-w$", "$H_0$"], name="P")

c.configure(color_params="$H_0$", shade=[True, True, False],
            shade_alpha=0.2, bar_shade=True, linestyles=["-", "--", "-"])
fig = c.plotter.plot(figsize=2.0, extents=[[0, 1], [0, 1.5]])

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
