# -*- coding: utf-8 -*-
"""
===================
Changing Linestyles
===================

Customise the plot line styles.

In this example we customise the line styles used, and make use of
the ability to pass lists of parameters to the configuration methods.

"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(1)
cov = normal(size=(3, 3))
data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)
data2 = data * 1.1 + 0.5

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$"]).add_chain(data2)
c.configure(linestyles=["-", "--"], linewidths=[1.0, 2.0],
            shade=[True, False], shade_alpha=[0.2, 0.0])
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
