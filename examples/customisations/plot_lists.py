# -*- coding: utf-8 -*-
"""
==================
Using List Options
==================

Utilise all the list options in the configuration!

This is a general example to illustrate that most parameters
that you can pass to the configuration methods accept lists.

"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(2)
cov = normal(size=(2, 2)) + np.identity(2)
d1 = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=100000)
cov = normal(size=(2, 2)) + np.identity(2)
d2 = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=100000)
cov = normal(size=(2, 2)) + np.identity(2)
d3 = multivariate_normal(normal(size=2), np.dot(cov, cov.T), size=1000000)

c = ChainConsumer()
c.add_chain(d1, parameters=["$x$", "$y$"])
c.add_chain(d2)
c.add_chain(d3)

c.configure(linestyles=["-", "--", "-"], linewidths=[1.0, 3.0, 1.0],
            bins=[3.0, 1.0, 1.0], colors=["#1E88E5", "#D32F2F", "#111111"],
            smooth=[0, 1, 2], shade=[True, True, False],
            shade_alpha=[0.2, 0.1, 0.0], bar_shade=[True, False, False])
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# List options are useful for when the properties of the chains are
# interconnected. But if you know the properties at the start,
# you can define them when adding the chains. List options will not
# override explicit chain configurations, so you can use the global
# configure to set options for all chains you haven't explicitly specified.
#
# Note here how even though we set 'all' chains to dotted lines of width 2, our third
# chain, with its explicit options, ignores that.

c = ChainConsumer()
c.add_chain(d1, parameters=["$x$", "$y$"]).add_chain(d2).add_chain(d3, linestyle="-", linewidth=5)

c.configure(linestyles=":", linewidths=2)
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.