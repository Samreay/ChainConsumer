# -*- coding: utf-8 -*-
"""
============
Truth Values
============

Plot truth values on top of your contours.

"""

###############################################################################
# You can specify truth values using a list (in the same order as the
# declared parameters).

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(2)
cov = 0.2 * normal(size=(3, 3)) + np.identity(3)
truth = normal(size=3)
data = multivariate_normal(truth, np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", r"$\beta$"])
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or you can specify truth values using a dictionary. This allows you to specify
# truth values for only some parameters. You can also customise the look
# of your truth lines.


c.configure_truth(color='w', ls=":", alpha=0.8)
fig2 = c.plotter.plot(truth={"$x$": truth[0], "$y$": truth[1]})
fig2.set_size_inches(0 + fig2.get_size_inches())  # Resize fig for doco. You don't need this.
