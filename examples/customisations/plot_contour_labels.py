# -*- coding: utf-8 -*-
"""
==============
Contour Labels
==============

Plot contours using labels.

You can set the contour_labels to display confidence levels, as shown below.

"""

from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer


data = multivariate_normal([0, 0], [[1, 0.5], [0.5, 1.0]], size=1000000)


c = ChainConsumer().add_chain(data).configure(contour_labels="confidence")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


###############################################################################
# Or you can plot in terms of sigma. Note that most people prefer stating
# the confidence levels, because of the ambiguity over sigma levels introduced
# by the `sigma2d` keyword.

c = ChainConsumer().add_chain(data).configure(contour_labels="sigma")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
