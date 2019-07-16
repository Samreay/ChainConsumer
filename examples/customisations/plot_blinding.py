# -*- coding: utf-8 -*-
"""
===================
Blinding Parameters
===================

You can blind parameters and not show axis labels very easily!

Just give ChainConsumer the `blind` parameter when plotting. You can specify `True` to blind all parameters,
or give it a string (or list of strings) detailing the specific parameters you want blinded!

"""

import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data = multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$"])
c.configure(colors=["g"])
fig = c.plotter.plot(blind="$y$")

fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
