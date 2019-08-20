# -*- coding: utf-8 -*-
"""
=============
No Histograms
=============

Sometimes marginalised histograms are not needed.

"""


from numpy.random import multivariate_normal, normal, seed
import numpy as np
from chainconsumer import ChainConsumer

seed(0)
cov = normal(size=(3, 3))
data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(data)
c.configure(plot_hists=False)
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

