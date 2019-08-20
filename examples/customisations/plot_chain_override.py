"""
====================
Overriding Configure
====================

You can specify display options when adding chains.

This is useful for when you are playing around with code, adding and removing chains
as you tweak the plot. Normally, this would involve modifying the lists passed into `configure`
if you wanted to keep a specific chain with a specific style. To make it easier, 
you can specify chain properties when addng them via `add_chain`. If set, these values override 
anything specified in configure (and # -*- coding: utf-8 -*-
thus override the default configure behaviour).

"""
import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data1 = multivariate_normal([-2, 0], [[1, 0], [0, 1]], size=100000)
data2 = multivariate_normal([4, -4], [[1, 0], [0, 1]], size=100000)
data3 = multivariate_normal([-2, -4], [[1, 0.7], [0.7, 1]], size=100000)

c = ChainConsumer()
c.add_chain(data1, parameters=["x", "y"], color="red", linestyle=":", name="Red dots")
c.add_chain(data2, parameters=["x", "y"], color="#4286f4", shade_alpha=1.0, name="Blue solid")
c.add_chain(data3, parameters=["x", "y"], color="lg", kde=1.5, linewidth=2.0, name="Green smoothed")

fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
