# -*- coding: utf-8 -*-
"""
=======================
Cmap and Custom Bins
=======================

Invoke the cmap colour scheme and choose how many bins to use with your data.

By default, the cmap colour scheme is used if you have many, many chains. You can
enable it before that point if you wish and pass in the cmap you want to use.

You can also pick how many bins you want to display your data with.

You can see that in this example, we pick too many bins and would not get good
summaries. If you simply want more (or less) bins than the default estimate,
if you input a float instead of an integer, the number of bins will simply scale
by that amount. For example, if the estimated picks 20 bins, and you set ``bins=1.5``
your plots and summaries would be calculated with 30 bins.

"""
import numpy as np
from numpy.random import normal, random, multivariate_normal
from chainconsumer import ChainConsumer


np.random.seed(0)
cov = 0.3 * random(size=(3, 3)) + np.identity(3)
data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)
cov = 0.3 * random(size=(3, 3)) + np.identity(3)
data2 = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)
cov = 0.3 * random(size=(3, 3)) + np.identity(3)
data3 = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)
cov = 0.3 * random(size=(3, 3)) + np.identity(3)
data4 = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)

c = ChainConsumer()
c.add_chain(data, name="A")
c.add_chain(data2, name="B")
c.add_chain(data3, name="C")
c.add_chain(data4, name="D")
c.configure(bins=50, cmap="plasma")
fig = c.plotter.plot(figsize=0.75)  # Also making the figure 75% of its original size, for fun

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
