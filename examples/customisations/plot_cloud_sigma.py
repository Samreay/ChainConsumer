"""
======================
Cloud and Sigma Levels
======================

Choose custom sigma levels and display point cloud.

In this example we display more sigma levels, turn on the point cloud, and
disable the parameter summaries on the top of the marginalised distributions.

Note that because of the very highly correlated distribution we have, it is
useful to increase the number of bins the plots are generated with, to capture the
thinness of the correlation.
"""

import numpy as np

from chainconsumer import ChainConsumer

rng = np.random.default_rng(1)
cov = rng.normal(size=(3, 3))
data = rng.multivariate_normal(rng.normal(size=3), np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$"])
c.configure(summary=False, bins=1.4, cloud=True, sigmas=np.linspace(0, 2, 10))
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
