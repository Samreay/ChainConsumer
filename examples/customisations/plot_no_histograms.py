"""
=============
No Histograms
=============

Sometimes marginalised histograms are not needed.

"""


import numpy as np

from chainconsumer import ChainConsumer

rng = np.random.default_rng(0)
cov = rng.normal(size=(3, 3))
data = rng.multivariate_normal(rng.normal(size=3), np.dot(cov, cov.T), size=100000)

c = ChainConsumer().add_chain(data)
c.configure_overrides(plot_hists=False)
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
