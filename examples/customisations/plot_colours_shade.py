"""
===================
Colours and Shading
===================

Choose custom colours and plot multiple chains with shading.

Normally when plotting more than one chain, shading is removed so
you can clearly see the outlines. However, you can turn shading back
on and modify the shade opacity if you prefer colourful plots.

Note that the contour shading and marginalised shading are separate
and are configured independently.

Colours should be given as hex colours.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chain_consumer import ChainConsumer

if __name__ == "__main__":
    np.random.seed(2)
    cov = normal(size=(2, 2)) + np.identity(2)
    data = multivariate_normal(normal(size=2), 0.5 * (cov + cov.T), size=100000)
    cov = normal(size=(2, 2)) + np.identity(2)
    data2 = multivariate_normal(normal(size=2), 0.5 * (cov + cov.T), size=100000)

    c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$"]).add_chain(data2)
    c.configure_general(colours=["#B32222", "#D1D10D"])
    c.configure_contour(contourf=True, contourf_alpha=0.5)
    c.configure_bar(shade=True)
    fig = c.plot()

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


