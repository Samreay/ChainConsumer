"""
======================
Cloud and Sigma Levels
======================

Choose custom sigma levels and display point cloud.

In this example we display more sigma levels, turn on the point cloud, and
disable the parameter summaries on the top of the marginalised distributions.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chain_consumer import ChainConsumer

if __name__ == "__main__":
    np.random.seed(1)
    cov = normal(size=(3, 3))
    data = multivariate_normal(normal(size=3), 0.5 * (cov + cov.T), size=100000)

    c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$"])
    c.configure_bar(summary=False)
    c.configure_contour(cloud=True, sigmas=np.linspace(0, 2, 10))
    fig = c.plot()

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
