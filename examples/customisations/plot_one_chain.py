"""
=========
One Chain
=========

Plot one chain with parameter names.

Because we are only plotting one chain, we will get
parameter bounds on the marginalised surfaces by
default.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer


if __name__ == "__main__":
    np.random.seed(0)
    cov = 1e2 * normal(size=(3, 3))
    data = multivariate_normal(1e3 * normal(size=3), 0.5 * (cov + cov.T), size=100000)

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    fig = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"]).plot()

    fig.set_size_inches(4.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
