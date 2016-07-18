"""
====================
Truncated Parameters
====================

If you have a large model, you don't need to plot all parameters at once.

Here we only plot the first two parameters. You could also simply pass the number two,
which means the *first* two parameters.
"""

import numpy as np
from numpy.random import normal, random, multivariate_normal
from chain_consumer import ChainConsumer


if __name__ == "__main__":
    np.random.seed(0)
    cov = random(size=(3, 3))
    data = multivariate_normal(normal(size=3), 0.5 * (cov + cov.T), size=200000)
    parameters = ["$x$", "$y$", "$z$"]
    c = ChainConsumer().add_chain(data, parameters=parameters)
    fig = c.plot(parameters=parameters[:2], figsize="page")

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
