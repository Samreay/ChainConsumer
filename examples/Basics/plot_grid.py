"""
==========
Grid Data!
==========

If you don't have Monte Carlo chains, and have grid evaluations instead, that's fine too!

Just flatten your grid, set the weights to the grid evaluation, and set the grid flag.

Here is a nice diamond that you get from modifying a simple multivariate normal distribution.

"""
import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer


if __name__ == "__main__":
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-7, 7, 100))
    xs, ys = xx.flatten(), yy.flatten()
    data = np.vstack((xs, ys)).T
    pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4 + np.abs(xs * ys)))

    c = ChainConsumer()
    c = ChainConsumer()
    c.add_chain(data, parameters=["$x$", "$y$"], weights=pdf, grid=True)
    fig = c.plot()

    fig.set_size_inches(3.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
