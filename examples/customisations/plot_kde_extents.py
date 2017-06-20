"""
========================
Gaussian KDE and Extents
========================

Smooth marginalised distributions with a Gaussian KDE, and pick custom extents.


Note that invoking the KDE on large data sets will significantly increase rendering time when
you have a large number of points.

You can see the increase in contour smoothness (without broadening) for when you have a
low number of samples in your chains!
"""

import numpy as np
from chainconsumer import ChainConsumer

if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.multivariate_normal([0.0, 4.0], [[1.0, -0.7], [-0.7, 1.5]], size=3000)

    c = ChainConsumer()
    c.add_chain(data, name="KDE on")
    c.add_chain(data + 1, name="KDE off")
    c.configure(kde=[True, False], shade_alpha=0.1, flip=False)
    fig = c.plotter.plot(extents=[(-2, 3), (0, 8)])

    fig.set_size_inches(4.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
