"""
========================
Gaussian KDE and Extents
========================

Smooth marginalised distributions with a Gaussian KDE, and pick custom extents.


Note that invoking the KDE on large data sets will significantly increase rendering time.
Also note that you can only invoke KDE on chains without varying weights. This limitation will
be lifted as soon as statsmodel, scipy or scikit-learn add a weighted Gaussian KDE.

Also note that if you pass a floating point number to bins, it multiplies the default bin size
(which is a function of number of steps in the chain) by that amount. If you give it an integer,
it will use that many bins.
"""

import numpy as np
from chain_consumer import ChainConsumer

if __name__ == "__main__":
    data = np.random.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=50000)

    c = ChainConsumer()
    c.add_chain(data)
    c.configure_general(bins=0.9, kde=True)
    c.plot(extents=[(-2, 4), (0, 10)])
