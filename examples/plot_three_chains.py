"""
============
Three Chains
============

Plot three chains together. Name the chains to get a legend.


"""
import numpy as np
from numpy.random import normal, random, multivariate_normal
from chain_consumer import ChainConsumer


if __name__ == "__main__":
    np.random.seed(0)
    cov = random(size=(3, 3))
    data = multivariate_normal(normal(size=3), 0.5 * (cov + cov.T), size=200000)
    cov = random(size=(3, 3))
    data2 = multivariate_normal(normal(size=3), 0.5 * (cov + cov.T), size=200000)
    cov = random(size=(3, 3))
    data3 = multivariate_normal(normal(size=3), 0.5 * (cov + cov.T), size=200000)

    # If the parameters are the same between chains, you can just pass it the
    # first time, and they will become the default parameters.
    fig = ChainConsumer()\
        .add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"], name="Test chain")\
        .add_chain(data2, name="Chain2")\
        .add_chain(data3, name="Chain3")\
        .plot()

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig because sphinx-gallery truncates them
