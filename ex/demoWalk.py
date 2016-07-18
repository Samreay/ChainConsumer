import numpy as np
from dessn.chain.chain import ChainConsumer


if __name__ == "__main__":
    ndim, nsamples = 3, 200000
    np.random.seed(1)
    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5

    ps = ["$x$", "$y$", r"$\epsilon$"]
    ChainConsumer().add_chain(data, parameters=ps)\
        .plot_walks(filename="demoWalks.png", truth=[0, 5, 0], convolve=100)
