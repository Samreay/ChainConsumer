import numpy as np
from dessn.chain.chain import ChainConsumer


if __name__ == "__main__":
    ndim, nsamples = 3, 200000
    np.random.seed(1)
    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    ChainConsumer()\
        .add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"])\
        .plot(filename="demoOneChain.png")
