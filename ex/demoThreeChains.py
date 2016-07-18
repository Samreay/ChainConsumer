import numpy as np
from dessn.chain.chain import ChainConsumer


if __name__ == "__main__":
    ndim, nsamples = 3, 200000
    np.random.seed(0)

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5

    data2 = np.random.randn(nsamples, ndim)
    data2[:, 0] -= 1
    data2[:, 2] += data2[:, 1]**2
    data2[:, 1] = data2[:, 1] * 2 - 5

    data3 = np.random.randn(nsamples, ndim)
    data3[:, 2] -= 1
    data3[:, 0] += np.abs(data3[:, 1])
    data3[:, 1] += 2
    data3[:, 1] = data3[:, 2] * 2 - 5

    # If the parameters are the same between chains, you can just pass it the
    # first time, and they will become the default parameters.
    c = ChainConsumer()\
        .add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"], name="Test chain")\
        .add_chain(data2, name="Chain2")\
        .add_chain(data3, name="Chain3")\
        .plot(filename="demoThreeChains.png")
