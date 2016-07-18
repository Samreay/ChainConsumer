import numpy as np
from dessn.chain.chain import ChainConsumer


if __name__ == "__main__":
    ndim, nsamples = 4, 200000
    np.random.seed(0)

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5
    data[:, 3] /= (np.abs(data[:, 1]) + 1)

    data2 = np.random.randn(nsamples, ndim)
    data2[:, 0] -= 1
    data2[:, 2] += data2[:, 1]**2
    data2[:, 1] = data2[:, 1] * 2 - 5
    data2[:, 3] = data2[:, 3] * 1.5 + 2

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    c = ChainConsumer()
    c.add_chain(data, parameters=["$x$", "$y$", r"$\alpha$", r"$\beta$"])
    c.add_chain(data2, parameters=["$x$", "$y$", r"$\alpha$", r"$\gamma$"])
    c.plot(display=True, filename="demoTwoDisjointChains.png")
