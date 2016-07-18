"""
Data
====

Fuck off 2
 """
import numpy as np


def get_data():
    ndim, nsamples = 4, 200000
    np.random.seed(0)

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5
    data[:, 3] /= (np.abs(data[:, 1]) + 1)

    data2 = np.random.randn(nsamples, ndim)
    data2[:, 0] -= 1
    data2[:, 2] += data2[:, 1] ** 2
    data2[:, 1] = data2[:, 1] * 2 - 5
    data2[:, 3] = data2[:, 3] * 1.5 + 2

    data3 = np.random.randn(nsamples, ndim)
    data3[:, 2] -= 1
    data3[:, 0] += np.abs(data3[:, 1])
    data3[:, 1] += 2
    data3[:, 1] = data3[:, 2] * 2 - 5

    data4 = (data[:] + 1.0) * 1.5

    return data, data2, data3, data4, ["$x$", "$y$", r"$\alpha$", r"$\beta$"]
