"""
============
Introduction
============

A trivial example using data from a multivariate normal.

We give truth values, parameter labels and set the figure size to
fit one column of a two column document.

"""

import numpy as np
# from chain_consumer import ChainConsumer
#
# mean = np.array([0.0, 4.0])
# cov = np.array([[1.0, 0.7], [0.7, 1.5]])
# data = np.random.multivariate_normal(mean, cov, size=100000)
#
# c = ChainConsumer()
# c.add_chain(data, parameters=["$x_1$", "$x_2$"])
# fig = c.plot(figsize="column", truth=[0.0, 4.0])

import matplotlib.pyplot as plt
plt.plot(np.arange(10))
