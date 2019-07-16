# -*- coding: utf-8 -*-
"""
===========
Plot Tables
===========

You can also get LaTeX tables for parameter summaries.

Turned into glorious LaTeX, we would get something like the following:

.. figure::     ../../examples/resources/table.png
    :align:     center

"""

###############################################################################
# The code to produce this, and the raw LaTeX, is given below:


import numpy as np
from chainconsumer import ChainConsumer


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
c.add_chain(data, parameters=["$x$", "$y$", r"$\alpha$", r"$\beta$"], name="Model A")
c.add_chain(data2, parameters=["$x$", "$y$", r"$\alpha$", r"$\gamma$"], name="Model B")
table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
print(table)
