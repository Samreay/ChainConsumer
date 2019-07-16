# -*- coding: utf-8 -*-
"""
===========================
Plot Parameter Correlations
===========================

You can also get LaTeX tables for parameter correlations.

Turned into glorious LaTeX, we would get something like the following:

.. figure::     ../../examples/resources/correlations.png
    :align:     center
    :width:     60%

"""

###############################################################################
# The code to produce this, and the raw LaTeX, is given below:


import numpy as np
from chainconsumer import ChainConsumer


cov = [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]]
data = np.random.multivariate_normal([0, 0, 1], cov, size=100000)
parameters = ["x", "y", "z"]
c = ChainConsumer()
c.add_chain(data, parameters=parameters)
latex_table = c.analysis.get_correlation_table()

print(latex_table)

