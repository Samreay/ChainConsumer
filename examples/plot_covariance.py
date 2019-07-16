# -*- coding: utf-8 -*-
"""
=========================
Plot Parameter Covariance
=========================

You can also get LaTeX tables for parameter covariance.

Turned into glorious LaTeX, we would get something like the following:

.. figure::     ../../examples/resources/covariance.png
    :align:     center
    :width:     60%

"""

###############################################################################
# The code to produce this, and the raw LaTeX, is given below:


import numpy as np
from chainconsumer import ChainConsumer


cov = [[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 3.0]]
data = np.random.multivariate_normal([0, 0, 1], cov, size=1000000)
parameters = ["x", "y", "z"]
c = ChainConsumer()
c.add_chain(data, parameters=parameters)
latex_table = c.analysis.get_covariance_table()
print(latex_table)

