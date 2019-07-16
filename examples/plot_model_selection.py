# -*- coding: utf-8 -*-
"""
=====================
Plot Model Comparison
=====================

You can also get LaTeX tables for model comparison.

Turned into glorious LaTeX, we would get something like the following:

.. figure::     ../../examples/resources/table_comparison.png
    :align:     center
    :width:     60%

"""

###############################################################################
# The code to produce this, and the raw LaTeX, is given below:


from scipy.stats import norm
from chainconsumer import ChainConsumer


n = 10000
d1 = norm.rvs(size=n)
p1 = norm.logpdf(d1)
p2 = norm.logpdf(d1, scale=1.1)

c = ChainConsumer()
c.add_chain(d1, posterior=p1, name="Model A", num_eff_data_points=n, num_free_params=4)
c.add_chain(d1, posterior=p2, name="Model B", num_eff_data_points=n, num_free_params=5)
c.add_chain(d1, posterior=p2, name="Model C", num_eff_data_points=n, num_free_params=4)
c.add_chain(d1, posterior=p1, name="Model D", num_eff_data_points=n, num_free_params=14)
table = c.comparison.comparison_table(caption="Model comparisons!")
print(table)
