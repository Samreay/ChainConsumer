# -*- coding: utf-8 -*-
"""
==================
Watermarking Plots
==================

Make it obvious that those results are preliminary!


It's easy to do, just supply a string to the `watermark` option when plotting your contours,
and remember that when using TeX `matplotlib` settings like `weight` don't do anything -
if you want bold text make it TeX bold.

The code for this is based off the preliminize github repo at
https://github.com/cpadavis/preliminize, which will add watermark to arbitrary
figures!

"""

import numpy as np
from numpy.random import multivariate_normal
from chainconsumer import ChainConsumer

np.random.seed(0)
data1 = multivariate_normal([3, 5], [[1, 0], [0, 1]], size=1000000)
data2 = multivariate_normal([5, 3], [[1, 0], [0, 1]], size=10000)


c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Good results")
c.add_chain(data2, name="Unfinished results")
fig = c.plotter.plot(watermark=r"\textbf{Preliminary}", figsize=2.0)


###############################################################################
# You can also control the text options sent to the matplotlib text call.

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Good results")
c.add_chain(data2, name="Unfinished results")
kwargs = {"color": "purple", "alpha": 1.0, "family": "sanserif", "usetex": False, "weight": "bold"}
c.configure(watermark_text_kwargs=kwargs, flip=True)
fig = c.plotter.plot(watermark="SECRET RESULTS", figsize=2.0)

