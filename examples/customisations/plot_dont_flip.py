# -*- coding: utf-8 -*-
"""
=====================
Flips, Ticks and Size
=====================

You can stop the second parameter rotating in the plot if you prefer squares!

Unlike the Introduction example, which shows the rotated plots, this example shows them
without the rotation.

Also, you can pass in a tuple for the figure size. We also demonstrate adding more
ticks to the axis in this example. Also, I change the colour to red, just for fun.
"""

import numpy as np
from chainconsumer import ChainConsumer

np.random.seed(0)
data = np.random.multivariate_normal([1.5, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=1000000)
data[:, 0] = np.abs(data[:, 0])

c = ChainConsumer().add_chain(data, parameters=["$x_1$", "$x_2$"])
c.configure(flip=False, max_ticks=10, colors="#D32F2F")
fig = c.plotter.plot(figsize=(6, 6))

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
