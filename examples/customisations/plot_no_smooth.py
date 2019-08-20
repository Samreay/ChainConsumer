# -*- coding: utf-8 -*-
"""
============
No Smoothing
============

We can turn off the default gaussian filter on marginalised distributions.

This can be done by setting ``smooth`` to either ``0``, ``None`` or ``False``.
Note that the parameter summaries also have smoothing turned off, and
thus summaries may change.

Fun colour change! And thicker lines!

"""

import numpy as np
from chainconsumer import ChainConsumer

data = np.random.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=100000)

c = ChainConsumer()
c.add_chain(data, parameters=["$x_1$", "$x_2$"])
c.configure(smooth=0, linewidths=2, colors="#673AB7")
fig = c.plotter.plot(figsize="column", truth=[0.0, 4.0])

# If we wanted to save to file, we would instead have written
# fig = c.plotter.plot(filename="location", figsize="column", truth=[0.0, 4.0])

# If we wanted to display the plot interactively...
# fig = c.plotter.plot(display=True, figsize="column", truth=[0.0, 4.0])

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
