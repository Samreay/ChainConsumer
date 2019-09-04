# -*- coding: utf-8 -*-
"""
=============
Plot Contours
=============

A trivial example using data from a multivariate normal.

We give truth values, parameter labels and set the figure size to
fit one column of a two column document.

It is important to note that (in future examples) we always call the
configuration methods *after* loading in all the data.

Note that the parameter summaries are all calculated from the chosen bin size and take
into account if the data is being smoothed or not. It is thus important to consider
whether you want smoothing enabled or (depending on your surfaces) more or less
bins than automatically estimated. See the extended customisation examples for
more information.

"""

import numpy as np
from chainconsumer import ChainConsumer

np.random.seed(0)
data = np.random.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=1000000)

c = ChainConsumer()
c.add_chain(data, parameters=["$x_1$", "$x_2$"])
fig = c.plotter.plot(figsize="column", truth=[0.0, 4.0])

# If we wanted to save to file, we would instead have written
# fig = c.plotter.plot(filename="location", figsize="column", truth=[0.0, 4.0])

# If we wanted to display the plot interactively...
# fig = c.plotter.plot(display=True, figsize="column", truth=[0.0, 4.0])

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
