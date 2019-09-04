# -*- coding: utf-8 -*-
"""
========
Logscale
========

For when linear is just not good enough.

You can set the `log_scale` property with a boolean, or specify a list of bools, or a list of parameter indexes,
or a list of parameter names, or a dictionary from parameter names to boolean values. Most things work, just give
it a crack.

"""

import numpy as np
from chainconsumer import ChainConsumer
from scipy.stats import lognorm

data = lognorm.rvs(0.95, loc=0, size=(100000, 2))

c = ChainConsumer()
c.add_chain(data, parameters=["$x_1$", "$x_2$"])

fig = c.plotter.plot(figsize="column", log_scales=True)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


###############################################################################
# It's not just for the main corner plot, you can do it anywhere.

fig = c.plotter.plot_walks(log_scales={"$x_1$": False})  # Dict example
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


fig = c.plotter.plot_distributions(log_scales=[True, False])  # list[bool] example
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

