# -*- coding: utf-8 -*-
"""
==========
Statistics
==========

Demonstrates the different statistics you can use with ChainConsumer.


"""

###############################################################################
# By default, ChainConsumer uses maximum likelihood statistics. Thus you do not
# need to explicitly enable maximum likelihood statistics. If you want to
# anyway, the keyword is `"max"`.

import numpy as np
from scipy.stats import skewnorm
from chainconsumer import ChainConsumer

# Lets create some data here to set things up
np.random.seed(0)
data = skewnorm.rvs(5, size=(1000000, 2))
parameters = ["$x$", "$y$"]


# Now the normal way of giving data is passing a numpy array and parameter separately
c = ChainConsumer().add_chain(data, parameters=parameters).configure(statistics="max")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable cumulative statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure(statistics="cumulative")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable mean statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure(statistics="mean")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable maximum symmetric statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure(statistics="max_symmetric")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable maximum closest statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure(statistics="max_shortest")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable maximum central statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure(statistics="max_central")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# We can also take advantage of the ability to pass lists to ChainConsumer's
# configuration to have report different statistics for different chains.
# Please note, I don't recommend you do this in practise, it is just begging
# for confusion.

c = ChainConsumer()
stats = list(c.analysis._summaries.keys())
for stat in stats:
    c.add_chain(data, parameters=parameters, name=stat.replace("_", " ").title())
c.configure(statistics=stats, bar_shade=True)
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
