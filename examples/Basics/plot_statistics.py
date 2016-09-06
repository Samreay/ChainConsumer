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
c = ChainConsumer().add_chain(data, parameters=parameters).configure_general(statistics="max")
fig = c.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable cumulative statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure_general(statistics="cumulative")
fig = c.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can enable mean statistics

c = ChainConsumer().add_chain(data, parameters=parameters).configure_general(statistics="mean")
fig = c.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# We can also take advantage of the ability to pass lists to ChainConsumer's
# configuration to have report different statistics for different chains.
# Please note, I don't recommend you do this in practise, it is just begging
# for confusion.

c = ChainConsumer().add_chain(data, parameters=parameters)
c.add_chain(data)
c.add_chain(data)
c.configure_general(statistics=["max", "mean", "cumulative"],
                    linestyles=["-", "--", ":"], linewidths=[1, 2, 3])
c.configure_bar(shade=True)
fig = c.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
