"""
============
Loading Data
============

Demonstrates the different ways of loading data!

If you want examples of loading grid data, see the grid data example!

"""

###############################################################################
# You can specify truth values using a list (in the same order as the
# declared parameters).

import numpy as np
import pandas as pd
from numpy.random import multivariate_normal
import tempfile
import os
from chainconsumer import ChainConsumer

# Lets create some data here to set things up
np.random.seed(4)
truth = [0, 5]
data = multivariate_normal(truth, np.eye(2), size=100000)
parameters = ["$x$", "$y$"]
df = pd.DataFrame(data, columns=parameters)

directory = tempfile._get_default_tempdir()
filename = next(tempfile._get_candidate_names())
filename1 = directory + os.sep + filename + ".csv"
filename2 = directory + os.sep + filename + ".npy"
df.to_csv(filename1, index=False)
np.save(filename2, data)

# Now the normal way of giving data is passing a # -*- coding: utf-8 -*-numpy array and parameter separately
c = ChainConsumer().add_chain(data, parameters=parameters)
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# You don't actually need to have them as a 2D array, if you have each parameter independently, just list em up!

x, y = data[:, 0], data[:, 1]
c = ChainConsumer().add_chain([x, y], parameters=parameters)
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Of course, a true master uses pandas everywhere
c = ChainConsumer().add_chain(df)
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


###############################################################################
# And yet we can do the same thing using a dictionary:

dictionary = {"$x$": data[:, 0], "$y$": data[:, 1]}
c = ChainConsumer().add_chain(dictionary)
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can pass a filename in containing a text dump of the chain

c = ChainConsumer().add_chain(filename1)
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can pass a filename for a file containing a binary numpy array

c = ChainConsumer().add_chain(filename2, parameters=parameters)
fig = c.plotter.plot(truth=truth)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
