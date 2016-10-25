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
from numpy.random import normal, multivariate_normal
import tempfile
import os
from chainconsumer import ChainConsumer

# Lets create some data here to set things up
np.random.seed(4)
cov = 0.2 * normal(size=(2, 2)) + np.identity(2)
truth = normal(size=2)
data = multivariate_normal(truth, 0.5 * (cov + cov.T), size=100000)
parameters = ["$x$", "$y$"]
directory = tempfile._get_default_tempdir()
filename = next(tempfile._get_candidate_names())
filename1 = directory + os.sep + filename + ".txt"
filename2 = directory + os.sep + filename + ".npy"
np.savetxt(filename1, data)
np.save(filename2, data)

# Now the normal way of giving data is passing a numpy array and parameter separately
c = ChainConsumer().add_chain(data, parameters=parameters)
fig = c.plot(truth=truth)
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# You don't actually need to have them as a 2D array, if you have each parameter independently, just list em up!

x, y = data[:, 0], data[:, 1]
c = ChainConsumer().add_chain([x, y], parameters=parameters)
fig = c.plot(truth=truth)
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


###############################################################################
# And yet we can do the same thing using a dictionary:

dictionary = {"$x$": data[:, 0], "$y$": data[:, 1]}
c = ChainConsumer().add_chain(dictionary)
fig = c.plot(truth=truth)
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can pass a filename in containing a text dump of the chain

c = ChainConsumer().add_chain(filename1, parameters=parameters)
fig = c.plot(truth=truth)
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Or we can pass a filename for a file containing a binary numpy array

c = ChainConsumer().add_chain(filename2, parameters=parameters)
fig = c.plot(truth=truth)
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
