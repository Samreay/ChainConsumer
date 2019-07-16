# -*- coding: utf-8 -*-
"""
=======================
Convergence Diagnostics
=======================

How to use the built in convergence diagnostic tests!

"""


###############################################################################
# Here we create some good and bad data, and then run convergence tests on it!


import numpy as np
from numpy.random import normal
from chainconsumer import ChainConsumer

np.random.seed(0)
# Here we have some nice data, and then some bad data,
# where the last part of the chain has walked off, and the first part
# of the chain isn't agreeing with anything else!
data_good = normal(size=100000)
data_bad = data_good.copy()
data_bad += np.linspace(-0.5, 0.5, 100000)
data_bad[98000:] += 2

# Lets load it into ChainConsumer, and pretend 10 walks went into making the chain
c = ChainConsumer()
c.add_chain(data_good, walkers=10, name="good")
c.add_chain(data_bad, walkers=10, name="bad")

# Now, lets check our convergence using the Gelman-Rubin statistic
gelman_rubin_converged = c.diagnostic.gelman_rubin()
# And also using the Geweke metric
geweke_converged = c.diagnostic.geweke()

# Lets just output the results too
print(gelman_rubin_converged, geweke_converged)

###############################################################################
# We can see that both the Gelman-Rubin and Geweke statistics failed.
# Note that by not specifying a chain when calling the diagnostics,
# they are invoked on *all* chains. For example, to invoke the statistic
# on only the second chain we can pass in either the chain index, or the chain
# name:

print(c.diagnostic.geweke(chain="bad"))

###############################################################################
# Finally, note that the statistics are set to fail easily. For example,
# if you have 10 chains and run `diagnostic_gelman_rubin` with the defaults,
# you will get false if *any* parameter of *any* chain has not converged.
# The printed output will let you know which chains and parameters are
# culpable.
