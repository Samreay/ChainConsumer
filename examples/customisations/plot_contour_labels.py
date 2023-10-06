"""
==============
Contour Labels
==============

Plot contours using labels.

You can set the contour_labels to display confidence levels, as shown below.

"""

import numpy as np

from chainconsumer import ChainConsumer

rng = np.random.default_rng(0)
data = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1.0]], size=1000000)


c = ChainConsumer().add_chain(data).configure_overrides(contour_labels="confidence")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


###############################################################################
# Or you can plot in terms of sigma. Note that most people prefer stating
# the confidence levels, because of the ambiguity over sigma levels introduced
# by the `sigma2d` keyword.

c = ChainConsumer().add_chain(data).configure_overrides(contour_labels="sigma")
fig = c.plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
