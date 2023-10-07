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
import pandas as pd

from chainconsumer import Chain, ChainConsumer

# Here's what you might start with
rng = np.random.default_rng(0)
data = rng.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=1000000)
df = pd.DataFrame(data, columns=["x_1", "x_2"])

# And how we give this to chainconsumer
c = ChainConsumer()
c.add_chain(Chain(samples=df, name="An Example Contour"))
fig = c.plotter.plot()

print("whoa")
