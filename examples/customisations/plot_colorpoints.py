"""
=============
Colour Points
=============

Add colour scatter to show an extra dimension.

If we have a secondary parameter that might not be best displayed
as a posterior surface and would be useful to instead give
context to other surfaces, we can select that point to give a
colour mapped scatter plot.

We can *also* display this as a posterior surface by setting
`plot_colour_params=True`, if we wanted.
"""

import numpy as np
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

if __name__ == "__main__":
    np.random.seed(1)
    cov = normal(size=(3, 3))
    data = multivariate_normal(normal(size=3), 0.5 * (cov + cov.T), size=100000)

    c = ChainConsumer().add_chain(data, parameters=["$x$", "$y$", "$z$"])
    c.configure(color_params="$z$")
    fig = c.plot(figsize=2.0)

    fig.set_size_inches(3.0 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
