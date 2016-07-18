"""
============
Introduction
============

A trivial example using data from a multivariate normal.

We give truth values, parameter labels and set the figure size to
fit one column of a two column document.

"""

import numpy as np
from chain_consumer import ChainConsumer

if __name__ == "__main__":
    data = np.random.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=100000)

    c = ChainConsumer()
    c.add_chain(data, parameters=["$x_1$", "$x_2$"])
    fig = c.plot(figsize="column", truth=[0.0, 4.0])

    # If we wanted to save to file, we would instead have written
    # fig = c.plot(filename="location", figsize="column", truth=[0.0, 4.0])

    # If we wanted to display the plot interactively...
    # fig = c.plot(display=True, figsize="column", truth=[0.0, 4.0])

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
