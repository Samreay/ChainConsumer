"""
==========
Plot Walks
==========

You can also plot the walks that your chains have undertaken.

This is a very helpful plot to create when determining if your chains have
converged and mixed. Below is an example walk from a Metropolis-Hastings run,
where we have set the optional parameters for the weights and posteriors,
giving the top two subplots.

.. figure::     ../../examples/resources/exampleWalk.png
    :align:     center

"""


###############################################################################
# To generate your own walk, with a 100 point smoothed walk overplotting,
# you can use the following code:

import numpy as np
from chain_consumer import ChainConsumer

if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.randn(10000, 2)
    data[:, 1] += 1

    ps = ["$x$", "$y$"]
    fig = ChainConsumer().add_chain(data, parameters=ps).plot_walks(truth=[0, 1], convolve=100)

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

