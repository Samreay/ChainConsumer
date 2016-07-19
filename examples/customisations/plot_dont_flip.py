"""
=====================
Flips, Ticks and Size
=====================

You can stop the second parameter rotating in the plot if you prefer squares!

Unlike the Introduction example, which shows the rotated plots, this example shows them
without the rotation.

Also, you can pass in a tuple for the figure size. We also demonstrate adding more
ticks to the axis in this example
"""

import numpy as np
from chainconsumer import ChainConsumer

if __name__ == "__main__":
    data = np.random.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=100000)

    c = ChainConsumer().add_chain(data, parameters=["$x_1$", "$x_2$"])
    c.configure_general(flip=False, max_ticks=10)
    fig = c.plot(figsize=(6, 6))

    fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
