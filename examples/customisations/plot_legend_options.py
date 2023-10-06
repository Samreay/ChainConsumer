"""
==============
Legend Options
==============

Legends are hard.

Because of that, you can pass any keywords to the legend call you want via `legend_kwargs`.
"""

import numpy as np

from chainconsumer import ChainConsumer

rng = np.random.default_rng(0)
data1 = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000000)
data2 = data1 + 2

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Chain 1")
c.add_chain(data2, parameters=["$x$", "$y$"], name="Chain 2")
c.configure_overrides(colors=["lb", "g"])
fig = c.plotter.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# If the linestyles are different and the colours are the same, the artists
# will reappear.

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Chain 1")
c.add_chain(data2, parameters=["$x$", "$y$"], name="Chain 2")
c.configure_overrides(colors=["lb", "lb"], linestyles=["-", "--"])
fig = c.plotter.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# You might also want to relocate the legend to another subplot if your
# contours don't have enough space for the legend!

c = ChainConsumer()
c.add_chain(data1, parameters=["$x$", "$y$"], name="Chain 1")
c.add_chain(data2, parameters=["$x$", "$y$"], name="Chain 2")
c.configure_overrides(
    linestyles=["-", "--"],
    sigmas=[0, 1, 2, 3],
    legend_kwargs={"loc": "upper left", "fontsize": 10},
    legend_color_text=False,
    legend_location=(0, 0),
)
fig = c.plotter.plot(figsize=1.5)
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
