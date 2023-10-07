"""
===================
Change Font Options
===================

Control tick rotation and font sizes.

Here the tick rotation has been turned off, ticks made smaller,
more ticks added, and label size increased!
"""

import numpy as np

from chainconsumer import ChainConsumer

rng = np.random.default_rng(0)
data = rng.multivariate_normal([0, 1, 2], np.eye(3) + 0.2, size=100000)

# If you pass in parameter labels and only one chain, you can also get parameter bounds
c = ChainConsumer()
c.add_chain(data, parameters=["$x$", "$y^2$", r"$\Omega_\beta$"], name="Example")
c.configure_overrides(diagonal_tick_labels=False, tick_font_size=8, label_font_size=25, max_ticks=8)
fig = c.plotter.plot(figsize="column", show_legend=True)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
