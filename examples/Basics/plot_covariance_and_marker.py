"""
===============================
Covariance, Fisher and Markers!
===============================

Sometimes you want to compare your data to a Fisher matrix projection,
or you just have some Gaussian you want also drawn.

Or maybe its just a random point you want to put on the plot.

It's all easy to do.

"""
# -*- coding: utf-8 -*-
from chainconsumer import ChainConsumer

mean = [1, 5]
cov = [[1, 1], [1, 3]]
parameters = ["a", "b"]

c = ChainConsumer()
c.add_covariance(mean, cov, parameters=parameters, name="Cov")
c.add_marker(mean, parameters=parameters, name="Marker!", marker_style="*", marker_size=100, color="r")
c.configure(usetex=False, serif=False)
fig = c.plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
