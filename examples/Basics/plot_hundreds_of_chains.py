# -*- coding: utf-8 -*-
"""
==================
Hundreds of Chains
==================

Sometimes you have a lot of results and want to see the distribution of your results.

When you have hundreds of chains, for example you've fit a model on 100 realsiations
of different data to validate your model, it's impractical to show hundreds of contours.

In this case, it is often desired to show the distribution of maximum likelihood points, which
helps quantify whether your model is biased and the statistical uncertainty of your model.

To do this, you will need to pass in the posterior values. By default, if you pass in enough chains (more than 20),
ChainConsumer will automatically only plot maximum posterior points for chains which have posteriors. You can
explicitly control this by setting `plot_point` and/or `plot_contour` when adding a chain.

Importantly, if you have a set of chains that represent the same thing, you can group them together
by giving the chains the same name. It is also good practise to set the same colour for these chains.

"""
# sphinx_gallery_thumbnail_number = 2

from scipy.stats import multivariate_normal
import numpy as np
from chainconsumer import ChainConsumer


c = ChainConsumer()
for i in range(1000):
   # Generate some data centered at a random location with uncertainty
   # equal to the scatter
   mean = [3, 8]
   cov = [[1.0, 0.5], [0.5, 2.0]]
   mean_scattered = multivariate_normal.rvs(mean=mean, cov=cov)
   data = multivariate_normal.rvs(mean=mean_scattered, cov=cov, size=1000)
   posterior = multivariate_normal.logpdf(data, mean=mean_scattered, cov=cov)
   c.add_chain(data, posterior=posterior, parameters=["$x$", "$y$"], color='r', name="Simulation validation")
fig = c.plotter.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# We can add multiple datasets, and even mix in plotting contours and points
# together. In this example, we generate two sets of data to plot two clusters
# of maximum posterior points. Additionally we show the contours of a
# 'representative' surface in amber.

c = ChainConsumer()
p = ["$x$", "$y$", "$z$"]
for i in range(200):
    # Generate some data centered at a random location with uncertainty
    # equal to the scatter
    mean = [3, 8, 4]
    cov = [[1.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 1.4]]
    mean_scattered = multivariate_normal.rvs(mean=mean, cov=cov)
    data = multivariate_normal.rvs(mean=mean_scattered, cov=cov, size=5000)
    data2 = data + multivariate_normal.rvs(mean=[8, -8, 7], cov=cov)
    posterior = multivariate_normal.logpdf(data, mean=mean_scattered, cov=cov)
    plot_contour = i == 0

    c.add_chain(data, posterior=posterior, parameters=p, color='p', name="Sim1")

    c.add_chain(data2, posterior=posterior, parameters=p, color='k',
                marker_style="+", marker_size=20, name="Sim2", marker_alpha=0.5)

c.add_chain(data + np.array([4, -4, 3]), parameters=p, posterior=posterior, name="Contour Too",
            plot_contour=True, plot_point=True, marker_style="*", marker_size=40,
            color="a", shade=True, shade_alpha=0.3, kde=True, linestyle="--", bar_shade=True)

c.configure(legend_artists=True)

fig = c.plotter.plot()
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# If you've loaded a whole host of chains in, but only want to focus on one
# set, you can also pick out all chains with the same name when plotting.

fig = c.plotter.plot(chains="Sim1")
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Finally, we should clarify what exactly the points mean! If you don't specify
# anything, by defaults the points represent the coordinates of the
# maximum posterior value. However, in high dimensional surfaces, this maximum
# value across all dimensions can be different to the maximum posterior value
# of a 2D slice. If we want to plot, instead of the global maximum as defined
# by the posterior values, the maximum point of each 2D slice, we can specify
# to `configure` that `global_point=False`.

c.configure(legend_artists=True, global_point=False)
fig = c.plotter.plot(chains="Sim1")
fig.set_size_inches(2.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

###############################################################################
# Note here that the histograms have disappeared. This is because the maximal
# point changes for each pair of parameters, and so none of the points can
# be used in a histogram. Whilst one could use the maximum point, marginalising
# across all parameters, this can be misleading if only two parameters
# are requested to be plotted. As such, we do not report histograms for
# the maximal 2D posterior points.
