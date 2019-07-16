# -*- coding: utf-8 -*-
"""
============
Plot Summary
============

Have a bunch of models and want to compare summaries, but in a plot instead of LaTeX? Can do!


"""

###############################################################################
# Lets add a bunch of chains represnting all these different models of ours.

import numpy as np
from chainconsumer import ChainConsumer


def get_instance():
    np.random.seed(0)
    c = ChainConsumer()
    parameters = ["$x$", r"$\Omega_\epsilon$", "$r^2(x_0)$"]
    for name in ["Ref. model", "Test A", "Test B", "Test C"]:
        # Add some random data
        mean = np.random.normal(loc=0, scale=3, size=3)
        sigma = np.random.uniform(low=1, high=3, size=3)
        data = np.random.multivariate_normal(mean=mean, cov=np.diag(sigma**2), size=100000)
        c.add_chain(data, parameters=parameters, name=name)
    return c

###############################################################################
# If we want the full shape of the distributions, well, thats the default
# behaviour!
c = get_instance()
c.configure(bar_shade=True)
c.plotter.plot_summary()

###############################################################################
# But lets make some changes. Say we don't like the colourful text. And we
# want errorbars, not distributions. And some fun truth values.

c = get_instance()
c.configure(legend_color_text=False)
c.configure_truth(ls=":", color="#FB8C00")
c.plotter.plot_summary(errorbar=True, truth=[[0], [-1, 1], [-2, 0, 2]])

###############################################################################
# Even better, lets use our reference model as the truth value and not plot
# it with the others

c = get_instance()
c.configure(legend_color_text=False)
c.configure_truth(ls="-", color="#555555")
c.plotter.plot_summary(errorbar=True, truth="Ref. model", include_truth_chain=False, extra_parameter_spacing=1.5)
