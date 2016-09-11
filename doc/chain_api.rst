
.. _chain_api:

==================
Chain Consumer API
==================

ChainConsumer has a number of different methods that can be access. They are broken into four main
categories:

General Methods
---------------
* :func:`chainconsumer.ChainConsumer.add_chain` - Add a chain!
* :func:`chainconsumer.ChainConsumer.divide_chain` - Split a chain into multiple chains to inspect each walk.

Plotting Methods
----------------
* :func:`chainconsumer.ChainConsumer.plot` - Plot the posterior surfaces
* :func:`chainconsumer.ChainConsumer.plot_walks` - Plot the walks to visually inspect convergence.
* :func:`chainconsumer.ChainConsumer.get_latex_table` - Return a LaTeX table of the parameter summaries.
* :func:`chainconsumer.ChainConsumer.get_parameter_text` - Return LaTeX text for specified parameter bounds.
* :func:`chainconsumer.ChainConsumer.get_summary` - Get the parameter bounds for your chains.


Configuration Methods
---------------------
* :func:`chainconsumer.ChainConsumer.configure_general` - Configure general settings.
* :func:`chainconsumer.ChainConsumer.configure_bar` - Configure the plots for the marginalised distributions.
* :func:`chainconsumer.ChainConsumer.configure_contour` - Configure the plots for the posterior surfaces.
* :func:`chainconsumer.ChainConsumer.configure_truth` - Configure how truth values are plotted.

Diagnostic Methods
------------------
* :func:`chainconsumer.ChainConsumer.diagnostic_gelman_rubin` - Run the Gelman-Rubin statistic on your chains.


The full documentation can be found below.

Full Documentation
------------------

.. autoclass:: chainconsumer.ChainConsumer
    :members:
    :undoc-members:
