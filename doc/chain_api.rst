
.. _chain_api:

==================
Chain Consumer API
==================

ChainConsumer has a number of different methods that can be access. In the latest version
of ChainConsumer, the increasing number of methods has had them put into smaller classes within
ChainConsumer.

Basic Methods
-------------

The methods found in the ChainConsumer class itself all relate to add, manipulating, and configuring
the chains fed in.

* :func:`chainconsumer.ChainConsumer.add_chain` - Add a chain!
* :func:`chainconsumer.ChainConsumer.add_marker` - Add a marker!
* :func:`chainconsumer.ChainConsumer.add_covariance` - Add a Gaussian to the mix.
* :func:`chainconsumer.ChainConsumer.divide_chain` - Split a chain into multiple chains to inspect each walk.
* :func:`chainconsumer.ChainConsumer.remove_chain` - Remove a chain.
* :func:`chainconsumer.ChainConsumer.configure` - Configure ChainConsumer.
* :func:`chainconsumer.ChainConsumer.configure_truth` - Configure how truth values are plotted.

Plotter Class
-------------

The plotter class, accessible via `chainConsumer.plotter` contains the methods
used for generating plots.


* :func:`chainconsumer.plotter.Plotter.plot` - Plot the posterior surfaces
* :func:`chainconsumer.plotter.Plotter.plot_walks` - Plot the walks to visually inspect convergence.
* :func:`chainconsumer.plotter.Plotter.plot_distributions` - Plot the marginalised distributions only.
* :func:`chainconsumer.plotter.Plotter.plot_summary` - Plot the marginalised distributions only.
* :func:`chainconsumer.plotter.Plotter.plot_contour` - Pass in an axis for a contour plot on an external figure.

Analysis Class
--------------

The plotter class, accessible via `chainConsumer.analysis` contains the methods
used for getting data or LaTeX analysis of the chains fed in.

* :func:`chainconsumer.analysis.Analysis.get_latex_table` - Return a LaTeX table of the parameter summaries.
* :func:`chainconsumer.analysis.Analysis.get_parameter_text` - Return LaTeX text for specified parameter bounds.
* :func:`chainconsumer.analysis.Analysis.get_summary` - Get the parameter bounds for your chains.
* :func:`chainconsumer.analysis.Analysis.get_max_posteriors` - Get the parameters for the point with greatest posterior.
* :func:`chainconsumer.analysis.Analysis.get_correlations` - Get the parameters and correlation matrix for a chain.
* :func:`chainconsumer.analysis.Analysis.get_correlation_table` - Get a chain's correlation matrix as a LaTeX table.
* :func:`chainconsumer.analysis.Analysis.get_covariance` - Get the parameters and covariance matrix for a chain.
* :func:`chainconsumer.analysis.Analysis.get_covariance_table` - Get a chain's covariance matrix as a LaTeX table.



Diagnostic Class
----------------

The plotter class, accessible via `chainConsumer.diagnostic` contains the methods
used for checking chain convergence

* :func:`chainconsumer.diagnostic.Diagnostic.gelman_rubin` - Run the Gelman-Rubin statistic on your chains.
* :func:`chainconsumer.diagnostic.Diagnostic.geweke` - Run the Geweke statistic on your chains.

Model Comparison Class
----------------------


The plotter class, accessible via `chainConsumer.comparison` contains the methods
used for comparing the chains from various models.

* :func:`chainconsumer.comparisons.Comparison.comparison.aic` - Return the AICc values for all chains.
* :func:`chainconsumer.comparisons.Comparison.comparison.bic` - Return the BIC values for all chains.
* :func:`chainconsumer.comparisons.Comparison.comparison.dic` - Return the DIC values for all chains.
* :func:`chainconsumer.comparisons.Comparison.comparison.comparison_table` - Return a LaTeX table comparing models as per the above methods.


The full documentation can be found below.

Full Documentation
------------------

.. autoclass:: chainconsumer.ChainConsumer
    :members:


------


.. autoclass:: chainconsumer.analysis.Analysis
    :members:


------


.. autoclass:: chainconsumer.comparisons.Comparison
    :members:


------


.. autoclass:: chainconsumer.diagnostic.Diagnostic
    :members:


------


.. autoclass:: chainconsumer.plotter.Plotter
    :members:


------
