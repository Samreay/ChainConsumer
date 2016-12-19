
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
* :func:`chainconsumer.ChainConsumer.remove_chain` - Remove a chain.

Plotting Methods
----------------
* :func:`chainconsumer.ChainConsumer.plot` - Plot the posterior surfaces
* :func:`chainconsumer.ChainConsumer.plot_walks` - Plot the walks to visually inspect convergence.
* :func:`chainconsumer.ChainConsumer.get_latex_table` - Return a LaTeX table of the parameter summaries.
* :func:`chainconsumer.ChainConsumer.get_parameter_text` - Return LaTeX text for specified parameter bounds.
* :func:`chainconsumer.ChainConsumer.get_summary` - Get the parameter bounds for your chains.


Configuration Methods
---------------------
* :func:`chainconsumer.ChainConsumer.configure` - Configure ChainConsumer.
* :func:`chainconsumer.ChainConsumer.configure_truth` - Configure how truth values are plotted.

Diagnostic Methods
------------------
* :func:`chainconsumer.ChainConsumer.diagnostic_gelman_rubin` - Run the Gelman-Rubin statistic on your chains.
* :func:`chainconsumer.ChainConsumer.diagnostic_geweke` - Run the Geweke statistic on your chains.

Model Selection Methods
-----------------------
* :func:`chainconsumer.ChainConsumer.comparison_aic` - Return the AICc values for all chains.
* :func:`chainconsumer.ChainConsumer.comparison_bic` - Return the BIC values for all chains.
* :func:`chainconsumer.ChainConsumer.comparison_dic` - Return the DIC values for all chains.
* :func:`chainconsumer.ChainConsumer.comparison_table` - Return a LaTeX table comparing models as per the above methods.


The full documentation can be found below.

Full Documentation
------------------

.. autoclass:: chainconsumer.ChainConsumer
    :members:
    :undoc-members:
