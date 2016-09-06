=============
ChainConsumer
=============

ChainConsumer is a python package designed to do one thing - consume the chains
output from Monte Carlo processes like MCMC. ChainConsumer can utilise these chains
to produce plots of the posterior surface inferred from the chain distributions,
to plot the chains as walks (to check for mixing and convergence), and to output
parameter summaries in the form of LaTeX tables.

To get things started, here is a basic example:

.. code-block:: python

    import numpy as np
    from chainconsumer import ChainConsumer

    mean = [0.0, 4.0]
    data = np.random.multivariate_normal(mean, [[1.0, 0.7], [0.7, 1.5]], size=100000)

    c = ChainConsumer()
    c.add_chain(data, parameters=["$x_1$", "$x_2$"])
    c.plot(filename="example.png", figsize="column", truth=mean)


The output figure is displayed below.

.. figure::     ../paper/example.png
   :align:     center
   :width:     80%

Check out the API and far more :ref:`examples-index` below:

Contents
--------

.. toctree::
   :maxdepth: 2

   chain_api
   usage
   examples/index

Installation
------------

ChainConsumer requires the following dependencies:

.. literalinclude:: ../requirements.txt

ChainConsumer can be installed as follows::

    pip install chainconsumer

Contributing
------------

Users that wish to contribute to this project may do so in a number of ways.
Firstly, for any feature requests, bugs or general ideas, please raise an issue
via `Github <https://github.com/samreay/ChainConsumer/issues>`_.

If you wish to contribute code to the project, please simple fork the project on
Github and then raise a pull request. Pull requests will be reviewed to determine
whether the changes are major or minor in nature, and to ensure all changes are tested.