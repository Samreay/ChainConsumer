=============
ChainConsumer
=============

I wrote this code after realising that the fantastic library
`corner <https://github.com/dfm/corner.py>`_ could not plot everything I
wanted to plot. And so here we are!

To get things started, here is a basic example:

.. code-block:: python

    import numpy as np
    from chain_consumer import ChainConsumer


    mean = [0.0, 4.0]
    data = np.random.multivariate_normal(mean, [[1.0, 0.7], [0.7, 1.5]], size=100000)

    c = ChainConsumer()
    c.add_chain(data, parameters=["$x_1$", "$x_2$"])
    c.plot(filename="example.png", figsize="column", truth=mean)


The output figure is displayed below.

.. figure::     ../example.png
   :align:     center
   :width:     70%

Contents
--------

.. toctree::
   :maxdepth: 2

   chain_api
   examples/index

ChainConsumer requires the following dependencies::

   matplotlib
   numpy
   scipy
   statsmodel
   pytest

Examples for how to use ChainConsumer are given below.

.. include:: examples/index.rst
.. raw:: html

        <div style='clear:both'></div>



