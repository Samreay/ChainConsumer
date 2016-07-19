=============
ChainConsumer
=============

I wrote this code after realising that the fantastic library
`corner <https://github.com/dfm/corner.py>`_ could not plot everything I
wanted to plot. And so here we are!

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

.. figure::     ../example.png
   :align:     center
   :width:     70%

Check out the API and far more :ref:`examples-index` below:

Contents
--------

.. toctree::
   :maxdepth: 2

   chain_api
   examples/index

ChainConsumer requires the following dependencies:

.. literalinclude:: ../requirements.txt

ChainConsumer can be installed as follows::

    pip install chainconsumer