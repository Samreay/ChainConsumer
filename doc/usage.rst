=====
Usage
=====

I recommend going straight to the :ref:`chain_api` and
the :ref:`examples-index` page for details on how to use ChainConsumer.

The Process
-----------

The process of using ChainConsumer should be straightforward:

1. Create an instance of ChainConsumer.
2. Add your chains to this instance.
3. Run convergence diagnostics, if desired.
4. Update the configurations if needed (make sure you do this *after* loading in the data).
5. Plot.

The main page and the examples page has code demonstrating these,
so I won't repeat it here.




Statistics
----------

An area of some difference in analyses is how to generate summary statistics
from marginalised posterior distributions. ChainConsumer comes equipped
with the several different methods that can be configured with the
`configure` method. The three methods are:

Maximum Likelihood Statistics
   The default statistic used by ChainConsumer, maximum likelihood statistics
   report asymmetric uncertainty, from the point of maximum likelihood to the
   iso-likelihood levels above and below the maximal point.
Cumulative Statistics
   For cumulative statistics , the lower :math:`1\sigma` confidence bound, mean value,
   and upper bound are respectively given by the value of the cumulative function
   at points :math:`C(x) = 0.15865`, :math:`0.5`, and :math:`0.84135`.
Mean Statistics
   Mean statistics report the same upper and lower confidence bound as cumulative
   statistics, however report symmetric error bars by having the primary statistic
   reported as the mean of the lower and upper bound.
Max Symmetric
   See `Figure 6(1) of Andrae (2010) <https://arxiv.org/pdf/1009.2755.pdf>`_. Maximum
   likelihood with error with symmetric errors to get the desired confidence interval.
Max Shortest
   See `Figure 6(2) of Andrae (2010) <https://arxiv.org/pdf/1009.2755.pdf>`_. Maximum
   likelihood with uncertainty bounds that minimise the distance between bounds.
Max Central
   See `Figure 6(3) of Andrae (2010) <https://arxiv.org/pdf/1009.2755.pdf>`_. Maximum
   likelihood with uncertainty bounds from the CDF (i.e. same as cumulative stats
   but the central point is the maximum likelihood point and not the :math:`x` such that
   :math:`C(x)=0.5`.


All three methods are illustrated below.

.. figure::     ../examples/resources/stats.png
   :align:     center
   :width:     80%

