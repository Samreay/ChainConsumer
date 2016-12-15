# [ChainConsumer](https://samreay.github.io/ChainConsumer)

[![Build Status](https://img.shields.io/travis/Samreay/ChainConsumer.svg?style=flat-square)](https://travis-ci.org/Samreay/ChainConsumer)
[![Coverage Status](https://coveralls.io/repos/github/Samreay/ChainConsumer/badge.svg?branch=master)](https://coveralls.io/github/Samreay/ChainConsumer?branch=master)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/dessn/abc/blob/master/LICENSE)
[![PyPi](https://img.shields.io/pypi/v/ChainConsumer.svg?style=flat)](https://pypi.python.org/pypi/ChainConsumer)
[![DOI](https://zenodo.org/badge/23430/Samreay/ChainConsumer.svg)](https://zenodo.org/badge/latestdoi/23430/Samreay/ChainConsumer)
[![JOSS](http://joss.theoj.org/papers/10.21105/joss.00045/status.svg?style=flat)](http://dx.doi.org/10.21105/joss.00045)

A new library to consume your fitting chains! Produce likelihood surfaces,
plot your walks to check convergence, output a LaTeX table of the
marginalised parameter distributions with uncertainties and significant
figures all done for you, or throw in a bunch of chains from different models
and perform some model selection!

[Click through to the online documentation](https://samreay.github.io/ChainConsumer)

### Installation

Install via `pip`:
    
    pip install chainconsumer

### Common Issues

Users on some Linux platforms have reported issues rendering plots using ChainConsumer. 
The common error states that `dvipng: not found`, and as per [StackOverflow](http://stackoverflow.com/a/32915992/3339667)
post, it can be solved by explicitly install the `matplotlib` dependency `dvipng` via `sudo apt-get install dvipng`.



### Update History

##### 0.15.1
* Bugfix for python 2.7


##### 0.15.0
* Adding `usetex` to `configure` method.
* When plotting walks, plots weights in log space if the mean weight is less than 0.1
* Adding AIC
* Adding BIC
* Adding DIC
* Adding method to output model comparison table.


##### 0.14.0
* Adding coloured scatter.
* Disallowing grid data and KDE.
* Adding more examples.
* Consolidating all configures into one method.
* Improved extent finding.
* Updating smoothing to use reflect and not constant.
* Improving `max` statistics being able to find ranges on cliff edges.
* Printing parameter summaries without parameter labels.

##### 0.13.3
* Removing ability to having vectorised dictionary inputs for grid data due to 2.7 compatibility issues.

##### 0.13.2
* Fixing bug when smoothing grid data.
* Adding more input options.
* Grids can now be specified using a list of parameter vectors.

##### 0.13.1
* Better determination of extents for data with extreme weighting.
* Able to scale figure size using float when plotting.

##### 0.13.0
* Modifying API defaults for smoothing with grid data.
* Allowing both smoothing and bins to be passed in as lists.

##### 0.12.0
* Adding support for grid data.

##### 0.11.3
* Fixing bug in Gelman-Rubin statistic

##### 0.11.2
* Improving text labels again.

##### 0.11.1
* Improving text labels for high value data.

##### 0.11.0
* Adding Gelman-Rubin and Geweke diagnostic methods.

##### 0.10.2
* Adding options for alternate statistics.

##### 0.10.1
* Updating setup so that dependencies are automatically installed.

##### 0.10.0
* Modifying the ``add_chain`` API, so that you can pass dictionaries!

##### 0.9.10
* Smarter extent tick labels and offsets.
* Adding list based line styles, widths, shading and opacity.
* Adding two more examples.

##### 0.9.9
* Preconfiguring logging.

##### 0.9.8
* Adding 2D Gaussian smoothing for the contour surfaces.
* Renaming ``contourf`` and ``contourf_alpha`` to ``shade`` and ``shade_alpha``.
* Updating some of the example plots.

##### 0.9.7
* Updating package setup scripts.

##### 0.9.6
* Updating package setup scripts.


##### 0.9.5
* Adding markdown paper.

##### 0.9.4
* Updating setup and package details

##### 0.9.3
* Initial zenodo release

##### 0.9.2
* Adding in smoothing, making it default
* Adding extra example to show how to remove smoothing.

##### 0.9.1
* Adding in tests

##### 0.9.0
* Initial PyPi push