# [ChainConsumer](https://samreay.github.io/ChainConsumer)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/eefe9aa7d4904306877be1e17b952f39)](https://www.codacy.com/app/samuelreay/ChainConsumer?utm_source=github.com&utm_medium=referral&utm_content=Samreay/ChainConsumer&utm_campaign=badger)
[![Build Status](https://img.shields.io/travis/Samreay/ChainConsumer.svg)](https://travis-ci.org/Samreay/ChainConsumer)
[![Coverage Status](https://codecov.io/gh/Samreay/ChainConsumer/branch/master/graph/badge.svg)](https://codecov.io/gh/Samreay/ChainConsumer)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/dessn/abc/blob/master/LICENSE)


[![PyPi](https://img.shields.io/pypi/v/ChainConsumer)](https://pypi.python.org/pypi/ChainConsumer)
[![Conda](https://anaconda.org/samreay/chainconsumer/badges/version.svg)](https://anaconda.org/samreay/chainconsumer)
[![DOI](https://zenodo.org/badge/23430/Samreay/ChainConsumer.svg)](https://zenodo.org/badge/latestdoi/23430/Samreay/ChainConsumer)
[![JOSS](http://joss.theoj.org/papers/10.21105/joss.00045/status.svg?style=flat)](http://dx.doi.org/10.21105/joss.00045)

A library to consume your fitting chains! Produce likelihood surfaces,
plot your walks to check convergence, output a LaTeX table of the
marginalised parameter distributions with uncertainties and significant
figures all done for you, or throw in a bunch of chains from different models
and perform some model selection!

[Click through to the online documentation](https://samreay.github.io/ChainConsumer)

### Installation

Install via `pip`:

    pip install chainconsumer

### Python Versions

Time has ticked on, and now only python 3.10 will be supported. This is because type hints are amazing.

### Developing

1. Clone repo
2. Run `make install`
3. Ensure that you set your python interpreter to the `.venv/bin/python`
4. Code away.

### Contributors

I would like to thank the following people for their contribution in issues, algorithms and code snippets
which have helped improve ChainConsumer:

* Simon Dupourqué (for his PRs and suggestions to make ChainConsumer better for all)
* Chris Davis (check out https://github.com/cpadavis/preliminize)
* Joe Zuntz
* Scott Dedelson
* Elizabeth Krause
* David Parkinson
* Caitlin Adams
* Tom McClintock
* Steven Murray
* J. Michael Burgess
* Matthew Kirby
* Michael Troxel
* Eduardo Rozo
* Warren Morningstar


### Common Issues

Users on some Linux platforms have reported issues rendering plots using ChainConsumer.
The common error states that `dvipng: not found`, and as per [StackOverflow](http://stackoverflow.com/a/32915992/3339667)
post, it can be solved by explicitly install the `matplotlib` dependency `dvipng` via `sudo apt-get install dvipng`.

If you are running on HPC or clusters where you can't install things yourself, users may run into issues where LaTeX or other optional dependencies aren't installed. In this case, set `usetex=False` in `configure` to request matplotlib not try to use TeX. If this does not work, also set `serif=False`, which has helped some uses.

### Update History

##### 1.2.1
* Fixing regression so gridded data has a default smooth of 0. Thanks Rick Kessler for helping debug this.

##### 1.2.0
* Adding truth marker values. Thanks @CaseyMcGrath for [#147](https://github.com/Samreay/ChainConsumer/pull/147)

##### 1.1.2
* Switch `scipy.integrate.simps` to `scipy.integrate.simpsons` for compatibility with new releases

##### 1.1.1
* Adding general `kwargs` to `add_marker` so one can pass `show_label_in_legend` and other customisations through.

##### 1.1.0
* Updating numpyro and arviz translators so you can specify variable names (`var_names`) to include or exclude.
* Adding `histogram_relative_height` property to chains to control histogram height.
* Adding `show_label_in_legend` so you can decide to give a chain a label but not have it pollute the legend.

##### 1.0.3
* Bug fix not using labels instead of column names for parameter title summaries.

##### 1.0.2
* Bug fix for chain divisions.

##### 1.0.1
* Bug fix for the truth labels.

##### 1.0.0
* Huge update to make full use of type hints, pydantic v2, and mkdocs

##### 0.34.0
* Making `usetex=False` and `serif=False` the defaults to reduce LaTeX errors.
##### 0.33.0
* Adding extra padding to bin extents for KDE and smoothing
* Updating watermarking to work with matplotlib v3.0.0+

##### 0.32.0
* Fixing matplotlib axis formatter issue.

##### 0.31.2, 0.31.3
* Conda-forge updates

##### 0.31.1
* Adding ability to display plot as a prior (1D only, no 2D)

##### 0.31.0
* Linking colorbar label font size to global label font size option. Thanks Yucheng-Zhang!
* Allowing chains to be passed in as a pandas DataFrame.
* Statsmodel update means we are now switching to Python 3 only support.

##### 0.30.1
* Updating `matplotlib` dependency version for conda install. Thanks He Jia!

##### 0.30.0
* Bug fix for specifying numeric `loc` to `legend_kwargs`
* Added `shift_params` when adding chains.

##### 0.29.1
* Potential bug fix for `log_space` feature.


##### 0.29.0
* Warning the user if `configure` is called multiple times.
* Allowing parameters to be a number when calling `get_latex_table`
* Adding log scales when plotting.
* Adding the ability to plot a contour on an arbitrary axis via new method `plot_contour`


##### 0.28.0
* Removing `rainbow` option and replacing with `cmap` so you can specify the cmap used, not just rainbow.
* Adding `zorder` configuration option to epxlicitly order contours.
* Adding extra checks to try and catch bad chains on load.

##### 0.27.0
* Now restores default `rcParams` for `usetex` and `font-family` after plotting.
* All logging now under logger name `chainconsumer` to make it easy to hide if needed.
* Formula for computing `shade_alpha` now uses sqrt(num_chains) instead of num_chains.
* `get_latex_table` now accepts a filename input to save the parameters to.
* Adding `add_covariance` to compliment `add_chain` - useful for Fisher matrix forecasts and similar. Just invert it first for me.
* Adding `add_marker` to allow easy inclusion of markers in the plots.

##### 0.26.3
* Adding ability to turn off chain names in `plot_summary`.

##### 0.26.2
* Fixing bug with `plot_walks` that required truth values.
* Fixing flaw in `configure` to allow for updating values.
* Fixing bug where summary values are cached without reference to the summary statistic method.

##### 0.26.1
* Adding ability to plot maximum points on 2D contour, not just global posterior maximum.
* Fixing truth dictionary mutation on `plot_walks`

##### 0.26.0
* Adding ability to pass in a power to raise the surface to for each chain.
* Adding methods to retrieve the maximum posterior point: `Analysis.get_max_posteriors`
* Adding ability to plot maximum posterior points. Can control `marker_size`, `marker_style`, `marker_alpha`, and whether to plot contours, points or both.
* Finishing migration of configuration options you can specify when adding chains rather than configuring all chains with `configure`.

##### 0.25.2
* (Attempting to) enable fully automated releases to Github, PyPI, Zenodo and conda.

##### 0.25.1
* (Attempting to) enable fully automated releases to Github, PyPI, Zenodo and conda.

##### 0.25.0
* Changing default `sigma2d` to `False`. *May chance how your plots are displayed*.
* Allowing format specification when adding chains.
* Making `yule_walker` (and thus all of `statsmodels`) a conditional import.
* Updating minimum version of requirements to reduce issues with install.

##### 0.24.3
* Fixing bug in label rendering for contour sigma labels.
* Improving parsing of `sigma` in `configure`, such that you don't need a leading zero.

##### 0.24.2
* Fixing bug in `get_correlations`.

##### 0.24.1
* Changing default colour order.
* Improving behaviour of `shade_gradient`.

##### 0.24.0
* Refactoring project structure.
* Updating colours for better legibility.
* Setting `shade=True` automatically if `shade_alpha` is overriden.

##### 0.23.2
* Removing `bbox_inches="tight"` due to a bug in matplotlib v2.1.0.
* Adding more colour shortcuts.

##### 0.23.1
* Making rainbow colours slightly more visible by darkening the yellow regions.

##### 0.23.0
* Can now pass a list of filenames to save out, to make generating a PNG and PDF option in one go easier
* Adding method `plot_summary`

##### 0.22.0
* Adding option to specify the confidence interval (area) for parameter summaries.
* Adding three extra methods for parameter summaries from Andrae 2010: max symmetric, max shortest and max central stats.

##### 0.21.7
* Fixing a bug that caused ChainConsumer to crash in some cases when you specified a number of parameters.

##### 0.21.6
* Fixing bug that made parameter ordering incorrect in some circumstances.

##### 0.21.5
* Fixing error when plotting walks with small weights.

##### 0.21.4
* Fixing issue where refactoring broke parameter blinding.

##### 0.21.3
* ChainConsumer now only finds extents of relevant parameters when plotting, instead of all parameters.

##### 0.21.2
* Updating extents so previous updates do something.

##### 0.21.1
* Adding example and code to deal with non-TeX watermarks.

##### 0.21.0
* Increasing extents again.
* Updating legend defaults.
* Code refactor.
* Can now specify which chains to plot when plotting contours.
* Adding watermark text.

##### 0.20.0
* Increase control over legend with kwargs.
* Can specify legend subplot location.
* Increased legend options with coloured text.
* Added `shade_gradient` option.
* Increase the amount of default extent given.

##### 0.19.4
* Adding ability to blind parameters.

##### 0.19.3
* Adding ability to plot contour levels, either in confidence levels or sigma.
* Changed shading defaults.

##### 0.19.2
* Legend gets placed in top right corner now when `plot_hsits` is `False` and there are only two parameters.

##### 0.19.1
* `sigma2d` correctly defaults to `True` now.

##### 0.19.0
* Adding log_weights to the detected colour parameters.
* Contours now support 1D Gaussian levels *and* 2D Gaussian levels (thanks @matthewkirby).

##### 0.18.0
* Adding Matched Elliptical Gaussian Kernel Density Estimator to replace statsmodels KDE.

##### 0.17.4
* Fixing bug in covariance calculation when getting the LaTeX table (did not affect contours)

##### 0.17.3
* Default figure size is now 1.5 inches per parameter, instead of 1. Also decreasing default font size, so that printing summaries is less likely to overlap surfaces.

##### 0.17.2
* Label font size now applies to legend.

##### 0.17.1
* Code quality improvements
* Documentation update

##### 0.17.0
* Refactoring ChainConsumer due to growing size.
* Improve bin limits to reduce overly large bins that form when some low-weight samples are located far away from the mean.
* Fixed issue generating text with one sided distributions.
* Adding ability to specify weights or posterior as the colour parameter.
* Color scatter with uniform weights doesn't have first plot a different color.
* Adding ability to control subplot spacing.
* Adding method `plot_distributions` to quickly plot marginalised distributions.

##### 0.16.5
* Fixing bug in Gelman-Rubin diagnostic. Thanks Warren!

##### 0.16.4
* Moving `rc` parameters before plot creation to fix issues with parallel plot generation.

##### 0.16.3
* Fixing an integer division bug where python 2 contour shading was setting to 0 alpha.

##### 0.16.2
* Fixing bug where tick font size was only honoured when ticks were on an angle.

##### 0.16.1
* Adding ability to specify label font size, tick font size, and whether the ticks should be on an angle.

##### 0.16.0
* Bug fix for those with latest `numpy` which removed a deprecated method I was using.
* Adding ability to get parameter covariance tables.

##### 0.15.7
* Adding ability to get parameter correlation tables.

##### 0.15.6
* Removing unnecessary debug output.

##### 0.15.5
* Can remove lists of chains properly now.

##### 0.15.4
* Adding ability to remove chains.

##### 0.15.3
* Adding ability to plot the walks of multiple chains together.

##### 0.15.2
* Removing unnecessary debug output.

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
