---
title: 'ChainConsumer'
tags:
  - Python
  - visualization
  - mcmc
authors:
  - name: Samuel Hinton
    orcid: 0000-0003-2071-9349
    affiliation: University of Queensland
date: 27 July 2016
bibliography: paper.bib
---

# Summary

ChainConsumer is a python package written to consume the output chains
of Monte-Carlo processes and fitting algorithms, such as the results
of MCMC. 

ChainConsumer's main function is to produce plots of the likelihood 
surface inferred from the supplied chain. In addition to showing
the two-dimensional marginalised likelihood surfaces, marginalised
parameter distributions are given, and maximum-likelihood statistics
are used to present parameter constraints. 


In addition to this, parameter constraints can be output
in the form of a LaTeX table. Finally, ChainConsumer also provides 
the functionality to plot the chains as a series of walks in 
parameter values, which provides an easy visual check on chain 
mixing and chain convergence.

Plotting is performed via the matplotlib library [@matplotlib], and 
makes use of various numpy [@numpy] and scipy [@scipy] functions. The
optional KDE feature makes use of [@statsmodels].

Code archives can be found on Zenodo at [@zenodo] and any
bugs or feature requests can be opened as issues on the Github
development page [@github].

-![Likelihood surfaces and marginalised distributions created by ChainConsumer.](example.png)


# References
