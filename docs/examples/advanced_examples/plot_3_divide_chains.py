"""
# Dividing Chains

It's common with algorithms like MH and other random walkers to have
multiple walkers each providing their own chains. Typically, you
want to verify each walker is burnt in, and then you put all of their
samples into one chain.

But, if your final samples are made up for four walkers each contributing
10k samples, you may want to inspect each walker's surface individually.

In this toy example, all the chains are from the same random generator,
so they're on top of each other. Except MCMC chains to not be as perfect.
"""
from chainconsumer import Chain, ChainConsumer, PlotConfig, make_sample

df = make_sample(num_dimensions=2, seed=3, num_points=40000)

c = ChainConsumer()
combined = Chain(samples=df, name="Model", walkers=4)
for chain in combined.divide():
    c.add_chain(chain)
c.set_plot_config(PlotConfig(flip=True))
fig = c.plotter.plot()
