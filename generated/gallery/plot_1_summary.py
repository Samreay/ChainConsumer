"""
# Introduction to Summaries

When you have a few chains and want to contrast them all with
each other, you probably want a summary plot.

To show you how they work, let's make some sample data that all
has the same average.
"""

from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth, make_sample

# Here's what you might start with
df_1 = make_sample(num_dimensions=4, seed=1)
df_2 = make_sample(num_dimensions=5, seed=2)
print(df_1.head())

# %% New cell
## Using distributions


# And now we give this to chainconsumer
c = ChainConsumer()
c.add_chain(Chain(samples=df_1, name="An Example Contour"))
c.add_chain(Chain(samples=df_2, name="A Different Contour"))
fig = c.plotter.plot_summary()

# %% New cell
# ## Using Errorbars
#
# Note that because the errorbar kwarg is specific to this function
# it is not part of the `PlotConfig` class.

fig = c.plotter.plot_summary(errorbar=True)

# %% New cell
# The other features of ChainConsumer should work with summaries too.
#
# For example, truth values should work just fine.

c.add_truth(Truth(location={"A": 0, "B": 1}, line_style=":", color="red"))
fig = c.plotter.plot_summary(errorbar=True, vertical_spacing_ratio=2.0)

# %% New cell
# And similarly, our overrides are generic and so effect this method too.
c.set_override(ChainConfig(bar_shade=False))
c.set_plot_config(PlotConfig(watermark="Preliminary", blind=["E"]))
fig = c.plotter.plot_summary()
