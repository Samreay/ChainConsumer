"""
# Introduction to Distributions

When you have a few chains and want to contrast them all with
each other, you probably want a summary plot.

To show you how they work, let's make some sample data that all
has the same average.
"""

from chainconsumer import Chain, ChainConsumer, Truth, make_sample

# Here's what you might start with
df_1 = make_sample(num_dimensions=4, seed=1, randomise_mean=True)
df_2 = make_sample(num_dimensions=5, seed=2, randomise_mean=True)
print(df_1.head())

# %% New cell
## Using distributions


# And now we give this to chainconsumer
c = ChainConsumer()
c.add_chain(Chain(samples=df_1, name="An Example Contour"))
fig = c.plotter.plot_distributions()

# %% Second cell
# If you want the summary stats you'll need to keep it just one
# chain. And if you don't want them, you can pass `summarise=False`
# to the `PlotConfig`.
#
# When you add a second chain, you'll see the summaries disappear.

c.add_chain(Chain(samples=df_2, name="Another contour!"))
c.add_truth(Truth(location={"A": 0, "B": 0}))
fig = c.plotter.plot_distributions(col_wrap=3, columns=["A", "B"])
