"""
# Introduction to Walks

Want to see if your chain is behaving nicely? Use a walk!
"""
from chainconsumer import Chain, ChainConsumer, Truth, make_sample

# Here's a sample dataset
df_1 = make_sample(num_dimensions=4, seed=1, randomise_mean=True, num_points=10000)

# And now we give this to chainconsumer
c = ChainConsumer()
c.add_chain(Chain(samples=df_1, name="An Example Contour"))
fig = c.plotter.plot_walks()

# %% Second cell
# You can add other chains in if you want, though it can get messy to see things.
#
# To reduce the mess, try turning on convolve, which will
# get you a smoothed out version of the walk. And truth lines are always nice.

c.add_truth(Truth(location={"A": 0, "B": 0}))
fig = c.plotter.plot_walks(convolve=100)
