"""
# Introduction to LaTeX Tables

Because typing those things out is a **massive pain in the ass.**

"""
from chainconsumer import Chain, ChainConsumer, Truth, make_sample

# Here's a sample dataset
n_1, n_2 = 100000, 200000
df_1 = make_sample(num_dimensions=2, seed=0, num_points=n_1)
df_2 = make_sample(num_dimensions=2, seed=1, num_points=n_2)


# Here's what the plot looks like:
c = ChainConsumer()
c.add_chain(Chain(samples=df_1, name="Model A", num_free_params=1, num_eff_data_points=n_1))
c.add_chain(Chain(samples=df_2, name="Model B", num_free_params=2, num_eff_data_points=n_2))
c.add_truth(Truth(location={"A": 0, "B": 1}))
fig = c.plotter.plot()

# %% Second cell
# # Comparing Models
#
# Provided you have the log posteriors, comparing models is easy.

latex_table = c.comparison.comparison_table()
print(latex_table)

# %%
# Granted, it's hard to read a LaTeX table. It'll come out something
# like this, though I took a screenshot a while ago and the data has changed.
# You get the idea though...
#
# ![](../../resources/comparison_table.png)
#
# # Summarising Parameters
#
# Alright, so what if you've compared models and you're happy and want
# to publish that paper!
#
# You can get a LaTeX table of the summary statistics as well.

print(c.analysis.get_latex_table())
# %%
# Which would look like this (though I saved this screenshot out a while ago too)
#
# ![](../../resources/summaries.png)
#
# And sometimes you might want this table transposed if you have a lot of
# parameters and not many models to compare.
#
print(c.analysis.get_latex_table(transpose=True, caption="The best table"))

# %%
#
# There are other things you can do if you dig around in the API, like
# correlations and covariance.
print(c.analysis.get_covariance_table("Model A"))
