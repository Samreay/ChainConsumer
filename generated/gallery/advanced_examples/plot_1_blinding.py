"""
# Blinding Plots


You can blind parameters and not show axis labels very easily!

Just give ChainConsumer the `blind` parameter when plotting. You can specify `True` to blind all parameters,
or give it a string (or list of strings) detailing the specific parameters you want blinded!

"""
from chainconsumer import Chain, ChainConsumer, PlotConfig, make_sample

df = make_sample(num_dimensions=4, seed=1)
c = ChainConsumer()
c.add_chain(Chain(samples=df, name="Blind Me!"))
c.set_plot_config(PlotConfig(blind=["A", "B"]))
fig = c.plotter.plot()

# %%
# Notice the blinding applies to all plots
fig = c.plotter.plot_summary()

# %%
fig = c.plotter.plot_walks()

# %%

fig = c.plotter.plot_distributions()
# %%
# And the LaTeX output
print(c.analysis.get_latex_table())
