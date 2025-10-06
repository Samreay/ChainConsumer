"""
# Multiple Truth Values

By default truth values are plotted with horizontal
and vertical lines. However, if you have multiple truth
values, this causes intersections which have no meaning.

This can be alleviated visually by ensuring your different
truth values are in different colours or line styles, but if
you end up having numerous truth values, the grid of intersecting
lines can still become an eyesore.

See the below for an example.
"""

from chainconsumer import Chain, ChainConsumer, Truth

cols = ["A", "B"]
chain1 = Chain.from_covariance([0.0, 0.0], [[1.0, -1], [-1, 2]], columns=cols, name="Contour 1")
chain2 = Chain.from_covariance([3.0, 3.0], [[1.0, 1], [1, 2]], columns=cols, name="Contour 2")
c = ChainConsumer()
c.add_chain(chain1)
c.add_chain(chain2)
c.add_truth(Truth(location={"A": 0, "B": 0}, color="#fb7185"))
c.add_truth(Truth(location={"A": 3, "B": 3}, color="#1f2937"))
fig = c.plotter.plot()


# %%
# To fix this, you have two options:
#
# 1. Add markers instead of truth values.
# 2. Tell the truth values to plot using a marker.
#
# To do option 2, simple pass in `marker` to the `Truth` object.
# This will ensure that your marginalised plots still have vertical
# lines at the truth values, while your contour plots will have markers.
#
# You also can pass in `marker_edge_width` and `marker_size` to control
# the appearance of the markers. Note though that matplotlib doesn't let
# you have marker edges for all markers, but if you try to increase
# the line width on a marker which doesn't support it, ChainConsumer
# will raise an error so you're not scratching your head over why the plots
# aren't changing!
c = ChainConsumer()
c.add_chain(chain1)
c.add_chain(chain2)
c.add_truth(Truth(location={"A": 0, "B": 0}, color="#fb7185", marker="o"))
c.add_truth(Truth(location={"A": 3, "B": 3}, color="#1f2937", marker="X"))
fig = c.plotter.plot()

# %%
# The above example will give you points on the subplots showing 2D slices
# of the marginalised posterior, and vertical lines in the 1D marginalised
# histograms. If this makes the histograms too noisy and you only want to
# see the points in the contour plots, consider swapping over to simply
# putting in markers.
#
# I also apologise for the inconsistent naming here - marker becomes
# marker_style. Sorry.
#
c = ChainConsumer()
c.add_chain(chain1)
c.add_marker(
    location={"A": 0, "B": 0},
    name="Truth1",
    color="#fb7185",
    marker_style="o",
)
c.add_marker(
    location={"A": 3, "B": 3},
    name="Truth2",
    color="#1f2937",
    marker_style="X",
)
fig = c.plotter.plot()
