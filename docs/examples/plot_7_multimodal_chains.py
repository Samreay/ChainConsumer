"""
# Multimodal distributions

`ChainConsumer` can handle cases where the distributions of your chains are multimodal.
"""

import numpy as np
import pandas as pd

from chainconsumer import Chain, ChainConsumer
from chainconsumer.statistics import SummaryStatistic

# %%
# First, let's build some dummy data

rng = np.random.default_rng(42)
size = 60_000

eta = rng.normal(loc=0.0, scale=0.8, size=size)

phi = np.asarray(
    [rng.gamma(shape=2.5, scale=0.4, size=size // 2) - 3.0, 3.0 - rng.gamma(shape=5.0, scale=0.35, size=(size // 2))]
).flatten()

rng.shuffle(phi)

df = pd.DataFrame({"eta": eta, "phi": phi})

# %%
# To build a multimodal chain, you simply have to pass `multimodal=True` when building the chain. To work, it requires
# you to specify `SummaryStatistic.HDI` as the summary statistic.

chain_multimodal = Chain(
    samples=df.copy(),
    name="posterior-multimodal",
    statistics=SummaryStatistic.HDI,
    multimodal=True,  # <- Here
)

# %%
# Now, if you add this `Chain` to a plotter, it will try to look for sub-intervals and display them.

cc = ChainConsumer()
cc.add_chain(chain_multimodal)
fig = cc.plotter.plot()

# %%
# Let's compare with what would happen if you don't use a multimodal chain. We use the same data as before but don't
# warn `ChainConsumer` that we expect the chains to be multimodal.

chain_unimodal = Chain(samples=df.copy(), name="posterior-unimodal", statistics=SummaryStatistic.HDI, multimodal=False)

cc.add_chain(chain_unimodal)
fig = cc.plotter.plot()

# %%
# Let's try with even more modes.

eta = np.asarray(
    [
        rng.normal(loc=-3, scale=0.8, size=size // 3),
        rng.normal(loc=0.0, scale=0.8, size=size // 3),
        rng.normal(loc=+3, scale=0.8, size=size // 3),
    ]
).flatten()


rng.shuffle(eta)

df = pd.DataFrame({"eta": eta, "phi": phi})

chain_multimodal = Chain(
    samples=df.copy(), name="posterior-multimodal", statistics=SummaryStatistic.HDI, multimodal=True
)

cc = ChainConsumer()
cc.add_chain(chain_multimodal)
fig = cc.plotter.plot()
fig.tight_layout()
