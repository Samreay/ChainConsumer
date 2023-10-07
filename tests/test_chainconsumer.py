import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from chainconsumer import Chain, ChainConsumer


class TestChainConsumer:
    rng = np.random.default_rng(1)
    n = 2000000
    data = rng.normal(loc=5.0, scale=1.5, size=n)
    data2 = rng.normal(loc=3, scale=1.0, size=n)
    data_combined = np.vstack((data, data2)).T
    data_skew = skewnorm.rvs(5, loc=1, scale=1.5, size=n)

    chain1 = Chain(samples=pd.DataFrame({"a": data}), name="A")
    chain2 = Chain(samples=pd.DataFrame({"b": data2}), name="B")
    chain3 = Chain(samples=pd.DataFrame({"c": data_skew}), name="C")
    chain4 = Chain(samples=pd.DataFrame(data_combined, columns=["a", "b"]), name="D")

    def get(self) -> ChainConsumer:
        c = ChainConsumer().add_chain(self.chain1).add_chain(self.chain2).add_chain(self.chain3).add_chain(self.chain4)
        return c

    def test_get_chain_name(self):
        assert self.get().get_chain(self.chain1.name) == self.chain1

    def test_get_chain_names(self):
        assert self.get().get_names() == ["A", "B", "C", "D"]

    def test_remove_chain_str(self):
        c = self.get()
        c.remove_chain("A")
        assert c.get_names() == ["B", "C", "D"]

    def test_remove_chain_obj(self):
        c = self.get()
        c.remove_chain(self.chain1)
        assert c.get_names() == ["B", "C", "D"]
