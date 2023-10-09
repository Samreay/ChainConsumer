import numpy as np
import pandas as pd
from scipy.stats import norm

from chainconsumer import Chain, ChainConsumer


class TestChain:
    rng = np.random.default_rng(1)
    n = 2000000
    data = rng.normal(loc=5.0, scale=1.5, size=n)
    data2 = rng.normal(loc=3, scale=1.0, size=n)

    chain1 = Chain(samples=pd.DataFrame(data, columns=["x"]), name="A")
    chain2 = Chain(samples=pd.DataFrame(data2, columns=["x"]), name="B")

    def test_plotter_extents1(self):
        c = ChainConsumer()
        c.add_chain(self.chain1)
        minv, maxv = c.plotter._get_parameter_extents("x", list(c._chains.values()))
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents2(self):
        c = ChainConsumer()
        c.add_chain(self.chain1)
        chain2 = self.chain2.model_copy()
        chain2.samples["x"] += 5
        chain2.samples = chain2.samples.rename(columns={"x": "y"})
        c.add_chain(chain2)
        minv, maxv = c.plotter._get_parameter_extents("x", list(c._chains.values()))
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents3(self):
        c = ChainConsumer()
        c.add_chain(self.chain1)
        chain2 = self.chain1.model_copy(deep=True)
        chain2.samples["x"] += 5
        chain2.name = "C"
        c.add_chain(chain2)
        minv, maxv = c.plotter._get_parameter_extents("x", list(c._chains.values()))
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (10.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents5(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        xs, ys = xx.flatten(), yy.flatten()
        chain = np.vstack((xs, ys)).T
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4))
        df = pd.DataFrame(chain, columns=["x", "y"])
        df["weight"] = pdf
        c = ChainConsumer()
        c.add_chain(Chain(samples=df, grid=True, name="grid"))
        minv, maxv = c.plotter._get_parameter_extents("x", list(c._chains.values()))
        assert np.isclose(minv, -3, atol=0.001)
        assert np.isclose(maxv, 3, atol=0.001)

    def test_plotter_extents6(self):
        c = ChainConsumer()
        for mid in np.linspace(-1, 1, 3):
            data = self.rng.normal(loc=0, size=1000)
            posterior = norm.logpdf(data)
            data += mid
            df = pd.DataFrame(data, columns=["x"]).assign(log_posterior=posterior)
            c.add_chain(Chain(samples=df, plot_point=True, plot_contour=False, name=f"point only {mid}"))

        minv, maxv = c.plotter._get_parameter_extents("x", list(c._chains.values()))
        assert np.isclose(minv, -1, atol=0.01)
        assert np.isclose(maxv, 1, atol=0.01)
