import numpy as np
from scipy.stats import norm

from chainconsumer import ChainConsumer


class TestChain:
    rng = np.random.default_rng(1)
    n = 2000000
    data = rng.normal(loc=5.0, scale=1.5, size=n)
    data2 = rng.normal(loc=3, scale=1.0, size=n)

    def test_plotter_extents1(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.configure_overrides()
        minv, maxv = c.plotter._get_parameter_extents("x", c._chains)
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents2(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.add_chain(self.data + 5, parameters=["y"])
        c.configure_overrides()
        minv, maxv = c.plotter._get_parameter_extents("x", c._chains)
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents3(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.add_chain(self.data + 5, parameters=["x"])
        c.configure_overrides()
        minv, maxv = c.plotter._get_parameter_extents("x", c._chains)
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (10.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents4(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.add_chain(self.data + 5, parameters=["y"])
        c.configure_overrides()
        minv, maxv = c.plotter._get_parameter_extents("x", c._chains[:1])
        assert np.isclose(minv, (5.0 - 1.5 * 3.7), atol=0.2)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.7), atol=0.2)

    def test_plotter_extents5(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        xs, ys = xx.flatten(), yy.flatten()
        chain = np.vstack((xs, ys)).T
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4))
        c = ChainConsumer()
        c.add_chain(chain, parameters=["x", "y"], weights=pdf, grid=True)
        c.configure_overrides()
        minv, maxv = c.plotter._get_parameter_extents("x", c._chains)
        assert np.isclose(minv, -3, atol=0.001)
        assert np.isclose(maxv, 3, atol=0.001)

    def test_plotter_extents6(self):
        c = ChainConsumer()
        for mid in np.linspace(-1, 1, 3):
            data = self.rng.normal(loc=0, size=1000)
            posterior = norm.logpdf(data)
            data += mid
            c.add_chain(data, parameters=["x"], posterior=posterior, plot_point=True, plot_contour=False)

        c.configure_overrides()
        minv, maxv = c.plotter._get_parameter_extents("x", c._chains)
        assert np.isclose(minv, -1, atol=0.01)
        assert np.isclose(maxv, 1, atol=0.01)
