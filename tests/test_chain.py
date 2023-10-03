import numpy as np
import pandas as pd
import pytest
from numpy.random import normal
from scipy.stats import norm

from chainconsumer.chain import Chain
from chainconsumer.chainconsumer import ChainConsumer


class TestChain:
    d = normal(size=(100, 3))
    d2 = normal(size=(1000000, 3))
    bad = d.copy()
    bad[0, 0] = np.nan
    p = ["a", "b", "c"]
    n = "A"
    w = np.ones(100)
    w2 = np.ones(1000000)

    def test_good_chain(self):
        Chain(self.d, self.p, self.n)

    def test_good_chain_weights1(self):
        Chain(self.d, self.p, self.n, self.w)

    def test_good_chain_weights2(self):
        Chain(self.d, self.p, self.n, self.w[None])

    def test_good_chain_weights3(self):
        Chain(self.d, self.p, self.n, self.w[None].T)

    def test_chain_with_bad_weights1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, weights=np.ones((50, 1)))

    def test_chain_with_bad_weights2(self):
        with pytest.raises(AssertionError):
            w = self.w.copy()
            w[10] = np.inf
            Chain(self.d, self.p, self.n, weights=w)

    def test_chain_with_bad_weights3(self):
        with pytest.raises(AssertionError):
            w = self.w.copy()
            w[10] = np.nan
            Chain(self.d, self.p, self.n, weights=w)

    def test_chain_with_bad_weights4(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, weights=np.ones((50, 2)))

    def test_chain_with_bad_name1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, 1)

    def test_chain_with_bad_name2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, None)

    def test_chain_with_bad_params1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p[:-1], self.n)

    def test_chain_with_bad_params2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, ["A", "B", 0], self.n)

    def test_chain_with_bad_params3(self):
        with pytest.raises(AssertionError):
            Chain(self.d, None, self.n)

    def test_chain_with_bad_chain_initial_success1(self):
        Chain(self.bad, self.p, self.n)

    def test_chain_with_bad_chain_initial_success2(self):
        c = Chain(self.bad, self.p, self.n)
        c.get_data(1)

    def test_chain_with_bad_chain_fails_on_access1(self):
        c = Chain(self.bad, self.p, self.n)
        with pytest.raises(AssertionError):
            c.get_data(0)

    def test_chain_with_bad_chain_fails_on_access2(self):
        c = Chain(self.bad, self.p, self.n)
        with pytest.raises(AssertionError):
            c.get_data(self.p[0])

    def test_good_grid(self):
        Chain(self.d, self.p, self.n, grid=False)

    def test_bad_grid1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, grid=0)

    def test_bad_grid2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, grid=None)

    def test_bad_grid3(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, grid="False")

    def test_good_walkers1(self):
        Chain(self.d, self.p, self.n, walkers=10)

    def test_good_walkers2(self):
        Chain(self.d, self.p, self.n, walkers=10.0)

    def test_bad_walkers1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, walkers=2000)

    def test_bad_walkers2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, walkers=11)

    def test_bad_walkers3(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, walkers="5")

    def test_bad_walkers4(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, walkers=2.5)

    def test_good_posterior1(self):
        Chain(self.d, self.p, self.n, posterior=np.ones(100))

    def test_good_posterior2(self):
        Chain(self.d, self.p, self.n, posterior=np.ones((100, 1)))

    def test_good_posterior3(self):
        Chain(self.d, self.p, self.n, posterior=np.ones((1, 100)))

    def test_bad_posterior1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, posterior=np.ones((2, 50)))

    def test_bad_posterior2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, posterior=np.ones(50))

    def test_bad_posterior3(self):
        posterior = np.ones(100)
        posterior[0] = np.nan
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, posterior=posterior)

    def test_bad_posterior4(self):
        posterior = np.ones(100)
        posterior[0] = np.inf
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, posterior=posterior)

    def test_bad_posterior5(self):
        posterior = np.ones(100)
        posterior[0] = -np.inf
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, posterior=posterior)

    def test_good_num_free_params1(self):
        Chain(self.d, self.p, self.n, num_free_params=2)

    def test_good_num_free_params2(self):
        Chain(self.d, self.p, self.n, num_free_params=2.0)

    def test_bad_num_free_params1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_free_params="2.5")

    def test_bad_num_free_params2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_free_params=np.inf)

    def test_bad_num_free_params3(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_free_params=np.nan)

    def test_bad_num_free_params4(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_free_params=-10)

    def test_good_num_eff_data_points1(self):
        Chain(self.d, self.p, self.n, num_eff_data_points=2)

    def test_good_num_eff_data_points2(self):
        Chain(self.d, self.p, self.n, num_eff_data_points=20.4)

    def test_bad_num_eff_data_points1(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_eff_data_points="2.5")

    def test_bad_num_eff_data_points2(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_eff_data_points=np.nan)

    def test_bad_num_eff_data_points3(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_eff_data_points=np.inf)

    def test_bad_num_eff_data_points4(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_eff_data_points=-100)

    def test_color_data_none(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, name=self.n, weights=self.w, posterior=np.ones(100))
        c.configure(color_params=None)
        chain = c.chains[0]
        assert chain.get_color_data() is None

    def test_color_data_p1(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, name=self.n, weights=self.w, posterior=np.ones(100))
        c.configure(color_params=self.p[0])
        chain = c.chains[0]
        assert np.all(chain.get_color_data() == self.d[:, 0])

    def test_color_data_w(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, name=self.n, weights=self.w, posterior=np.ones(100))
        c.configure(color_params="weights")
        chain = c.chains[0]
        assert np.all(chain.get_color_data() == self.w)

    def test_color_data_logw(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, name=self.n, weights=self.w, posterior=np.ones(100))
        c.configure(color_params="log_weights")
        chain = c.chains[0]
        assert np.all(chain.get_color_data() == np.log(self.w))

    def test_color_data_posterior(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, name=self.n, weights=self.w, posterior=np.ones(100))
        c.configure(color_params="posterior")
        chain = c.chains[0]
        assert np.all(chain.get_color_data() == np.ones(100))

    def test_override_color(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, color="#4286f4")
        c.configure()
        assert c.chains[0].config["color"] == "#4286f4"

    def test_override_linewidth(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, linewidth=2.0)
        c.configure(linewidths=[100])
        assert c.chains[0].config["linewidth"] == 100

    def test_override_linestyle(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, linestyle="--")
        c.configure()
        assert c.chains[0].config["linestyle"] == "--"

    def test_override_shade_alpha(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, shade_alpha=0.8)
        c.configure()
        assert c.chains[0].config["shade_alpha"] == 0.8

    def test_override_kde(self):
        c = ChainConsumer()
        c.add_chain(self.d, parameters=self.p, kde=2.0)
        c.configure()
        assert c.chains[0].config["kde"] == 2.0

    def test_override_kde_grid(self):
        c = ChainConsumer()
        x, y = np.linspace(0, 10, 10), np.linspace(0, 10, 10)
        z = np.ones((10, 10))
        c.add_chain([x, y], weights=z, grid=True, kde=2.0)
        c.configure()
        assert not c.chains[0].config["kde"]

    def test_cache_invalidation(self):
        c = ChainConsumer()
        c.add_chain(normal(size=(1000000, 1)), parameters=["a"])
        c.configure(summary_area=0.68)
        summary1 = c.analysis.get_summary()
        c.configure(summary_area=0.95)
        summary2 = c.analysis.get_summary()
        assert np.isclose(summary1["a"][0], -1, atol=0.03)
        assert np.isclose(summary2["a"][0], -2, atol=0.03)
        assert np.isclose(summary1["a"][2], 1, atol=0.03)
        assert np.isclose(summary2["a"][2], 2, atol=0.03)

    def test_pass_in_dataframe1(self):
        df = pd.DataFrame(self.d2, columns=self.p)
        c = ChainConsumer()
        c.add_chain(df)
        summary1 = c.analysis.get_summary()
        assert np.isclose(summary1["a"][0], -1, atol=0.03)
        assert np.isclose(summary1["a"][1], 0, atol=0.05)
        assert np.isclose(summary1["a"][2], 1, atol=0.03)
        assert np.isclose(summary1["b"][0], -1, atol=0.03)
        assert np.isclose(summary1["c"][0], -1, atol=0.03)

    def test_pass_in_dataframe2(self):
        df = pd.DataFrame(self.d2, columns=self.p)
        df["weight"] = self.w2
        c = ChainConsumer()
        c.add_chain(df)
        summary1 = c.analysis.get_summary()
        assert np.isclose(summary1["a"][0], -1, atol=0.03)
        assert np.isclose(summary1["a"][1], 0, atol=0.05)
        assert np.isclose(summary1["a"][2], 1, atol=0.03)
        assert np.isclose(summary1["b"][0], -1, atol=0.03)
        assert np.isclose(summary1["c"][0], -1, atol=0.03)

    def test_pass_in_dataframe3(self):
        data = np.random.uniform(-4, 6, size=(1000000, 1))
        weight = norm.pdf(data)
        df = pd.DataFrame(data, columns=["a"])
        df["weight"] = weight
        c = ChainConsumer()
        c.add_chain(df)
        summary1 = c.analysis.get_summary()
        assert np.isclose(summary1["a"][0], -1, atol=0.03)
        assert np.isclose(summary1["a"][1], 0, atol=0.05)
        assert np.isclose(summary1["a"][2], 1, atol=0.03)
