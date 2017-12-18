import numpy as np
from numpy.random import normal
import pytest

from chainconsumer.chain import Chain


class TestChain():
    d = normal(size=(100, 3))
    bad = d.copy()
    bad[0, 0] = np.nan
    p = ["a", "b", "c"]
    n = "A"
    w = np.ones(100)

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

    def test_bad_num_eff_data_points3(self):
        with pytest.raises(AssertionError):
            Chain(self.d, self.p, self.n, num_eff_data_points=-100)
