import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from chainconsumer.chain import Chain


class TestChain:
    rng = np.random.default_rng(0)
    d = rng.normal(size=(100, 3))
    d2 = rng.normal(size=(1000000, 3))
    p = ("a", "b", "c")
    df = pd.DataFrame(d, columns=p)
    dfw = df.assign(weight=1)
    dfp = df.assign(log_posterior=1)
    dfp_bad = dfp.copy()
    dfp_bad["log_posterior"][0] = np.nan
    bad = df.copy()
    bad["a"][0] = np.nan
    n = "A"

    def test_good_chain(self):
        Chain(samples=self.df, name=self.n)

    def test_good_chain_weights1(self):
        Chain(samples=self.dfw, name=self.n)

    def test_chain_with_bad_weights1(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.dfw, name=self.n, weight_column="potato")

    def test_chain_with_bad_weights2(self):
        with pytest.raises(ValidationError):
            df2 = self.dfw.copy()
            df2["weight"][10] = np.inf
            Chain(samples=df2, name=self.n)

    def test_chain_with_bad_params2(self):
        with pytest.raises(ValidationError):
            df2 = self.df.copy()
            df2.columns = ["A", "B", 0]
            Chain(samples=df2, name=self.n)

    def test_chain_with_bad_chain(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.bad, name=self.n)

    def test_good_grid(self):
        Chain(samples=self.df, name=self.n, grid=False)

    def test_good_walkers1(self):
        Chain(samples=self.df, name=self.n, walkers=10)

    def test_bad_walkers1(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.df, name=self.n, walkers=2000)

    def test_bad_walkers2(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.df, name=self.n, walkers=11)

    def test_good_posterior1(self):
        Chain(samples=self.dfp, name=self.n)

    def test_bad_posterior1(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.dfp_bad, name=self.n)

    def test_good_num_free_params1(self):
        Chain(samples=self.df, name=self.n, num_free_params=2)

    def test_bad_num_free_params4(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.df, name=self.n, num_free_params=-10)

    def test_good_num_eff_data_points1(self):
        Chain(samples=self.df, name=self.n, num_eff_data_points=2)

    def test_good_num_eff_data_points2(self):
        Chain(samples=self.df, name=self.n, num_eff_data_points=20.4)

    def test_bad_num_eff_data_points2(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.df, name=self.n, num_eff_data_points=np.nan)

    def test_bad_num_eff_data_points3(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.df, name=self.n, num_eff_data_points=np.inf)

    def test_bad_num_eff_data_points4(self):
        with pytest.raises(ValidationError):
            Chain(samples=self.df, name=self.n, num_eff_data_points=-100)

    def test_color_data_p1(self):
        chain = Chain(samples=self.df, name=self.n, color_param="a")
        color_data = chain.color_data
        assert color_data is not None
        assert np.allclose(self.df["a"].to_numpy(), color_data)

    def test_divide(self):
        n_walkers = 10
        result = Chain(samples=self.df, name=self.n, walkers=n_walkers).divide()
        assert len(result) == n_walkers

        for chain in result:
            assert chain.walkers == 1
