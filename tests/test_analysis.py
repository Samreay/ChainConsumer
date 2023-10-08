import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from chainconsumer import Bound, Chain, ChainConfig, ChainConsumer


class TestChain:
    rng = np.random.default_rng(1)
    n = 3000000
    data = rng.normal(loc=5.0, scale=1.5, size=n)
    data2 = rng.normal(loc=3, scale=1.0, size=n)
    data_combined = np.vstack((data, data2)).T
    data_skew = skewnorm.rvs(5, loc=1, scale=1.5, size=n)

    chain = Chain(samples=pd.DataFrame(data, columns=["x"]), name="a")
    chain2 = Chain(samples=pd.DataFrame(data2, columns=["x"]), name="b")
    chain_combined = Chain(samples=pd.DataFrame(data_combined, columns=["a", "b"]), name="combined")
    chain_skew = Chain(samples=pd.DataFrame(data_skew, columns=["x"]), name="skew")  # type: ignore

    def test_summary(self):
        tolerance = 4e-2
        consumer = ChainConsumer()
        consumer.add_chain(Chain(samples=pd.DataFrame(self.data[::10], columns=["x"]), name="a", kde=True))
        summary = consumer.analysis.get_summary()
        actual = summary["a"]["x"].array
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary_no_smooth(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.chain)
        consumer.set_override(ChainConfig(smooth=0, bins=100))
        summary = consumer.analysis.get_summary()
        actual = summary["a"]["x"].array
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary2(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.chain_combined)
        c2 = self.chain_combined.model_copy()
        c2.name = "chain2"
        consumer.add_chain(c2)
        summary = consumer.analysis.get_summary()

        assert len(summary) == 2
        assert "combined" in summary
        assert "chain2" in summary
        assert "a" in summary["combined"]
        assert "b" in summary["combined"]
        assert "a" in summary["chain2"]
        assert "b" in summary["chain2"]

        expected1 = np.array([3.5, 5.0, 6.5])
        expected2 = np.array([2.0, 3.0, 4.0])
        diff1 = np.abs(expected1 - summary["combined"]["a"].array)
        diff2 = np.abs(expected2 - summary["chain2"]["b"].array)
        assert np.all(diff1 < tolerance)
        assert np.all(diff2 < tolerance)

    def test_summary_disjoint(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.chain)
        c2 = self.chain.model_copy()
        c2.name = "chain2"
        c2.samples = c2.samples.rename(columns={"x": "y"})
        consumer.add_chain(c2)
        summary = consumer.analysis.get_summary(columns=["x"])
        assert len(summary) == 2  # Two chains
        assert not summary["chain2"]  # but this one has no cols
        actual = summary["a"]["x"].array
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary_power(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        data = self.rng.normal(loc=0, scale=np.sqrt(2), size=1000000)
        consumer.add_chain(Chain(samples=pd.DataFrame(data, columns=["x"]), name="A", power=2.0))
        summary = consumer.analysis.get_summary()
        actual = summary["A"]["x"].array
        expected = np.array([-1.0, 0.0, 1.0])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_output_text(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.chain)
        vals = consumer.analysis.get_summary()["a"]
        text = consumer.analysis.get_parameter_text(vals["x"])
        assert text == r"5.0\pm 1.5"

    def test_output_text_asymmetric(self):
        bound = Bound.from_array([1.0, 2.0, 3.5])
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(bound)
        assert text == r"2.0^{+1.5}_{-1.0}"

    def test_output_format1(self):
        p1 = [1.0e-1, 2.0e-1, 3.5e-1]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"0.20^{+0.15}_{-0.10}"

    def test_output_format2(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"0.020^{+0.015}_{-0.010}"

    def test_output_format3(self):
        p1 = [1.0e-3, 2.0e-3, 3.5e-3]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{-3}"

    def test_output_format4(self):
        p1 = [1.0e3, 2.0e3, 3.5e3]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{3}"

    def test_output_format5(self):
        p1 = [1.1e6, 2.2e6, 3.3e6]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"\left( 2.2\pm 1.1 \right) \times 10^{6}"

    def test_output_format6(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1), wrap=True)
        assert text == r"$0.020^{+0.015}_{-0.010}$"

    def test_output_format7(self):
        p1 = [None, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == ""

    def test_output_format8(self):
        p1 = [-1, -0.0, 1]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"0.0\pm 1.0"

    def test_output_format9(self):
        x = 123456.789
        d = 123.321
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"123460\pm 120"

    def test_output_format10(self):
        x = 123456.789
        d = 1234.321
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"\left( 123.5\pm 1.2 \right) \times 10^{3}"

    def test_output_format11(self):
        x = 222.222
        d = 111.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"220\pm 110"

    def test_output_format12(self):
        x = 222.222
        d = 11.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"222\pm 11"

    def test_output_format13(self):
        x = 2222.222
        d = 11.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"2222\pm 11"

    def test_output_format14(self):
        x = 222.222
        d = 1.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"222.2\pm 1.1"

    def test_output_format15(self):
        x = 222.222
        d = 0.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"222.22\pm 0.11"

    def test_output_format16(self):
        x = 222.2222222
        d = 0.0111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"222.222\pm 0.011"

    def test_output_format17(self):
        p1 = [1.0, 1.0, 2.0]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"1.0^{+1.0}_{-0.0}"

    def test_output_format18(self):
        p1 = [10000.0, 10000.0, 10000.0]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"\left( 1.0\pm 0.0 \right) \times 10^{4}"

    def test_output_format19(self):
        p1 = [1.0, 2.0, 2.0]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(Bound.from_array(p1))
        assert text == r"2.0^{+0.0}_{-1.0}"

    def test_divide_chains_name(self):
        data = np.concatenate((self.rng.normal(loc=0.0, size=100000), self.rng.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        num_walkers = 2
        chain = Chain(samples=pd.DataFrame(data, columns=["x"]), walkers=num_walkers, name="test")
        for c in chain.divide():
            consumer.add_chain(c)
        means = [0, 1.0]
        stats = consumer.analysis.get_summary()

        for i in range(num_walkers):
            name = f"test Walker {i}"
            assert name in stats
            array = stats[name]["x"]
            assert np.all(np.abs(array.center - means[i]) < 1e-1)
            assert np.abs(consumer.get_chain(name).get_data("x").mean() - means[i]) < 1e-2

    # def test_stats_max_cliff(self):
    #     tolerance = 5e-2
    #     n = 100000
    #     data = np.linspace(0, 10, n)
    #     weights = norm.pdf(data, 1, 2)
    #     consumer = ChainConsumer()
    #     consumer.add_chain(data, weights=weights)
    #     consumer.configure_overrides(statistics="max", bins=4.0, smooth=1)
    #     summary = consumer.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     expected = np.array([0.0, 1.0, 2.73])
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_stats_mean_normal(self):
    #     tolerance = 5e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data)
    #     consumer.configure_overrides(statistics="mean")
    #     summary = consumer.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     expected = np.array([3.5, 5.0, 6.5])
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_stats_cum_normal(self):
    #     tolerance = 5e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data)
    #     consumer.configure_overrides(statistics="cumulative")
    #     summary = consumer.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     expected = np.array([3.5, 5.0, 6.5])
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_reject_bad_satst(self):
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data)
    #     with pytest.raises(AssertionError):
    #         consumer.configure_overrides(statistics="monkey")

    # def test_stats_max_skew(self):
    #     tolerance = 3e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data_skew)
    #     consumer.configure_overrides(statistics="max")
    #     summary = consumer.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     expected = np.array([1.01, 1.55, 2.72])
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_stats_mean_skew(self):
    #     tolerance = 3e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data_skew)
    #     consumer.configure_overrides(statistics="mean")
    #     summary = consumer.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     expected = np.array([1.27, 2.19, 3.11])
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_stats_cum_skew(self):
    #     tolerance = 3e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data_skew)
    #     consumer.configure_overrides(statistics="cumulative")
    #     summary = consumer.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     expected = np.array([1.27, 2.01, 3.11])
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_stats_list_skew(self):
    #     tolerance = 3e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain(self.data_skew)
    #     consumer.add_chain(self.data_skew)
    #     consumer.configure_overrides(statistics=["cumulative", "mean"])
    #     summary = consumer.analysis.get_summary()
    #     actual0 = np.array(next(iter(summary[0].values())))
    #     actual1 = np.array(next(iter(summary[1].values())))
    #     expected0 = np.array([1.27, 2.01, 3.11])
    #     expected1 = np.array([1.27, 2.19, 3.11])
    #     diff0 = np.abs(expected0 - actual0)
    #     diff1 = np.abs(expected1 - actual1)
    #     assert np.all(diff0 < tolerance)
    #     assert np.all(diff1 < tolerance)

    # def test_weights(self):
    #     tolerance = 3e-2
    #     samples = np.linspace(-4, 4, 200000)
    #     weights = norm.pdf(samples)
    #     c = ChainConsumer()
    #     c.add_chain(samples, weights=weights)
    #     expected = np.array([-1.0, 0.0, 1.0])
    #     summary = c.analysis.get_summary()
    #     actual = np.array(next(iter(summary.values())))
    #     diff = np.abs(expected - actual)
    #     assert np.all(diff < tolerance)

    # def test_grid_data(self):
    #     x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
    #     xx, yy = np.meshgrid(x, y, indexing="ij")
    #     xs, ys = xx.flatten(), yy.flatten()
    #     chain = np.vstack((xs, ys)).T
    #     pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4))
    #     c = ChainConsumer()
    #     c.add_chain(chain, parameters=["x", "y"], weights=pdf, grid=True)
    #     summary = c.analysis.get_summary()
    #     x_sum = summary["x"]
    #     y_sum = summary["y"]
    #     expected_x = np.array([-1.0, 0.0, 1.0])
    #     expected_y = np.array([-2.0, 0.0, 2.0])
    #     threshold = 0.1
    #     assert np.all(np.abs(expected_x - x_sum) < threshold)
    #     assert np.all(np.abs(expected_y - y_sum) < threshold)

    # def test_grid_list_input(self):
    #     x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
    #     xx, yy = np.meshgrid(x, y, indexing="ij")
    #     pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
    #     c = ChainConsumer()
    #     c.add_chain([x, y], parameters=["x", "y"], weights=pdf, grid=True)
    #     summary = c.analysis.get_summary()
    #     x_sum = summary["x"]
    #     y_sum = summary["y"]
    #     expected_x = np.array([-1.0, 0.0, 1.0])
    #     expected_y = np.array([-2.0, 0.0, 2.0])
    #     threshold = 0.05
    #     assert np.all(np.abs(expected_x - x_sum) < threshold)
    #     assert np.all(np.abs(expected_y - y_sum) < threshold)

    # def test_grid_dict_input(self):
    #     x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
    #     xx, yy = np.meshgrid(x, y, indexing="ij")
    #     pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
    #     c = ChainConsumer()
    #     with pytest.raises(AssertionError):
    #         c.add_chain({"x": x, "y": y}, weights=pdf, grid=True)

    # def test_grid_dict_input2(self):
    #     x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
    #     xx, yy = np.meshgrid(x, y, indexing="ij")
    #     pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
    #     c = ChainConsumer()
    #     c.add_chain({"x": xx.flatten(), "y": yy.flatten()}, weights=pdf.flatten(), grid=True)
    #     summary = c.analysis.get_summary()
    #     x_sum = summary["x"]
    #     y_sum = summary["y"]
    #     expected_x = np.array([-1.0, 0.0, 1.0])
    #     expected_y = np.array([-2.0, 0.0, 2.0])
    #     threshold = 0.05
    #     assert np.all(np.abs(expected_x - x_sum) < threshold)
    #     assert np.all(np.abs(expected_y - y_sum) < threshold)

    # def test_normal_list_input(self):
    #     tolerance = 5e-2
    #     consumer = ChainConsumer()
    #     consumer.add_chain([self.data, self.data2], parameters=["x", "y"])
    #     # consumer.configure(bins=1.6)
    #     summary = consumer.analysis.get_summary()
    #     actual1 = summary["x"]
    #     actual2 = summary["y"]
    #     expected1 = np.array([3.5, 5.0, 6.5])
    #     expected2 = np.array([2.0, 3.0, 4.0])
    #     diff1 = np.abs(expected1 - actual1)
    #     diff2 = np.abs(expected2 - actual2)
    #     assert np.all(diff1 < tolerance)
    #     assert np.all(diff2 < tolerance)

    # def test_grid_3d(self):
    #     x, y, z = np.linspace(-3, 3, 30), np.linspace(-3, 3, 30), np.linspace(-3, 3, 30)
    #     xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    #     pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy + zz * zz))
    #     c = ChainConsumer()
    #     c.add_chain([x, y, z], parameters=["x", "y", "z"], weights=pdf, grid=True)
    #     summary = c.analysis.get_summary()
    #     expected = np.array([-1.0, 0.0, 1.0])
    #     for k in summary:
    #         assert np.all(np.abs(summary[k] - expected) < 0.2)

    # def test_correlations_1d(self):
    #     data = self.rng.normal(0, 1, size=100000)
    #     parameters = ["x"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     p, cor = c.analysis.get_correlations()
    #     assert p[0] == "x"
    #     assert np.isclose(cor[0, 0], 1, atol=1e-2)
    #     assert cor.shape == (1, 1)

    # def test_correlations_2d(self):
    #     data = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100000)
    #     parameters = ["x", "y"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     p, cor = c.analysis.get_correlations()
    #     assert p[0] == "x"
    #     assert p[1] == "y"
    #     assert np.isclose(cor[0, 0], 1, atol=1e-2)
    #     assert np.isclose(cor[1, 1], 1, atol=1e-2)
    #     assert np.abs(cor[0, 1]) < 0.01
    #     assert cor.shape == (2, 2)

    # def test_correlations_3d(self):
    #     data = self.rng.multivariate_normal([0, 0, 1], [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]], size=100000)
    #     parameters = ["x", "y", "z"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters, name="chain1")
    #     p, cor = c.analysis.get_correlations(chain="chain1", parameters=["y", "z", "x"])
    #     assert p[0] == "y"
    #     assert p[1] == "z"
    #     assert p[2] == "x"
    #     assert np.isclose(cor[0, 0], 1, atol=1e-2)
    #     assert np.isclose(cor[1, 1], 1, atol=1e-2)
    #     assert np.isclose(cor[2, 2], 1, atol=1e-2)
    #     assert cor.shape == (3, 3)
    #     assert np.abs(cor[0, 1] - 0.3) < 0.01
    #     assert np.abs(cor[0, 2] - 0.5) < 0.01
    #     assert np.abs(cor[1, 2] - 0.2) < 0.01

    # def test_correlations_2d_non_unitary(self):
    #     data = self.rng.multivariate_normal([0, 0], [[4, 0], [0, 4]], size=100000)
    #     parameters = ["x", "y"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     p, cor = c.analysis.get_correlations()
    #     assert p[0] == "x"
    #     assert p[1] == "y"
    #     assert np.isclose(cor[0, 0], 1, atol=1e-2)
    #     assert np.isclose(cor[1, 1], 1, atol=1e-2)
    #     assert np.abs(cor[0, 1]) < 0.01
    #     assert cor.shape == (2, 2)

    # def test_correlation_latex_table(self):
    #     data = self.rng.multivariate_normal([0, 0, 1], [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]], size=1000000)
    #     parameters = ["x", "y", "z"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     latex_table = c.analysis.get_correlation_table()

    #     actual = r"""\begin{table}
    #         \centering
    #         \caption{Parameter Correlations}
    #         \label{tab:parameter_correlations}
    #         \begin{tabular}{c|ccc}
    #              & x & y & z\\
    #             \hline
    #                x  &  1.00 &  0.50 &  0.20 \\
    #                y  &  0.50 &  1.00 &  0.30 \\
    #                z  &  0.20 &  0.30 &  1.00 \\
    #             \hline
    #         \end{tabular}
    #     \end{table}"""
    #     assert latex_table.replace(" ", "") == actual.replace(" ", "")

    # def test_covariance_1d(self):
    #     data = self.rng.normal(0, 2, size=2000000)
    #     parameters = ["x"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     p, cor = c.analysis.get_covariance()
    #     assert p[0] == "x"
    #     assert np.isclose(cor[0, 0], 4, atol=1e-2)
    #     assert cor.shape == (1, 1)

    # def test_covariance_2d(self):
    #     data = self.rng.multivariate_normal([0, 0], [[3, 0], [0, 9]], size=2000000)
    #     parameters = ["x", "y"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     p, cor = c.analysis.get_covariance()
    #     assert p[0] == "x"
    #     assert p[1] == "y"
    #     assert np.isclose(cor[0, 0], 3, atol=2e-2)
    #     assert np.isclose(cor[1, 1], 9, atol=2e-2)
    #     assert np.isclose(cor[0, 1], 0, atol=2e-2)
    #     assert cor.shape == (2, 2)

    # def test_covariance_3d(self):
    #     cov = [[3, 0.5, 0.2], [0.5, 4, 0.3], [0.2, 0.3, 5]]
    #     data = self.rng.multivariate_normal([0, 0, 1], cov, size=2000000)
    #     parameters = ["x", "y", "z"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters, name="chain1")
    #     p, cor = c.analysis.get_covariance(chain="chain1", parameters=["y", "z", "x"])
    #     assert p[0] == "y"
    #     assert p[1] == "z"
    #     assert p[2] == "x"
    #     assert np.isclose(cor[0, 0], 4, atol=2e-2)
    #     assert np.isclose(cor[1, 1], 5, atol=2e-2)
    #     assert np.isclose(cor[2, 2], 3, atol=2e-2)
    #     assert cor.shape == (3, 3)
    #     assert np.abs(cor[0, 1] - 0.3) < 0.01
    #     assert np.abs(cor[0, 2] - 0.5) < 0.01
    #     assert np.abs(cor[1, 2] - 0.2) < 0.01

    # def test_covariance_latex_table(self):
    #     cov = [[2, 0.5, 0.2], [0.5, 3, 0.3], [0.2, 0.3, 4.0]]
    #     data = self.rng.multivariate_normal([0, 0, 1], cov, size=20000000)
    #     parameters = ["x", "y", "z"]
    #     c = ChainConsumer()
    #     c.add_chain(data, parameters=parameters)
    #     latex_table = c.analysis.get_covariance_table()

    #     actual = r"""\begin{table}
    #         \centering
    #         \caption{Parameter Covariance}
    #         \label{tab:parameter_covariance}
    #         \begin{tabular}{c|ccc}
    #              & x & y & z\\
    #             \hline
    #                x  &  2.00 &  0.50 &  0.20 \\
    #                y  &  0.50 &  3.00 &  0.30 \\
    #                z  &  0.20 &  0.30 &  4.00 \\
    #             \hline
    #         \end{tabular}
    #     \end{table}"""
    #     assert latex_table.replace(" ", "") == actual.replace(" ", "")

    # def test_fail_if_more_parameters_than_data(self):
    #     with pytest.raises(AssertionError):
    #         ChainConsumer().add_chain(self.data_combined, parameters=["x", "y", "z"])

    # def test_covariant_covariance_calc(self):
    #     data1 = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
    #     data2 = self.rng.multivariate_normal([0, 0], [[2, 1], [1, 2]], size=10000)
    #     weights = np.concatenate((np.ones(10000), np.zeros(10000)))
    #     data = np.concatenate((data1, data2))
    #     c = ChainConsumer()
    #     c.add_chain(data, weights=weights, parameters=["x", "y"])
    #     p, cor = c.analysis.get_covariance()
    #     assert p[0] == "x"
    #     assert p[1] == "y"
    #     assert np.isclose(cor[0, 0], 1, atol=4e-2)
    #     assert np.isclose(cor[1, 1], 1, atol=4e-2)
    #     assert np.isclose(cor[0, 1], 0, atol=4e-2)
    #     assert cor.shape == (2, 2)

    # def test_2d_levels(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(sigmas=[0, 1, 2], sigma2d=True)
    #     levels = c.plotter._get_levels()
    #     assert np.allclose(levels, [0, 0.39, 0.86], atol=0.01)

    # def test_1d_levels(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(sigmas=[0, 1, 2], sigma2d=False)
    #     levels = c.plotter._get_levels()
    #     assert np.allclose(levels, [0, 0.68, 0.95], atol=0.01)

    # def test_summary_area(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     summary = c.analysis.get_summary()["0"]
    #     expected = [3.5, 5, 6.5]
    #     assert np.all(np.isclose(summary, expected, atol=0.1))

    # def test_summary_area_default(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(summary_area=0.6827)
    #     summary = c.analysis.get_summary()["0"]
    #     expected = [3.5, 5, 6.5]
    #     assert np.all(np.isclose(summary, expected, atol=0.1))

    # def test_summary_area_95(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(summary_area=0.95)
    #     summary = c.analysis.get_summary()["0"]
    #     expected = [2, 5, 8]
    #     assert np.all(np.isclose(summary, expected, atol=0.1))

    # def test_summary_max_symmetric_1(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(statistics="max_symmetric")
    #     summary = c.analysis.get_summary()["0"]
    #     expected = [3.5, 5, 6.5]
    #     assert np.all(np.isclose(summary, expected, atol=0.1))
    #     assert np.isclose(summary[2] - summary[1], summary[1] - summary[0])

    # def test_summary_max_symmetric_2(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data_skew)
    #     summary_area = 0.6827
    #     c.configure_overrides(statistics="max_symmetric", bins=1.0, summary_area=summary_area)
    #     summary = c.analysis.get_summary()["0"]

    #     xs = np.linspace(0, 2, 1000)
    #     pdf = skewnorm.pdf(xs, 5, 1, 1.5)
    #     xmax = xs[pdf.argmax()]
    #     cdf_top = skewnorm.cdf(summary[2], 5, 1, 1.5)
    #     cdf_bottom = skewnorm.cdf(summary[0], 5, 1, 1.5)
    #     area = cdf_top - cdf_bottom

    #     assert np.isclose(xmax, summary[1], atol=0.05)
    #     assert np.isclose(area, summary_area, atol=0.05)
    #     assert np.isclose(summary[2] - summary[1], summary[1] - summary[0])

    # def test_summary_max_symmetric_3(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data_skew)
    #     summary_area = 0.95
    #     c.configure_overrides(statistics="max_symmetric", bins=1.0, summary_area=summary_area)
    #     summary = c.analysis.get_summary()["0"]

    #     xs = np.linspace(0, 2, 1000)
    #     pdf = skewnorm.pdf(xs, 5, 1, 1.5)
    #     xmax = xs[pdf.argmax()]
    #     cdf_top = skewnorm.cdf(summary[2], 5, 1, 1.5)
    #     cdf_bottom = skewnorm.cdf(summary[0], 5, 1, 1.5)
    #     area = cdf_top - cdf_bottom

    #     assert np.isclose(xmax, summary[1], atol=0.05)
    #     assert np.isclose(area, summary_area, atol=0.05)
    #     assert np.isclose(summary[2] - summary[1], summary[1] - summary[0])

    # def test_summary_max_shortest_1(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(statistics="max_shortest")
    #     summary = c.analysis.get_summary()["0"]
    #     expected = [3.5, 5, 6.5]
    #     assert np.all(np.isclose(summary, expected, atol=0.1))

    # def test_summary_max_shortest_2(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data_skew)
    #     summary_area = 0.6827
    #     c.configure_overrides(statistics="max_shortest", bins=1.0, summary_area=summary_area)
    #     summary = c.analysis.get_summary()["0"]

    #     xs = np.linspace(-1, 5, 1000)
    #     pdf = skewnorm.pdf(xs, 5, 1, 1.5)
    #     cdf = skewnorm.cdf(xs, 5, 1, 1.5)
    #     x2 = interp1d(cdf, xs, bounds_error=False, fill_value=np.inf)(cdf + summary_area)
    #     dist = x2 - xs
    #     ind = np.argmin(dist)
    #     x0 = xs[ind]
    #     x2 = x2[ind]
    #     xmax = xs[pdf.argmax()]

    #     assert np.isclose(xmax, summary[1], atol=0.05)
    #     assert np.isclose(x0, summary[0], atol=0.05)
    #     assert np.isclose(x2, summary[2], atol=0.05)

    # def test_summary_max_shortest_3(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data_skew)
    #     summary_area = 0.95
    #     c.configure_overrides(statistics="max_shortest", bins=1.0, summary_area=summary_area)
    #     summary = c.analysis.get_summary()["0"]

    #     xs = np.linspace(-1, 5, 1000)
    #     pdf = skewnorm.pdf(xs, 5, 1, 1.5)
    #     cdf = skewnorm.cdf(xs, 5, 1, 1.5)
    #     x2 = interp1d(cdf, xs, bounds_error=False, fill_value=np.inf)(cdf + summary_area)
    #     dist = x2 - xs
    #     ind = np.argmin(dist)
    #     x0 = xs[ind]
    #     x2 = x2[ind]
    #     xmax = xs[pdf.argmax()]

    #     assert np.isclose(xmax, summary[1], atol=0.05)
    #     assert np.isclose(x0, summary[0], atol=0.05)
    #     assert np.isclose(x2, summary[2], atol=0.05)

    # def test_summary_max_central_1(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data)
    #     c.configure_overrides(statistics="max_central")
    #     summary = c.analysis.get_summary()["0"]
    #     expected = [3.5, 5, 6.5]
    #     assert np.all(np.isclose(summary, expected, atol=0.1))

    # def test_summary_max_central_2(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data_skew)
    #     summary_area = 0.6827
    #     c.configure_overrides(statistics="max_central", bins=1.0, summary_area=summary_area)
    #     summary = c.analysis.get_summary()["0"]

    #     xs = np.linspace(-1, 5, 1000)
    #     pdf = skewnorm.pdf(xs, 5, 1, 1.5)
    #     cdf = skewnorm.cdf(xs, 5, 1, 1.5)
    #     xval = interp1d(cdf, xs)([0.5 - 0.5 * summary_area, 0.5 + 0.5 * summary_area])
    #     xmax = xs[pdf.argmax()]

    #     assert np.isclose(xmax, summary[1], atol=0.05)
    #     assert np.isclose(xval[0], summary[0], atol=0.05)
    #     assert np.isclose(xval[1], summary[2], atol=0.05)

    # def test_summary_max_central_3(self):
    #     c = ChainConsumer()
    #     c.add_chain(self.data_skew)
    #     summary_area = 0.95
    #     c.configure_overrides(statistics="max_central", bins=1.0, summary_area=summary_area)
    #     summary = c.analysis.get_summary()["0"]

    #     xs = np.linspace(-1, 5, 1000)
    #     pdf = skewnorm.pdf(xs, 5, 1, 1.5)
    #     cdf = skewnorm.cdf(xs, 5, 1, 1.5)
    #     xval = interp1d(cdf, xs)([0.5 - 0.5 * summary_area, 0.5 + 0.5 * summary_area])
    #     xmax = xs[pdf.argmax()]

    #     assert np.isclose(xmax, summary[1], atol=0.05)
    #     assert np.isclose(xval[0], summary[0], atol=0.05)
    #     assert np.isclose(xval[1], summary[2], atol=0.05)

    # def test_max_likelihood_1(self):
    #     c = ChainConsumer()
    #     data = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
    #     posterior = norm.logpdf(data).sum(axis=1)
    #     data[:, 1] += 1
    #     c.add_chain(data, parameters=["x", "y"], posterior=posterior, name="A")
    #     result = c.analysis.get_max_posteriors()
    #     x, y = result["x"], result["y"]
    #     assert np.isclose(x, 0, atol=0.05)
    #     assert np.isclose(y, 1, atol=0.05)

    # def test_max_likelihood_2(self):
    #     c = ChainConsumer()
    #     data = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
    #     posterior = norm.logpdf(data).sum(axis=1)
    #     data[:, 1] += 2
    #     c.add_chain(data, parameters=["x", "y"], posterior=posterior, name="A")
    #     c.add_chain(data + 3, parameters=["x", "y"], name="B")
    #     result = c.analysis.get_max_posteriors(parameters=["x", "y"], chains="A")
    #     x, y = result["x"], result["y"]
    #     assert np.isclose(x, 0, atol=0.05)
    #     assert np.isclose(y, 2, atol=0.05)

    # def test_max_likelihood_3(self):
    #     c = ChainConsumer()
    #     data = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
    #     posterior = norm.logpdf(data).sum(axis=1)
    #     data[:, 1] += 3
    #     c.add_chain(data, parameters=["x", "y"], posterior=posterior, name="A")
    #     c.add_chain(data + 3, parameters=["x", "y"], name="B")
    #     result = c.analysis.get_max_posteriors(chains="A")
    #     x, y = result["x"], result["y"]
    #     assert np.isclose(x, 0, atol=0.05)
    #     assert np.isclose(y, 3, atol=0.05)

    # def test_max_likelihood_4(self):
    #     c = ChainConsumer()
    #     data = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
    #     posterior = norm.logpdf(data).sum(axis=1)
    #     data[:, 1] += 2
    #     c.add_chain(data, parameters=["x", "y"], posterior=posterior, name="A")
    #     c.add_chain(data + 3, parameters=["x", "y"], name="B")
    #     result = c.analysis.get_max_posteriors(parameters="x", chains="A", squeeze=False)
    #     assert len(result) == 1
    #     x = result[0]["x"]
    #     assert np.isclose(x, 0, atol=0.05)

    # def test_max_likelihood_5_failure(self):
    #     c = ChainConsumer()
    #     data = self.rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
    #     data[:, 1] += 2
    #     c.add_chain(data, parameters=["x", "y"], name="A")
    #     result = c.analysis.get_max_posteriors(parameters="x", chains="A")
    #     assert result is None
