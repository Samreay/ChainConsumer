import os
import tempfile

import numpy as np
from scipy.stats import skewnorm, norm, multivariate_normal
import pytest

from chainconsumer import ChainConsumer


class TestChain(object):
    np.random.seed(1)
    n = 2000000
    data = np.random.normal(loc=5.0, scale=1.5, size=n)
    data2 = np.random.normal(loc=3, scale=1.0, size=n)
    data_combined = np.vstack((data, data2)).T
    data_skew = skewnorm.rvs(5, loc=1, scale=1.5, size=n)

    def test_summary(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data[::10])
        consumer.configure(kde=True)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary_no_smooth(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(smooth=0, bins=2.4)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary2(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_combined, parameters=["a", "b"], name="chain1")
        consumer.add_chain(self.data_combined, name="chain2")
        summary = consumer.get_summary()
        k1 = list(summary[0].keys())
        k2 = list(summary[1].keys())
        assert len(k1) == 2
        assert "a" in k1
        assert "b" in k1
        assert len(k2) == 2
        assert "a" in k2
        assert "b" in k2
        expected1 = np.array([3.5, 5.0, 6.5])
        expected2 = np.array([2.0, 3.0, 4.0])
        diff1 = np.abs(expected1 - np.array(list(summary[0]["a"])))
        diff2 = np.abs(expected2 - np.array(list(summary[0]["b"])))
        assert np.all(diff1 < tolerance)
        assert np.all(diff2 < tolerance)

    def test_summary1(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(bins=1.6)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_output_text(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data, parameters=["a"])
        vals = consumer.get_summary()["a"]
        text = consumer.get_parameter_text(*vals)
        assert text == r"5.0\pm 1.5"

    def test_output_text_asymmetric(self):
        p1 = [1.0, 2.0, 3.5]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"2.0^{+1.5}_{-1.0}"

    def test_output_format1(self):
        p1 = [1.0e-1, 2.0e-1, 3.5e-1]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"0.20^{+0.15}_{-0.10}"

    def test_output_format2(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"0.020^{+0.015}_{-0.010}"

    def test_output_format3(self):
        p1 = [1.0e-3, 2.0e-3, 3.5e-3]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{-3}"

    def test_output_format4(self):
        p1 = [1.0e3, 2.0e3, 3.5e3]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{3}"

    def test_output_format5(self):
        p1 = [1.1e6, 2.2e6, 3.3e6]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 2.2\pm 1.1 \right) \times 10^{6}"

    def test_output_format6(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1, wrap=True)
        assert text == r"$0.020^{+0.015}_{-0.010}$"

    def test_output_format7(self):
        p1 = [None, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == ""

    def test_output_format8(self):
        p1 = [-1, -0.0, 1]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"0.0\pm 1.0"

    def test_output_format9(self):
        x = 123456.789
        d = 123.321
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"123460\pm 120"

    def test_output_format10(self):
        x = 123456.789
        d = 1234.321
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 123.5\pm 1.2 \right) \times 10^{3}"

    def test_output_format11(self):
        x = 222.222
        d = 111.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"220\pm 110"

    def test_output_format12(self):
        x = 222.222
        d = 11.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"222\pm 11"

    def test_output_format13(self):
        x = 2222.222
        d = 11.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"2222\pm 11"

    def test_output_format14(self):
        x = 222.222
        d = 1.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"222.2\pm 1.1"

    def test_output_format15(self):
        x = 222.222
        d = 0.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"222.22\pm 0.11"

    def test_output_format16(self):
        x = 222.2222222
        d = 0.0111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"222.222\pm 0.011"

    def test_file_loading1(self):
        data = self.data[:1000]
        directory = tempfile._get_default_tempdir()
        filename = next(tempfile._get_candidate_names())
        filename = directory + os.sep + filename + ".txt"
        np.savetxt(filename, data)
        consumer = ChainConsumer()
        consumer.add_chain(filename)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        assert np.abs(actual[1] - 5.0) < 0.5

    def test_file_loading2(self):
        data = self.data[:1000]
        directory = tempfile._get_default_tempdir()
        filename = next(tempfile._get_candidate_names())
        filename = directory + os.sep + filename + ".npy"
        np.save(filename, data)
        consumer = ChainConsumer()
        consumer.add_chain(filename)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        assert np.abs(actual[1] - 5.0) < 0.5

    def test_using_list(self):
        data = self.data.tolist()
        c = ChainConsumer()
        c.add_chain(data)
        summary = c.get_summary()
        actual = np.array(list(summary.values())[0])
        assert np.abs(actual[1] - 5.0) < 0.1

    def test_using_dict(self):
        data = {"x": self.data, "y": self.data2}
        c = ChainConsumer()
        c.add_chain(data)
        summary = c.get_summary()
        print(c._chains[0].shape)
        deviations = np.abs([summary["x"][1] - 5, summary["y"][1] - 3])
        assert np.all(deviations < 0.1)

    def test_summary_when_no_parameter_names(self):
        c = ChainConsumer()
        c.add_chain(self.data)
        summary = c.get_summary()
        assert list(summary.keys()) == [0]

    def test_squeeze_squeezes(self):
        sum = ChainConsumer().add_chain(self.data).get_summary()
        assert isinstance(sum, dict)

    def test_squeeze_doesnt(self):
        sum = ChainConsumer().add_chain(self.data).get_summary(squeeze=False)
        assert isinstance(sum, list)
        assert len(sum) == 1

    def test_squeeze_doesnt_squeeze_multi(self):
        c = ChainConsumer()
        c.add_chain(self.data).add_chain(self.data)
        sum = c.get_summary()
        assert isinstance(sum, list)
        assert len(sum) == 2

    def test_dictionary_and_parameters_fail(self):
        with pytest.raises(AssertionError):
            ChainConsumer().add_chain({"x": self.data}, parameters=["$x$"])

    def test_convergence_failure(self):
        data = np.concatenate((np.random.normal(loc=0.0, size=10000),
                              np.random.normal(loc=4.0, size=10000)))
        consumer = ChainConsumer()
        consumer.add_chain(data)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        print(actual)
        assert actual[0] is None and actual[2] is None

    def test_divide_chains_default(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        num_walkers = 2
        print(consumer._walkers)

        consumer.add_chain(data, walkers=num_walkers)

        c = consumer.divide_chain()
        c.configure()
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.get_summary()[i].values())[0]
            assert np.abs(stats[1] - means[i]) < 1e-1
            assert np.abs(c._chains[i][:, 0].mean() - means[i]) < 1e-2

    def test_divide_chains_index(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        num_walkers = 2
        consumer.add_chain(data, walkers=num_walkers)

        c = consumer.divide_chain(chain=0)
        c.configure()
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.get_summary()[i].values())[0]
            assert np.abs(stats[1] - means[i]) < 1e-1
            assert np.abs(c._chains[i][:, 0].mean() - means[i]) < 1e-2

    def test_divide_chains_name(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        num_walkers = 2
        consumer.add_chain(data, walkers=num_walkers, name="test")
        c = consumer.divide_chain(chain="test")
        c.configure()
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.get_summary()[i].values())[0]
            assert np.abs(stats[1] - means[i]) < 1e-1
            assert np.abs(c._chains[i][:, 0].mean() - means[i]) < 1e-2

    def test_divide_chains_fail(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=2)
        with pytest.raises(ValueError):
            consumer.divide_chain(chain=2.0)

    def test_divide_chains_name_fail(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=2)
        with pytest.raises(AssertionError):
            c = consumer.divide_chain(chain="notexist")

    def test_stats_max_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(statistics="max")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_max_cliff(self):
        tolerance = 5e-2
        n = 100000
        data = np.linspace(0, 10, n)
        weights = norm.pdf(data, 1, 2)
        consumer = ChainConsumer()
        consumer.add_chain(data, weights=weights)
        consumer.configure(statistics="max", bins=4.0, smooth=1)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([0.0, 1.0, 2.73])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_mean_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(statistics="mean")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_cum_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(statistics="cumulative")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_reject_bad_satst(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        with pytest.raises(AssertionError):
            consumer.configure(statistics="monkey")

    def test_stats_max_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics="max")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.01, 1.55, 2.72])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_mean_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics="mean")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.27, 2.19, 3.11])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_cum_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics="cumulative")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.27, 2.01, 3.11])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_list_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics=["cumulative", "mean"])
        summary = consumer.get_summary()
        actual0 = np.array(list(summary[0].values())[0])
        actual1 = np.array(list(summary[1].values())[0])
        expected0 = np.array([1.27, 2.01, 3.11])
        expected1 = np.array([1.27, 2.19, 3.11])
        diff0 = np.abs(expected0 - actual0)
        diff1 = np.abs(expected1 - actual1)
        assert np.all(diff0 < tolerance)
        assert np.all(diff1 < tolerance)

    def test_weights(self):
        tolerance = 2e-2
        samples = np.linspace(-4, 4, 200000)
        weights = norm.pdf(samples)
        c = ChainConsumer()
        c.add_chain(samples, weights=weights)
        expected = np.array([-1.0, 0.0, 1.0])
        summary = c.get_summary()
        actual = np.array(list(summary.values())[0])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_gelman_rubin_index(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4)
        assert consumer.diagnostic_gelman_rubin(chain=0)

    def test_gelman_rubin_index_not_converged(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000))).T
        data[80000:, :] *= 2
        data[80000:, :] += 1
        consumer = ChainConsumer()

        consumer.add_chain(data, walkers=4)
        assert not consumer.diagnostic_gelman_rubin(chain=0)

    def test_gelman_rubin_index_not_converged(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000))).T
        data[:, 0] += np.linspace(0, 10, 100000)
        consumer = ChainConsumer()

        consumer.add_chain(data, walkers=8)
        assert not consumer.diagnostic_gelman_rubin(chain=0)

    def test_gelman_rubin_index_fails(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4)
        with pytest.raises(AssertionError):
            consumer.diagnostic_gelman_rubin(chain=10)

    def test_gelman_rubin_name(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="testchain")
        assert consumer.diagnostic_gelman_rubin(chain="testchain")

    def test_gelman_rubin_name_fails(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="testchain")
        with pytest.raises(AssertionError):
            consumer.diagnostic_gelman_rubin(chain="testchain2")

    def test_gelman_rubin_unknown_fails(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="testchain")
        with pytest.raises(ValueError):
            consumer.diagnostic_gelman_rubin(chain=np.pi)

    def test_gelman_rubin_default(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="c1")
        consumer.add_chain(data, walkers=4, name="c2")
        consumer.add_chain(data, walkers=4, name="c3")
        assert consumer.diagnostic_gelman_rubin()

    def test_gelman_rubin_default_not_converge(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="c1")
        consumer.add_chain(data, walkers=4, name="c2")
        data2 = data.copy()
        data2[:, 0] += np.linspace(-5, 5, 100000)
        consumer.add_chain(data2, walkers=4, name="c3")
        assert not consumer.diagnostic_gelman_rubin()

    def test_geweke_index(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=20, name="c1")
        assert consumer.diagnostic_geweke(chain=0)

    def test_geweke_index_failed(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        data[98000:, :] += 0.3
        consumer.add_chain(data, walkers=20, name="c1")
        assert not consumer.diagnostic_geweke(chain=0)

    def test_geweke_default(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=20, name="c1")
        consumer.add_chain(data, walkers=20, name="c2")
        assert consumer.diagnostic_geweke(chain=0)

    def test_geweke_default_failed(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=20, name="c1")
        data2 = data.copy()
        data2[98000:, :] += 0.3
        consumer.add_chain(data2, walkers=20, name="c2")
        assert not consumer.diagnostic_geweke()

    def test_grid_data(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        xs, ys = xx.flatten(), yy.flatten()
        chain = np.vstack((xs, ys)).T
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4))
        c = ChainConsumer()
        c.add_chain(chain, parameters=['x', 'y'], weights=pdf, grid=True)
        summary = c.get_summary()
        x_sum = summary['x']
        y_sum = summary['y']
        expected_x = np.array([-1.0, 0.0, 1.0])
        expected_y = np.array([-2.0, 0.0, 2.0])
        threshold = 0.1
        assert np.all(np.abs(expected_x - x_sum) < threshold)
        assert np.all(np.abs(expected_y - y_sum) < threshold)

    def test_extents(self):
        xs = np.random.normal(size=100000)
        weights = np.ones(xs.shape)
        low, high = ChainConsumer()._get_extent(xs, weights)
        threshold = 0.1
        assert np.abs(low + 3) < threshold
        assert np.abs(high - 3) < threshold

    def test_extents_weighted(self):
        xs = np.random.uniform(low=-4, high=4, size=100000)
        weights = norm.pdf(xs)
        low, high = ChainConsumer()._get_extent(xs, weights)
        threshold = 0.1
        assert np.abs(low + 3) < threshold
        assert np.abs(high - 3) < threshold

    def test_grid_list_input(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
        c = ChainConsumer()
        c.add_chain([x, y], parameters=['x', 'y'], weights=pdf, grid=True)
        summary = c.get_summary()
        x_sum = summary['x']
        y_sum = summary['y']
        expected_x = np.array([-1.0, 0.0, 1.0])
        expected_y = np.array([-2.0, 0.0, 2.0])
        threshold = 0.05
        assert np.all(np.abs(expected_x - x_sum) < threshold)
        assert np.all(np.abs(expected_y - y_sum) < threshold)

    def test_grid_dict_input(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
        c = ChainConsumer()
        with pytest.raises(AssertionError):
            c.add_chain({'x': x, 'y': y}, weights=pdf, grid=True)

    def test_grid_dict_input2(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
        c = ChainConsumer()
        c.add_chain({'x': xx.flatten(), 'y': yy.flatten()}, weights=pdf.flatten(), grid=True)
        summary = c.get_summary()
        x_sum = summary['x']
        y_sum = summary['y']
        expected_x = np.array([-1.0, 0.0, 1.0])
        expected_y = np.array([-2.0, 0.0, 2.0])
        threshold = 0.05
        assert np.all(np.abs(expected_x - x_sum) < threshold)
        assert np.all(np.abs(expected_y - y_sum) < threshold)

    def test_normal_list_input(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain([self.data, self.data2], parameters=['x', 'y'])
        # consumer.configure(bins=1.6)
        summary = consumer.get_summary()
        actual1 = summary['x']
        actual2 = summary['y']
        expected1 = np.array([3.5, 5.0, 6.5])
        expected2 = np.array([2.0, 3.0, 4.0])
        diff1 = np.abs(expected1 - actual1)
        diff2 = np.abs(expected2 - actual2)
        assert np.all(diff1 < tolerance)
        assert np.all(diff2 < tolerance)

    def test_grid_3d(self):
        x, y, z = np.linspace(-3, 3, 30), np.linspace(-3, 3, 30), np.linspace(-3, 3, 30)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy + zz * zz))
        c = ChainConsumer()
        c.add_chain([x, y, z], parameters=['x', 'y', 'z'], weights=pdf, grid=True)
        summary = c.get_summary()
        expected = np.array([-1.0, 0.0, 1.0])
        for k in summary:
            assert np.all(np.abs(summary[k] - expected) < 0.2)
