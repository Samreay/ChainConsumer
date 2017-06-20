import os
import tempfile

import numpy as np
from numpy import meshgrid
from scipy.interpolate import interp1d
from scipy.stats import skewnorm, norm
import pytest

from chainconsumer import ChainConsumer
from chainconsumer.helpers import get_extents
from chainconsumer.kde import MegKDE


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
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary_no_smooth(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(smooth=0, bins=2.4)
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary2(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_combined, parameters=["a", "b"], name="chain1")
        consumer.add_chain(self.data_combined, name="chain2")
        summary = consumer.analysis.get_summary()
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
        consumer.configure(bins=0.8)
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_output_text(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data, parameters=["a"])
        consumer.configure(bins=0.8)
        vals = consumer.analysis.get_summary()["a"]
        text = consumer.analysis.get_parameter_text(*vals)
        assert text == r"5.0\pm 1.5"

    def test_output_text_asymmetric(self):
        p1 = [1.0, 2.0, 3.5]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"2.0^{+1.5}_{-1.0}"

    def test_output_format1(self):
        p1 = [1.0e-1, 2.0e-1, 3.5e-1]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"0.20^{+0.15}_{-0.10}"

    def test_output_format2(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"0.020^{+0.015}_{-0.010}"

    def test_output_format3(self):
        p1 = [1.0e-3, 2.0e-3, 3.5e-3]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{-3}"

    def test_output_format4(self):
        p1 = [1.0e3, 2.0e3, 3.5e3]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{3}"

    def test_output_format5(self):
        p1 = [1.1e6, 2.2e6, 3.3e6]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"\left( 2.2\pm 1.1 \right) \times 10^{6}"

    def test_output_format6(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1, wrap=True)
        assert text == r"$0.020^{+0.015}_{-0.010}$"

    def test_output_format7(self):
        p1 = [None, 2.0e-2, 3.5e-2]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == ""

    def test_output_format8(self):
        p1 = [-1, -0.0, 1]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"0.0\pm 1.0"

    def test_output_format9(self):
        x = 123456.789
        d = 123.321
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"123460\pm 120"

    def test_output_format10(self):
        x = 123456.789
        d = 1234.321
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"\left( 123.5\pm 1.2 \right) \times 10^{3}"

    def test_output_format11(self):
        x = 222.222
        d = 111.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"220\pm 110"

    def test_output_format12(self):
        x = 222.222
        d = 11.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"222\pm 11"

    def test_output_format13(self):
        x = 2222.222
        d = 11.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"2222\pm 11"

    def test_output_format14(self):
        x = 222.222
        d = 1.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"222.2\pm 1.1"

    def test_output_format15(self):
        x = 222.222
        d = 0.111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"222.22\pm 0.11"

    def test_output_format16(self):
        x = 222.2222222
        d = 0.0111
        p1 = [x - d, x, x + d]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"222.222\pm 0.011"

    def test_output_format17(self):
        p1 = [1.0, 1.0, 2.0]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"1.0^{+1.0}_{-0.0}"

    def test_output_format18(self):
        p1 = [10000.0, 10000.0, 10000.0]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"\left( 1.0\pm 0.0 \right) \times 10^{4}"

    def test_output_format19(self):
        p1 = [1.0, 2.0, 2.0]
        consumer = ChainConsumer()
        text = consumer.analysis.get_parameter_text(*p1)
        assert text == r"2.0^{+0.0}_{-1.0}"

    def test_file_loading1(self):
        data = self.data[:1000]
        directory = tempfile._get_default_tempdir()
        filename = next(tempfile._get_candidate_names())
        filename = directory + os.sep + filename + ".txt"
        np.savetxt(filename, data)
        consumer = ChainConsumer()
        consumer.add_chain(filename)
        summary = consumer.analysis.get_summary()
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
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        assert np.abs(actual[1] - 5.0) < 0.5

    def test_using_list(self):
        data = self.data.tolist()
        c = ChainConsumer()
        c.add_chain(data)
        summary = c.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        assert np.abs(actual[1] - 5.0) < 0.1

    def test_using_dict(self):
        data = {"x": self.data, "y": self.data2}
        c = ChainConsumer()
        c.add_chain(data)
        summary = c.analysis.get_summary()
        print(c._chains[0].shape)
        deviations = np.abs([summary["x"][1] - 5, summary["y"][1] - 3])
        assert np.all(deviations < 0.1)

    def test_summary_when_no_parameter_names(self):
        c = ChainConsumer()
        c.add_chain(self.data)
        summary = c.analysis.get_summary()
        assert list(summary.keys()) == [0]

    def test_squeeze_squeezes(self):
        sum = ChainConsumer().add_chain(self.data).analysis.get_summary()
        assert isinstance(sum, dict)

    def test_squeeze_doesnt(self):
        sum = ChainConsumer().add_chain(self.data).analysis.get_summary(squeeze=False)
        assert isinstance(sum, list)
        assert len(sum) == 1

    def test_squeeze_doesnt_squeeze_multi(self):
        c = ChainConsumer()
        c.add_chain(self.data).add_chain(self.data)
        sum = c.analysis.get_summary()
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
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        assert actual[0] is None and actual[2] is None

    def test_divide_chains_default(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        num_walkers = 2
        consumer.add_chain(data, walkers=num_walkers)

        c = consumer.divide_chain()
        c.configure(bins=0.7)
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.analysis.get_summary()[i].values())[0]
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
        c.configure(bins=0.7)
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.analysis.get_summary()[i].values())[0]
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
        c.configure(bins=0.7)
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.analysis.get_summary()[i].values())[0]
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
        data = np.concatenate((np.random.normal(loc=0.0, size=200000),
                               np.random.normal(loc=1.0, size=200000)))
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=2)
        with pytest.raises(AssertionError):
            c = consumer.divide_chain(chain="notexist")

    def test_stats_max_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(statistics="max")
        summary = consumer.analysis.get_summary()
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
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([0.0, 1.0, 2.73])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_mean_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(statistics="mean")
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_cum_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure(statistics="cumulative")
        summary = consumer.analysis.get_summary()
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
        tolerance = 3e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics="max")
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.01, 1.55, 2.72])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_mean_skew(self):
        tolerance = 3e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics="mean")
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.27, 2.19, 3.11])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_cum_skew(self):
        tolerance = 3e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics="cumulative")
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.27, 2.01, 3.11])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_list_skew(self):
        tolerance = 3e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.add_chain(self.data_skew)
        consumer.configure(statistics=["cumulative", "mean"])
        summary = consumer.analysis.get_summary()
        actual0 = np.array(list(summary[0].values())[0])
        actual1 = np.array(list(summary[1].values())[0])
        expected0 = np.array([1.27, 2.01, 3.11])
        expected1 = np.array([1.27, 2.19, 3.11])
        diff0 = np.abs(expected0 - actual0)
        diff1 = np.abs(expected1 - actual1)
        assert np.all(diff0 < tolerance)
        assert np.all(diff1 < tolerance)

    def test_weights(self):
        tolerance = 3e-2
        samples = np.linspace(-4, 4, 200000)
        weights = norm.pdf(samples)
        c = ChainConsumer()
        c.add_chain(samples, weights=weights)
        expected = np.array([-1.0, 0.0, 1.0])
        summary = c.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_gelman_rubin_index(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4)
        assert consumer.diagnostic.gelman_rubin(chain=0)

    def test_gelman_rubin_index_not_converged(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        data[80000:, :] *= 2
        data[80000:, :] += 1
        consumer = ChainConsumer()

        consumer.add_chain(data, walkers=4)
        assert not consumer.diagnostic.gelman_rubin(chain=0)

    def test_gelman_rubin_index_not_converged(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        data[:, 0] += np.linspace(0, 10, 100000)
        consumer = ChainConsumer()

        consumer.add_chain(data, walkers=8)
        assert not consumer.diagnostic.gelman_rubin(chain=0)

    def test_gelman_rubin_index_fails(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4)
        with pytest.raises(AssertionError):
            consumer.diagnostic.gelman_rubin(chain=10)

    def test_gelman_rubin_name(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="testchain")
        assert consumer.diagnostic.gelman_rubin(chain="testchain")

    def test_gelman_rubin_name_fails(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="testchain")
        with pytest.raises(AssertionError):
            consumer.diagnostic.gelman_rubin(chain="testchain2")

    def test_gelman_rubin_unknown_fails(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="testchain")
        with pytest.raises(ValueError):
            consumer.diagnostic.gelman_rubin(chain=np.pi)

    def test_gelman_rubin_default(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="c1")
        consumer.add_chain(data, walkers=4, name="c2")
        consumer.add_chain(data, walkers=4, name="c3")
        assert consumer.diagnostic.gelman_rubin()

    def test_gelman_rubin_default_not_converge(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=4, name="c1")
        consumer.add_chain(data, walkers=4, name="c2")
        data2 = data.copy()
        data2[:, 0] += np.linspace(-5, 5, 100000)
        consumer.add_chain(data2, walkers=4, name="c3")
        assert not consumer.diagnostic.gelman_rubin()

    def test_geweke_index(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=20, name="c1")
        assert consumer.diagnostic.geweke(chain=0)

    def test_geweke_index_failed(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        data[98000:, :] += 0.3
        consumer.add_chain(data, walkers=20, name="c1")
        assert not consumer.diagnostic.geweke(chain=0)

    def test_geweke_default(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=20, name="c1")
        consumer.add_chain(data, walkers=20, name="c2")
        assert consumer.diagnostic.geweke(chain=0)

    def test_geweke_default_failed(self):
        data = np.vstack((np.random.normal(loc=0.0, size=100000),
                          np.random.normal(loc=1.0, size=100000))).T
        consumer = ChainConsumer()
        consumer.add_chain(data, walkers=20, name="c1")
        data2 = data.copy()
        data2[98000:, :] += 0.3
        consumer.add_chain(data2, walkers=20, name="c2")
        assert not consumer.diagnostic.geweke()

    def test_grid_data(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        xs, ys = xx.flatten(), yy.flatten()
        chain = np.vstack((xs, ys)).T
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4))
        c = ChainConsumer()
        c.add_chain(chain, parameters=['x', 'y'], weights=pdf, grid=True)
        summary = c.analysis.get_summary()
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
        low, high = get_extents(xs, weights)
        threshold = 0.2
        assert np.abs(low + 3) < threshold
        assert np.abs(high - 3) < threshold

    def test_extents_weighted(self):
        xs = np.random.uniform(low=-4, high=4, size=100000)
        weights = norm.pdf(xs)
        low, high = get_extents(xs, weights)
        threshold = 0.1
        assert np.abs(low + 3) < threshold
        assert np.abs(high - 3) < threshold

    def test_grid_list_input(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xx * xx + yy * yy / 4))
        c = ChainConsumer()
        c.add_chain([x, y], parameters=['x', 'y'], weights=pdf, grid=True)
        summary = c.analysis.get_summary()
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
        summary = c.analysis.get_summary()
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
        summary = consumer.analysis.get_summary()
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
        summary = c.analysis.get_summary()
        expected = np.array([-1.0, 0.0, 1.0])
        for k in summary:
            assert np.all(np.abs(summary[k] - expected) < 0.2)

    def test_aic_fail_no_posterior(self):
        d = norm.rvs(size=1000)
        c = ChainConsumer()
        c.add_chain(d, num_eff_data_points=1000, num_free_params=1)
        aics = c.comparison.aic()
        assert len(aics) == 1
        assert aics[0] is None

    def test_aic_fail_no_data_points(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1)
        aics = c.comparison.aic()
        assert len(aics) == 1
        assert aics[0] is None

    def test_aic_fail_no_num_params(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_eff_data_points=1000)
        aics = c.comparison.aic()
        assert len(aics) == 1
        assert aics[0] is None

    def test_aic_0(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        aics = c.comparison.aic()
        assert len(aics) == 1
        assert aics[0] == 0

    def test_aic_posterior_dependence(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        p2 = norm.logpdf(d, scale=2)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        c.add_chain(d, posterior=p2, num_free_params=1, num_eff_data_points=1000)
        aics = c.comparison.aic()
        assert len(aics) == 2
        assert aics[0] == 0
        expected = 2 * np.log(2)
        assert np.isclose(aics[1], expected, atol=1e-3)

    def test_aic_parameter_dependence(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        c.add_chain(d, posterior=p, num_free_params=2, num_eff_data_points=1000)
        aics = c.comparison.aic()
        assert len(aics) == 2
        assert aics[0] == 0
        expected = 1 * 2 + (2.0 * 2 * 3 / (1000 - 2 - 1)) - (2.0 * 1 * 2 / (1000 - 1 - 1))
        assert np.isclose(aics[1], expected, atol=1e-3)

    def test_aic_data_dependence(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=500)
        aics = c.comparison.aic()
        assert len(aics) == 2
        assert aics[0] == 0
        expected = (2.0 * 1 * 2 / (500 - 1 - 1)) - (2.0 * 1 * 2 / (1000 - 1 - 1))
        assert np.isclose(aics[1], expected, atol=1e-3)

    def test_bic_fail_no_posterior(self):
        d = norm.rvs(size=1000)
        c = ChainConsumer()
        c.add_chain(d, num_eff_data_points=1000, num_free_params=1)
        bics = c.comparison.bic()
        assert len(bics) == 1
        assert bics[0] is None

    def test_bic_fail_no_data_points(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1)
        bics = c.comparison.bic()
        assert len(bics) == 1
        assert bics[0] is None

    def test_bic_fail_no_num_params(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_eff_data_points=1000)
        bics = c.comparison.bic()
        assert len(bics) == 1
        assert bics[0] is None

    def test_bic_0(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        bics = c.comparison.bic()
        assert len(bics) == 1
        assert bics[0] == 0

    def test_bic_posterior_dependence(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        p2 = norm.logpdf(d, scale=2)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        c.add_chain(d, posterior=p2, num_free_params=1, num_eff_data_points=1000)
        bics = c.comparison.bic()
        assert len(bics) == 2
        assert bics[0] == 0
        expected = 2 * np.log(2)
        assert np.isclose(bics[1], expected, atol=1e-3)

    def test_bic_parameter_dependence(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        c.add_chain(d, posterior=p, num_free_params=2, num_eff_data_points=1000)
        bics = c.comparison.bic()
        assert len(bics) == 2
        assert bics[0] == 0
        expected = np.log(1000)
        assert np.isclose(bics[1], expected, atol=1e-3)

    def test_bic_data_dependence(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
        c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=500)
        bics = c.comparison.bic()
        assert len(bics) == 2
        assert bics[1] == 0
        expected = np.log(1000) - np.log(500)
        assert np.isclose(bics[0], expected, atol=1e-3)

    def test_bic_data_dependence2(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p, num_free_params=2, num_eff_data_points=1000)
        c.add_chain(d, posterior=p, num_free_params=3, num_eff_data_points=500)
        bics = c.comparison.bic()
        assert len(bics) == 2
        assert bics[0] == 0
        expected = 3 * np.log(500) - 2 * np.log(1000)
        assert np.isclose(bics[1], expected, atol=1e-3)

    def test_dic_fail_no_posterior(self):
        d = norm.rvs(size=1000)
        c = ChainConsumer()
        c.add_chain(d, num_eff_data_points=1000, num_free_params=1)
        dics = c.comparison.dic()
        assert len(dics) == 1
        assert dics[0] is None

    def test_dic_0(self):
        d = norm.rvs(size=1000)
        p = norm.logpdf(d)
        c = ChainConsumer()
        c.add_chain(d, posterior=p)
        dics = c.comparison.dic()
        assert len(dics) == 1
        assert dics[0] == 0

    def test_dic_posterior_dependence(self):
        d = norm.rvs(size=1000000)
        p = norm.logpdf(d)
        p2 = norm.logpdf(d, scale=2)
        c = ChainConsumer()
        c.add_chain(d, posterior=p)
        c.add_chain(d, posterior=p2)
        bics = c.comparison.dic()
        assert len(bics) == 2
        assert bics[1] == 0
        dic1 = 2 * np.mean(-2 * p) + 2 * norm.logpdf(0)
        dic2 = 2 * np.mean(-2 * p2) + 2 * norm.logpdf(0, scale=2)
        assert np.isclose(bics[0], dic1 - dic2, atol=1e-3)

    def test_remove_last_chain(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.add_chain(self.data * 2)
        consumer.remove_chain()
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_first_chain(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data * 2)
        consumer.add_chain(self.data)
        consumer.remove_chain(chain=0)
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_chain_by_name(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data * 2, name="a")
        consumer.add_chain(self.data, name="b")
        consumer.remove_chain(chain="a")
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_chain_recompute_params(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data * 2, parameters=["p1"], name="a")
        consumer.add_chain(self.data, parameters=["p2"], name="b")
        consumer.remove_chain(chain="a")
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        assert "p2" in summary
        assert "p1" not in summary
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_multiple_chains(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data * 2, parameters=["p1"], name="a")
        consumer.add_chain(self.data, parameters=["p2"], name="b")
        consumer.add_chain(self.data * 3, parameters=["p3"], name="c")
        consumer.remove_chain(chain=["a", "c"])
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        assert "p2" in summary
        assert "p1" not in summary
        assert "p3" not in summary
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_multiple_chains2(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data * 2, parameters=["p1"], name="a")
        consumer.add_chain(self.data, parameters=["p2"], name="b")
        consumer.add_chain(self.data * 3, parameters=["p3"], name="c")
        consumer.remove_chain(chain=[0, 2])
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        assert "p2" in summary
        assert "p1" not in summary
        assert "p3" not in summary
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_multiple_chains3(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data * 2, parameters=["p1"], name="a")
        consumer.add_chain(self.data, parameters=["p2"], name="b")
        consumer.add_chain(self.data * 3, parameters=["p3"], name="c")
        consumer.remove_chain(chain=["a", 2])
        consumer.configure()
        summary = consumer.analysis.get_summary()
        assert isinstance(summary, dict)
        assert "p2" in summary
        assert "p1" not in summary
        assert "p3" not in summary
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_remove_multiple_chains_fails(self):
        with pytest.raises(AssertionError):
            ChainConsumer().add_chain(self.data).remove_chain(chain=[0, 0])

    def test_correlations_1d(self):
        data = np.random.normal(0, 1, size=100000)
        parameters = ["x"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters)
        p, cor = c.analysis.get_correlations()
        assert p[0] == "x"
        print(cor)
        assert np.isclose(cor[0, 0], 1)
        assert cor.shape == (1, 1)

    def test_correlations_2d(self):
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100000)
        parameters = ["x", "y"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters)
        p, cor = c.analysis.get_correlations()
        assert p[0] == "x"
        assert p[1] == "y"
        assert np.isclose(cor[0, 0], 1)
        assert np.isclose(cor[1, 1], 1)
        assert np.abs(cor[0, 1]) < 0.01
        assert cor.shape == (2, 2)

    def test_correlations_3d(self):
        data = np.random.multivariate_normal([0, 0, 1], [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]], size=100000)
        parameters = ["x", "y", "z"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters, name="chain1")
        p, cor = c.analysis.get_correlations(chain="chain1", parameters=["y", "z", "x"])
        assert p[0] == "y"
        assert p[1] == "z"
        assert p[2] == "x"
        assert np.isclose(cor[0, 0], 1)
        assert np.isclose(cor[1, 1], 1)
        assert np.isclose(cor[2, 2], 1)
        assert cor.shape == (3, 3)
        assert np.abs(cor[0, 1] - 0.3) < 0.01
        assert np.abs(cor[0, 2] - 0.5) < 0.01
        assert np.abs(cor[1, 2] - 0.2) < 0.01

    def test_correlation_latex_table(self):
        data = np.random.multivariate_normal([0, 0, 1], [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1.0]], size=100000)
        parameters = ["x", "y", "z"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters)
        latex_table = c.analysis.get_correlation_table()

        actual = r"""\begin{table}
            \centering
            \caption{Parameter Correlations}
            \label{tab:parameter_correlations}
            \begin{tabular}{c|ccc}
                 & x & y & z\\
                \hline
                   x  &  1.00 &  0.50 &  0.20 \\
                   y  &  0.50 &  1.00 &  0.30 \\
                   z  &  0.20 &  0.30 &  1.00 \\
                \hline
            \end{tabular}
        \end{table}"""
        assert latex_table.replace(" ", "") == actual.replace(" ", "")

    def test_covariance_1d(self):
        data = np.random.normal(0, 2, size=2000000)
        parameters = ["x"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters)
        p, cor = c.analysis.get_covariance()
        assert p[0] == "x"
        assert np.isclose(cor[0, 0], 4, atol=1e-2)
        assert cor.shape == (1, 1)

    def test_covariance_2d(self):
        data = np.random.multivariate_normal([0, 0], [[3, 0], [0, 9]], size=2000000)
        parameters = ["x", "y"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters)
        p, cor = c.analysis.get_covariance()
        assert p[0] == "x"
        assert p[1] == "y"
        assert np.isclose(cor[0, 0], 3, atol=2e-2)
        assert np.isclose(cor[1, 1], 9, atol=2e-2)
        assert np.isclose(cor[0, 1], 0, atol=2e-2)
        assert cor.shape == (2, 2)

    def test_covariance_3d(self):
        cov = [[3, 0.5, 0.2], [0.5, 4, 0.3], [0.2, 0.3, 5]]
        data = np.random.multivariate_normal([0, 0, 1], cov, size=2000000)
        parameters = ["x", "y", "z"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters, name="chain1")
        p, cor = c.analysis.get_covariance(chain="chain1", parameters=["y", "z", "x"])
        assert p[0] == "y"
        assert p[1] == "z"
        assert p[2] == "x"
        assert np.isclose(cor[0, 0], 4, atol=2e-2)
        assert np.isclose(cor[1, 1], 5, atol=2e-2)
        assert np.isclose(cor[2, 2], 3, atol=2e-2)
        assert cor.shape == (3, 3)
        assert np.abs(cor[0, 1] - 0.3) < 0.01
        assert np.abs(cor[0, 2] - 0.5) < 0.01
        assert np.abs(cor[1, 2] - 0.2) < 0.01

    def test_covariance_latex_table(self):
        cov = [[2, 0.5, 0.2], [0.5, 3, 0.3], [0.2, 0.3, 4.0]]
        data = np.random.multivariate_normal([0, 0, 1], cov, size=20000000)
        parameters = ["x", "y", "z"]
        c = ChainConsumer()
        c.add_chain(data, parameters=parameters)
        latex_table = c.analysis.get_covariance_table()

        actual = r"""\begin{table}
            \centering
            \caption{Parameter Covariance}
            \label{tab:parameter_covariance}
            \begin{tabular}{c|ccc}
                 & x & y & z\\
                \hline
                   x  &  2.00 &  0.50 &  0.20 \\
                   y  &  0.50 &  3.00 &  0.30 \\
                   z  &  0.20 &  0.30 &  4.00 \\
                \hline
            \end{tabular}
        \end{table}"""
        assert latex_table.replace(" ", "") == actual.replace(" ", "")

    def test_fail_if_more_parameters_than_data(self):
        with pytest.raises(AssertionError):
            ChainConsumer().add_chain(self.data_combined, parameters=["x", "y", "z"])

    def test_shade_alpha_algorithm1(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure()
        alphas = consumer.config["shade_alpha"]
        assert len(alphas) == 1
        assert alphas[0] == 1.0

    def test_shade_alpha_algorithm2(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.add_chain(self.data)
        consumer.configure()
        alphas = consumer.config["shade_alpha"]
        assert len(alphas) == 2
        assert alphas[0] == np.sqrt(1.0 / 2.0)
        assert alphas[1] == np.sqrt(1.0 / 2.0)

    def test_shade_alpha_algorithm3(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.add_chain(self.data)
        consumer.add_chain(self.data)
        consumer.configure()
        alphas = consumer.config["shade_alpha"]
        assert len(alphas) == 3
        assert alphas[0] == np.sqrt(1.0 / 3.0)
        assert alphas[1] == np.sqrt(1.0 / 3.0)
        assert alphas[2] == np.sqrt(1.0 / 3.0)

    def test_plotter_extents1(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.configure()
        minv, maxv = c.plotter._get_parameter_extents("x", [0])
        assert np.isclose(minv, (5.0 - 1.5 * 3.1), atol=0.1)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.1), atol=0.1)

    def test_plotter_extents2(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.add_chain(self.data + 5, parameters=["y"])
        c.configure()
        minv, maxv = c.plotter._get_parameter_extents("x", [0, 1])
        assert np.isclose(minv, (5.0 - 1.5 * 3.1), atol=0.1)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.1), atol=0.1)

    def test_plotter_extents3(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.add_chain(self.data + 5, parameters=["x"])
        c.configure()
        minv, maxv = c.plotter._get_parameter_extents("x", [0, 1])
        assert np.isclose(minv, (5.0 - 1.5 * 3.1), atol=0.1)
        assert np.isclose(maxv, (10.0 + 1.5 * 3.1), atol=0.1)

    def test_plotter_extents4(self):
        c = ChainConsumer()
        c.add_chain(self.data, parameters=["x"])
        c.add_chain(self.data + 5, parameters=["y"])
        c.configure()
        minv, maxv = c.plotter._get_parameter_extents("x", [0])
        assert np.isclose(minv, (5.0 - 1.5 * 3.1), atol=0.1)
        assert np.isclose(maxv, (5.0 + 1.5 * 3.1), atol=0.1)

    def test_plotter_extents5(self):
        x, y = np.linspace(-3, 3, 200), np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        xs, ys = xx.flatten(), yy.flatten()
        chain = np.vstack((xs, ys)).T
        pdf = (1 / (2 * np.pi)) * np.exp(-0.5 * (xs * xs + ys * ys / 4))
        c = ChainConsumer()
        c.add_chain(chain, parameters=['x', 'y'], weights=pdf, grid=True)
        c.configure()
        minv, maxv = c.plotter._get_parameter_extents("x", [0])
        assert np.isclose(minv, -3, atol=0.001)
        assert np.isclose(maxv, 3, atol=0.001)

    def test_covariant_covariance_calc(self):
        data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=10000)
        data2 = np.random.multivariate_normal([0, 0], [[2, 1], [1, 2]], size=10000)
        weights = np.concatenate((np.ones(10000), np.zeros(10000)))
        data = np.concatenate((data1, data2))
        c = ChainConsumer()
        c.add_chain(data, weights=weights, parameters=["x", "y"])
        p, cor = c.analysis.get_covariance()
        assert p[0] == "x"
        assert p[1] == "y"
        assert np.isclose(cor[0, 0], 1, atol=2e-2)
        assert np.isclose(cor[1, 1], 1, atol=2e-2)
        assert np.isclose(cor[0, 1], 0, atol=2e-2)
        assert cor.shape == (2, 2)

    def test_megkde_1d_basic(self):
        # Draw from normal, fit KDE, see if sampling from kde's pdf recovers norm
        np.random.seed(0)
        data = np.random.normal(loc=0, scale=1.0, size=2000)
        xs = np.linspace(-3, 3, 100)
        ys = MegKDE(data).evaluate(xs)
        cs = ys.cumsum()
        cs /= cs[-1]
        cs[0] = 0
        samps = interp1d(cs, xs)(np.random.uniform(size=10000))
        mu, std = norm.fit(samps)
        assert np.isclose(mu, 0, atol=0.1)
        assert np.isclose(std, 1.0, atol=0.1)

    def test_megkde_1d_uniform_weight(self):
        # Draw from normal, fit KDE, see if sampling from kde's pdf recovers norm
        np.random.seed(0)
        data = np.random.normal(loc=0, scale=1.0, size=2000)
        xs = np.linspace(-3, 3, 100)
        ys = MegKDE(data, weights=np.ones(2000)).evaluate(xs)
        cs = ys.cumsum()
        cs /= cs[-1]
        cs[0] = 0
        samps = interp1d(cs, xs)(np.random.uniform(size=10000))
        mu, std = norm.fit(samps)
        assert np.isclose(mu, 0, atol=0.1)
        assert np.isclose(std, 1.0, atol=0.1)

    def test_megkde_1d_changing_weights(self):
        # Draw from normal, fit KDE, see if sampling from kde's pdf recovers norm
        np.random.seed(0)
        xs = np.linspace(-3, 3, 100)
        weights = norm.pdf(xs)
        ys = MegKDE(xs, weights=weights).evaluate(xs)
        cs = ys.cumsum()
        cs /= cs[-1]
        cs[0] = 0
        samps = interp1d(cs, xs)(np.random.uniform(size=10000))
        mu, std = norm.fit(samps)
        assert np.isclose(mu, 0, atol=0.1)
        assert np.isclose(std, 1.0, atol=0.1)

    def test_megkde_2d_basic(self):
        # Draw from normal, fit KDE, see if sampling from kde's pdf recovers norm
        np.random.seed(1)
        data = np.random.multivariate_normal([0, 1], [[1.0, 0.], [0., 0.75**2]], size=10000)
        xs, ys = np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)
        xx, yy = meshgrid(xs, ys, indexing='ij')
        samps = np.vstack((xx.flatten(), yy.flatten())).T
        zs = MegKDE(data).evaluate(samps).reshape(xx.shape)
        zs_x = zs.sum(axis=1)
        zs_y = zs.sum(axis=0)
        cs_x = zs_x.cumsum()
        cs_x /= cs_x[-1]
        cs_x[0] = 0
        cs_y = zs_y.cumsum()
        cs_y /= cs_y[-1]
        cs_y[0] = 0
        samps_x = interp1d(cs_x, xs)(np.random.uniform(size=10000))
        samps_y = interp1d(cs_y, ys)(np.random.uniform(size=10000))
        mu_x, std_x = norm.fit(samps_x)
        mu_y, std_y = norm.fit(samps_y)
        assert np.isclose(mu_x, 0, atol=0.1)
        assert np.isclose(std_x, 1.0, atol=0.1)
        assert np.isclose(mu_y, 1, atol=0.1)
        assert np.isclose(std_y, 0.75, atol=0.1)
