import os
import tempfile

import numpy as np
from scipy.stats import skewnorm
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
        consumer.configure_general(kde=True)
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary_no_smooth(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure_general(smooth=0, bins=2.4)
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
        consumer.configure_general(bins=1.6)
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
        print(c.chains[0].shape)
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

    def test_divide_chains(self):
        np.random.seed(0)
        data = np.concatenate((np.random.normal(loc=0.0, size=100000),
                               np.random.normal(loc=1.0, size=100000)))
        consumer = ChainConsumer()
        consumer.add_chain(data)
        num_walkers = 2

        c = consumer.divide_chain(2)
        c.configure_general()
        means = [0, 1.0]
        for i in range(num_walkers):
            stats = list(c.get_summary()[i].values())[0]
            assert np.abs(stats[1] - means[i]) < 1e-1
            assert np.abs(c.chains[i][:, 0].mean() - means[i]) < 1e-2

    def test_stats_max_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure_general(statistics="max")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_mean_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure_general(statistics="mean")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_cum_normal(self):
        tolerance = 5e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure_general(statistics="cumulative")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_reject_bad_satst(self):
        consumer = ChainConsumer()
        consumer.add_chain(self.data)
        with pytest.raises(AssertionError):
            consumer.configure_general(statistics="monkey")

    def test_stats_max_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure_general(statistics="max")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.01, 1.55, 2.72])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_mean_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure_general(statistics="mean")
        summary = consumer.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([1.27, 2.19, 3.11])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_stats_cum_skew(self):
        tolerance = 2e-2
        consumer = ChainConsumer()
        consumer.add_chain(self.data_skew)
        consumer.configure_general(statistics="cumulative")
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
        consumer.configure_general(statistics=["cumulative", "mean"])
        summary = consumer.get_summary()
        actual0 = np.array(list(summary[0].values())[0])
        actual1 = np.array(list(summary[1].values())[0])
        expected0 = np.array([1.27, 2.01, 3.11])
        expected1 = np.array([1.27, 2.19, 3.11])
        diff0 = np.abs(expected0 - actual0)
        diff1 = np.abs(expected1 - actual1)
        assert np.all(diff0 < tolerance)
        assert np.all(diff1 < tolerance)
