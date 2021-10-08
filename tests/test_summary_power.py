import os
import tempfile

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skewnorm, norm
import pytest

from chainconsumer import ChainConsumer

class TestChain(object):
    np.random.seed(1)
    n = 2000000
    data = np.random.normal(loc=5.0, scale=1.5, size=n)
    data2 = np.random.normal(loc=3, scale=1.0, size=n)
    data_combined = np.vstack((data, data2)).T
    data_skew = skewnorm.rvs(5, loc=1, scale=1.5, size=n)

    def test_summary_power(self):
        tolerance = 4e-2
        consumer = ChainConsumer()
        data = np.random.normal(loc=0, scale=np.sqrt(2), size=1000000)
        consumer.add_chain(data, power=2.0)
        summary = consumer.analysis.get_summary()
        actual = np.array(list(summary.values())[0])
        expected = np.array([-1.0, 0.0, 1.0])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)