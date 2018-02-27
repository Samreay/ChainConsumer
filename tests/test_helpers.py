import numpy as np
from scipy.stats import norm

from chainconsumer.helpers import get_extents


def test_extents():
    xs = np.random.normal(size=1000000)
    weights = np.ones(xs.shape)
    low, high = get_extents(xs, weights)
    threshold = 0.5
    assert np.abs(low + 4) < threshold
    assert np.abs(high - 4) < threshold


def test_extents_weighted():
    xs = np.random.uniform(low=-4, high=4, size=1000000)
    weights = norm.pdf(xs)
    low, high = get_extents(xs, weights)
    threshold = 0.5
    assert np.abs(low + 4) < threshold
    assert np.abs(high - 4) < threshold


def test_extents_summary():
    xs = np.random.normal(size=1000000)
    low, high = get_extents(xs, np.ones(xs.shape), plot=True, wide_extents=False)
    threshold = 0.1
    assert np.abs(low + 1.644855) < threshold
    assert np.abs(high - 1.644855) < threshold
