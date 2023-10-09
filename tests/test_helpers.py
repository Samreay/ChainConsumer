import numpy as np
import pytest
from scipy.stats import norm

from chainconsumer.helpers import get_extents


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_extents(rng):
    xs = rng.normal(size=1000000)
    weights = np.ones(xs.shape)
    low, high = get_extents(xs, weights)
    threshold = 0.5
    assert np.abs(low + 4) < threshold
    assert np.abs(high - 4) < threshold


def test_extents_weighted(rng):
    xs = rng.uniform(low=-4, high=4, size=1000000)
    weights = norm.pdf(xs)
    low, high = get_extents(xs, weights)
    threshold = 0.5
    assert np.abs(low + 4) < threshold
    assert np.abs(high - 4) < threshold


def test_extents_summary(rng):
    xs = rng.normal(size=1000000)
    low, high = get_extents(xs, np.ones(xs.shape), plot=True, wide_extents=False)
    threshold = 0.1
    assert np.abs(low + 1.644855) < threshold
    assert np.abs(high - 1.644855) < threshold
