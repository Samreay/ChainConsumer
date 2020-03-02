import numpy as np
import pytest

from chainconsumer import ChainConsumer


def test_gelman_rubin_index():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4)
    assert consumer.diagnostic.gelman_rubin(chain=0)


def test_gelman_rubin_index_not_converged():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    data[80000:, :] *= 2
    data[80000:, :] += 1
    consumer = ChainConsumer()

    consumer.add_chain(data, walkers=4)
    assert not consumer.diagnostic.gelman_rubin(chain=0)


def test_gelman_rubin_index_not_converged():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    data[:, 0] += np.linspace(0, 10, 100000)
    consumer = ChainConsumer()

    consumer.add_chain(data, walkers=8)
    assert not consumer.diagnostic.gelman_rubin(chain=0)


def test_gelman_rubin_index_fails():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4)
    with pytest.raises(AssertionError):
        consumer.diagnostic.gelman_rubin(chain=10)


def test_gelman_rubin_name():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="testchain")
    assert consumer.diagnostic.gelman_rubin(chain="testchain")


def test_gelman_rubin_name_fails():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="testchain")
    with pytest.raises(AssertionError):
        consumer.diagnostic.gelman_rubin(chain="testchain2")


def test_gelman_rubin_unknown_fails():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="testchain")
    with pytest.raises(ValueError):
        consumer.diagnostic.gelman_rubin(chain=np.pi)


def test_gelman_rubin_default():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="c1")
    consumer.add_chain(data, walkers=4, name="c2")
    consumer.add_chain(data, walkers=4, name="c3")
    assert consumer.diagnostic.gelman_rubin()


def test_gelman_rubin_default_not_converge():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="c1")
    consumer.add_chain(data, walkers=4, name="c2")
    data2 = data.copy()
    data2[:, 0] += np.linspace(-5, 5, 100000)
    consumer.add_chain(data2, walkers=4, name="c3")
    assert not consumer.diagnostic.gelman_rubin()


def test_geweke_index():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=20, name="c1")
    assert consumer.diagnostic.geweke(chain=0)


def test_geweke_index_failed():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    data[98000:, :] += 0.5
    consumer.add_chain(data, walkers=20, name="c1")
    assert not consumer.diagnostic.geweke(chain=0)


def test_geweke_default():
    np.random.seed(0)
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=20, name="c1")
    consumer.add_chain(data, walkers=20, name="c2")
    assert consumer.diagnostic.geweke(chain=0)


def test_geweke_default_failed():
    data = np.vstack((np.random.normal(loc=0.0, size=100000),
                      np.random.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=20, name="c1")
    data2 = data.copy()
    data2[98000:, :] += 0.3
    consumer.add_chain(data2, walkers=20, name="c2")
    assert not consumer.diagnostic.geweke()