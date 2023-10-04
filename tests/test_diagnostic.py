import numpy as np
import pytest

from chainconsumer import ChainConsumer


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng()


def test_gelman_rubin_index(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4)
    assert consumer.diagnostic.gelman_rubin(chain=0)


def test_gelman_rubin_index_not_converged(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    data[80000:, :] *= 2
    data[80000:, :] += 1
    consumer = ChainConsumer()

    consumer.add_chain(data, walkers=4)
    assert not consumer.diagnostic.gelman_rubin(chain=0)


def test_gelman_rubin_index_not_converged2(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    data[:, 0] += np.linspace(0, 10, 100000)
    consumer = ChainConsumer()

    consumer.add_chain(data, walkers=8)
    assert not consumer.diagnostic.gelman_rubin(chain=0)


def test_gelman_rubin_index_fails(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4)
    with pytest.raises(AssertionError):
        consumer.diagnostic.gelman_rubin(chain=10)


def test_gelman_rubin_name(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="testchain")
    assert consumer.diagnostic.gelman_rubin(chain="testchain")


def test_gelman_rubin_name_fails(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="testchain")
    with pytest.raises(AssertionError):
        consumer.diagnostic.gelman_rubin(chain="testchain2")


def test_gelman_rubin_unknown_fails(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="testchain")
    with pytest.raises(ValueError):
        consumer.diagnostic.gelman_rubin(chain=np.pi)


def test_gelman_rubin_default(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="c1")
    consumer.add_chain(data, walkers=4, name="c2")
    consumer.add_chain(data, walkers=4, name="c3")
    assert consumer.diagnostic.gelman_rubin()


def test_gelman_rubin_default_not_converge(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=4, name="c1")
    consumer.add_chain(data, walkers=4, name="c2")
    data2 = data.copy()
    data2[:, 0] += np.linspace(-5, 5, 100000)
    consumer.add_chain(data2, walkers=4, name="c3")
    assert not consumer.diagnostic.gelman_rubin()


def test_geweke_index(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=20, name="c1")
    assert consumer.diagnostic.geweke(chain=0)


def test_geweke_index_failed(rng):
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    data[98000:, :] += 0.5
    consumer.add_chain(data, walkers=20, name="c1")
    assert not consumer.diagnostic.geweke(chain=0)


def test_geweke_default(rng):
    generator = np.random.default_rng(0)
    data = np.vstack((generator.normal(loc=0.0, size=100000), generator.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=20, name="c1")
    consumer.add_chain(data, walkers=20, name="c2")
    assert consumer.diagnostic.geweke(chain=0)


def test_geweke_default_failed():
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(data, walkers=20, name="c1")
    data2 = data.copy()
    data2[98000:, :] += 0.3
    consumer.add_chain(data2, walkers=20, name="c2")
    assert not consumer.diagnostic.geweke()
