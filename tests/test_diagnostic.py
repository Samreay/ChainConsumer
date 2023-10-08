import numpy as np
import pandas as pd
import pytest

from chainconsumer import Chain, ChainConsumer


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def good_chain(rng) -> Chain:
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    chain = Chain(samples=pd.DataFrame(data, columns=["a", "b"]), name="A", walkers=4)
    return chain


@pytest.fixture
def bad_chain(rng) -> Chain:
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    data[80000:, :] *= 2
    data[80000:, :] += 1
    chain = Chain(samples=pd.DataFrame(data, columns=["a", "b"]), name="A", walkers=4)
    return chain


@pytest.fixture
def good_cc(good_chain: Chain) -> ChainConsumer:
    return ChainConsumer().add_chain(good_chain)


@pytest.fixture
def good_cc2(good_chain: Chain) -> ChainConsumer:
    c2 = good_chain.model_copy()
    c2.name = "B"
    return ChainConsumer().add_chain(good_chain).add_chain(c2)


@pytest.fixture
def bad_cc(bad_chain: Chain) -> ChainConsumer:
    return ChainConsumer().add_chain(bad_chain)


def test_gelman_rubin_index(good_cc: ChainConsumer) -> None:
    assert good_cc.diagnostic.gelman_rubin()


def test_gelman_rubin_index2(good_cc2: ChainConsumer) -> None:
    res = good_cc2.diagnostic.gelman_rubin()
    assert res
    assert res.passed
    assert "A" in res.results
    assert res.results["A"]
    assert "B" in res.results
    assert res.results["B"]


def test_gelman_rubin_index_not_converged(bad_cc: ChainConsumer) -> None:
    assert not bad_cc.diagnostic.gelman_rubin()


def test_gelman_rubin_index_not_converged2(rng) -> None:
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    data[:, 0] += np.linspace(0, 10, 100000)
    consumer = ChainConsumer()

    consumer.add_chain(Chain(samples=pd.DataFrame(data, columns=["A", "B"]), name="B", walkers=8))
    res = consumer.diagnostic.gelman_rubin()
    assert not res
    assert not res.passed
    assert "B" in res.results
    assert not res.results["B"]


def test_geweke_index(good_cc: ChainConsumer) -> None:
    assert good_cc.diagnostic.geweke()


def test_geweke_index_failed(bad_cc: ChainConsumer) -> None:
    assert not bad_cc.diagnostic.geweke()


def test_geweke_default(good_cc2: ChainConsumer) -> None:
    res = good_cc2.diagnostic.geweke()
    assert res
    assert res.passed
    assert "A" in res.results
    assert res.results["A"]
    assert "B" in res.results
    assert res.results["B"]


def test_geweke_default_failed(rng: np.random.Generator) -> None:
    data = np.vstack((rng.normal(loc=0.0, size=100000), rng.normal(loc=1.0, size=100000))).T
    consumer = ChainConsumer()
    consumer.add_chain(Chain(samples=pd.DataFrame(data, columns=["a", "b"]), walkers=20, name="c1"))
    data2 = data.copy()
    data2[98000:, :] += 0.3
    consumer.add_chain(Chain(samples=pd.DataFrame(data2, columns=["a", "b"]), walkers=20, name="c2"))
    assert not consumer.diagnostic.geweke()
