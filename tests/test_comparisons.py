import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from chainconsumer import Chain, ChainConsumer


@pytest.fixture
def cc_noposterior() -> ChainConsumer:
    d = norm.rvs(size=1000)
    df = pd.DataFrame({"a": d})
    chain = Chain(samples=df, name="A")
    return ChainConsumer().add_chain(chain)


@pytest.fixture
def cc_no_effective_data_points() -> ChainConsumer:
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    df = pd.DataFrame({"a": d, "log_posterior": p})
    chain = Chain(samples=df, name="A")
    return ChainConsumer().add_chain(chain)


@pytest.fixture
def cc_no_free_params() -> ChainConsumer:
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    df = pd.DataFrame({"a": d, "log_posterior": p})
    chain = Chain(samples=df, name="A", num_eff_data_points=1000)
    return ChainConsumer().add_chain(chain)


@pytest.fixture
def cc() -> ChainConsumer:
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    df = pd.DataFrame({"a": d, "log_posterior": p})
    chain = Chain(samples=df, name="A", num_eff_data_points=1000, num_free_params=1)
    return ChainConsumer().add_chain(chain)


def test_aic_fail_no_posterior(cc_noposterior) -> None:
    aics = cc_noposterior.comparison.aic()
    assert len(aics) == 0


def test_aic_fail_no_data_points(cc_no_effective_data_points) -> None:
    aics = cc_no_effective_data_points.comparison.aic()
    assert len(aics) == 0


def test_aic_fail_no_num_params(cc_no_free_params) -> None:
    aics = cc_no_free_params.comparison.aic()
    assert len(aics) == 0


def test_aic_0(cc) -> None:
    aics = cc.comparison.aic()
    assert len(aics) == 1
    assert aics["A"] == 0


def test_aic_posterior_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    df = chain1.samples.assign(log_posterior=lambda x: norm.logpdf(x["a"], scale=2))
    cc.add_chain(Chain(samples=df, name="B", num_eff_data_points=1000, num_free_params=1))
    aics = cc.comparison.aic()
    assert len(aics) == 2
    assert aics["A"] == 0
    expected = 2 * np.log(2)
    assert np.isclose(aics["B"], expected, atol=1e-3)


def test_aic_parameter_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    chain2 = chain1.model_copy()
    chain2.num_free_params = 2
    chain2.name = "B"
    cc.add_chain(chain2)

    aics = cc.comparison.aic()
    assert len(aics) == 2
    assert aics["A"] == 0
    expected = 1 * 2 + (2.0 * 2 * 3 / (1000 - 2 - 1)) - (2.0 * 1 * 2 / (1000 - 1 - 1))
    assert np.isclose(aics["B"], expected, atol=1e-3)


def test_aic_data_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    chain2 = chain1.model_copy()
    chain2.num_eff_data_points = 500
    chain2.name = "B"
    cc.add_chain(chain2)

    aics = cc.comparison.aic()
    assert len(aics) == 2
    assert aics["A"] == 0
    expected = (2.0 * 1 * 2 / (500 - 1 - 1)) - (2.0 * 1 * 2 / (1000 - 1 - 1))
    assert np.isclose(aics["B"], expected, atol=1e-3)


def test_bic_fail_no_posterior(cc_noposterior: ChainConsumer) -> None:
    bics = cc_noposterior.comparison.bic()
    assert len(bics) == 0


def test_bic_fail_no_data_points(cc_no_effective_data_points: ChainConsumer) -> None:
    bics = cc_no_effective_data_points.comparison.bic()
    assert len(bics) == 0


def test_bic_fail_no_num_params(cc_no_free_params: ChainConsumer) -> None:
    bics = cc_no_free_params.comparison.bic()
    assert len(bics) == 0


def test_bic_0(cc: ChainConsumer) -> None:
    bics = cc.comparison.bic()
    assert len(bics) == 1
    assert bics["A"] == 0


def test_bic_posterior_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    df = chain1.samples.assign(log_posterior=lambda x: norm.logpdf(x["a"], scale=2))
    cc.add_chain(Chain(samples=df, name="B", num_eff_data_points=1000, num_free_params=1))
    bics = cc.comparison.bic()
    assert len(bics) == 2
    assert bics["A"] == 0
    expected = 2 * np.log(2)
    assert np.isclose(bics["B"], expected, atol=1e-3)


def test_bic_parameter_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    chain2 = chain1.model_copy()
    chain2.num_free_params = 2
    chain2.name = "B"
    cc.add_chain(chain2)
    bics = cc.comparison.bic()
    assert len(bics) == 2
    assert bics["A"] == 0
    expected = np.log(1000)
    assert np.isclose(bics["B"], expected, atol=1e-3)


def test_bic_data_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    chain2 = chain1.model_copy()
    chain2.num_eff_data_points = 500
    chain2.name = "B"
    cc.add_chain(chain2)
    bics = cc.comparison.bic()
    assert len(bics) == 2
    assert bics["B"] == 0
    expected = np.log(1000) - np.log(500)
    assert np.isclose(bics["A"], expected, atol=1e-3)


def test_bic_data_dependence2(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    chain1.num_free_params = 2
    chain2 = chain1.model_copy()
    chain2.num_eff_data_points = 500
    chain2.num_free_params = 3
    chain2.name = "B"
    cc.add_chain(chain2)

    bics = cc.comparison.bic()
    assert len(bics) == 2
    assert bics["A"] == 0
    expected = 3 * np.log(500) - 2 * np.log(1000)
    assert np.isclose(bics["B"], expected, atol=1e-3)


def test_dic_fail_no_posterior(cc_noposterior: ChainConsumer) -> None:
    dics = cc_noposterior.comparison.dic()
    assert len(dics) == 0


def test_dic_0(cc: ChainConsumer) -> None:
    dics = cc.comparison.dic()
    assert len(dics) == 1
    assert dics["A"] == 0


def test_dic_posterior_dependence(cc: ChainConsumer) -> None:
    chain1 = cc.get_chain("A")
    df = chain1.samples.assign(log_posterior=lambda x: norm.logpdf(x["a"], scale=2))
    cc.add_chain(Chain(samples=df, name="B", num_eff_data_points=1000, num_free_params=1))
    bics = cc.comparison.bic()
    assert len(bics) == 2
    assert bics["A"] == 0
    dic1 = 2 * np.mean(-2 * chain1.log_posterior) + 2 * norm.logpdf(0)  # type: ignore
    dic2 = 2 * np.mean(-2 * chain1.log_posterior) + 2 * norm.logpdf(0, scale=2)  # type: ignore
    assert np.isclose(bics["B"], dic1 - dic2, atol=1e-3)
