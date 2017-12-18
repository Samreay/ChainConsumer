import numpy as np
from scipy.stats import norm

from chainconsumer import ChainConsumer


def test_aic_fail_no_posterior():
    d = norm.rvs(size=1000)
    c = ChainConsumer()
    c.add_chain(d, num_eff_data_points=1000, num_free_params=1)
    aics = c.comparison.aic()
    assert len(aics) == 1
    assert aics[0] is None


def test_aic_fail_no_data_points():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p, num_free_params=1)
    aics = c.comparison.aic()
    assert len(aics) == 1
    assert aics[0] is None


def test_aic_fail_no_num_params():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p, num_eff_data_points=1000)
    aics = c.comparison.aic()
    assert len(aics) == 1
    assert aics[0] is None


def test_aic_0():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
    aics = c.comparison.aic()
    assert len(aics) == 1
    assert aics[0] == 0


def test_aic_posterior_dependence():
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


def test_aic_parameter_dependence():
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


def test_aic_data_dependence():
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


def test_bic_fail_no_posterior():
    d = norm.rvs(size=1000)
    c = ChainConsumer()
    c.add_chain(d, num_eff_data_points=1000, num_free_params=1)
    bics = c.comparison.bic()
    assert len(bics) == 1
    assert bics[0] is None


def test_bic_fail_no_data_points():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p, num_free_params=1)
    bics = c.comparison.bic()
    assert len(bics) == 1
    assert bics[0] is None


def test_bic_fail_no_num_params():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p, num_eff_data_points=1000)
    bics = c.comparison.bic()
    assert len(bics) == 1
    assert bics[0] is None


def test_bic_0():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p, num_free_params=1, num_eff_data_points=1000)
    bics = c.comparison.bic()
    assert len(bics) == 1
    assert bics[0] == 0


def test_bic_posterior_dependence():
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


def test_bic_parameter_dependence():
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


def test_bic_data_dependence():
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


def test_bic_data_dependence2():
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


def test_dic_fail_no_posterior():
    d = norm.rvs(size=1000)
    c = ChainConsumer()
    c.add_chain(d, num_eff_data_points=1000, num_free_params=1)
    dics = c.comparison.dic()
    assert len(dics) == 1
    assert dics[0] is None


def test_dic_0():
    d = norm.rvs(size=1000)
    p = norm.logpdf(d)
    c = ChainConsumer()
    c.add_chain(d, posterior=p)
    dics = c.comparison.dic()
    assert len(dics) == 1
    assert dics[0] == 0


def test_dic_posterior_dependence():
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