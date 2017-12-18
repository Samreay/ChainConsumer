import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from chainconsumer.kde import MegKDE


def test_megkde_1d_basic():
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


def test_megkde_1d_uniform_weight():
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


def test_megkde_1d_changing_weights():
    # Draw from normal, fit KDE, see if sampling from kde's pdf recovers norm
    np.random.seed(0)
    xs = np.linspace(-3, 3, 1000)
    weights = norm.pdf(xs)
    ys = MegKDE(xs, weights=weights).evaluate(xs)
    cs = ys.cumsum()
    cs /= cs[-1]
    cs[0] = 0
    samps = interp1d(cs, xs)(np.random.uniform(size=10000))
    mu, std = norm.fit(samps)
    assert np.isclose(mu, 0, atol=0.1)
    assert np.isclose(std, 1.0, atol=0.1)


def test_megkde_2d_basic():
    # Draw from normal, fit KDE, see if sampling from kde's pdf recovers norm
    np.random.seed(1)
    data = np.random.multivariate_normal([0, 1], [[1.0, 0.], [0., 0.75 ** 2]], size=10000)
    xs, ys = np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
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
