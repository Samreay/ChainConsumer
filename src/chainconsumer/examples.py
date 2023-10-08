import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mv


def make_sample(
    num_dimensions: int = 2,
    seed: int | None = None,
    randomise_mean: bool = False,
    num_points: int = 1000000,
) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    vals = gen.random((num_dimensions, num_dimensions)) - 0.5
    cov = np.dot(vals, vals.T)
    diag = np.sqrt(np.diag(cov))
    outer = np.outer(diag, diag)
    cor = cov / outer
    means = np.arange(num_dimensions) * 1.0
    if randomise_mean:
        means += gen.uniform(-1, 1, num_dimensions)
    norm = mv(mean=means, cov=cor)
    samples = norm.rvs(size=num_points)
    # Use the letters of the alphabet as the column names
    columns = [chr(65 + i) for i in range(num_dimensions)]
    return pd.DataFrame(samples, columns=columns).assign(log_posterior=norm.logpdf(samples))


if __name__ == "__main__":
    make_sample()
