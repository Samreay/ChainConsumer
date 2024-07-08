import numpy as np
from pydantic import Field
from scipy.stats import normaltest

from .base import BetterBase
from .chain import Chain, ChainName
from .log import logger


class TestResult(BetterBase):
    passed: bool = Field(default=..., description="Whether or not the test passed in general")
    results: dict[ChainName, bool] = Field(
        default=..., description="For each chain, whether the test passed and its numerical value"
    )

    def __bool__(self) -> bool:
        return self.passed


def _sanitise_chains(
    chains: list[Chain | ChainName] | Chain | ChainName | None, parent: "ChainConsumer"
) -> list[Chain]:
    if chains is None:
        return list(parent._chains.values())
    elif isinstance(chains, list):
        return [c if isinstance(c, Chain) else parent.get_chain(c) for c in chains]
    return [parent.get_chain(chains) if isinstance(chains, str) else chains]


class Diagnostic:
    def __init__(self, parent: "ChainConsumer"):
        self.parent: ChainConsumer = parent

    def gelman_rubin(
        self, chains: list[Chain | ChainName] | Chain | ChainName | None = None, threshold: float = 0.05
    ) -> TestResult:
        r"""Runs the Gelman Rubin diagnostic on the supplied chains.

        Args:
            chains: Which chain to run the diagnostic on. None will run on all chains
            threshold: The maximum deviation permitted from 1 for the final value $\hat{R}$

        Returns:
            The test results

        Notes:

            I follow PyMC in calculating the Gelman-Rubin statistic, where,
            having :math:`m` chains of length :math:`n`, we compute

            .. math::

                B = \frac{n}{m-1} \sum_{j=1}^{m} \left(\bar{\theta}_{.j} - \bar{\theta}_{..}\right)^2

                W = \frac{1}{m} \sum_{j=1}^{m} \left[ \frac{1}{n-1} \sum_{i=1}^{n} \left( \theta_{ij} - \bar{\theta_{.j}}\right)^2 \right]

            where :math:`\theta` represents each model parameter. We then compute
            :math:`\hat{V} = \frac{n_1}{n}W + \frac{1}{n}B`, and have our convergence ratio
            :math:`\hat{R} = \sqrt{\frac{\hat{V}}{W}}`. We check that for all parameters,
            this ratio deviates from unity by less than the supplied threshold.
        """  # noqa: E501
        final_chains = _sanitise_chains(chains, parent=self.parent)
        if len(final_chains) > 1:
            results = [self.gelman_rubin(c, threshold=threshold) for c in final_chains]
            passed = all([r.passed for r in results])
            combined_dict: dict[ChainName, bool] = {}
            for result in results:
                combined_dict |= result.results
            return TestResult(passed=passed, results=combined_dict)
        chain = final_chains[0]

        num_walkers = chain.walkers
        parameters = chain.data_columns
        name = chain.name
        data = chain.data_samples
        split_samples = np.split(data, num_walkers)
        assert num_walkers > 1, "Cannot run Gelman-Rubin statistic with only one walker"
        m = 1.0 * len(split_samples)
        n = 1.0 * split_samples[0].shape[0]
        all_mean = np.mean(data.to_numpy(), axis=0)
        chain_means = np.array([np.mean(c, axis=0) for c in split_samples])
        chain_var = np.array([np.var(c, axis=0, ddof=1) for c in split_samples])
        b = n / (m - 1) * ((chain_means - all_mean) ** 2).sum(axis=0)
        w = (1 / m) * chain_var.sum(axis=0)
        var = (n - 1) * w / n + b / n
        v = var + b / (n * m)
        r: float = np.sqrt(v / w)

        passed = np.abs(r - 1) < threshold
        logger.info(f"Gelman-Rubin Statistic values for chain {name}")
        for p, v, pas in zip(parameters, r, passed):
            param = "Param %d" % p if isinstance(p, int) else p
            logger.info(f"{param}: {v:7.5f} ({'Passed' if pas else 'Failed'})")
        all_passed: bool = np.all(passed)
        return TestResult(passed=all_passed, results={chain.name: all_passed})

    def geweke(
        self,
        chains: list[Chain | ChainName] | Chain | ChainName | None = None,
        first: float = 0.1,
        last: float = 0.5,
        threshold: float = 0.05,
    ) -> TestResult:
        """Runs the Geweke diagnostic on the supplied chains.

        Args:
            chains: Which chain to run the diagnostic on. None will run on all chains
            first: The amount of the start of the chain to use
            last: The end amount of the chain to use
            threshold: The p-value to use when testing for normality.

        Returns:
            The test results
        """
        final_chains = _sanitise_chains(chains, parent=self.parent)
        if len(final_chains) > 1:
            results = [self.geweke(c, first=first, last=last, threshold=threshold) for c in final_chains]
            passed = all([r.passed for r in results])
            combined_dict = {}
            for r in results:
                combined_dict |= r.results
            return TestResult(passed=passed, results=combined_dict)
        chain = final_chains[0]

        num_walkers = chain.walkers
        assert (
            num_walkers is not None and num_walkers > 0
        ), "You need to specify the number of walkers to use the Geweke diagnostic."
        name = chain.name
        data = chain.data_samples
        split_samples = np.split(data.to_numpy(), num_walkers)
        n = 1.0 * split_samples[0].shape[0]
        n_start = int(np.floor(first * n))
        n_end = int(np.floor((1 - last) * n))
        mean_start = np.array([np.mean(c[:n_start, i]) for c in split_samples for i in range(c.shape[1])])
        var_start = np.array(
            [self._spec(c[:n_start, i]) / c[:n_start, i].size for c in split_samples for i in range(c.shape[1])]
        )
        mean_end = np.array([np.mean(c[n_end:, i]) for c in split_samples for i in range(c.shape[1])])
        var_end = np.array(
            [self._spec(c[n_end:, i]) / c[n_end:, i].size for c in split_samples for i in range(c.shape[1])]
        )
        zs = (mean_start - mean_end) / (np.sqrt(var_start + var_end))
        _, pvalue = normaltest(zs)
        logger.info(f"Gweke Statistic for chain {name} has p-value {pvalue:e}")
        passed = pvalue > threshold
        return TestResult(passed=passed, results={chain.name: passed})

    # Method of estimating spectral density following PyMC.
    # See https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py
    def _spec(self, x, order=2):
        from statsmodels.regression.linear_model import yule_walker

        beta, sigma = yule_walker(x, order)  # type: ignore
        return sigma**2 / (1.0 - np.sum(beta)) ** 2


if __name__ == "__main__":
    from chainconsumer import ChainConsumer
