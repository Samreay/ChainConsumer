import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random
import emcee
import numpy as np
import scipy.stats as stats
import arviz as az

from chainconsumer.translator import df_from_arviz_id, df_from_numpyro_mcmc, df_from_emcee_sampler


def run_numpyro_mcmc(n_steps, n_chains):
    """
    Generate dummy data for testing using numpyro's inference process
    """

    observed_data = np.random.normal(loc=0, scale=1, size=100)

    def model(data=None):
        # Prior
        mu = numpyro.sample('mu', dist.Normal(0, 10))
        sigma = numpyro.sample('sigma', dist.HalfNormal(10))

        # Likelihood
        with numpyro.plate('data', size=len(data)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=data)

    # Running MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=n_steps, num_chains=n_chains, progress_bar=False)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, data=observed_data)

    return mcmc


def run_emcee_mcmc(n_steps, n_chains):
    """
    Generate dummy data for testing using emcee's inference process
    """

    observed_data = np.random.normal(loc=0, scale=1, size=100)

    def log_likelihood(theta, data):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        return np.sum(stats.norm.logpdf(data, mu, sigma))

    def log_prior(theta):
        mu, log_sigma = theta
        if -10 < mu < 10 and 0 < log_sigma < 10:
            return 0.0
        return -np.inf

    def log_probability(theta, data):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, data)

    nwalkers = n_chains
    ndim = 2
    p0 = np.random.normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(observed_data,))
    sampler.run_mcmc(p0, n_steps, progress=False)

    return sampler


class TestTranslators:

    n_steps: int = 2000
    n_chains: int = 4
    n_params: int = 2

    def test_arviz_translator(self):

        numpyro_mcmc = run_numpyro_mcmc(self.n_steps, self.n_chains)
        arviz_id = az.from_numpyro(numpyro_mcmc)
        df = df_from_arviz_id(arviz_id)

        assert df.shape == (self.n_steps * self.n_chains, self.n_params)

    def test_numpyro_translator(self):

        numpyro_mcmc = run_numpyro_mcmc(self.n_steps, self.n_chains)
        df = df_from_numpyro_mcmc(numpyro_mcmc)
        assert df.shape == (self.n_steps * self.n_chains, self.n_params)

    def test_emcee_translator(self):

        emcee_sampler = run_emcee_mcmc(self.n_steps, self.n_chains)
        df = df_from_emcee_sampler(emcee_sampler)
        assert df.shape == (self.n_steps * self.n_chains, self.n_params)
