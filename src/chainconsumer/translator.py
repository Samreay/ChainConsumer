import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    try :
        from numpyro.infer import MCMC
    except ImportError:
        raise ImportError('numpyro is not installed (see : https://num.pyro.ai/en/stable/getting_started.html#installation)')

    try :
        from arviz import InferenceData
    except ImportError:
        raise ImportError('arviz is not installed (see : https://python.arviz.org/en/stable/getting_started/Installation.html)')

    try :
        from emcee import EnsembleSampler
    except ImportError:
        raise ImportError('emcee is not installed (see : https://emcee.readthedocs.io/en/stable/user/install/)')


def df_from_arviz_id(inference_data: "InferenceData") -> pd.DataFrame:

    return inference_data.to_dataframe(groups="posterior").drop(columns=["chain", "draw"])


def df_from_numpyro_mcmc(mcmc: "MCMC") -> pd.DataFrame:

    samples = {key: np.ravel(value) for key, value in mcmc.get_samples().items()}

    return pd.DataFrame.from_dict(samples)


def df_from_emcee_sampler(sampler: "EnsembleSampler") -> pd.DataFrame:

    return pd.DataFrame.from_dict(sampler.get_chain(flat=True))
