from scipy.interpolate import griddata
import numpy as np
import logging


class Comparison(object):
    def __init__(self, parent):
        self.parent = parent
        self._logger = logging.getLogger(__name__)

    def comparison_dic(self):
        r""" Returns the corrected Deviance Information Criterion (DIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, this method will return `None` for that chain. **Note that
        the DIC metric is only valid on posterior surfaces which closely resemble multivariate normals!**
        Formally, we follow Liddle (2007) and first define *Bayesian complexity* as

        .. math::
            p_D = \bar{D}(\theta) - D(\bar{\theta}),

        where :math:`D(\theta) = -2\ln(P(\theta)) + C` is the deviance, where :math:`P` is the posterior
        and :math:`C` a constant. From here the DIC is defined as

        .. math::
            DIC \equiv D(\bar{\theta}) + 2p_D = \bar{D}(\theta) + p_D.

        Returns
        -------
        list[float]
            A list of all the DIC values - one per chain, in the order in which the chains were added.

        References
        ----------
        [1] Andrew R. Liddle, "Information criteria for astrophysical model selection", MNRAS (2007)
        """
        dics = []
        dics_bool = []
        for i, p in enumerate(self.parent._posteriors):
            if p is None:
                dics_bool.append(False)
                self._logger.warn("You need to set the posterior for chain %s to get the DIC" %
                                  self.parent._get_chain_name(i))
            else:
                dics_bool.append(True)
                chain = self.parent._chains[i]
                num_params = chain.shape[1]
                means = np.array([np.average(chain[:, ii], weights=self.parent._weights[i]) for ii in range(num_params)])
                d = -2 * p
                d_of_mean = griddata(chain, d, means, method='nearest')[0]
                mean_d = np.average(d, weights=self.parent._weights[i])
                p_d = mean_d - d_of_mean
                dic = mean_d + p_d
                dics.append(dic)
        if len(dics) > 0:
            dics -= np.min(dics)
        dics_fin = []
        i = 0
        for b in dics_bool:
            if not b:
                dics_fin.append(None)
            else:
                dics_fin.append(dics[i])
                i += 1
        return dics_fin

    def comparison_bic(self):
        r""" Returns the corrected Bayesian Information Criterion (BIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the BIC is defined as

        .. math::
            BIC \equiv -2\ln(P) + k \ln(N),

        where :math:`P` represents the posterior, :math:`k` the number of model parameters and :math:`N`
        the number of independent data points used in the model fitting.

        Returns
        -------
        list[float]
            A list of all the BIC values - one per chain, in the order in which the chains were added.
        """
        bics = []
        bics_bool = []
        for i, (p, n_data, n_free) in enumerate(zip(self.parent._posteriors, self.parent._num_data, self.parent._num_free)):
            if p is None or n_data is None or n_free is None:
                bics_bool.append(False)
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                self._logger.warn("You need to set %s for chain %s to get the BIC" %
                                  (missing[:-2], self.parent._get_chain_name(i)))
            else:
                bics_bool.append(True)
                bics.append(n_free * np.log(n_data) - 2 * np.max(p))
        if len(bics) > 0:
            bics -= np.min(bics)
        bics_fin = []
        i = 0
        for b in bics_bool:
            if not b:
                bics_fin.append(None)
            else:
                bics_fin.append(bics[i])
                i += 1
        return bics_fin

    def comparison_aic(self):
        r""" Returns the corrected Akaike Information Criterion (AICc) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the AIC is defined as

        .. math::
            AIC \equiv -2\ln(P) + 2k,

        where :math:`P` represents the posterior, and :math:`k` the number of model parameters. The AICc
        is then defined as

        .. math::
            AIC_c \equiv AIC + \frac{2k(k+1)}{N-k-1},

        where :math:`N` represents the number of independent data points used in the model fitting.
        The AICc is a correction for the AIC to take into account finite chain sizes.

        Returns
        -------
        list[float]
            A list of all the AICc values - one per chain, in the order in which the chains were added.
        """
        aics = []
        aics_bool = []
        for i, (p, n_data, n_free) in enumerate(zip(self.parent._posteriors, self.parent._num_data, self.parent._num_free)):
            if p is None or n_data is None or n_free is None:
                aics_bool.append(False)
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                self._logger.warn("You need to set %s for chain %s to get the AIC" %
                                  (missing[:-2], self.parent._get_chain_name(i)))
            else:
                aics_bool.append(True)
                c_cor = (1.0 * n_free * (n_free + 1) / (n_data - n_free - 1))
                aics.append(2.0 * (n_free + c_cor - np.max(p)))
        if len(aics) > 0:
            aics -= np.min(aics)
        aics_fin = []
        i = 0
        for b in aics_bool:
            if not b:
                aics_fin.append(None)
            else:
                aics_fin.append(aics[i])
                i += 1
        return aics_fin
