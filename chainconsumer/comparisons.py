# -*- coding: utf-8 -*-
from scipy.interpolate import griddata
import numpy as np
import logging

from .helpers import get_latex_table_frame


class Comparison(object):
    def __init__(self, parent):
        self.parent = parent
        self._logger = logging.getLogger("chainconsumer")

    def dic(self):
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
        for i, chain in enumerate(self.parent.chains):
            p = chain.posterior
            if p is None:
                dics_bool.append(False)
                self._logger.warn("You need to set the posterior for chain %s to get the DIC" % chain.name)
            else:
                dics_bool.append(True)
                num_params = chain.chain.shape[1]
                means = np.array([np.average(chain.chain[:, ii], weights=chain.weights) for ii in range(num_params)])
                d = -2 * p
                d_of_mean = griddata(chain.chain, d, means, method='nearest')[0]
                mean_d = np.average(d, weights=chain.weights)
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

    def bic(self):
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
        for i, chain in enumerate(self.parent.chains):
            p, n_data, n_free = chain.posterior, chain.num_eff_data_points, chain.num_free_params
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
                                  (missing[:-2], chain.name))
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

    def aic(self):
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
        for i, chain in enumerate(self.parent.chains):
            p, n_data, n_free = chain.posterior, chain.num_eff_data_points, chain.num_free_params
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
                                  (missing[:-2], chain.name))
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

    def comparison_table(self, caption=None, label="tab:model_comp", hlines=True,
                         aic=True, bic=True, dic=True, sort="bic", descending=True):  # pragma: no cover
        """
        Return a LaTeX ready table of model comparisons.

        Parameters
        ----------
        caption : str, optional
            The table caption to insert.
        label : str, optional
            The table label to insert.
        hlines : bool, optional
            Whether to insert hlines in the table or not.
        aic : bool, optional
            Whether to include a column for AICc or not.
        bic : bool, optional
            Whether to include a column for BIC or not.
        dic : bool, optional
            Whether to include a column for DIC or not.
        sort : str, optional
            How to sort the models. Should be one of "bic", "aic" or "dic".
        descending : bool, optional
            The sort order.

        Returns
        -------
        str
            A LaTeX table to be copied into your document.
        """

        if sort == "bic":
            assert bic, "You cannot sort by BIC if you turn it off"
        if sort == "aic":
            assert aic, "You cannot sort by AIC if you turn it off"
        if sort == "dic":
            assert dic, "You cannot sort by DIC if you turn it off"

        if caption is None:
            caption = ""
        if label is None:
            label = ""

        base_string = get_latex_table_frame(caption, label)
        end_text = " \\\\ \n"
        num_cols = 1 + (1 if aic else 0) + (1 if bic else 0)
        column_text = "c" * (num_cols + 1)
        center_text = ""
        hline_text = "\\hline\n"
        if hlines:
            center_text += hline_text
        center_text += "\tModel" + (" & AIC" if aic else "") + (" & BIC " if bic else "") \
                       + (" & DIC " if dic else "") + end_text
        if hlines:
            center_text += "\t" + hline_text
        if aic:
            aics = self.aic()
        else:
            aics = np.zeros(len(self.parent.chains))
        if bic:
            bics = self.bic()
        else:
            bics = np.zeros(len(self.parent.chains))
        if dic:
            dics = self.dic()
        else:
            dics = np.zeros(len(self.parent.chains))

        if sort == "bic":
            to_sort = bics
        elif sort == "aic":
            to_sort = aics
        elif sort == "dic":
            to_sort = dics
        else:
            raise ValueError("sort %s not recognised, must be dic, aic or dic" % sort)

        good = [i for i, t in enumerate(to_sort) if t is not None]
        names = [self.parent.chains[g].name for g in good]
        aics = [aics[g] for g in good]
        bics = [bics[g] for g in good]
        to_sort = bics if sort == "bic" else aics

        indexes = np.argsort(to_sort)

        if descending:
            indexes = indexes[::-1]

        for i in indexes:
            line = "\t" + names[i]
            if aic:
                line += "  &  %5.1f  " % aics[i]
            if bic:
                line += "  &  %5.1f  " % bics[i]
            if dic:
                line += "  &  %5.1f  " % dics[i]
            line += end_text
            center_text += line
        if hlines:
            center_text += "\t" + hline_text

        return base_string % (column_text, center_text)