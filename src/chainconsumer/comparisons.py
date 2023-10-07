from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .helpers import get_latex_table_frame
from .log import logger


class Comparison:
    def __init__(self, parent: "ChainConsumer"):
        self.parent = parent

    def dic(self) -> dict[str, float]:
        r"""Returns the corrected Deviance Information Criterion (DIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, this method will return `None` for that chain. **Note that
        the DIC metric is only valid on posterior surfaces which closely resemble multivariate normals!**
        Formally, we follow Liddle (2007) and first define *Bayesian complexity* as

        .. math::
            p_D = \bar{D}(\theta) - D(\bar{\theta}),

        where :math:`D(\theta) = -2\ln(P(\theta)) + C` is the deviance, where :math:`P` is the posterior
        and :math:`C` a constant. From here the DIC is defined as

        .. math::
            DIC \equiv D(\bar{\theta}) + 2p_D = \bar{D}(\theta) + p_D.

        Returns:
            dict[str, float]: A dict of chain name to DIC value.

        References
        ----------
        [1] Andrew R. Liddle, "Information criteria for astrophysical model selection", MNRAS (2007)
        """
        dics = {}
        for name, chain in self.parent._chains.items():
            p = chain.log_posterior
            if p is None:
                logger.warning("You need to set the posterior for chain %s to get the DIC" % chain.name)
            else:
                num_params = chain.samples.shape[1]
                means = np.array([np.average(chain.samples[:, ii], weights=chain.weights) for ii in range(num_params)])
                d = -2 * p
                d_of_mean = griddata(chain.samples, d, means, method="nearest")[0]
                mean_d = np.average(d, weights=chain.weights)
                p_d = mean_d - d_of_mean
                dic = mean_d + p_d
                dics[name] = dic
        if dics:
            min_dic = np.min(list(dics.values()))
            for name in dics:
                dics[name] -= min_dic
        return dics

    def bic(self) -> dict[str, float]:
        r"""Returns the corrected Bayesian Information Criterion (BIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the BIC is defined as

        .. math::
            BIC \equiv -2\ln(P) + k \ln(N),

        where :math:`P` represents the posterior, :math:`k` the number of model parameters and :math:`N`
        the number of independent data points used in the model fitting.

        Returns:
            dict[str, float]: A dict of chain name to BIC value.

        """
        bics = {}
        for name, chain in self.parent._chains.items():
            p, n_data, n_free = chain.log_posterior, chain.num_eff_data_points, chain.num_free_params
            if p is None or n_data is None or n_free is None:
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                logger.warning(f"You need to set {missing[:-2]} for chain {name} to get the BIC")
            else:
                bics[name] = n_free * np.log(n_data) - 2 * np.max(p)
        if bics:
            min_bic = np.min(list(bics.values()))
            for name in bics:
                bics[name] -= min_bic

        return bics

    def aic(self) -> dict[str, float]:
        r"""Returns the corrected Akaike Information Criterion (AICc) for all chains loaded into ChainConsumer.

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

        Returns:
            dict[str, float]: A dict of chain name to AIC value.

        """
        aics = {}
        for name, chain in self.parent._chains.items():
            p, n_data, n_free = chain.log_posterior, chain.num_eff_data_points, chain.num_free_params
            if p is None or n_data is None or n_free is None:
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                logger.warning(f"You need to set {missing[:-2]} for chain {chain.name} to get the AIC")
            else:
                c_cor = 1.0 * n_free * (n_free + 1) / (n_data - n_free - 1)
                aics[name] = 2.0 * (n_free + c_cor - np.max(p))
        if aics:
            min_aic = np.min(list(aics.values()))
            for name in aics:
                aics[name] -= min_aic
        return aics

    def comparison_table(
        self,
        caption: str | None = None,
        label: str = "tab:model_comp",
        hlines: bool = True,
        aic: bool = True,
        bic: bool = True,
        dic: bool = True,
        sort: Literal["bic", "aic", "dic"] = "bic",
        descending: bool = True,
    ) -> str:  # pragma: no cover
        """
        Return a LaTeX ready table of model comparisons.

        Args:
            caption (str, optional): The table caption to insert. Defaults to None.
            label (str, optional): The table label to insert. Defaults to "tab:model_comp".
            hlines (bool, optional): Whether to insert hlines in the table or not. Defaults to True.
            aic (bool, optional): Whether to include a column for AICc or not. Defaults to True.
            bic (bool, optional): Whether to include a column for BIC or not. Defaults to True.
            dic (bool, optional): Whether to include a column for DIC or not. Defaults to True.
            sort (str, optional): How to sort the models. Should be one of "bic", "aic" or "dic". Defaults to "bic".
            descending (bool, optional): The sort order. Defaults to True.

        Returns:
            str: A LaTeX table to be copied into your document.
        """

        if sort == "bic":
            assert bic, "You cannot sort by BIC if you turn it off"
        if sort == "aic":
            assert aic, "You cannot sort by AIC if you turn it off"
        if sort == "dic":
            assert dic, "You cannot sort by DIC if you turn it off"
        assert sort in ["bic", "aic", "dic"], f"sort {sort} not recognised, must be dic, aic or dic"

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
        center_text += (
            "\tModel" + (" & AIC" if aic else "") + (" & BIC " if bic else "") + (" & DIC " if dic else "") + end_text
        )
        if hlines:
            center_text += "\t" + hline_text

        series = {}
        if aic:
            series["aic"] = self.aic()
        if bic:
            series["bic"] = self.bic()
        if dic:
            series["dic"] = self.dic()

        df = pd.DataFrame(series).sort_values(by=sort, ascending=not descending)
        for name, row in df.iterrows():
            chain_name: str = str(name)
            line = "\t" + chain_name
            if aic:
                line += f"  &  {row['aic']:5.1f}  "
            if bic:
                line += f"  &  {row['bic']:5.1f}  "
            if dic:
                line += f"  &  {row['dic']:5.1f}  "
            line += end_text
            center_text += line
        if hlines:
            center_text += "\t" + hline_text

        return base_string % (column_text, center_text)


if __name__ == "__main__":
    from .chainconsumer import ChainConsumer
