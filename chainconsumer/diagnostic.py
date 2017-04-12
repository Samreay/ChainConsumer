import numpy as np
import logging
from scipy.stats import normaltest
from statsmodels.regression.linear_model import yule_walker


class Diagnostic(object):
    def __init__(self, parent):
        self.parent = parent
        self._logger = logging.getLogger(__name__)

    def diagnostic_gelman_rubin(self, chain=None, threshold=0.05):
        r""" Runs the Gelman Rubin diagnostic on the supplied chains.

        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        threshold : float, optional
            The maximum deviation permitted from 1 for the final value
            :math:`\hat{R}`

        Returns
        -------
        float
            whether or not the chains pass the test

        Notes
        -----

        I follow PyMC in calculating the Gelman-Rubin statistic, where,
        having :math:`m` chains of length :math:`n`, we compute

        .. math::

            B = \frac{n}{m-1} \sum_{j=1}^{m} \left(\bar{\theta}_{.j} - \bar{\theta}_{..}\right)^2

            W = \frac{1}{m} \sum_{j=1}^{m} \left[ \frac{1}{n-1} \sum_{i=1}^{n} \left( \theta_{ij} - \bar{\theta_{.j}}\right)^2 \right]

        where :math:`\theta` represents each model parameter. We then compute
        :math:`\hat{V} = \frac{n_1}{n}W + \frac{1}{n}B`, and have our convergence ratio
        :math:`\hat{R} = \sqrt{\frac{\hat{V}}{W}}`. We check that for all parameters,
        this ratio deviates from unity by less than the supplied threshold.
        """
        if chain is None:
            keys = [n if n is not None else i for i, n in enumerate(self.parent._names)]
            return np.all([self.diagnostic_gelman_rubin(k, threshold=threshold) for k in keys])
        index = self.parent._get_chain(chain)
        num_walkers = self.parent._walkers[index]
        parameters = self.parent._parameters[index]
        name = self.parent._names[index] if self.parent._names[index] is not None else "%d" % index
        chain = self.parent._chains[index]
        chains = np.split(chain, num_walkers)
        assert num_walkers > 1, "Cannot run Gelman-Rubin statistic with only one walker"
        m = 1.0 * len(chains)
        n = 1.0 * chains[0].shape[0]
        all_mean = np.mean(chain, axis=0)
        chain_means = np.array([np.mean(c, axis=0) for c in chains])
        chain_std = np.array([np.std(c, axis=0) for c in chains])
        b = n / (m - 1) * ((chain_means - all_mean) ** 2).sum(axis=0)
        w = (1 / m) * chain_std.sum(axis=0)
        var = (n - 1) * w / n + b / n
        R = np.sqrt(var / w)
        passed = np.abs(R - 1) < threshold
        print("Gelman-Rubin Statistic values for chain %s" % name)
        for p, v, pas in zip(parameters, R, passed):
            param = "Param %d" % p if isinstance(p, int) else p
            print("%s: %7.5f (%s)" % (param, v, "Passed" if pas else "Failed"))
        return np.all(passed)

    def diagnostic_geweke(self, chain=None, first=0.1, last=0.5, threshold=0.05):
        """ Runs the Geweke diagnostic on the supplied chains.

        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        first : float, optional
            The amount of the start of the chain to use
        last : float, optional
            The end amount of the chain to use
        threshold : float, optional
            The p-value to use when testing for normality.

        Returns
        -------
        float
            whether or not the chains pass the test

        """
        if chain is None:
            keys = [n if n is not None else i for i, n in enumerate(self.parent._names)]
            return np.all([self.diagnostic_geweke(k, threshold=threshold) for k in keys])
        index = self.parent._get_chain(chain)
        num_walkers = self.parent._walkers[index]
        assert num_walkers is not None and num_walkers > 0, \
            "You need to specify the number of walkers to use the Geweke diagnostic."
        name = self.parent._names[index] if self.parent._names[index] is not None else "%d" % index
        chain = self.parent._chains[index]
        chains = np.split(chain, num_walkers)
        n = 1.0 * chains[0].shape[0]
        n_start = int(np.floor(first * n))
        n_end = int(np.floor((1 - last) * n))
        mean_start = np.array([np.mean(c[:n_start, i])
                               for c in chains for i in range(c.shape[1])])
        var_start = np.array([self._spec(c[:n_start, i]) / c[:n_start, i].size
                              for c in chains for i in range(c.shape[1])])
        mean_end = np.array([np.mean(c[n_end:, i])
                             for c in chains for i in range(c.shape[1])])
        var_end = np.array([self._spec(c[n_end:, i]) / c[n_end:, i].size
                            for c in chains for i in range(c.shape[1])])
        zs = (mean_start - mean_end) / (np.sqrt(var_start + var_end))
        stat, pvalue = normaltest(zs)
        print("Gweke Statistic for chain %s has p-value %e" % (name, pvalue))
        return pvalue > threshold

    # Method of estimating spectral density following PyMC.
    # See https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py
    def _spec(self, x, order=2):
        beta, sigma = yule_walker(x, order)
        return sigma ** 2 / (1. - np.sum(beta)) ** 2

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

        base_string = self.parent._get_latex_table(caption, label)
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
            aics = self.comparison_aic()
        else:
            aics = np.zeros(len(self.parent._chains))
        if bic:
            bics = self.comparison_bic()
        else:
            bics = np.zeros(len(self.parent._chains))
        if dic:
            dics = self.comparison_dic()
        else:
            dics = np.zeros(len(self.parent._chains))

        if sort == "bic":
            to_sort = bics
        elif sort == "aic":
            to_sort = aics
        elif sort == "dic":
            to_sort = dics
        else:
            raise ValueError("sort %s not recognised, must be dic, aic or dic" % sort)

        good = [i for i, t in enumerate(to_sort) if t is not None]
        names = [self.parent._names[g] for g in good]
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