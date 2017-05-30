import logging
import numpy as np


def get_parameter_text(lower, maximum, upper, wrap=False):
    """ Generates LaTeX appropriate text from marginalised parameter bounds.

    Parameters
    ----------
    lower : float
        The lower bound on the parameter
    maximum : float
        The value of the parameter with maximum probability
    upper : float
        The upper bound on the parameter
    wrap : bool
        Wrap output text in dollar signs for LaTeX

    Returns
    -------
    str
        The formatted text given the parameter bounds
    """
    if lower is None or upper is None:
        return ""
    upper_error = upper - maximum
    lower_error = maximum - lower
    resolution = min(np.floor(np.log10(np.abs(upper_error))),
                     np.floor(np.log10(np.abs(lower_error))))
    factor = 0
    fmt = "%0.1f"
    r = 1
    if np.abs(resolution) > 2:
        factor = -resolution
    if resolution == 2:
        fmt = "%0.0f"
        factor = -1
        r = 0
    if resolution == 1:
        fmt = "%0.0f"
    if resolution == -1:
        fmt = "%0.2f"
        r = 2
    elif resolution == -2:
        fmt = "%0.3f"
        r = 3
    upper_error *= 10 ** factor
    lower_error *= 10 ** factor
    maximum *= 10 ** factor
    upper_error = round(upper_error, r)
    lower_error = round(lower_error, r)
    maximum = round(maximum, r)
    if maximum == -0.0:
        maximum = 0.0
    if resolution == 2:
        upper_error *= 10 ** -factor
        lower_error *= 10 ** -factor
        maximum *= 10 ** -factor
        factor = 0
        fmt = "%0.0f"
    upper_error_text = fmt % upper_error
    lower_error_text = fmt % lower_error
    if upper_error_text == lower_error_text:
        text = r"%s\pm %s" % (fmt, "%s") % (maximum, lower_error_text)
    else:
        text = r"%s^{+%s}_{-%s}" % (fmt, "%s", "%s") % \
               (maximum, upper_error_text, lower_error_text)
    if factor != 0:
        text = r"\left( %s \right) \times 10^{%d}" % (text, -factor)
    if wrap:
        text = "$%s$" % text
    return text


def get_extents(data, weight):
    hist, be = np.histogram(data, weights=weight, bins=1000, normed=True)
    bc = 0.5 * (be[1:] + be[:-1])
    cdf = hist.cumsum()
    cdf = cdf / cdf.max()
    icdf = (1 - cdf)[::-1]
    threshold = 1e-3
    i1 = np.where(cdf > threshold)[0][0]
    i2 = np.where(icdf > threshold)[0][0]
    return bc[i1], bc[-i2]