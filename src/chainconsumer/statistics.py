from enum import Enum


class SummaryStatistic(Enum):
    MAX = "max"
    """The max value summary statistic is the default. The central point is set
    to your maximum likelihood, and the upper and lower bounds are determined by
    finding an iso-likelihood surface which encapsulates the required volume."""

    MAX_CENTRAL = "max_central"
    """As per the MAX method, this has the centre point at the maximum likelihood.
    However the lower and upper values come from the CDF, like the cumulative method."""

    CUMULATIVE = "cumulative"
    """The lower, central, and upper bound are determined by finding where on the marginalised
    sample CDF the points lie. This means the central point is the median."""

    MEAN = "mean"
    """As per the cumulative method, except the central value is placed in the midpoint between
    the upper and lower boundary. Not recommended, but was requested."""

    HDI = "hdi"
    """Use the highest density interval. Finds the narrowest interval covering the requested mass."""
