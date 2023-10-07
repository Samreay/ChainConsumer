from enum import Enum


class SummaryStatistic(Enum):
    MAX = "max"
    MEAN = "mean"
    CUMULATIVE = "cumulative"
    MAX_SYMMETRIC = "max_symmetric"
    MAX_SHORTEST = "max_shortest"
    MAX_CENTRAL = "max_central"
