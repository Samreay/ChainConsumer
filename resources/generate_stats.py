import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from scipy.stats import gamma

from chainconsumer import Chain, ChainConsumer
from chainconsumer.statistics import SummaryStatistic

# Activate latex text rendering
rc("font", family="serif", serif=["Computer Modern Roman"], size=13)
rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

x = np.linspace(0, 5, 100)

loc = 4
scale = 0.45

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, height_ratios=[0.5, 0.5], figsize=(5, 5))
axs[0].plot(x, gamma.pdf(x, a=loc, scale=scale), color="black")
axs[1].plot(x, gamma.cdf(x, a=loc, scale=scale), color="black")


axs[1].set_xlabel("$x$")
axs[0].set_ylabel("$P(x)$")
axs[1].set_ylabel("$C(x)$")
axs[0].set_xlim(0, 5.0)
axs[0].set_ylim(0, 0.6)
axs[1].set_ylim(0, 1)

samples = pd.DataFrame.from_dict({"gamma": gamma.rvs(size=10_000_000, a=loc, scale=scale)})

summary_list = [
    (SummaryStatistic.MAX, "MAX"),
    (SummaryStatistic.CUMULATIVE, "CUMULATIVE"),
    (SummaryStatistic.MEAN, "MEAN"),
    (SummaryStatistic.HDI, "HDI"),
]

chains = []

for summary, name in summary_list:
    chains.append(Chain(samples=samples, statistics=summary, name=name))

cc = ChainConsumer()

summary_result = cc.analysis.get_summary(chains=chains, columns=["gamma"])

for (_summary, name), color, linestyle, marker_style in zip(
    summary_list,
    ["r", "g", "b", "y"],
    [":", "--", "-", "-."],
    ["o", "^", "s", "*"],
    strict=False,
):
    bound = summary_result[name]["gamma"]

    x_min, x_mid, x_max = bound.lower, bound.center, bound.upper

    axs[0].scatter(x_mid, gamma.pdf(x_mid, a=loc, scale=scale), label=name, zorder=10, color=color, marker=marker_style)
    axs[1].scatter(x_mid, gamma.cdf(x_mid, a=loc, scale=scale), zorder=10, color=color, marker=marker_style)

    axs[0].vlines(
        x=x_min, ymin=0, ymax=gamma.pdf(x_min, a=loc, scale=scale), color=color, linestyle=linestyle, alpha=0.5
    )
    axs[0].vlines(
        x=x_max, ymin=0, ymax=gamma.pdf(x_max, a=loc, scale=scale), color=color, linestyle=linestyle, alpha=0.5
    )

    axs[1].hlines(
        xmin=0, xmax=x_min, y=gamma.cdf(x_min, a=loc, scale=scale), color=color, linestyle=linestyle, alpha=0.5
    )
    axs[1].hlines(
        xmin=0, xmax=x_max, y=gamma.cdf(x_max, a=loc, scale=scale), color=color, linestyle=linestyle, alpha=0.5
    )

axs[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("stats.png", bbox_inches="tight")
