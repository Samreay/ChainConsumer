import logging

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from .analysis import SummaryStatistic
from .base import BetterBase
from .colors import ColourInput, colors


class Chain(BetterBase):
    chain: pd.DataFrame = Field(
        default=...,
        description="The chain data as a pandas DataFrame",
    )
    name: str = Field(
        default=...,
        description="The name of the chain",
    )
    column_labels: dict[str, str] = Field(
        default={}, description="A dictionary mapping column names to labels. If not set, will use the column names."
    )
    weight_column: str = Field(
        default="weights",
        description="The name of the weight column, if it exists",
    )
    posterior_column: str = Field(
        default="posterior",
        description="The name of the log posterior column, if it exists",
    )
    walkers: int = Field(
        default=1,
        ge=1,
        description="The number of walkers in the chain",
    )
    grid: bool = Field(
        default=False,
        description="Whether the chain is a sampled grid or not",
    )
    num_free_params: int | None = Field(
        default=None,
        description="The number of free parameters in the chain",
        ge=0,
    )
    num_eff_data_points: float | None = Field(
        default=None,
        description="The number of effective data points",
        ge=0,
    )
    power: float = Field(
        default=1.0,
        description="Raise the posterior surface to this. Useful for inflating or deflating uncertainty for debugging.",
    )

    statistics: SummaryStatistic = Field(
        default=SummaryStatistic.MAX,
        description="The summary statistic to use",
    )

    color: ColourInput | None = Field(default=None, description="The color of the chain")
    linestyle: str | None = Field(default=None, description="The line style of the chain")
    linewidth: float | None = Field(default=None, description="The line width of the chain")
    cloud: bool | None = Field(default=False, description="Whether to show the cloud of the chain")
    shade: bool | None = Field(default=True, description="Whether to shade the chain")
    shade_alpha: float | None = Field(default=None, description="The alpha of the shading")
    shade_gradient: float | None = Field(default=None, description="The contrast between contour levels")
    bar_shade: bool | None = Field(default=None, description="Whether to shade marginalised distributions")
    bins: int | float | None = Field(default=None, description="The number of bins to use for histograms")
    kde: int | float | bool | None = Field(default=False, description="The bandwidth for KDEs")
    smooth: int | float | bool | None = Field(default=3, description="The smoothing for histograms.")
    color_params: str | None = Field(default=None, description="The parameter (column) to use for coloring")
    plot_color_params: bool | None = Field(default=None, description="Whether to plot the color parameter")
    cmap: str | None = Field(default=None, description="The colormap to use for shading")
    num_cloud: int | float | None = Field(default=None, description="The number of points in the cloud")
    plot_contour: bool | None = Field(default=True, description="Whether to plot contours")
    plot_point: bool | None = Field(default=False, description="Whether to plot points")
    show_as_1d_prior: bool | None = Field(default=False, description="Whether to show as a 1D prior")
    marker_style: str | None = Field(default=None, description="The marker style to use")
    marker_size: int | float | None = Field(default=None, description="The marker size to use")
    marker_alpha: int | float | None = Field(default=None, description="The marker alpha to use")
    zorder: int | None = Field(default=None, description="The zorder to use")

    shift_params: bool = Field(
        default=False,
        description="Whether to shift the parameters by subtracting each parameters mean",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def max_posterior_row(self) -> pd.Series | None:
        if self.posterior_column not in self.chain.columns:
            logging.warning("No posterior column found, cannot find max posterior row")
            return None
        argmax = self.chain[self.posterior_column].argmax()
        return self.chain.loc[argmax]

    @property
    def labels(self) -> list[str]:
        return [self.column_labels.get(col, col) for col in self.chain.columns]

    @property
    def weights(self) -> np.ndarray:
        return self.chain[self.weight_column].to_numpy()

    @property
    def log_posterior(self) -> np.ndarray | None:
        if self.posterior_column not in self.chain.columns:
            return None
        return self.chain[self.posterior_column].to_numpy()

    @property
    def color_data(self) -> np.ndarray | None:
        if self.color_params is None:
            return None
        return self.chain[self.color_params].to_numpy()

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: str | np.ndarray | list[float] | None) -> str | None:
        if v is None:
            return None
        return colors.format(v)

    @model_validator(mode="after")
    def validate_model(self) -> "Chain":
        assert not self.chain.empty, "Your chain is empty. This is not ideal."

        # If weights aren't set, add them all as one
        if self.weight_column not in self.chain:
            self.chain[self.weight_column] = 1.0
        else:
            assert np.all(self.weights > 0), "Weights must be positive and non-zero"
            assert np.all(np.isfinite(self.weights)), "Weights must be finite"

        # Apply the mean shift if it is set to true
        if self.shift_params:
            for param in self.chain:
                self.chain[param] -= np.average(self.chain[param], weights=self.weights)  # type: ignore

        # Check the walkers
        assert self.chain.shape[0] % self.walkers == 0, (
            f"Chain {self.name} has {self.chain.shape[0]} steps, "
            "which is not divisible by {self.walkers} walkers. This is not good."
        )

        # And the log posterior
        if self.log_posterior is not None:
            assert np.all(np.isfinite(self.log_posterior)), f"Chain {self.name} has NaN or inf in the log-posterior"

        # And if the color_params are set, ensure they're in the dataframe
        if self.color_params is not None:
            assert (
                self.color_params in self.chain.columns
            ), f"Chain {self.name} does not have color parameter {self.color_params}"

        return self

    def get_data(self, columns: list[str] | str):
        if isinstance(columns, str):
            columns = [columns]
        return self.chain[columns]
