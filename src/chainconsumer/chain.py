from __future__ import annotations

import logging
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from .analysis import SummaryStatistic
from .base import BetterBase
from .colors import ColorInput, colors

ChainName: TypeAlias = str
ColumnName: TypeAlias = str


class MaxPosterior(BetterBase):
    log_posterior: float
    coordinate: dict[ColumnName, float]

    @property
    def vec_coordinate(self) -> np.ndarray:
        return np.array(list(self.coordinate.values()))


class Named2DMatrix(BetterBase):
    columns: list[str]
    matrix: np.ndarray  # type: ignore


class ChainConfig(BetterBase):
    """The configuration for a chain. This is used to set the default values for
    plotting chains, and is also used to store the configuration of a chain.

    Note that some attributes are defaulted to None instead of their type hint.
    Like color. This indicates that this parameter should be inferred if not explicitly
    set, and that this inference requires knowledge of the other chains. For example,
    if you have two chains, you probably want them to be different colors.
    """

    statistics: SummaryStatistic = Field(default=SummaryStatistic.MAX, description="The summary statistic to use")
    summary_area: float = Field(default=0.6827, description="The area to use for summary statistics")
    sigmas: list[float] = Field(default=None, description="The sigmas to use for summary statistics")
    color: ColorInput = Field(default=None, description="The color of the chain")
    linestyle: str = Field(default="-", description="The line style of the chain")
    linewidth: float = Field(default=1.0, description="The line width of the chain")
    cloud: bool = Field(default=False, description="Whether to show the cloud of the chain")
    show_contour_labels: bool = Field(default=False, description="Whether to show contour labels")
    shade: bool = Field(default=None, description="Whether to shade the chain")
    shade_alpha: float = Field(default=None, description="The alpha of the shading")
    shade_gradient: float = Field(default=1.0, description="The contrast between contour levels")
    bar_shade: bool = Field(default=None, description="Whether to shade marginalised distributions")
    bins: int | float = Field(
        default=1.0, description="The number of bins to use for histograms. If a float, used to scale the default bins"
    )
    kde: int | float | bool = Field(default=False, description="The bandwidth for KDEs")
    smooth: int = Field(default=3, description="The smoothing for histograms. Set to 0 for no smoothing")
    color_param: str | None = Field(default=None, description="The parameter (column) to use for coloring")
    cmap: str = Field(default="viridis", description="The colormap to use for shading cloud points")
    num_cloud: int | float = Field(default=10000, description="The number of points in the cloud")
    plot_cloud: bool = Field(default=False, description="Whether to plot the cloud")
    plot_contour: bool = Field(default=True, description="Whether to plot contours")
    plot_point: bool = Field(default=False, description="Whether to plot points")
    show_as_1d_prior: bool = Field(default=False, description="Whether to show as a 1D prior")
    marker_style: str = Field(default=None, description="The marker style to use")
    marker_size: int | float = Field(default=None, description="The marker size to use")
    marker_alpha: int | float = Field(default=None, description="The marker alpha to use")
    zorder: int = Field(default=None, description="The zorder to use")
    shift_params: bool = Field(
        default=False,
        description="Whether to shift the parameters by subtracting each parameters mean",
    )

    @field_validator("color")
    @classmethod
    def convert_color(cls, v: ColorInput | None) -> str | None:
        if v is None:
            return None
        return colors.format(v)

    def apply_if_none(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

    def apply(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class Chain(ChainConfig):
    samples: pd.DataFrame = Field(
        default=...,
        description="The chain data as a pandas DataFrame",
    )
    name: ChainName = Field(
        default=...,
        description="The name of the chain",
    )

    weight_column: ColumnName = Field(
        default="weights",
        description="The name of the weight column, if it exists",
    )
    posterior_column: ColumnName = Field(
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

    @property
    def skip(self) -> bool:
        return self.samples.empty or not (self.plot_contour or self.plot_cloud or self.plot_point)

    @property
    def max_posterior_row(self) -> pd.Series | None:
        if self.posterior_column not in self.samples.columns:
            logging.warning("No posterior column found, cannot find max posterior row")
            return None
        argmax = self.samples[self.posterior_column].argmax()
        return self.samples.loc[argmax]

    @property
    def weights(self) -> np.ndarray:
        return self.samples[self.weight_column].to_numpy()

    @property
    def log_posterior(self) -> np.ndarray | None:
        if self.posterior_column not in self.samples.columns:
            return None
        return self.samples[self.posterior_column].to_numpy()

    @property
    def color_data(self) -> np.ndarray | None:
        if self.color_param is None:
            return None
        return self.samples[self.color_param].to_numpy()

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: str | np.ndarray | list[float] | None) -> str | None:
        if v is None:
            return None
        return colors.format(v)

    @model_validator(mode="after")
    def validate_model(self) -> Chain:
        assert not self.samples.empty, "Your chain is empty. This is not ideal."

        # If weights aren't set, add them all as one
        if self.weight_column not in self.samples:
            self.samples[self.weight_column] = 1.0
        else:
            assert np.all(self.weights > 0), "Weights must be positive and non-zero"
            assert np.all(np.isfinite(self.weights)), "Weights must be finite"

        # Apply the mean shift if it is set to true
        if self.shift_params:
            for param in self.samples:
                self.samples[param] -= np.average(self.samples[param], weights=self.weights)  # type: ignore

        # Check the walkers
        assert self.samples.shape[0] % self.walkers == 0, (
            f"Chain {self.name} has {self.samples.shape[0]} steps, "
            "which is not divisible by {self.walkers} walkers. This is not good."
        )

        # And the log posterior
        if self.log_posterior is not None:
            assert np.all(np.isfinite(self.log_posterior)), f"Chain {self.name} has NaN or inf in the log-posterior"

        # And if the color_params are set, ensure they're in the dataframe
        if self.color_param is not None:
            assert (
                self.color_param in self.samples.columns
            ), f"Chain {self.name} does not have color parameter {self.color_param}"

        return self

    def get_data(self, columns: str) -> pd.Series[float]:
        return self.samples[columns]

    @classmethod
    def from_covariance(
        cls,
        mean: np.ndarray,
        covariance: np.ndarray,
        columns: list[str],
        name: str,
        **kwargs: dict[str, Any],
    ) -> Chain:
        """Generate samples as per mean and covariance supplied. Useful for Fisher matrix forecasts.

        Args:
            mean (np.ndarray): The an array of mean values.
            covariance (np.ndarray): The 2D array describing the covariance.
                Dimensions should agree with the `mean` input.
            columns (list[str]): A list of parameter names, one for each column (dimension) in the mean array.
            name (str): The name of the chain. Defaults to None.
            kwargs: Any other arguments to pass to the Chain constructor.

        Returns:
            Chain: The generated chain.
        """
        rng = np.random.default_rng()
        samples = rng.multivariate_normal(mean, covariance, size=1000000)
        df = pd.DataFrame(samples, columns=columns)
        return cls(samples=df, name=name, **kwargs)  # type: ignore

    def divide(self) -> list[Chain]:
        """Returns a ChainConsumer instance containing all the walks of a given chain
        as individual chains themselves.

        This method might be useful if, for example, your chain was made using
        MCMC with 4 walkers. To check the sampling of all 4 walkers agree, you could
        call this to get a ChainConsumer instance with one chain for ech of the
        four walks. If you then plot, hopefully all four contours
        you would see agree.

        Returns:
            list[Chain]: One chain per walker, split evenly
        """
        assert self.walkers > 1, "Cannot divide a chain with only one walker"
        assert not self.grid, "Cannot divide a grid chain"

        splits = np.split(self.samples, self.walkers)
        chains = []
        for i, split in enumerate(splits):
            df = pd.DataFrame(split, columns=self.samples.columns)
            options = self.model_dump(exclude={"samples", "name"})
            chain = Chain(samples=df, name=f"{self.name} Walker {i}", **options)
            chains.append(chain)

        return chains

    def get_max_posterior_point(self) -> MaxPosterior | None:
        if self.max_posterior_row is None:
            return None
        row = self.max_posterior_row.to_dict()
        log_posterior = row.pop(self.posterior_column)
        return MaxPosterior(log_posterior=log_posterior, coordinate=row)

    def get_covariance(self, columns: list[str] | None) -> Named2DMatrix:
        if columns is None:
            columns = list(self.samples.columns)
        cov = np.cov(self.samples[columns], rowvar=False, aweights=self.weights)
        return Named2DMatrix(columns=columns, matrix=cov)

    def get_correlation(self, columns: list[str] | None) -> Named2DMatrix:
        cov = self.get_covariance(columns)
        diag = np.sqrt(np.diag(cov.matrix))
        divisor = diag[None, :] * diag[:, None]
        correlations = cov.matrix / divisor
        return Named2DMatrix(columns=cov.columns, matrix=correlations)
