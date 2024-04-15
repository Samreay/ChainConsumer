"""# Chains and ChainConfigs

This file is where I expect most people will focus their time. It contains a general configuration class,
which stores non-unique things that chains use. Like line styles, colours, etc.

It is then extended by the `Chain` class, which contains the actual data.

There are also a few helper functions and objects in here, like the `MaxPosterior` class which
provides the log posterior and the coordinate at which it can be found for the chain."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import arviz as az
import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from .base import BetterBase
from .color_finder import ColorInput, colors
from .statistics import SummaryStatistic

if TYPE_CHECKING:
    # Extra packages imported for type checking
    import arviz
    import emcee
    import numpyro

ChainName: TypeAlias = str
ColumnName: TypeAlias = str


class Named2DMatrix(BetterBase):
    """A 2D matrix with named columns. Used for covariance and correlation matrices."""

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
    summary_area: float = Field(default=0.6827, ge=0, le=1.0, description="The area to use for summary statistics")
    sigmas: list[float] = Field(default=[0, 1, 2], description="The sigmas to use for summary statistics")
    color: ColorInput | None = Field(default=None, description="The color of the chain")  # type: ignore
    linestyle: str = Field(default="-", description="The line style of the chain")
    linewidth: float = Field(default=1.0, description="The line width of the chain")
    show_contour_labels: bool = Field(default=False, description="Whether to show contour labels")
    shade: bool = Field(default=True, description="Whether to shade the chain")
    shade_alpha: float = Field(default=0.5, description="The alpha of the shading")
    shade_gradient: float = Field(default=1.0, description="The contrast between contour levels")
    bar_shade: bool = Field(default=True, description="Whether to shade marginalised distributions")
    bins: int | None = Field(default=None, description="The number of bins to use for histograms.")
    kde: int | float | bool = Field(default=False, description="The bandwidth for KDEs")
    smooth: int = Field(default=3, description="The smoothing for histograms. Set to 0 for no smoothing")
    color_param: str | None = Field(default=None, description="The parameter (column) to use for coloring")
    cmap: str = Field(default="viridis", description="The colormap to use for shading cloud points")
    num_cloud: int | float = Field(default=10000, description="The number of points in the cloud")
    plot_cloud: bool = Field(default=False, description="Whether to plot the cloud")
    plot_contour: bool = Field(default=True, description="Whether to plot contours")
    plot_point: bool = Field(default=False, description="Whether to plot points")
    marker_style: str = Field(default=".", description="The marker style to use")
    marker_size: int | float = Field(default=10.0, ge=1, description="The marker size to use")
    marker_alpha: int | float = Field(default=1.0, description="The marker alpha to use")
    zorder: int = Field(default=10, description="The zorder to use")
    shift_params: bool = Field(
        default=False,
        description="Whether to shift the parameters by subtracting each parameters mean",
    )

    @field_validator("color")
    @classmethod
    def _convert_color(cls, v: ColorInput | None) -> str | None:
        if v is None:
            return None
        return colors.format(v)

    def _apply_if_not_user_set(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key not in self._user_specified:
                setattr(self, key, value)

    def _apply(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class Chain(ChainConfig):
    """The numerical chain with its configuration."""

    samples: pd.DataFrame = Field(
        default=...,
        description="The chain data as a pandas DataFrame",
    )
    name: ChainName = Field(
        default=...,
        description="The name of the chain",
    )

    weight_column: ColumnName = Field(
        default="weight",
        description="The name of the weight column, if it exists",
    )
    posterior_column: ColumnName = Field(
        default="log_posterior",
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
    def data_columns(self) -> list[str]:
        """The columns in the dataframe which are not weights or posteriors."""
        results = []
        for c in self.samples.columns:
            if c in {self.weight_column, self.posterior_column}:
                continue
            if c.lower() in {
                "weight",
                "weights",
                "posterior",
                "posteriors",
                "log_weights",
                "log_posterior",
                "log_posteriors",
            }:
                continue
            results.append(c)
        return results

    @property
    def data_samples(self) -> pd.DataFrame:
        """The subsection of the dataframe with data points (ie excluding weights and posterior)"""
        return self.samples[self.data_columns]

    @property
    def plotting_columns(self) -> list[str]:
        """The columns to be plotted, which are the dataframe columns
        with the weights, posterior and colour coloumns removed."""
        cols = self.data_columns
        if not self.plot_cloud:
            return cols
        return [c for c in cols if c != self.color_param]

    @property
    def skip(self) -> bool:
        """If the chain will be skipped in plotting because it has nothing to plot."""
        return self.samples.empty or not (self.plot_contour or self.plot_cloud or self.plot_point)

    @property
    def max_posterior_row(self) -> pd.Series | None:
        """The row of samples which correspond to the maximum posterior value.
        None if the posterior is not supplied."""
        if self.posterior_column not in self.samples.columns:
            logging.warning("No posterior column found, cannot find max posterior")
            return None
        argmax = self.samples[self.posterior_column].argmax()
        return self.samples.loc[argmax]  # type: ignore

    @property
    def weights(self) -> np.ndarray:
        """The column of weights in the samples."""
        return self.samples[self.weight_column].to_numpy()

    @property
    def log_posterior(self) -> np.ndarray | None:
        """The column of log posteriors in the samples. None if not set."""
        if self.posterior_column not in self.samples.columns:
            return None
        return self.samples[self.posterior_column].to_numpy()

    @property
    def color_data(self) -> np.ndarray | None:
        """The data from the color column. None if not set."""
        if self.color_param is None:
            return None
        return self.samples[self.color_param].to_numpy()

    @field_validator("color")
    @classmethod
    def _validate_color(cls, v: str | np.ndarray | list[float] | None) -> str | None:
        if v is None:
            return None
        return colors.format(v)

    @field_validator("samples")
    @classmethod
    def _copy_df(cls, v: pd.DataFrame) -> pd.DataFrame:
        return v.copy()

    @model_validator(mode="after")
    def _validate_model(self) -> Chain:
        assert not self.samples.empty, "Your chain is empty. This is not ideal."

        # If weights aren't set, add them all as one
        if self.weight_column not in self.samples:
            assert (
                self.weight_column == "weight"
            ), f"weight column has been changed to {self.weight_column}, but its not in the dataframe"

            self.samples[self.weight_column] = 1.0
        else:
            assert np.all(self.weights > 0), "Weights must be positive and non-zero"

        for column in self.samples.columns:
            assert isinstance(column, str), f"Column {column} is not a string"
            assert np.all(np.isfinite(self.samples[column])), f"Column {column} has NaN or inf in it"

        # Apply the mean shift if it is set to true
        if self.shift_params:
            for param in self.samples:
                self.samples[param] -= np.average(self.samples[param], weights=self.weights)  # type: ignore

        # Check the walkers
        assert self.samples.shape[0] % self.walkers == 0, (
            f"Chain {self.name} has {self.samples.shape[0]} steps, "
            "which is not divisible by {self.walkers} walkers. This is not good."
        )

        # And if the color_params are set, ensure they're in the dataframe
        if self.color_param is not None:
            assert (
                self.color_param in self.samples.columns
            ), f"Chain {self.name} does not have color parameter {self.color_param}"

        # more nan checks
        if self.num_eff_data_points is not None:
            assert np.isfinite(self.num_eff_data_points), "num_eff_data_points is not finite"

        if self.num_free_params is not None:
            assert np.isfinite(self.num_free_params), "num_free_params is not finite"

        return self

    def get_data(self, column: str) -> pd.Series[float]:
        """Extracts a single columns from the samples dataframe."""
        return self.samples[column]

    @classmethod
    def from_covariance(
        cls,
        mean: np.ndarray | list[float],
        covariance: np.ndarray | list[list[float]],
        columns: list[ColumnName],
        name: ChainName,
        **kwargs: Any,
    ) -> Chain:
        """Generate samples as per mean and covariance supplied. Useful for Fisher matrix forecasts.

        Args:
            mean: The an array of mean values.
            covariance: The 2D array describing the covariance.
                Dimensions should agree with the `mean` input.
            columns: A list of parameter names, one for each column (dimension) in the mean array.
            name: The name of the chain.
            kwargs: Any other arguments to pass to the Chain constructor.

        Returns:
            The generated chain.
        """
        rng = np.random.default_rng()
        samples = rng.multivariate_normal(mean, covariance, size=1000000)  # type: ignore
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
            One chain per walker, split evenly
        """
        assert self.walkers > 1, "Cannot divide a chain with only one walker"
        assert not self.grid, "Cannot divide a grid chain"

        splits = np.split(self.samples, self.walkers)
        chains = []
        for i, split in enumerate(splits):
            df = pd.DataFrame(split, columns=self.samples.columns)
            options = self.model_dump(exclude={"samples", "name", "walkers"})
            if "color" in options:
                options.pop("color")
            chain = Chain(samples=df, name=f"{self.name} Walker {i}", **options)
            chains.append(chain)

        return chains

    def get_max_posterior_point(self) -> MaxPosterior | None:
        """Returns the maximum posterior point in the chain. If the posterior

        Returns:
            MaxPosterior: The maximum posterior point
        """
        if self.max_posterior_row is None:
            return None
        row = self.max_posterior_row.to_dict()
        log_posterior = row.pop(self.posterior_column)
        row = {k: v for k, v in row.items() if k in self.plotting_columns}
        return MaxPosterior(log_posterior=log_posterior, coordinate=row)

    def get_covariance(self, columns: list[str] | None = None) -> Named2DMatrix:
        """Returns the covariance matrix of the chain.

        Args:
            columns: The columns to use. None means all data columns.

        Returns:
            Named2DMatrix: The covariance matrix
        """
        if columns is None:
            columns = self.data_columns
        cov = np.cov(self.samples[columns], rowvar=False, aweights=self.weights)
        return Named2DMatrix(columns=columns, matrix=cov)

    def get_correlation(self, columns: list[str] | None = None) -> Named2DMatrix:
        """Returns the correlation matrix of the chain.

        Args:
            columns: The columns to use. None means all data columns.

        Returns:
            Named2DMatrix: The correlation matrix
        """
        cov = self.get_covariance(columns)
        diag = np.sqrt(np.diag(cov.matrix))
        divisor = diag[None, :] * diag[:, None]  # type: ignore
        correlations = cov.matrix / divisor
        return Named2DMatrix(columns=cov.columns, matrix=correlations)

    @classmethod
    def from_emcee(
        cls,
        sampler: emcee.EnsembleSampler,
        columns: list[str],
        name: str,
        thin: int = 1,
        discard: int = 0,
        **kwargs: Any,
    ) -> Chain:
        """Constructor from an emcee sampler

        Args:
            sampler: The emcee sampler
            columns: The names of the parameters
            name: The name of the chain
            thin: The thinning to apply to the chain
            discard: The number of steps to discard from the start of the chain
            kwargs: Any other arguments to pass to the Chain constructor.

        Returns:
            A ChainConsumer Chain made from the emcee samples
        """
        chain: np.ndarray = sampler.get_chain(flat=True, thin=thin, discard=discard)  # type: ignore
        df = pd.DataFrame.from_dict({col: val for col, val in zip(columns, chain.T)})

        return cls(samples=df, name=name, **kwargs)

    @classmethod
    def from_numpyro(
        cls,
        mcmc: numpyro.infer.MCMC,
        name: str,
        var_names: list[str] = [],
        **kwargs: Any,
    ) -> Chain:
        """Constructor from numpyro samples

        Args:
            mcmc: The numpyro sampler
            name: The name of the chain
            var_names: The names of the parameters to include in the chain. If the entries of var_names start with ~,
            they are excluded from the variables. If empty, all parameters are included.
            kwargs: Any other arguments to pass to the Chain constructor.

        Returns:
            A ChainConsumer Chain made from numpyro samples
        """

        var_names = _filter_var_names(var_names, list(mcmc.get_samples().keys()))
        df = pd.DataFrame.from_dict(
            {key: np.ravel(value) for key, value in mcmc.get_samples().items() if key in var_names}
        )
        return cls(samples=df, name=name, **kwargs)

    @classmethod
    def from_arviz(
        cls,
        arviz_id: arviz.InferenceData,
        name: str,
        var_names: list[str] = [],
        **kwargs: Any,
    ) -> Chain:
        """Constructor from an arviz InferenceData object

        Args:
            arviz_id: The arviz inference data
            name: The name of the chain
            var_names: The names of the parameters to include in the chain. If the entries of var_names start with ~,
            they are excluded from the variables. If empty, all parameters are included.
            kwargs: Any other arguments to pass to the Chain constructor.

        Returns:
            A ChainConsumer Chain made from the arviz chain
        """

        var_names = _filter_var_names(var_names, list(arviz_id.posterior.keys()))
        reduced_id = az.extract(arviz_id, var_names=var_names, group="posterior")
        df = reduced_id.to_dataframe().drop(columns=["chain", "draw"])

        return cls(samples=df, name=name, **kwargs)


class MaxPosterior(BetterBase):
    """A class that bundles the value of the
    log posterior with the coordinate you can find it at."""

    log_posterior: float
    coordinate: dict[ColumnName, float]

    @property
    def vec_coordinate(self) -> np.ndarray:
        """The coordinate as a numpy array, in the order the columns were given."""
        return np.array(list(self.coordinate.values()))


def _filter_var_names(var_names: list[str], all_vars: list[str]):
    """
    Helper function to return the var_names to allows filtering parameters names.
    """

    if not var_names:
        return all_vars

    elif var_names:
        if not (all([var.startswith("~") for var in var_names]) or all([not var.startswith("~") for var in var_names])):
            raise ValueError(
                "all values in var_names must start with ~ to exclude a subset OR none of them to keep a subset"
            )

        if all([var.startswith("~") for var in var_names]):
            # remove the ~ from the var names
            var_names = [var[1:] for var in var_names]
            var_names = [var for var in all_vars if var not in var_names]

            return var_names

        else:
            # keep var_names as is but check if var is in all_vars
            var_names = [var for var in all_vars if var in var_names]
            return var_names
