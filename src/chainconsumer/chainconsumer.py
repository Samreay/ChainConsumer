from typing import Any

import numpy as np
import pandas as pd

from .analysis import Analysis
from .chain import Chain, ChainConfig, ChainName, ColumnName
from .colors import ColorInput, colors
from .comparisons import Comparison
from .diagnostic import Diagnostic
from .helpers import get_bins
from .plotter import PlotConfig, Plotter
from .truth import Truth

__all__ = ["ChainConsumer"]


class ChainConsumer:
    """A class for consuming chains produced by an MCMC walk. Or grid searches. To make plots,
    figures, tables, diagnostics, you name it."""

    def __init__(self) -> None:
        self._chains: dict[ChainName, Chain] = {}
        self._truths: list[Truth] = []
        self._global_chain_override: ChainConfig | None = None

        self.plotter = Plotter(self)
        """Use this to access all the plotting functions"""
        self.diagnostic = Diagnostic(self)
        """Use this to access your diagnostics to see if chains have converged."""
        self.comparison = Comparison(self)
        """Use this to compare chains to each other, like ranking the AIC, BIC, and DIC."""
        self.analysis = Analysis(self)
        """Use this to access the analysis functions, like getting summary statistics from your chains."""

    @property
    def _all_columns(self) -> list[str]:
        """All the columns across all chains"""
        return list(set([c for chain in self._chains.values() for c in chain.samples.columns]))

    def add_truth(self, truth: Truth) -> "ChainConsumer":
        """Add a truth to ChainConsumer.

        Args:
            truth: The truth to add.

        Returns:
            Itself, to allow chaining calls.
        """
        self._truths.append(truth)
        return self

    def add_chain(self, chain: Chain) -> "ChainConsumer":
        """Add a chain to ChainConsumer.

        Args:
            chain: The chain to add.

        Returns:
            Itself, to allow chaining calls.
        """
        key = chain.name
        assert key not in self._chains, f"Chain with name {key} already exists!"
        self._chains[key] = chain
        return self

    def set_plot_config(self, plot_config: PlotConfig) -> "ChainConsumer":
        """Set the plot config for ChainConsumer.

        Args:
            plot_config: The plot config to use.

        Returns:
            Itself, to allow chaining calls.
        """
        self.plotter.set_config(plot_config)
        return self

    def add_marker(
        self,
        location: dict[ColumnName, float],
        name: str,
        color: ColorInput | None = None,
        marker_size: float = 20.0,
        marker_style: str = ".",
        marker_alpha: float = 1.0,
    ) -> "ChainConsumer":
        r"""Add a marker to the plot at the given location.

        Args:
            location: The location of the marker.
            name: The name of the marker.
            color: The colour of the marker. Defaults to None.
            marker_size: The size of the marker. Defaults to 20.0.
            marker_style: The style of the marker. Defaults to ".".
            marker_alpha: The alpha of the marker. Defaults to 1.0.


        Returns:
            Itself, to allow chaining calls.
        """

        samples = pd.DataFrame(location, index=[0])
        samples["weight"] = 1.0
        samples["log_posterior"] = 1.0
        kwargs = {}
        if color is not None:
            kwargs["color"] = color
        chain = Chain(
            samples=samples,
            name=name,
            marker_size=marker_size,
            marker_style=marker_style,
            marker_alpha=marker_alpha,
            plot_contour=False,
            plot_point=True,
            **kwargs,
        )
        return self.add_chain(chain)

    def remove_chain(self, remove: str | Chain) -> "ChainConsumer":
        r"""Removes a chain from ChainConsumer.

        Args:
            remove: The name of the chain to remove, or the chain itself.

        Returns:
            Itself, to allow chaining calls.
        """
        if isinstance(remove, Chain):
            remove = remove.name

        assert remove in self._chains, f"Chain with name {remove} does not exist!"
        self._chains.pop(remove)
        return self

    def set_override(
        self,
        override: ChainConfig,
    ) -> "ChainConsumer":
        """Apply a custom override config

        Args:
            override: The override config. Defaults to None.

        Returns:
            Itself, to allow chaining calls.
        """
        self._global_chain_override = override
        return self

    def _get_final_chains(self) -> dict[ChainName, Chain]:
        # Copy the original chain list
        final_chains = {k: v.model_copy() for k, v in self._chains.items()}
        num_chains = len(self._chains)

        # Note we only have to override things without a default
        # and things which should change as the number of chains change
        global_config: dict[str, Any] = {}
        global_config["bar_shade"] = num_chains < 5
        global_config["sigmas"] = [0, 1, 2]
        global_config["shade"] = num_chains < 5
        global_config["shade_alpha"] = 1.0 / np.sqrt(num_chains)

        for _, chain in final_chains.items():
            # copy global config into local config
            local_config = global_config.copy()

            if isinstance(chain.bins, float):
                chain.bins = int(chain.bins * get_bins(chain))

            # Reduce shade alpha if we're showing contour labels
            if chain.show_contour_labels:
                local_config["shade_alpha"] *= 0.5

            # Check to see if the color is set
            if chain.color is None:
                local_config["color"] = next(colors.next_colour())

            chain._apply_if_none(**local_config)

            # Apply user overrides
            if self._global_chain_override is not None:
                chain._apply(**self._global_chain_override.get_user_specified_dump())

        return final_chains

    def get_chain(self, name: str) -> Chain:
        """Get a chain by name.

        Args:
            name: The name of the chain.

        Returns:
            The chain.
        """
        assert name in self._chains, f"Chain with name {name} does not exist!"
        return self._chains[name]

    def get_names(self) -> list[str]:
        """Get the names of all chains.

        Returns:
        The names of all chains."""
        return list(self._chains.keys())
