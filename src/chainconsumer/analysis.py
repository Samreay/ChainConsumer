from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from pydantic import Field
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import root_scalar

from .base import BetterBase
from .chain import Chain, ChainName, ColumnName, MaxPosterior, Named2DMatrix
from .helpers import get_bins, get_grid_bins, get_latex_table_frame, get_smoothed_bins
from .kde import MegKDE
from .statistics import SummaryStatistic


def _mask_to_intervals(
    x: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Turn a mask indexed on x to a list of intervals
    """
    if mask.size == 0:
        return []

    change = np.diff(mask.astype(int))
    starts = np.where(change == 1)[0] + 1  # False -> True
    ends = np.where(change == -1)[0]  # True  -> False

    # If we start inside an interval, prepend 0
    if mask[0]:
        starts = np.concatenate(([0], starts))

    # If we end inside an interval, append last index
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask) - 1]))

    intervals = [(float(x[s]), float(x[e])) for s, e in zip(starts, ends, strict=True) if x[e] > x[s]]

    return intervals


class Bound(BetterBase):
    lower: float | None = Field(default=None)
    center: float | None = Field(default=None)
    upper: float | None = Field(default=None)

    @property
    def array(self) -> np.ndarray:
        return np.array(
            [
                self.lower if self.lower is not None else np.nan,
                self.center if self.center is not None else np.nan,
                self.upper if self.upper is not None else np.nan,
            ]
        )

    @property
    def all_none(self) -> bool:
        return self.lower is None and self.center is None and self.upper is None

    @classmethod
    def from_array(cls, array: np.ndarray | list[float]) -> Bound:
        assert len(array) == 3, "Array must have 3 elements"
        lower, center, upper = array
        return cls(lower=lower, center=center, upper=upper)


class Analysis:
    def __init__(self, parent: ChainConsumer):
        self.parent = parent
        self._logger = logging.getLogger("chainconsumer")

        self._summaries: dict[SummaryStatistic, Callable[[Chain, ColumnName], Bound | None]] = {
            SummaryStatistic.MAX: self.get_parameter_summary_max,
            SummaryStatistic.MEAN: self.get_parameter_summary_mean,
            SummaryStatistic.CUMULATIVE: self.get_parameter_summary_cumulative,
            SummaryStatistic.MAX_CENTRAL: self.get_parameter_summary_max_central,
            SummaryStatistic.HDI: self.get_parameter_summary_hdi,
        }

    def get_latex_table(
        self,
        chains: list[ChainName | Chain] | None = None,
        columns: list[ColumnName] | None = None,
        transpose: bool = False,
        caption: str | None = None,
        label: str = "tab:model_params",
        hlines: bool = True,
        blank_fill: str = "--",
        filename: str | Path | None = None,
    ) -> str:  # pragma: no cover
        """Generates a LaTeX table from parameter summaries.

        Args:
            chains:
                Used to specify which chain to show if more than one chain is loaded in.
                Can be an integer, specifying the
                chain index, or a str, specifying the chain name.
            columns:
                If set, only creates a plot for those specific parameters (if list). If an
                integer is given, only plots the fist so many parameters.
            transpose : bool, optional
                Defaults to False, which gives each column as a parameter, each chain (framework)
                as a row. You can swap it so that you have a parameter each row and a framework
                each column by setting this to True
            caption : str, optional
                If you want to generate a caption for the table through Python, use this.
                Defaults to an empty string
            label : str, optional
                If you want to generate a label for the table through Python, use this.
                Defaults to an empty string
            hlines : bool, optional
                Inserts ``\\hline`` before and after the header, and at the end of table.
            blank_fill : str, optional
                If a framework does not have a particular parameter, will fill that cell of
                the table with this string.
            filename : str | Path, optional
                The file to save the output string to

        Returns:
            str: the LaTeX table.
        """
        final_chains = self.parent.plotter._sanitise_chains(chains)
        final_columns = self.parent.plotter._sanitise_columns(columns, final_chains)
        blind = self.parent.plotter._sanitise_blinds(self.parent.plotter.config.blind, final_columns)

        final_columns = [c for c in final_columns if c not in blind]
        num_chains = len(final_chains)
        num_parameters = len(final_columns)
        fit_values = self.get_summary(chains=final_chains)
        if label is None:
            label = ""
        if caption is None:
            caption = ""

        end_text = " \\\\ \n"
        column_text = "c" * (num_chains + 1) if transpose else "c" * (num_parameters + 1)

        center_text = ""
        hline_text = "\\hline\n"
        if hlines:
            center_text += hline_text + "\t\t"
        if transpose:
            center_text += " & ".join(["Parameter"] + [c.name for c in final_chains]) + end_text
            if hlines:
                center_text += "\t\t" + hline_text
            for p in final_columns:
                arr = ["\t\t" + self.parent.plotter.config.get_label(p)]
                for _, column_results in fit_values.items():
                    if p in column_results:
                        arr.append(self.get_parameter_text(column_results[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        else:
            center_text += (
                " & ".join(["Model", *[self.parent.plotter.config.get_label(c) for c in final_columns]]) + end_text
            )
            if hlines:
                center_text += "\t\t" + hline_text
            for name, chain_res in fit_values.items():
                arr = ["\t\t" + name]
                for p in final_columns:
                    if p in chain_res:
                        arr.append(self.get_parameter_text(chain_res[p], wrap=True))
                    else:
                        arr.append(blank_fill)
                center_text += " & ".join(arr) + end_text
        if hlines:
            center_text += "\t\t" + hline_text
        final_text = get_latex_table_frame(caption, label) % (column_text, center_text)

        if filename is not None:
            if isinstance(filename, str):
                filename = Path(filename)
            with Path.open(filename, "w") as f:
                f.write(final_text)

        return final_text

    def get_summary(
        self,
        chains: list[Chain] | None = None,
        columns: list[ColumnName] | None = None,
    ) -> dict[ChainName, dict[ColumnName, Bound]]:
        """Gets a summary of the marginalised parameter distributions.

        Args:
            columns (list[str], optional): A list of parameters which to generate summaries for.
            chains (dict[str, Chain] | list[str], optional): A list of chains to generate summaries for.

        Returns:
            dict[ChainName, dict[ColumnName, Bound]]: A map from chain name to column name to bound.
        """
        results = {}
        if chains is None:
            chains = self.parent.plotter._sanitise_chains(None, include_skip=True)
        if columns is None:
            columns = self.parent.plotter._sanitise_columns(None, chains)

        for chain in chains:
            res = {}
            params_to_find = columns if columns is not None else chain.data_columns
            for p in params_to_find:
                if p not in chain.samples:
                    continue
                summary = self.get_parameter_summary(chain, p)
                res[p] = summary

                if chain.multimodal:
                    intervals = self.get_parameter_hdi_intervals(chain, p)

                    # If there is a single interval, we skip
                    if len(intervals) < 2:
                        continue

                    multimodal_bounds = self.get_parameter_multimodal_bounds(
                        chain,
                        p,
                        intervals=intervals,
                    )

                    if multimodal_bounds is not None:
                        res[p] = multimodal_bounds
                        continue

                res[p] = summary
            results[chain.name] = res

        return results

    def get_max_posteriors(self, chains: dict[str, Chain] | list[str] | None = None) -> dict[ChainName, MaxPosterior]:
        """Gets the maximum posterior point in parameter space from the passed parameters.

        Requires the chains to have set `posterior` values.

        Args:
            chains (dict[str, Chain] | list[str], optional): A list of chains to generate summaries for.

        Returns:
            dict[ChainName, MaxPosterior]: A map from chain name to max posterior point.
        """

        results = {}
        if chains is None:
            chains = self.parent._chains
        if isinstance(chains, list):
            chains = {c: self.parent._chains[c] for c in chains}

        for chain_name, chain in chains.items():
            max_posterior = chain.get_max_posterior_point()
            if max_posterior is None:
                continue
            results[chain_name] = max_posterior

        return results

    def get_parameter_summary(self, chain: Chain, column: ColumnName) -> Bound | None:
        callback = self._summaries[chain.statistics]
        return callback(chain, column)

    def get_correlation_table(
        self,
        chain: str | Chain,
        columns: list[str] | None = None,
        caption: str = "Parameter Correlations",
        label: str = "tab:parameter_correlations",
    ) -> str:
        """
        Gets a LaTeX table of parameter correlations.

        Args:
        chain (str|Chain, optional_: The chain index or name. Defaults to first chain.
        columns (list[str], optional): The list of parameters to compute correlations. Defaults to all columns
        caption (str, optional): The LaTeX table caption.
        label (str, optional): The LaTeX table label.

        Returns:
            str: The LaTeX table ready to go!
        """
        if isinstance(chain, str):
            assert chain in self.parent._chains, f"Chain {chain} not found!"
            chain = self.parent._chains[chain]
        if chain is None:
            assert len(self.parent._chains) == 1, "You must specify a chain if there are multiple chains"
            chain = next(iter(self.parent._chains.values()))

        correlations = chain.get_correlation(columns=columns)
        return self._get_2d_latex_table(correlations, caption, label)

    def get_covariance_table(
        self,
        chain: str | Chain,
        columns: list[str] | None = None,
        caption: str = "Parameter Covariance",
        label: str = "tab:parameter_covariance",
    ) -> str:
        """
        Gets a LaTeX table of parameter covariances.

        Args:
        chain (str|Chain, optional_: The chain index or name. Defaults to first chain.
        columns (list[str], optional): The list of parameters to compute covariances on. Defaults to all columns
        caption (str, optional): The LaTeX table caption.
        label (str, optional): The LaTeX table label.

        Returns:
            str: The LaTeX table ready to go!
        """
        if isinstance(chain, str):
            assert chain in self.parent._chains, f"Chain {chain} not found!"
            chain = self.parent._chains[chain]
        if chain is None:
            assert len(self.parent._chains) == 1, "You must specify a chain if there are multiple chains"
            chain = next(iter(self.parent._chains.values()))

        covariance = chain.get_covariance(columns=columns)
        return self._get_2d_latex_table(covariance, caption, label)

    def _get_smoothed_histogram(
        self,
        chain: Chain,
        column: ColumnName,
        pad: bool = False,
        *,
        use_kde: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = chain.get_data(column)
        if chain.grid:
            bins = get_grid_bins(data)
        else:
            bins, _ = get_smoothed_bins(chain.smooth_value, get_bins(chain), data, chain.weights, pad=pad)

        hist, edges = np.histogram(data, bins=bins, density=True, weights=chain.weights)
        if chain.power is not None:
            hist = hist**chain.power
        edge_centers = 0.5 * (edges[1:] + edges[:-1])
        xs = np.linspace(edge_centers[0], edge_centers[-1], 10000)

        if chain.smooth_value:
            hist = gaussian_filter(hist, chain.smooth_value, mode="reflect")
        if use_kde is None:
            use_kde = bool(chain.kde)

        if use_kde:
            kde_xs = np.linspace(edge_centers[0], edge_centers[-1], max(200, int(bins.max())))
            factor = chain.kde if isinstance(chain.kde, int | float) else 1.0
            ys = MegKDE(data.to_numpy(), chain.weights, factor=factor).evaluate(kde_xs)
            area = simps(ys, x=kde_xs)
            ys = ys / area
            ys = interp1d(kde_xs, ys, kind="linear")(xs)
        else:
            ys = interp1d(edge_centers, hist, kind="linear")(xs)
        cs = ys.cumsum()
        cs /= cs.max()
        return xs, ys, cs

    def _get_2d_latex_table(self, named_matrix: Named2DMatrix, caption: str, label: str) -> str:
        parameters = [self.parent.plotter.config.get_label(c) for c in named_matrix.columns]
        matrix = named_matrix.matrix
        latex_table = get_latex_table_frame(caption=caption, label=label)
        column_def = "c|%s" % ("c" * len(parameters))
        hline_text = "        \\hline\n"

        table = ""
        table += " & ".join(["", *parameters]) + "\\\\ \n"
        table += hline_text
        max_len = max([len(s) for s in parameters])
        format_string = "        %%%ds" % max_len
        for p, row in zip(parameters, matrix, strict=False):
            table += format_string % p
            for r in row:
                table += f" & {r:5.2f}"
            table += " \\\\ \n"
        table += hline_text
        return latex_table % (column_def, table)

    def get_parameter_text(
        self,
        bound: Bound | Sequence[Bound],
        wrap: bool = False,
        *,
        label: str | None = None,
    ) -> str:
        """Format marginal parameter bounds for display.

        Args:
            bound:
                The bound (or list of bounds) to format.
            wrap:
                Wrap each formatted expression in LaTeX dollar signs.
            label:
                Optional parameter label to prepend. For multimodal results the
                label is placed on its own line.

        Returns:
            The formatted string. Returns an empty string when the input contains
            no finite limits.
        """

        if bound is None:
            return ""

        # Fallback to single bound behavior is there is only one mode identified
        if isinstance(bound, Sequence) and len(bound) < 2:
            bound = bound[0]

        if isinstance(bound, Sequence) and not isinstance(bound, Bound):
            bounds = [b for b in bound if isinstance(b, Bound) and not b.all_none]
            if not bounds:
                return ""

            lines: list[str] = []
            if label:
                lines.append(f"${label}$" if wrap else label)

            for index, sub_bound in enumerate(bounds, start=1):
                entry = Analysis._format_single_bound(sub_bound, use_pm=False)
                if not entry:
                    continue
                if wrap:
                    entry = f"${entry}$"
                lines.append(f"I{index}: {entry}")

            return "\n".join(lines)

        if bound.lower is None or bound.upper is None or bound.center is None:
            return ""

        text = self._format_single_bound(bound, use_pm=True)

        if label:
            text = f"{label} = {text}"

        if wrap:
            return f"${text}$"
        return text

    @staticmethod
    def _format_single_bound(bound: Bound, *, use_pm: bool) -> str:
        upper_error = bound.upper - bound.center
        lower_error = bound.center - bound.lower
        if upper_error != 0 and lower_error != 0:
            resolution = min(np.floor(np.log10(np.abs(upper_error))), np.floor(np.log10(np.abs(lower_error))))
        elif upper_error == 0 and lower_error != 0:
            resolution = np.floor(np.log10(np.abs(lower_error)))
        elif upper_error != 0 and lower_error == 0:
            resolution = np.floor(np.log10(np.abs(upper_error)))
        else:
            resolution = np.floor(np.log10(np.abs(bound.center)))
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
        upper_error *= 10**factor
        lower_error *= 10**factor
        maximum = bound.center * 10**factor
        upper_error = round(upper_error, r)
        lower_error = round(lower_error, r)
        maximum = round(maximum, r)
        if maximum == -0.0:
            maximum = 0.0
        if resolution == 2:
            upper_error *= 10**-factor
            lower_error *= 10**-factor
            maximum *= 10**-factor
            factor = 0
            fmt = "%0.0f"
        upper_error_text = fmt % upper_error
        lower_error_text = fmt % lower_error
        if use_pm and upper_error_text == lower_error_text:
            text = r"{}\pm {}".format(fmt, "%s") % (maximum, lower_error_text)
        else:
            text = r"{}^{{+{}}}_{{-{}}}".format(fmt, "%s", "%s") % (maximum, upper_error_text, lower_error_text)
        if factor != 0:
            text = r"\left( %s \right) \times 10^{%d}" % (text, -factor)
        return text

    def get_parameter_summary_mean(self, chain: Chain, column: ColumnName) -> Bound | None:
        xs, _, cs = self._get_smoothed_histogram(chain, column)
        vals = [0.5 - chain.summary_area / 2, 0.5, 0.5 + chain.summary_area / 2]
        bounds = interp1d(cs, xs)(vals)
        bounds[1] = 0.5 * (bounds[0] + bounds[2])
        return Bound(lower=bounds[0], center=bounds[1], upper=bounds[2])

    def get_parameter_summary_cumulative(self, chain: Chain, column: ColumnName) -> Bound | None:
        xs, _, cs = self._get_smoothed_histogram(chain, column)
        vals = [0.5 - chain.summary_area / 2, 0.5, 0.5 + chain.summary_area / 2]
        bounds = interp1d(cs, xs)(vals)
        return Bound(lower=bounds[0], center=bounds[1], upper=bounds[2])

    def get_parameter_summary_hdi(self, chain: Chain, column: ColumnName) -> Bound:
        data = chain.get_data(column).to_numpy()
        n_samples = data.size

        if n_samples <= 512:  # Arbitrary low sample warning
            warnings.warn(
                (
                    f"Only {n_samples} samples available to compute an HDI for column '{column}' "
                    f"in chain '{chain.name}'. Results may be unreliable; consider enabling KDE or "
                    "providing more samples."
                ),
                UserWarning,
                stacklevel=2,
            )

        xs, _, cs = self._get_smoothed_histogram(chain, column, pad=True)

        cdf_points = np.concatenate(([0.0], cs))
        x_points = np.concatenate(([xs[0]], xs))

        eps = 1e-12
        best_width = float("inf")
        best_lower = float(x_points[0])
        best_upper = float(x_points[-1])
        best_start_mass = 0.0
        best_end_mass = 1.0

        for start_idx, start_mass in enumerate(cdf_points[:-1]):
            required = start_mass + chain.summary_area
            if required > 1.0 + eps:
                break

            # Smallest index with cdf_points[end_idx] >= required
            end_idx = np.searchsorted(cdf_points, required, side="left")

            # Ensure at least one point is in the interval
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            if end_idx >= cdf_points.size:
                break

            # If still slightly under target, move one step right if possible
            if cdf_points[end_idx] - start_mass < chain.summary_area - eps and end_idx + 1 < cdf_points.size:
                end_idx += 1

            lower = float(x_points[start_idx])
            upper = float(x_points[end_idx])
            width = upper - lower
            if width <= eps:
                continue

            if width < best_width - eps:
                best_width = width
                best_lower = lower
                best_upper = upper
                best_start_mass = float(start_mass)
                best_end_mass = float(cdf_points[end_idx])

        interval_mass = best_end_mass - best_start_mass

        if interval_mass <= eps:
            center = 0.5 * (best_lower + best_upper)

        else:
            center_mass = best_start_mass + 0.5 * interval_mass
            center = float(np.interp(center_mass, cdf_points, x_points, left=best_lower, right=best_upper))

        return Bound(lower=best_lower, center=center, upper=best_upper)

    def get_parameter_hdi_intervals(self, chain: Chain, column: ColumnName) -> list[tuple[float, float]]:
        """Return highest-density intervals for a marginal distribution.

        Multimodal chains yield one interval per disjoint density band, whereas unimodal chains
        return a single contiguous interval.
        """
        summary = self.get_parameter_summary_hdi(chain, column)
        default_interval = [(summary.lower, summary.upper)]
        xs, ys, _ = self._get_smoothed_histogram(chain, column, pad=True, use_kde=False)

        # We look for the threshold that is the root of this function
        def mass_diff(threshold, density, xs, target):
            mask = density >= threshold
            mass_above_threshold = float(simps(np.where(mask, density, 0.0), x=xs))
            return mass_above_threshold - target

        area = simps(ys, x=xs)
        density = ys / area

        sol = root_scalar(
            mass_diff,
            bracket=(0.0, float(np.max(density))),
            args=(density, xs, chain.summary_area),
            method="bisect",
            xtol=5e-4,
        )

        threshold = sol.root
        mask = density >= threshold

        intervals = _mask_to_intervals(xs, mask)

        return intervals if intervals else default_interval

    def get_parameter_multimodal_bounds(
        self,
        chain: Chain,
        column: ColumnName,
        intervals: list[tuple[float, float]],
    ) -> list[Bound]:
        """
        Convert multimodal HDI bands into `Bound` instances.
        """

        xs, ys, _ = self._get_smoothed_histogram(
            chain,
            column,
            pad=True,
            use_kde=False if chain.multimodal else None,
        )

        lower_limit, upper_limit = float(xs.min()), float(xs.max())

        bounds = []

        for lower_raw, upper_raw in intervals:
            lower, upper = max(lower_raw, lower_limit), min(upper_raw, upper_limit)
            mask = (xs >= lower) & (xs <= upper)

            if np.any(mask):
                idx = int(np.argmax(ys[mask]))
                center = float(xs[mask][idx])

            else:
                center = float(0.5 * (lower + upper))

            bounds.append(Bound(lower=float(lower), center=center, upper=float(upper)))

        return bounds

    def get_parameter_summary_max(self, chain: Chain, column: ColumnName) -> Bound | None:
        xs, ys, cs = self._get_smoothed_histogram(chain, column)
        n_pad = 1000
        x_start = xs[0] * np.ones(n_pad)
        x_end = xs[-1] * np.ones(n_pad)
        y_start = np.linspace(0, ys[0], n_pad)
        y_end = np.linspace(ys[-1], 0, n_pad)
        xs = np.concatenate((x_start, xs, x_end))
        ys = np.concatenate((y_start, ys, y_end))
        cs = ys.cumsum()
        cs = cs / cs.max()
        start_index = ys.argmax()
        max_val = ys[start_index]
        min_val = 0
        threshold = 0.003
        x1 = None
        x2 = None
        count = 0
        while x1 is None:
            mid = (max_val + min_val) / 2.0
            count += 1
            try:
                if count > 50:
                    raise ValueError("Failed to converge")  # noqa: TRY301
                i1 = start_index - np.where(ys[:start_index][::-1] < mid)[0][0]
                i2 = start_index + np.where(ys[start_index:] < mid)[0][0]
                area = cs[i2] - cs[i1]
                deviation = np.abs(area - chain.summary_area)
                if deviation < threshold:
                    x1 = float(xs[i1])
                    x2 = float(xs[i2])
                elif area < chain.summary_area:
                    max_val = mid
                elif area > chain.summary_area:
                    min_val = mid
            except ValueError:
                self._logger.warning(f"Parameter {column} in chain {chain.name} is not constrained")
                return Bound(lower=None, center=float(xs[start_index]), upper=None)

        return Bound(lower=x1, center=float(xs[start_index]), upper=x2)

    def get_parameter_summary_max_central(self, chain, parameter):
        xs, ys, cs = self._get_smoothed_histogram(chain, parameter)

        c_to_x = interp1d(cs, xs)
        max_index = ys.argmax()
        x = xs[max_index]

        vals = [0.5 - 0.5 * chain.summary_area, 0.5 + 0.5 * chain.summary_area]
        xvals = c_to_x(vals)

        return Bound(lower=xvals[0], center=x, upper=xvals[1])


if __name__ == "__main__":
    from .chainconsumer import ChainConsumer
