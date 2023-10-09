from typing import Any

from pydantic import Field

from ..base import BetterBase
from ..chain import ColumnName


class PlotConfig(BetterBase):
    labels: dict[ColumnName, str] = Field(default={}, description="Labels for parameters")
    max_ticks: int = Field(default=5, ge=0, description="Maximum number of ticks to use on axes")
    plot_hists: bool = Field(default=True, description="Whether to plot the 1D histograms")
    flip: bool = Field(default=False, description="Whether to flip the 1D histograms")
    serif: bool = Field(default=False, description="Whether to use a serif font")
    usetex: bool = Field(default=False, description="Whether to use LaTeX for text rendering")
    diagonal_tick_labels: bool = Field(default=True, description="Whether to show tick labels on the diagonal")
    label_font_size: int = Field(default=12, ge=0, description="Font size for axis labels")
    tick_font_size: int = Field(default=10, ge=0, description="Font size for axis ticks")
    spacing: float | None = Field(default=None, ge=0, description="Spacing between subplots")
    contour_label_font_size: int = Field(default=10, ge=0, description="Font size for contour labels")
    show_legend: bool | None = Field(
        default=None,
        description="Whether to show the legend. None means determine automatically",
    )
    legend_kwargs: dict[str, Any] = Field(default={}, description="Kwargs to pass to the legend")
    legend_location: tuple[int, int] | None = Field(default=None, description="Which subplot to put the legend in")
    legend_artists: bool | None = Field(default=None, description="Whether to show artists in the legend")
    legend_color_text: bool = Field(default=True, description="Whether to color the legend text")
    watermark: str | None = Field(default=None, description="Watermark text to add to the plot")
    watermark_text_kwargs: dict[str, Any] = Field(default={}, description="Kwargs to pass to the watermark text")
    summarise: bool = Field(default=True, description="Whether to annotate the plot with summary statistics")
    summary_font_size: int = Field(default=12, ge=0, description="Font size for parameter summaries")
    sigma2d: bool | None = Field(
        default=None,
        description=(
            "Whether to use 2D sigmas for summary statistics. Ie in 2D a 1sigma contour"
            r" does *not* encapsulate 68% of the volume, it covers 39.3% of the volume."
        ),
    )
    blind: bool | list[str] = Field(default=False, description="Whether to blind some parameters")
    log_scales: list[ColumnName] = Field(default=[], description="Whether to use log scales for some parameters")
    extents: dict[ColumnName, tuple[float, float]] = Field(
        default={}, description="Extents for parameters. Any you don't specify are determined automatically"
    )
    dpi: int = Field(default=300, ge=0, description="DPI for the figure")

    @property
    def legend_kwargs_final(self) -> dict[str, Any]:
        default = {
            "labelspacing": 0.3,
            "loc": "upper right",
            "frameon": False,
            "fontsize": self.label_font_size,
            "handlelength": 1,
            "handletextpad": 0.2,
            "borderaxespad": 0.0,
        }
        return default | self.legend_kwargs

    @property
    def watermark_text_kwargs_final(self) -> dict[str, Any]:
        default = {
            "color": "#333333",
            "alpha": 0.7,
            "verticalalignment": "center",
            "horizontalalignment": "center",
            "weight": "bold",
        }
        return default | self.watermark_text_kwargs

    def get_label(self, column: ColumnName) -> str:
        return self.labels.get(column, column)
