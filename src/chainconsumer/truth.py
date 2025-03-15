from typing import Any

import pandas as pd
from matplotlib.lines import Line2D
from pydantic import Field, ValidationError, field_validator, model_validator

from .base import BetterBase
from .color_finder import ColorInput

_DEFAULT_EDGE_WIDTH = 1.0


class Truth(BetterBase):
    location: dict[str, float] = Field(
        default=...,
        description="The truth value, either as dictionary or pandas series which will be converted to a dict)",
    )
    name: str | None = Field(default=None, description="The name of the truth line")
    color: ColorInput = Field(default="black", description="The color of the truth line")
    line_width: float = Field(default=1.0, description="The width of the truth line")
    line_style: str = Field(default="--", description="The style of the truth line")
    alpha: float = Field(default=1.0, description="The alpha of the truth line")
    zorder: int = Field(default=100, description="The zorder of the truth line")
    marker: str | None = Field(default=None, description="The truth marker style for the contour plots")
    marker_size: float = Field(default=150.0, description="The truth marker size for the contour plots")
    marker_edge_width: float = Field(
        default=_DEFAULT_EDGE_WIDTH, description="The truth marker edge width for the contour plots"
    )

    @field_validator("location")
    @classmethod
    def _ensure_dict(cls, v):
        if isinstance(v, dict):
            return v
        elif isinstance(v, pd.Series):
            return v.to_dict()
        raise ValidationError("Truth must be a dict or a pandas Series")

    @model_validator(mode="after")
    def validate_model(self):
        if self.marker is not None and self.is_filled_marker and self.marker_edge_width > _DEFAULT_EDGE_WIDTH:
            msg = (
                f"It seems you are trying to make the marker {self.marker} thicker. "
                "Alas, this is not possible. Matplotlib only lets you make the marker "
                "edge thicker if the marker is filled. Which, FYI, means picking from "
                "one of the following markers: "
                f"{', '.join(Line2D.filled_markers)}"
            )
            raise ValueError(msg)
        return self

    @property
    def is_filled_marker(self) -> bool:
        return self.marker in Line2D.filled_markers

    @property
    def _kwargs(self) -> dict[str, Any]:
        return {
            "ls": self.line_style,
            "c": self.color,
            "lw": self.line_width,
            "alpha": self.alpha,
            "zorder": self.zorder,
        }

    @property
    def _marker_kwargs(self) -> dict[str, Any]:
        result = {
            "marker": self.marker,
            "s": self.marker_size,
            "color": self.color,
            "alpha": self.alpha,
            "zorder": self.zorder,
        }
        if self.is_filled_marker:
            result["edgecolor"] = self.color
            result["facecolor"] = self.color
            result["linewidth"] = self.marker_edge_width
        return result
