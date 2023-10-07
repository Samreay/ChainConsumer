from typing import Any

import pandas as pd
from pydantic import Field, ValidationError, field_validator

from .base import BetterBase


class Truth(BetterBase):
    location: dict[str, float] = Field(
        default=..., description="The truth value, either as dictionary or pandas series"
    )
    truth_name: str | None = Field(default=None, description="The name of the truth line")
    line_color: str = Field(default="black", description="The color of the truth line")
    line_width: float = Field(default=1.0, description="The width of the truth line")
    line_alpha: float = Field(default=1.0, description="The alpha of the truth line")
    line_style: str = Field(default="--", description="The style of the truth line")
    line_zorder: int = Field(default=100, description="The zorder of the truth line")
    name: str | None = Field(default=None, description="The name of the truth line")

    @field_validator("location")
    @classmethod
    def ensure_dict(cls, v):
        if isinstance(v, dict):
            return v
        elif isinstance(v, pd.Series):
            return v.to_dict()
        raise ValidationError("Truth must be a dict or a pandas Series")

    @property
    def kwargs(self) -> dict[str, Any]:
        return {
            "ls": self.line_style,
            "c": self.line_color,
            "lw": self.line_width,
            "alpha": self.line_alpha,
            "zorder": self.line_zorder,
        }
