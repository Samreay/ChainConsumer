import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from .config import PlotConfig


def add_watermark(
    fig: Figure,
    axes: Axes | None,
    fig_size: tuple[float, float],
    config: PlotConfig,
    size_scale: float = 1.0,
) -> None:  # pragma: no cover
    """Add a watermark to a figure or axis."""
    # Code based off github repository https://github.com/cpadavis/preliminize
    if config.watermark is None:
        return
    dx, dy = fig_size
    dy, dx = dy * config.dpi, dx * config.dpi
    rotation = 180 / np.pi * np.arctan2(-dy, dx)
    property_dict = config.watermark_text_kwargs_final

    keys_in_font_dict = ["family", "style", "variant", "weight", "stretch", "size"]
    fontdict = {k: property_dict[k] for k in keys_in_font_dict if k in property_dict}
    font_prop = FontProperties(**fontdict)
    usetex = property_dict.get("usetex", config.usetex)
    if usetex:
        px, py, scale = 0.5, 0.5, 1.0
    else:
        px, py, scale = 0.5, 0.5, 0.8

    bb0 = TextPath((0, 0), config.watermark, size=50, prop=font_prop, usetex=usetex).get_extents()
    bb1 = TextPath((0, 0), config.watermark, size=51, prop=font_prop, usetex=usetex).get_extents()
    dw = (bb1.width - bb0.width) * (config.dpi / 100)
    dh = (bb1.height - bb0.height) * (config.dpi / 100)
    size = np.sqrt(dy**2 + dx**2) / (dh * abs(dy / dx) + dw) * 0.7 * scale * size_scale
    if axes is not None:
        if usetex:
            size *= 0.7
        else:
            size *= 0.8
    size = int(size)
    if axes is None:
        fig.text(px, py, config.watermark, fontdict=property_dict, rotation=rotation, fontsize=size)
    else:
        axes.text(
            px, py, config.watermark, transform=axes.transAxes, fontdict=property_dict, rotation=rotation, fontsize=size
        )
