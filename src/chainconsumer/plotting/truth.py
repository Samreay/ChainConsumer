from matplotlib.axes import Axes

from ..chain import ColumnName
from ..truth import Truth


def plot_truths(ax: Axes, truths: list[Truth], px: ColumnName | None = None, py: ColumnName | None = None) -> None:
    for truth in truths:
        if px is not None:
            val_x = truth.location.get(px)
            if val_x is not None:
                ax.axvline(val_x, **truth._kwargs)
        if py is not None:
            val_y = truth.location.get(py)
            if val_y is not None:
                ax.axhline(val_y, **truth._kwargs)
