from matplotlib.axes import Axes

from ..chain import ColumnName
from ..truth import Truth


def plot_truths(ax: Axes, truths: list[Truth], px: ColumnName | None = None, py: ColumnName | None = None) -> None:
    for truth in truths:
        val_x: float | None = truth.location.get(px)  # type: ignore
        val_y: float | None = truth.location.get(py)  # type: ignore
        plot_marker = val_x is not None and val_y is not None and truth.marker is not None
        # If there's no truth value, skip
        if val_x is None and val_y is None:
            continue

        # If there are both, plot the line
        if plot_marker:
            ax.scatter(val_x, val_y, **truth._marker_kwargs)  # type: ignore
        else:
            if val_x is not None:
                ax.axvline(val_x, **truth._kwargs)
            if val_y is not None:
                ax.axhline(val_y, **truth._kwargs)
