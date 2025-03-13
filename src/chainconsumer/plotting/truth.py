from matplotlib.axes import Axes

from ..chain import ColumnName
from ..truth import Truth


def plot_truths(ax: Axes, truths: list[Truth], px: ColumnName | None = None, py: ColumnName | None = None) -> None:
    for truth in truths:
        # DEFAULT PLOTTING STYLE
        if truth.off_diagonal_marker is None:
            for truth in truths:
                if px is not None:
                    val_x = truth.location.get(px)
                    if val_x is not None:
                        ax.axvline(val_x, **truth._kwargs)
                if py is not None:
                    val_y = truth.location.get(py)
                    if val_y is not None:
                        ax.axhline(val_y, **truth._kwargs)
        # ALTERNATE PLOTTING STYLE
        if truth.off_diagonal_marker:
            if px is not None:
                # Plot vertical line ONLY along diagonal
                if py is None:
                    val_x = truth.location.get(px)
                    if val_x is not None:
                        ax.axvline(val_x, **truth._kwargs)
            # Plot specified markers on the off-diagonals
            if py is not None:
                val_x = truth.location.get(px)
                val_y = truth.location.get(py)
                ax.scatter(val_x, val_y, marker=truth.off_diagonal_marker, s=truth.off_diagonal_marker_size, c=truth._kwargs['c'], alpha=truth._kwargs['alpha'], zorder=truth._kwargs['zorder'])
