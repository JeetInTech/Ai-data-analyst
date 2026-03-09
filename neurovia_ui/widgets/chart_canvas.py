"""
NeuroviaI Chart Canvas Widget
Matplotlib figure embedded in Qt with dark NeuroviaI styling.
"""

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np

from neurovia_ui.theme import Colors


# Configure matplotlib for dark theme globally
plt.rcParams.update({
    "figure.facecolor": Colors.BG_DARK,
    "axes.facecolor": Colors.BG_BASE,
    "axes.edgecolor": Colors.BORDER,
    "axes.labelcolor": Colors.TEXT_SECONDARY,
    "text.color": Colors.TEXT_PRIMARY,
    "xtick.color": Colors.TEXT_MUTED,
    "ytick.color": Colors.TEXT_MUTED,
    "grid.color": Colors.BORDER,
    "grid.alpha": 0.5,
    "legend.facecolor": Colors.BG_CARD,
    "legend.edgecolor": Colors.BORDER,
    "legend.labelcolor": Colors.TEXT_PRIMARY,
    "font.family": "Segoe UI",
    "font.size": 10,
})


# Neurovia color cycle for charts
NEUROVIA_COLORS = [
    "#10B981",  # Emerald
    "#3B82F6",  # Blue
    "#06B6D4",  # Cyan
    "#8B5CF6",  # Purple
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#F97316",  # Orange
    "#6366F1",  # Indigo
]


class ChartCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas styled for Neurovia dark theme."""

    def __init__(self, parent=None, width: float = 8, height: float = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=Colors.BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)

    def _style_axes(self, ax):
        ax.set_facecolor(Colors.BG_BASE)
        ax.tick_params(colors=Colors.TEXT_MUTED, labelsize=9)
        ax.xaxis.label.set_color(Colors.TEXT_SECONDARY)
        ax.yaxis.label.set_color(Colors.TEXT_SECONDARY)
        ax.title.set_color(Colors.TEXT_PRIMARY)
        ax.title.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_color(Colors.BORDER)
        ax.grid(True, alpha=0.3, color=Colors.BORDER)

    def clear_and_get_axes(self, rows=1, cols=1):
        """Clear figure and return fresh axes (single or grid)."""
        self.fig.clear()
        if rows == 1 and cols == 1:
            ax = self.fig.add_subplot(111)
            self._style_axes(ax)
            self.axes = ax
            return ax
        axes = []
        for i in range(rows * cols):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self._style_axes(ax)
            axes.append(ax)
        self.axes = axes[0]
        return axes

    def refresh(self):
        """Redraw the canvas with tight layout."""
        try:
            self.fig.tight_layout(pad=1.5)
        except (ValueError, RuntimeError):
            pass
        self.draw()

    def export_chart(self, filepath: str, dpi: int = 200):
        """Save the current figure to a file (PNG, SVG, PDF)."""
        self.fig.savefig(filepath, dpi=dpi, facecolor=self.fig.get_facecolor(),
                         edgecolor='none', bbox_inches='tight')
