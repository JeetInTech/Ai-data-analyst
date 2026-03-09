"""
Visualization Page — Interactive chart builder with multiple chart types.
"""

import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QComboBox,
    QGridLayout, QSizePolicy, QSplitter, QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, EmptyState
from neurovia_ui.widgets.chart_canvas import ChartCanvas, NEUROVIA_COLORS


CHART_TYPES = [
    # ── 2D Charts ──
    ("histogram",    "📊  Histogram"),
    ("scatter",      "⊙  Scatter Plot"),
    ("line",         "📈  Line Chart"),
    ("bar",          "▐  Bar Chart"),
    ("box",          "☐  Box Plot"),
    ("heatmap",      "▦  Correlation Heatmap"),
    ("pie",          "◉  Pie Chart"),
    ("violin",       "🎻  Violin Plot"),
    ("pair",         "⊞  Pair Plot"),
    ("area",         "▤  Area Chart"),
    ("radar",        "◎  Radar Chart"),
    ("waterfall",    "▥  Waterfall Chart"),
    ("treemap",      "▦  Treemap"),
    ("bubble",       "⊕  Bubble Chart"),
    ("density",      "▧  Density Plot"),
    # ── 3D Charts ──
    ("3d_scatter",   "🔮  3D Scatter"),
    ("3d_bar",       "🔮  3D Bar Chart"),
    ("3d_surface",   "🔮  3D Surface"),
    ("3d_wireframe", "🔮  3D Wireframe"),
    ("3d_contour",   "🔮  3D Contour"),
]

# Chart themes for matplotlib styling
CHART_THEMES = {
    "NeuroviaI Dark": {
        "fig_face": Colors.BG_DARK,
        "axes_face": Colors.BG_BASE,
        "text": Colors.TEXT_PRIMARY,
        "grid": Colors.BORDER,
        "tick": Colors.TEXT_MUTED,
        "label": Colors.TEXT_SECONDARY,
    },
    "Midnight Blue": {
        "fig_face": "#0a0e27",
        "axes_face": "#0f1538",
        "text": "#e0e7ff",
        "grid": "#1e2a5a",
        "tick": "#6b7db3",
        "label": "#8b9fd4",
    },
    "Cyberpunk": {
        "fig_face": "#0d0221",
        "axes_face": "#150533",
        "text": "#ff00ff",
        "grid": "#2a0845",
        "tick": "#bc13fe",
        "label": "#ff6ec7",
    },
    "Light Professional": {
        "fig_face": "#fafbfc",
        "axes_face": "#ffffff",
        "text": "#1a1a2e",
        "grid": "#e1e5ea",
        "tick": "#6c757d",
        "label": "#495057",
    },
    "Warm Earth": {
        "fig_face": "#1a1410",
        "axes_face": "#231c14",
        "text": "#f5e6d3",
        "grid": "#3d3226",
        "tick": "#a08060",
        "label": "#c4a882",
    },
}


class VisualizationPage(QWidget):
    """Chart builder with type selection, axis configuration, and live preview."""

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._df: pd.DataFrame = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._empty = EmptyState(
            "▣", "No Data for Visualization",
            "Import a dataset to create visualizations.",
        )
        layout.addWidget(self._empty)

        self._content = QWidget()
        self._content.hide()
        content_layout = QHBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # ── Left: Controls ──
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setFixedWidth(300)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(16)

        left_layout.addWidget(SectionHeader("Chart Type"))

        self._type_combo = QComboBox()
        for key, label in CHART_TYPES:
            self._type_combo.addItem(label, key)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        left_layout.addWidget(self._type_combo)

        left_layout.addWidget(SectionHeader("X Axis"))
        self._x_combo = QComboBox()
        self._x_combo.currentIndexChanged.connect(self._update_chart)
        left_layout.addWidget(self._x_combo)

        left_layout.addWidget(SectionHeader("Y Axis"))
        self._y_combo = QComboBox()
        self._y_combo.currentIndexChanged.connect(self._update_chart)
        left_layout.addWidget(self._y_combo)

        left_layout.addWidget(SectionHeader("Color By"))
        self._color_combo = QComboBox()
        self._color_combo.addItem("None", None)
        self._color_combo.currentIndexChanged.connect(self._update_chart)
        left_layout.addWidget(self._color_combo)

        # Z axis (for 3D charts)
        self._z_header = SectionHeader("Z Axis")
        left_layout.addWidget(self._z_header)
        self._z_combo = QComboBox()
        self._z_combo.currentIndexChanged.connect(self._update_chart)
        left_layout.addWidget(self._z_combo)
        self._z_header.hide()
        self._z_combo.hide()

        # Chart theme
        left_layout.addWidget(SectionHeader("Chart Theme"))
        self._theme_combo = QComboBox()
        for name in CHART_THEMES:
            self._theme_combo.addItem(name)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        left_layout.addWidget(self._theme_combo)

        # Generate button
        self._generate_btn = QPushButton("   ▶  Generate Chart   ")
        self._generate_btn.setProperty("accent", "primary")
        self._generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._generate_btn.setFixedHeight(44)
        self._generate_btn.clicked.connect(self._update_chart)
        left_layout.addWidget(self._generate_btn)

        # Quick charts
        left_layout.addWidget(SectionHeader("Quick Charts"))

        self._auto_btn = QPushButton("⚡  Auto Visualize All")
        self._auto_btn.setProperty("accent", "secondary")
        self._auto_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._auto_btn.clicked.connect(self._auto_visualize)
        left_layout.addWidget(self._auto_btn)

        self._corr_btn = QPushButton("▦  Correlation Matrix")
        self._corr_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._corr_btn.clicked.connect(self._show_correlation)
        left_layout.addWidget(self._corr_btn)

        self._dist_btn = QPushButton("📊  All Distributions")
        self._dist_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dist_btn.clicked.connect(self._show_distributions)
        left_layout.addWidget(self._dist_btn)

        left_layout.addStretch()
        left_scroll.setWidget(left)
        content_layout.addWidget(left_scroll)

        # ── Separator ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        content_layout.addWidget(sep)

        # ── Right: Chart display ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)

        self._chart_title = QLabel("Select a chart type and columns")
        self._chart_title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_MD, QFont.Weight.DemiBold))
        right_layout.addWidget(self._chart_title)

        self._canvas = ChartCanvas(width=10, height=7)
        right_layout.addWidget(self._canvas)

        # ── Export buttons ──
        export_row = QHBoxLayout()
        export_row.setSpacing(8)

        self._export_png_btn = QPushButton("📥  Export PNG")
        self._export_png_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_png_btn.clicked.connect(lambda: self._export_chart("png"))
        export_row.addWidget(self._export_png_btn)

        self._export_svg_btn = QPushButton("📥  Export SVG")
        self._export_svg_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_svg_btn.clicked.connect(lambda: self._export_chart("svg"))
        export_row.addWidget(self._export_svg_btn)

        self._export_data_btn = QPushButton("📥  Export Data CSV")
        self._export_data_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_data_btn.clicked.connect(self._export_data_csv)
        export_row.addWidget(self._export_data_btn)

        export_row.addStretch()
        right_layout.addLayout(export_row)

        content_layout.addWidget(right)
        content_layout.setStretchFactor(right, 1)

        layout.addWidget(self._content)

    def set_dataframe(self, df: pd.DataFrame):
        self._df = df
        self._empty.hide()
        self._content.show()

        cols = list(df.columns)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)

        for combo in [self._x_combo, self._y_combo, self._z_combo]:
            combo.blockSignals(True)
            combo.clear()
            for c in cols:
                combo.addItem(c)
            combo.blockSignals(False)

        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        self._color_combo.addItem("None", None)
        for c in cols:
            self._color_combo.addItem(c)
        self._color_combo.blockSignals(False)

        if len(num_cols) >= 2:
            self._x_combo.setCurrentText(num_cols[0])
            self._y_combo.setCurrentText(num_cols[1])
        elif len(cols) >= 2:
            self._x_combo.setCurrentIndex(0)
            self._y_combo.setCurrentIndex(1)

    def _on_type_changed(self, idx):
        chart_type = self._type_combo.currentData()
        single_axis = chart_type in ("histogram", "box", "pie", "violin", "density", "treemap")
        is_3d = chart_type.startswith("3d_")
        self._y_combo.setEnabled(not single_axis)
        # Show Z axis for 3D charts
        self._z_header.setVisible(is_3d)
        self._z_combo.setVisible(is_3d)
        self._update_chart()

    def _show_chart_info(self, ax, message: str, is_3d: bool = False):
        """Display an informational message on the chart when data is unsuitable."""
        if is_3d:
            ax.text2D(0.5, 0.5, message, transform=ax.transAxes,
                      ha="center", va="center", color=Colors.TEXT_MUTED,
                      fontsize=11, linespacing=1.8)
        else:
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                    ha="center", va="center", color=Colors.TEXT_MUTED,
                    fontsize=11, linespacing=1.8)

    def _update_chart(self):
        if self._df is None:
            return

        chart_type = self._type_combo.currentData()
        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        color_col = self._color_combo.currentData()

        if not x_col:
            return

        ax = self._canvas.clear_and_get_axes()
        df = self._df

        try:
            if chart_type == "histogram":
                data = df[x_col].dropna()
                if pd.api.types.is_numeric_dtype(data):
                    ax.hist(data, bins=min(50, max(10, len(data) // 20)),
                            color=NEUROVIA_COLORS[0], alpha=0.85, edgecolor="none")
                else:
                    vc = data.value_counts().head(20)
                    ax.bar(range(len(vc)), vc.values, color=NEUROVIA_COLORS[:len(vc)])
                    ax.set_xticks(range(len(vc)))
                    ax.set_xticklabels([str(v)[:15] for v in vc.index], rotation=45, ha="right")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Count")
                self._chart_title.setText(f"Histogram: {x_col}")

            elif chart_type == "scatter":
                if color_col and color_col in df.columns:
                    groups = df[color_col].unique()[:10]
                    for i, g in enumerate(groups):
                        mask = df[color_col] == g
                        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                                   c=NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)],
                                   alpha=0.6, s=20, label=str(g)[:20])
                    ax.legend(fontsize=8, framealpha=0.8)
                else:
                    ax.scatter(df[x_col], df[y_col], c=NEUROVIA_COLORS[0], alpha=0.6, s=20)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                self._chart_title.setText(f"Scatter: {x_col} vs {y_col}")

            elif chart_type == "line":
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    ax.plot(df[x_col].values, df[y_col].values,
                            color=NEUROVIA_COLORS[0], linewidth=1.5, alpha=0.9)
                    ax.fill_between(range(len(df)), df[y_col].values,
                                    alpha=0.1, color=NEUROVIA_COLORS[0])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    self._show_chart_info(ax, f"Line Chart requires a numeric Y axis.\n"
                                         f"'{y_col}' is not numeric — select a numeric column.")
                self._chart_title.setText(f"Line: {y_col} over {x_col}")

            elif chart_type == "bar":
                if pd.api.types.is_numeric_dtype(df[x_col]):
                    bins = pd.cut(df[x_col], bins=10)
                    if y_col and y_col in df.columns:
                        means = df.groupby(bins, observed=True)[y_col].mean()
                    else:
                        means = bins.value_counts().sort_index()
                    ax.bar(range(len(means)), means.values, color=NEUROVIA_COLORS[:len(means)])
                    ax.set_xticks(range(len(means)))
                    ax.set_xticklabels([str(v)[:12] for v in means.index], rotation=45, ha="right", fontsize=8)
                else:
                    vc = df[x_col].value_counts().head(15)
                    ax.bar(range(len(vc)), vc.values, color=NEUROVIA_COLORS[:len(vc)])
                    ax.set_xticks(range(len(vc)))
                    ax.set_xticklabels([str(v)[:15] for v in vc.index], rotation=45, ha="right", fontsize=8)
                ax.set_xlabel(x_col)
                self._chart_title.setText(f"Bar: {x_col}")

            elif chart_type == "box":
                num = df.select_dtypes(include=[np.number])
                if len(num.columns) > 0:
                    data = [num[c].dropna().values for c in num.columns[:10]]
                    bp = ax.boxplot(data, patch_artist=True, labels=[c[:12] for c in num.columns[:10]])
                    for i, patch in enumerate(bp["boxes"]):
                        patch.set_facecolor(NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)])
                        patch.set_alpha(0.7)
                    ax.tick_params(axis="x", rotation=45)
                else:
                    self._show_chart_info(ax, "Box Plot requires numeric columns.\nNo numeric columns found in the dataset.")
                self._chart_title.setText("Box Plot: Numeric Columns")

            elif chart_type == "heatmap":
                self._show_correlation()
                return

            elif chart_type == "pie":
                vc = df[x_col].value_counts().head(8)
                colors = NEUROVIA_COLORS[:len(vc)]
                ax.pie(vc.values, labels=[str(v)[:15] for v in vc.index],
                       colors=colors, autopct="%1.1f%%", startangle=90,
                       textprops={"color": Colors.TEXT_PRIMARY, "fontsize": 9})
                self._chart_title.setText(f"Pie: {x_col}")

            elif chart_type == "violin":
                num = df.select_dtypes(include=[np.number])
                if len(num.columns) > 0:
                    data = [num[c].dropna().values for c in num.columns[:8]]
                    parts = ax.violinplot(data, showmeans=True, showmedians=True)
                    for i, pc in enumerate(parts.get("bodies", [])):
                        pc.set_facecolor(NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)])
                        pc.set_alpha(0.7)
                    ax.set_xticks(range(1, len(num.columns[:8]) + 1))
                    ax.set_xticklabels([c[:12] for c in num.columns[:8]], rotation=45)
                else:
                    self._show_chart_info(ax, "Violin Plot requires numeric columns.\nNo numeric columns found in the dataset.")
                self._chart_title.setText("Violin Plot: Numeric Columns")

            elif chart_type == "pair":
                self._show_pair_plot()
                return

            elif chart_type == "area":
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    ax.fill_between(range(len(df)), df[y_col].values,
                                    alpha=0.4, color=NEUROVIA_COLORS[0])
                    ax.plot(df[y_col].values, color=NEUROVIA_COLORS[0], linewidth=1.5)
                    ax.set_xlabel("Index")
                    ax.set_ylabel(y_col)
                else:
                    self._show_chart_info(ax, f"Area Chart requires a numeric Y axis.\n"
                                         f"'{y_col}' is not numeric — select a numeric column.")
                self._chart_title.setText(f"Area: {y_col}")

            # ── New 2D chart types ──

            elif chart_type == "radar":
                num = df.select_dtypes(include=[np.number])
                cols = list(num.columns[:8])
                if len(cols) >= 3:
                    values = num[cols].mean().values
                    # Normalize to 0-1
                    vmin, vmax = values.min(), values.max()
                    if vmax > vmin:
                        values = (values - vmin) / (vmax - vmin)
                    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
                    values = np.concatenate([values, [values[0]]])
                    angles += angles[:1]

                    ax.remove()
                    ax = self._canvas.fig.add_subplot(111, polar=True)
                    ax.set_facecolor(self._get_theme()["axes_face"])
                    ax.fill(angles, values, color=NEUROVIA_COLORS[0], alpha=0.25)
                    ax.plot(angles, values, color=NEUROVIA_COLORS[0], linewidth=2)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels([c[:12] for c in cols], fontsize=9,
                                       color=self._get_theme()["text"])
                    ax.tick_params(colors=self._get_theme()["tick"])
                else:
                    self._show_chart_info(ax, f"Radar Chart requires 3+ numeric columns.\n"
                                         f"Found {len(cols)} numeric column(s).")
                self._chart_title.setText("Radar: Mean of Numeric Columns")

            elif chart_type == "waterfall":
                vc = df[x_col].value_counts().head(10)
                cumulative = vc.values.cumsum()
                starts = np.concatenate([[0], cumulative[:-1]])
                colors = [NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)] for i in range(len(vc))]
                ax.bar(range(len(vc)), vc.values, bottom=starts, color=colors, width=0.7)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels([str(v)[:12] for v in vc.index], rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Cumulative Count")
                self._chart_title.setText(f"Waterfall: {x_col}")

            elif chart_type == "treemap":
                try:
                    import matplotlib.pyplot as _plt
                    vc = df[x_col].value_counts().head(12)
                    sizes = vc.values
                    labels = [f"{str(v)[:15]}\n{s:,}" for v, s in zip(vc.index, sizes)]
                    colors = [NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)] for i in range(len(vc))]
                    # Manual treemap using squarify layout
                    import squarify
                    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85,
                                  ax=ax, text_kwargs={"fontsize": 9, "color": "white"})
                    ax.axis("off")
                except ImportError:
                    # Fallback: horizontal stacked bar as treemap approximation
                    vc = df[x_col].value_counts().head(12)
                    total = vc.sum()
                    left = 0
                    for i, (label, val) in enumerate(vc.items()):
                        width = val / total
                        ax.barh(0, width, left=left, height=0.6,
                                color=NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)])
                        if width > 0.05:
                            ax.text(left + width / 2, 0, f"{str(label)[:10]}\n{val:,}",
                                    ha="center", va="center", fontsize=8, color="white")
                        left += width
                    ax.set_yticks([])
                    ax.set_xlabel("Proportion")
                self._chart_title.setText(f"Treemap: {x_col}")

            elif chart_type == "bubble":
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    z_col = self._z_combo.currentText()
                    if z_col and z_col in df.columns and pd.api.types.is_numeric_dtype(df[z_col]):
                        sizes = df[z_col].fillna(0).abs()
                    else:
                        sizes = pd.Series(50, index=df.index)
                    # Normalize sizes
                    smin, smax = sizes.min(), sizes.max()
                    if smax > smin:
                        sizes = 20 + (sizes - smin) / (smax - smin) * 400
                    else:
                        sizes = 50
                    ax.scatter(df[x_col], df[y_col], s=sizes, c=NEUROVIA_COLORS[0],
                               alpha=0.5, edgecolors=NEUROVIA_COLORS[1], linewidth=0.5)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    non_num = [c for c in [x_col, y_col] if not pd.api.types.is_numeric_dtype(df[c])]
                    self._show_chart_info(ax, f"Bubble Chart requires numeric X and Y axes.\n"
                                         f"Non-numeric: {', '.join(non_num)}")
                self._chart_title.setText(f"Bubble: {x_col} vs {y_col}")

            elif chart_type == "density":
                data = df[x_col].dropna()
                if pd.api.types.is_numeric_dtype(data) and len(data) > 1:
                    from scipy.stats import gaussian_kde
                    density = gaussian_kde(data)
                    xs = np.linspace(data.min(), data.max(), 300)
                    ax.fill_between(xs, density(xs), alpha=0.4, color=NEUROVIA_COLORS[0])
                    ax.plot(xs, density(xs), color=NEUROVIA_COLORS[0], linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Density")
                else:
                    self._show_chart_info(ax, f"Density Plot requires a numeric column.\n"
                                         f"'{x_col}' is categorical — try selecting a numeric column.")
                self._chart_title.setText(f"Density: {x_col}")

            # ── 3D Charts ──

            elif chart_type == "3d_scatter":
                self._canvas.fig.clear()
                ax3 = self._canvas.fig.add_subplot(111, projection="3d")
                self._style_3d_axes(ax3)
                z_col = self._z_combo.currentText()
                cols_to_check = [x_col, y_col]
                if z_col and z_col in df.columns:
                    cols_to_check.append(z_col)
                non_num = [c for c in cols_to_check if not pd.api.types.is_numeric_dtype(df[c])]
                if non_num:
                    self._show_chart_info(ax3, f"3D Scatter requires all numeric columns.\n"
                                         f"Non-numeric: {', '.join(non_num)}\n\n"
                                         f"Select numeric columns for X, Y, and Z axes.", is_3d=True)
                elif len(cols_to_check) < 3:
                    self._show_chart_info(ax3, "Select a Z Axis column for 3D Scatter.\n"
                                         "Choose a numeric column from the Z Axis dropdown.", is_3d=True)
                else:
                    x_data = df[x_col].values.astype(float)
                    y_data = df[y_col].values.astype(float)
                    z_data = df[z_col].values.astype(float)
                    mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data)
                    if mask.sum() > 0:
                        ax3.scatter(x_data[mask], y_data[mask], z_data[mask],
                                    c=NEUROVIA_COLORS[0], alpha=0.6, s=20, edgecolors="none")
                    ax3.set_xlabel(x_col, fontsize=9)
                    ax3.set_ylabel(y_col, fontsize=9)
                    ax3.set_zlabel(z_col, fontsize=9)
                self._chart_title.setText(f"3D Scatter: {x_col} × {y_col} × {z_col if z_col else '?'}")
                self._canvas.refresh()
                return

            elif chart_type == "3d_bar":
                self._canvas.fig.clear()
                ax3 = self._canvas.fig.add_subplot(111, projection="3d")
                self._style_3d_axes(ax3)
                vc = df[x_col].value_counts().head(12)
                if len(vc) == 0:
                    self._show_chart_info(ax3, "No data available for 3D Bar chart.", is_3d=True)
                else:
                    x_pos = np.arange(len(vc), dtype=float)
                    y_pos = np.zeros(len(vc), dtype=float)
                    z_pos = np.zeros(len(vc), dtype=float)
                    dx = dy = 0.6
                    dz = vc.values.astype(float)
                    colors = [NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)] for i in range(len(vc))]
                    ax3.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.85)
                    ax3.set_xticks(x_pos)
                    ax3.set_xticklabels([str(v)[:10] for v in vc.index], rotation=20, fontsize=8)
                    ax3.set_ylabel("")
                    ax3.set_zlabel("Count", fontsize=9)
                self._chart_title.setText(f"3D Bar: {x_col}")
                self._canvas.refresh()
                return

            elif chart_type == "3d_surface":
                self._canvas.fig.clear()
                ax3 = self._canvas.fig.add_subplot(111, projection="3d")
                self._style_3d_axes(ax3)
                num = df.select_dtypes(include=[np.number])
                if len(num.columns) >= 2:
                    c1, c2 = num.columns[0], num.columns[1]
                    x_vals = np.linspace(float(num[c1].min()), float(num[c1].max()), 40)
                    y_vals = np.linspace(float(num[c2].min()), float(num[c2].max()), 40)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    from scipy.interpolate import griddata
                    z_col = self._z_combo.currentText()
                    if z_col and z_col in num.columns:
                        Z = griddata((num[c1].values.astype(float), num[c2].values.astype(float)),
                                     num[z_col].values.astype(float), (X, Y), method="cubic", fill_value=0)
                    else:
                        Z = np.sin(np.sqrt(X ** 2 + Y ** 2) * 0.1) * float(num[c1].std())
                    ax3.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none")
                    ax3.set_xlabel(c1, fontsize=9)
                    ax3.set_ylabel(c2, fontsize=9)
                    if z_col and z_col in num.columns:
                        ax3.set_zlabel(z_col, fontsize=9)
                else:
                    self._show_chart_info(ax3, f"3D Surface requires 2+ numeric columns.\n"
                                         f"Found {len(num.columns)} numeric column(s).", is_3d=True)
                self._chart_title.setText("3D Surface")
                self._canvas.refresh()
                return

            elif chart_type == "3d_wireframe":
                self._canvas.fig.clear()
                ax3 = self._canvas.fig.add_subplot(111, projection="3d")
                self._style_3d_axes(ax3)
                num = df.select_dtypes(include=[np.number])
                if len(num.columns) >= 2:
                    c1, c2 = num.columns[0], num.columns[1]
                    x_vals = np.linspace(float(num[c1].min()), float(num[c1].max()), 30)
                    y_vals = np.linspace(float(num[c2].min()), float(num[c2].max()), 30)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    from scipy.interpolate import griddata
                    z_col = self._z_combo.currentText()
                    if z_col and z_col in num.columns:
                        Z = griddata((num[c1].values.astype(float), num[c2].values.astype(float)),
                                     num[z_col].values.astype(float), (X, Y), method="cubic", fill_value=0)
                    else:
                        Z = np.sin(np.sqrt(X ** 2 + Y ** 2) * 0.1) * float(num[c1].std())
                    ax3.plot_wireframe(X, Y, Z, color=NEUROVIA_COLORS[0], alpha=0.7, linewidth=0.5)
                    ax3.set_xlabel(c1, fontsize=9)
                    ax3.set_ylabel(c2, fontsize=9)
                else:
                    self._show_chart_info(ax3, f"3D Wireframe requires 2+ numeric columns.\n"
                                         f"Found {len(num.columns)} numeric column(s).", is_3d=True)
                self._chart_title.setText("3D Wireframe")
                self._canvas.refresh()
                return

            elif chart_type == "3d_contour":
                num = df.select_dtypes(include=[np.number])
                if len(num.columns) >= 2:
                    c1, c2 = num.columns[0], num.columns[1]
                    x_vals = np.linspace(float(num[c1].min()), float(num[c1].max()), 50)
                    y_vals = np.linspace(float(num[c2].min()), float(num[c2].max()), 50)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    from scipy.interpolate import griddata
                    z_col = self._z_combo.currentText()
                    if z_col and z_col in num.columns:
                        Z = griddata((num[c1].values.astype(float), num[c2].values.astype(float)),
                                     num[z_col].values.astype(float), (X, Y), method="cubic", fill_value=0)
                    else:
                        Z = np.sin(np.sqrt(X ** 2 + Y ** 2) * 0.1) * float(num[c1].std())
                    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
                    ax.contour(X, Y, Z, levels=20, colors="white", linewidths=0.3, alpha=0.5)
                    self._canvas.fig.colorbar(cf, ax=ax, shrink=0.8)
                    ax.set_xlabel(c1)
                    ax.set_ylabel(c2)
                else:
                    self._show_chart_info(ax, f"Contour Plot requires 2+ numeric columns.\n"
                                         f"Found {len(num.columns)} numeric column(s).")
                self._chart_title.setText("Contour Plot")

        except Exception as e:
            try:
                ax_err = self._canvas.clear_and_get_axes()
                ax_err.text(0.5, 0.5, f"Chart Error: {str(e)[:120]}", ha="center", va="center",
                            color=Colors.ERROR, fontsize=11, transform=ax_err.transAxes,
                            linespacing=1.5)
            except Exception:
                pass
            self._chart_title.setText("Chart Error")

        self._canvas.refresh()

    # ── Theme & 3D helpers ──

    def _get_theme(self) -> dict:
        name = self._theme_combo.currentText()
        return CHART_THEMES.get(name, CHART_THEMES["NeuroviaI Dark"])

    def _on_theme_changed(self, theme_name: str):
        theme = CHART_THEMES.get(theme_name, CHART_THEMES["NeuroviaI Dark"])
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.facecolor": theme["fig_face"],
            "axes.facecolor": theme["axes_face"],
            "text.color": theme["text"],
            "axes.labelcolor": theme["label"],
            "xtick.color": theme["tick"],
            "ytick.color": theme["tick"],
            "grid.color": theme["grid"],
        })
        self._canvas.fig.set_facecolor(theme["fig_face"])
        self._update_chart()

    def _style_3d_axes(self, ax3):
        theme = self._get_theme()
        ax3.set_facecolor(theme["axes_face"])
        ax3.tick_params(colors=theme["tick"], labelsize=8)
        ax3.xaxis.label.set_color(theme["label"])
        ax3.yaxis.label.set_color(theme["label"])
        ax3.zaxis.label.set_color(theme["label"])
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor(theme["grid"])
        ax3.yaxis.pane.set_edgecolor(theme["grid"])
        ax3.zaxis.pane.set_edgecolor(theme["grid"])
        ax3.grid(True, alpha=0.3, color=theme["grid"])

    def _show_correlation(self):
        if self._df is None:
            return
        ax = self._canvas.clear_and_get_axes()
        num = self._df.select_dtypes(include=[np.number])
        if len(num.columns) < 2:
            ax.text(0.5, 0.5, "Need 2+ numeric columns", ha="center", va="center",
                    color=Colors.TEXT_MUTED)
        else:
            corr = num.corr()
            im = ax.imshow(corr, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels([c[:12] for c in corr.columns], rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([c[:12] for c in corr.columns], fontsize=8)
            self._canvas.fig.colorbar(im, ax=ax, shrink=0.8)
            # Add correlation values
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else Colors.TEXT_MUTED)
        self._chart_title.setText("Correlation Heatmap")
        self._canvas.refresh()

    def _show_distributions(self):
        if self._df is None:
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns[:9]
        n = len(num_cols)
        if n == 0:
            return
        rows = int(np.ceil(n / 3))
        cols = min(n, 3)
        axes = self._canvas.clear_and_get_axes(rows, cols)
        if not isinstance(axes, list):
            axes = [axes]
        for i, col in enumerate(num_cols):
            if i < len(axes):
                data = self._df[col].dropna()
                axes[i].hist(data, bins=30, color=NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)],
                             alpha=0.8, edgecolor="none")
                axes[i].set_title(col[:20], fontsize=9, color=Colors.TEXT_PRIMARY)
        self._chart_title.setText("All Distributions")
        self._canvas.refresh()

    def _show_pair_plot(self):
        if self._df is None:
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns[:4]
        n = len(num_cols)
        if n < 2:
            return
        axes = self._canvas.clear_and_get_axes(n, n)
        for i in range(n):
            for j in range(n):
                ax = axes[i * n + j]
                if i == j:
                    ax.hist(self._df[num_cols[i]].dropna(), bins=20,
                            color=NEUROVIA_COLORS[i], alpha=0.7, edgecolor="none")
                else:
                    ax.scatter(self._df[num_cols[j]], self._df[num_cols[i]],
                               c=NEUROVIA_COLORS[0], alpha=0.3, s=5)
                if j == 0:
                    ax.set_ylabel(num_cols[i][:10], fontsize=7)
                if i == n - 1:
                    ax.set_xlabel(num_cols[j][:10], fontsize=7)
        self._chart_title.setText("Pair Plot")
        self._canvas.refresh()

    def _auto_visualize(self):
        if self._df is None:
            return
        num_cols = self._df.select_dtypes(include=[np.number]).columns
        cat_cols = self._df.select_dtypes(include=["object", "category"]).columns

        n_plots = min(6, len(num_cols) + min(2, len(cat_cols)))
        if n_plots == 0:
            return

        rows = int(np.ceil(n_plots / 3))
        cols = min(n_plots, 3)
        axes = self._canvas.clear_and_get_axes(rows, cols)
        if not isinstance(axes, list):
            axes = [axes]

        plot_idx = 0
        # Histograms for numeric
        for i, col in enumerate(num_cols[:4]):
            if plot_idx >= len(axes):
                break
            data = self._df[col].dropna()
            axes[plot_idx].hist(data, bins=30, color=NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)],
                                alpha=0.8, edgecolor="none")
            axes[plot_idx].set_title(col[:20], fontsize=9, color=Colors.TEXT_PRIMARY)
            plot_idx += 1

        # Bar charts for categorical
        for i, col in enumerate(cat_cols[:2]):
            if plot_idx >= len(axes):
                break
            vc = self._df[col].value_counts().head(8)
            axes[plot_idx].barh(range(len(vc)), vc.values,
                                color=NEUROVIA_COLORS[:len(vc)])
            axes[plot_idx].set_yticks(range(len(vc)))
            axes[plot_idx].set_yticklabels([str(v)[:15] for v in vc.index], fontsize=8)
            axes[plot_idx].set_title(col[:20], fontsize=9, color=Colors.TEXT_PRIMARY)
            axes[plot_idx].invert_yaxis()
            plot_idx += 1

        self._chart_title.setText("Auto Visualization Overview")
        self._canvas.refresh()

    # ── Export helpers ──

    def _export_chart(self, fmt: str):
        """Export current chart as PNG or SVG."""
        ext_map = {"png": "PNG Files (*.png)", "svg": "SVG Files (*.svg)"}
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", f"chart.{fmt}", ext_map.get(fmt, "All Files (*)")
        )
        if path:
            self._canvas.export_chart(path)

    def _export_data_csv(self):
        """Export the current dataframe as CSV."""
        if self._df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "data.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            if path.endswith(".xlsx"):
                self._df.to_excel(path, index=False)
            else:
                self._df.to_csv(path, index=False)
