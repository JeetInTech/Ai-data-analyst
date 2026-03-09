"""
Data Explorer Page — Interactive data preview with profiling and column stats.
"""

import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QSplitter,
    QListWidget, QListWidgetItem, QGridLayout, QSizePolicy,
    QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, EmptyState, Chip
from neurovia_ui.widgets.data_table import DataTable
from neurovia_ui.widgets.chart_canvas import ChartCanvas, NEUROVIA_COLORS


class DataExplorerPage(QWidget):
    """Data exploration with table view, column profiling, and distribution charts."""

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._df: pd.DataFrame = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Empty state (shown when no data) ──
        self._empty = EmptyState(
            "🔍", "No Data Loaded",
            "Import a dataset first to explore it here.",
            "Go to Import"
        )
        layout.addWidget(self._empty)

        # ── Main content (hidden until data loaded) ──
        self._content = QWidget()
        self._content.hide()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter: left = table, right = column detail
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: data table
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 8, 16)

        header_row = QHBoxLayout()
        self._title = QLabel("Data Preview")
        self._title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_LG, QFont.Weight.Bold))
        header_row.addWidget(self._title)
        header_row.addStretch()

        self._shape_label = QLabel("")
        self._shape_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        header_row.addWidget(self._shape_label)

        self._export_btn = QPushButton("📥  Export")
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.clicked.connect(self._export_data)
        header_row.addWidget(self._export_btn)

        left_layout.addLayout(header_row)

        self._table = DataTable()
        left_layout.addWidget(self._table)

        splitter.addWidget(left)

        # Right panel: column details
        right = QWidget()
        right.setMinimumWidth(320)
        right.setMaximumWidth(450)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 16, 16, 16)

        right_layout.addWidget(SectionHeader("Columns"))

        self._col_list = QListWidget()
        self._col_list.setFixedHeight(200)
        self._col_list.currentRowChanged.connect(self._on_column_selected)
        right_layout.addWidget(self._col_list)

        right_layout.addWidget(SectionHeader("Column Profile"))

        # Stats card
        self._stats_card = Card()
        self._stats_grid = QGridLayout()
        self._stats_grid.setSpacing(8)
        self._stats_card.card_layout().addLayout(self._stats_grid)
        right_layout.addWidget(self._stats_card)

        # Mini chart for selected column
        self._mini_chart = ChartCanvas(width=4, height=2.5, dpi=80)
        right_layout.addWidget(self._mini_chart)

        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        content_layout.addWidget(splitter)
        layout.addWidget(self._content)

    def set_dataframe(self, df: pd.DataFrame):
        self._df = df
        self._empty.hide()
        self._content.show()

        self._table.set_dataframe(df)
        self._shape_label.setText(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

        # Populate column list
        self._col_list.clear()
        for col in df.columns:
            dtype = str(df[col].dtype)
            icon = self._dtype_icon(dtype)
            item = QListWidgetItem(f"{icon}  {col}  ({dtype})")
            self._col_list.addItem(item)

        if len(df.columns) > 0:
            self._col_list.setCurrentRow(0)

    def _dtype_icon(self, dtype: str) -> str:
        if "int" in dtype or "float" in dtype:
            return "🔢"
        if "datetime" in dtype:
            return "📅"
        if "bool" in dtype:
            return "✓"
        return "🔤"

    def _on_column_selected(self, row: int):
        if self._df is None or row < 0 or row >= len(self._df.columns):
            return
        col = self._df.columns[row]
        series = self._df[col]
        self._show_column_stats(col, series)
        self._show_column_chart(col, series)

    def _show_column_stats(self, col: str, series: pd.Series):
        # Clear old stats
        while self._stats_grid.count():
            item = self._stats_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        stats = [
            ("Type", str(series.dtype)),
            ("Non-null", f"{series.count():,} / {len(series):,}"),
            ("Missing", f"{series.isnull().sum():,} ({series.isnull().mean() * 100:.1f}%)"),
            ("Unique", f"{series.nunique():,}"),
        ]

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            stats += [
                ("Mean", f"{desc.get('mean', 0):.4g}"),
                ("Std", f"{desc.get('std', 0):.4g}"),
                ("Min", f"{desc.get('min', 0):.4g}"),
                ("Max", f"{desc.get('max', 0):.4g}"),
                ("Median", f"{desc.get('50%', 0):.4g}"),
            ]
        else:
            top = series.value_counts().head(3)
            if len(top) > 0:
                stats.append(("Top Value", f"{top.index[0]} ({top.iloc[0]:,})"))

        for i, (key, val) in enumerate(stats):
            k = QLabel(key)
            k.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_XS}px;")
            v = QLabel(val)
            v.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_XS}px;")
            self._stats_grid.addWidget(k, i, 0)
            self._stats_grid.addWidget(v, i, 1)

    def _show_column_chart(self, col: str, series: pd.Series):
        ax = self._mini_chart.clear_and_get_axes()
        try:
            clean = series.dropna()
            if len(clean) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=Colors.TEXT_MUTED, fontsize=12)
            elif pd.api.types.is_numeric_dtype(series):
                ax.hist(clean, bins=min(50, max(10, len(clean) // 20)),
                        color=NEUROVIA_COLORS[0], alpha=0.8, edgecolor="none")
                ax.set_title(f"Distribution: {col}", fontsize=10, pad=8)
            else:
                top = clean.value_counts().head(10)
                colors = NEUROVIA_COLORS[:len(top)]
                ax.barh(range(len(top)), top.values, color=colors)
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels([str(v)[:20] for v in top.index], fontsize=8)
                ax.set_title(f"Top Values: {col}", fontsize=10, pad=8)
                ax.invert_yaxis()
        except Exception:
            ax.text(0.5, 0.5, "Cannot plot", ha="center", va="center",
                    color=Colors.TEXT_MUTED, fontsize=12)
        self._mini_chart.refresh()

    def _export_data(self):
        """Export the current dataframe as CSV or Excel."""
        if self._df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "data_export.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            if path.endswith(".xlsx"):
                self._df.to_excel(path, index=False)
            else:
                self._df.to_csv(path, index=False)
