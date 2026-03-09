"""
NeuroVia Data Table Widget
High-performance pandas DataFrame viewer using QTableView + QAbstractTableModel.
"""

import pandas as pd
from PySide6.QtWidgets import (
    QTableView, QVBoxLayout, QHBoxLayout, QWidget,
    QLineEdit, QLabel, QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QColor, QFont

from neurovia_ui.theme import Colors, Fonts


class PandasModel(QAbstractTableModel):
    """Model adapter for displaying a pandas DataFrame in QTableView."""

    MAX_DISPLAY_ROWS = 10_000  # Limit for display performance

    def __init__(self, df: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self._display_df = self._df.head(self.MAX_DISPLAY_ROWS)

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self._display_df = self._df.head(self.MAX_DISPLAY_ROWS)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._display_df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._display_df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        value = self._display_df.iloc[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value):
                return "—"
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)

        if role == Qt.ItemDataRole.ForegroundRole:
            if pd.isna(value):
                return QColor(Colors.TEXT_MUTED)
            return QColor(Colors.TEXT_PRIMARY)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._display_df.columns[section])
            return str(section + 1)
        if role == Qt.ItemDataRole.FontRole and orientation == Qt.Orientation.Horizontal:
            f = QFont(Fonts.FAMILY, Fonts.SIZE_XS)
            f.setBold(True)
            return f
        return None

    @property
    def full_dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def total_rows(self) -> int:
        return len(self._df)

    @property
    def is_truncated(self) -> bool:
        return len(self._df) > self.MAX_DISPLAY_ROWS


class DataTable(QWidget):
    """Complete data table widget with search, row info, and sorting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search columns or values...")
        self._search.setFixedWidth(300)
        self._search.textChanged.connect(self._on_search)
        toolbar.addWidget(self._search)

        toolbar.addStretch()

        self._info_label = QLabel("No data loaded")
        self._info_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_XS}px;")
        toolbar.addWidget(self._info_label)

        layout.addLayout(toolbar)

        # ── Table ──
        self._model = PandasModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setSortingEnabled(True)
        self._table.setShowGrid(True)
        self._table.verticalHeader().setDefaultSectionSize(36)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setMinimumSectionSize(80)

        layout.addWidget(self._table)

    def set_dataframe(self, df: pd.DataFrame):
        self._model.set_dataframe(df)
        total = self._model.total_rows
        shown = self._model.rowCount()
        cols = self._model.columnCount()
        trunc = " (showing first 10,000)" if self._model.is_truncated else ""
        self._info_label.setText(f"{total:,} rows × {cols} columns{trunc}")

    def get_dataframe(self) -> pd.DataFrame:
        return self._model.full_dataframe

    def _on_search(self, text: str):
        self._proxy.setFilterKeyColumn(-1)  # Search all columns
        self._proxy.setFilterFixedString(text)
