"""
Data Import Page — Drag-and-drop file import with format support info.
"""

import os
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QFileDialog,
    QSizePolicy, QGridLayout, QApplication,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, Chip


SUPPORTED_FORMATS = {
    ".csv":     ("CSV",            "Comma-separated values"),
    ".tsv":     ("TSV",            "Tab-separated values"),
    ".xlsx":    ("Excel",          "Microsoft Excel workbook"),
    ".xls":     ("Excel Legacy",   "Microsoft Excel 97-2003"),
    ".json":    ("JSON",           "JavaScript Object Notation"),
    ".parquet": ("Parquet",        "Apache Parquet columnar format"),
    ".feather": ("Feather",        "Apache Arrow Feather format"),
}


class DropZone(QFrame):
    """Drag-and-drop zone for file imports."""

    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon = QLabel("📂")
        icon.setObjectName("drop_zone_icon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        text = QLabel("Drop your data file here")
        text.setObjectName("drop_zone_text")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)

        hint = QLabel("or click Browse to select a file")
        hint.setObjectName("drop_zone_hint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        # Format chips
        chip_row = QHBoxLayout()
        chip_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chip_row.setSpacing(8)
        for ext in [".csv", ".xlsx", ".json", ".parquet"]:
            chip_row.addWidget(Chip(ext.upper(), "primary"))
        layout.addLayout(chip_row)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                f"#drop_zone {{ border-color: {Colors.PRIMARY}; "
                f"background-color: {Colors.PRIMARY_DIM}; }}"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.file_dropped.emit(path)


class DataImportPage(QWidget):
    """Data import page with drag-drop and file browser."""

    data_loaded = Signal(object, str)  # (DataFrame, filepath)

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(24)

        # Header
        title = QLabel("Import Data")
        title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XL, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel("Load your dataset from a file. Supports CSV, Excel, JSON, Parquet, and more.")
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        layout.addWidget(desc)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._load_file)
        layout.addWidget(self._drop_zone)

        # Browse button
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._browse_btn = QPushButton("   Browse Files   ")
        self._browse_btn.setProperty("accent", "primary")
        self._browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_btn.setFixedHeight(44)
        self._browse_btn.clicked.connect(self._browse)
        btn_row.addWidget(self._browse_btn)

        layout.addLayout(btn_row)

        # Status
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # File info card (hidden initially)
        self._info_card = Card()
        self._info_card.hide()
        self._info_grid = QGridLayout()
        self._info_grid.setSpacing(12)
        self._info_card.card_layout().addWidget(
            SectionHeader("Loaded Dataset")
        )
        self._info_card.card_layout().addLayout(self._info_grid)
        layout.addWidget(self._info_card)

        # Supported formats
        layout.addWidget(SectionHeader("Supported Formats"))
        fmt_card = Card()
        fmt_grid = QGridLayout()
        fmt_grid.setSpacing(8)
        for i, (ext, (name, desc_text)) in enumerate(SUPPORTED_FORMATS.items()):
            lbl = QLabel(f"  {ext}")
            lbl.setStyleSheet(f"color: {Colors.PRIMARY}; font-weight: bold;")
            fmt_grid.addWidget(lbl, i, 0)
            fmt_grid.addWidget(QLabel(f"{name} — {desc_text}"), i, 1)
        fmt_card.card_layout().addLayout(fmt_grid)
        layout.addWidget(fmt_card)

        layout.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _browse(self):
        exts = " ".join(f"*{e}" for e in SUPPORTED_FORMATS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", "",
            f"Data Files ({exts});;All Files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, filepath: str):
        self._status.setText(f"Loading {os.path.basename(filepath)}...")
        self._status.setStyleSheet(f"color: {Colors.INFO};")
        QApplication.processEvents()

        try:
            ext = Path(filepath).suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(filepath, low_memory=False)
            elif ext == ".tsv":
                df = pd.read_csv(filepath, sep="\t", low_memory=False)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(filepath)
            elif ext == ".json":
                df = pd.read_json(filepath)
            elif ext == ".parquet":
                df = pd.read_parquet(filepath)
            elif ext == ".feather":
                df = pd.read_feather(filepath)
            else:
                self._status.setText(f"Unsupported format: {ext}")
                self._status.setStyleSheet(f"color: {Colors.ERROR};")
                return

            self._state["current_df"] = df
            self._state["current_file"] = filepath
            self._state["sessions"] = self._state.get("sessions", 0) + 1

            self._show_file_info(filepath, df)
            self._status.setText(f"✓ Successfully loaded {os.path.basename(filepath)}")
            self._status.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")

            self.data_loaded.emit(df, filepath)

        except Exception as e:
            self._status.setText(f"Error loading file: {e}")
            self._status.setStyleSheet(f"color: {Colors.ERROR};")

    def _show_file_info(self, filepath: str, df: pd.DataFrame):
        # Clear old info
        while self._info_grid.count():
            item = self._info_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        rows, cols = df.shape
        mem = df.memory_usage(deep=True).sum()
        missing = df.isnull().sum().sum()
        missing_pct = (missing / (rows * cols)) * 100 if rows * cols > 0 else 0
        dtypes = df.dtypes.value_counts()

        infos = [
            ("File", os.path.basename(filepath)),
            ("Rows", f"{rows:,}"),
            ("Columns", f"{cols}"),
            ("Memory", f"{mem / 1024 / 1024:.1f} MB" if mem > 1024 * 1024 else f"{mem / 1024:.1f} KB"),
            ("Missing Values", f"{missing:,} ({missing_pct:.1f}%)"),
            ("Column Types", ", ".join(f"{k}: {v}" for k, v in dtypes.items())),
        ]

        for i, (key, val) in enumerate(infos):
            k = QLabel(key)
            k.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-weight: bold;")
            v = QLabel(val)
            v.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
            self._info_grid.addWidget(k, i, 0)
            self._info_grid.addWidget(v, i, 1)

        self._info_card.show()
