"""
Data Cleaning Page — Step-by-step cleaning pipeline with live previews.
"""

import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QCheckBox,
    QComboBox, QGridLayout, QProgressBar, QSizePolicy,
    QApplication, QGroupBox, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, EmptyState


class CleaningWorker(QThread):
    """Runs data cleaning in a background thread."""
    progress = Signal(int, str)
    finished = Signal(object, dict)  # (cleaned_df, report)
    error = Signal(str)

    def __init__(self, df: pd.DataFrame, options: dict):
        super().__init__()
        self._df = df.copy()
        self._options = options

    def run(self):
        try:
            report = {"steps": [], "before": self._df.shape}
            total_steps = sum(self._options.values())
            step = 0

            # 1. Remove duplicates
            if self._options.get("remove_duplicates"):
                step += 1
                self.progress.emit(int(step / total_steps * 100), "Removing duplicates...")
                before = len(self._df)
                self._df = self._df.drop_duplicates()
                removed = before - len(self._df)
                report["steps"].append(f"Removed {removed:,} duplicate rows")

            # 2. Handle missing values
            if self._options.get("handle_missing"):
                step += 1
                self.progress.emit(int(step / total_steps * 100), "Handling missing values...")
                for col in self._df.columns:
                    missing_pct = self._df[col].isnull().mean()
                    if missing_pct > 0.9:
                        self._df.drop(columns=[col], inplace=True)
                        report["steps"].append(f"Dropped column '{col}' (>{90}% missing)")
                    elif missing_pct > 0:
                        if pd.api.types.is_numeric_dtype(self._df[col]):
                            self._df[col].fillna(self._df[col].median(), inplace=True)
                        else:
                            self._df[col].fillna(self._df[col].mode().iloc[0] if len(self._df[col].mode()) > 0 else "Unknown", inplace=True)
                report["steps"].append("Imputed remaining missing values (median/mode)")

            # 3. Fix data types
            if self._options.get("fix_dtypes"):
                step += 1
                self.progress.emit(int(step / total_steps * 100), "Fixing data types...")
                converted = 0
                for col in self._df.select_dtypes(include=["object"]).columns:
                    try:
                        self._df[col] = pd.to_numeric(self._df[col])
                        converted += 1
                    except (ValueError, TypeError):
                        try:
                            self._df[col] = pd.to_datetime(self._df[col], format="mixed")
                            converted += 1
                        except (ValueError, TypeError):
                            pass
                report["steps"].append(f"Auto-converted {converted} columns to proper types")

            # 4. Handle outliers
            if self._options.get("handle_outliers"):
                step += 1
                self.progress.emit(int(step / total_steps * 100), "Handling outliers...")
                capped = 0
                for col in self._df.select_dtypes(include=[np.number]).columns:
                    q1 = self._df[col].quantile(0.25)
                    q3 = self._df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        outliers = ((self._df[col] < lower) | (self._df[col] > upper)).sum()
                        if outliers > 0:
                            self._df[col] = self._df[col].clip(lower, upper)
                            capped += 1
                report["steps"].append(f"Capped outliers in {capped} numeric columns (IQR method)")

            # 5. Normalize text
            if self._options.get("normalize_text"):
                step += 1
                self.progress.emit(int(step / total_steps * 100), "Normalizing text...")
                text_cols = self._df.select_dtypes(include=["object"]).columns
                for col in text_cols:
                    self._df[col] = self._df[col].astype(str).str.strip()
                report["steps"].append(f"Trimmed whitespace in {len(text_cols)} text columns")

            # 6. Remove constant columns
            if self._options.get("remove_constant"):
                step += 1
                self.progress.emit(int(step / total_steps * 100), "Removing constant columns...")
                const_cols = [c for c in self._df.columns if self._df[c].nunique() <= 1]
                self._df.drop(columns=const_cols, inplace=True)
                report["steps"].append(f"Removed {len(const_cols)} constant columns")

            report["after"] = self._df.shape
            self.progress.emit(100, "Done!")
            self.finished.emit(self._df, report)

        except Exception as e:
            self.error.emit(str(e))


class CleaningPage(QWidget):
    """Data cleaning pipeline with configurable steps."""

    data_cleaned = Signal(object)  # cleaned DataFrame

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Empty state
        self._empty = EmptyState(
            "✦", "No Data to Clean",
            "Import a dataset first, then come here to clean it.",
        )
        layout.addWidget(self._empty)

        # Main content
        self._content = QWidget()
        self._content.hide()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(32, 28, 32, 28)
        inner_layout.setSpacing(20)

        # Header
        title = QLabel("Data Cleaning")
        title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XL, QFont.Weight.Bold))
        inner_layout.addWidget(title)

        self._dataset_info = QLabel("")
        self._dataset_info.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        inner_layout.addWidget(self._dataset_info)

        # Cleaning options
        inner_layout.addWidget(SectionHeader("Cleaning Steps"))
        options_card = Card()

        self._checks = {}
        steps = [
            ("remove_duplicates", "Remove Duplicate Rows", "Identify and remove exact duplicate rows"),
            ("handle_missing", "Handle Missing Values", "Drop columns >90% missing, impute rest with median/mode"),
            ("fix_dtypes", "Fix Data Types", "Auto-detect and convert numeric and datetime columns"),
            ("handle_outliers", "Handle Outliers (IQR)", "Cap extreme values using the Interquartile Range method"),
            ("normalize_text", "Normalize Text", "Trim whitespace in text columns"),
            ("remove_constant", "Remove Constant Columns", "Drop columns with only one unique value"),
        ]

        for key, label, desc in steps:
            row = QHBoxLayout()
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet(f"font-weight: bold;")
            row.addWidget(cb)
            row.addStretch()
            desc_label = QLabel(desc)
            desc_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_XS}px;")
            row.addWidget(desc_label)
            options_card.card_layout().addLayout(row)
            self._checks[key] = cb

        inner_layout.addWidget(options_card)

        # Run button + progress
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("   ▶  Run Cleaning Pipeline   ")
        self._run_btn.setProperty("accent", "primary")
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setFixedHeight(48)
        self._run_btn.clicked.connect(self._run_cleaning)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        inner_layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(12)
        self._progress.hide()
        inner_layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self._progress_label.hide()
        inner_layout.addWidget(self._progress_label)

        # Report
        inner_layout.addWidget(SectionHeader("Cleaning Report"))
        self._report_card = Card()
        self._report_text = QLabel("Run the pipeline to see results.")
        self._report_text.setWordWrap(True)
        self._report_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self._report_card.card_layout().addWidget(self._report_text)
        inner_layout.addWidget(self._report_card)

        # Download cleaned data
        dl_row = QHBoxLayout()
        self._dl_csv_btn = QPushButton("📥  Download Cleaned CSV")
        self._dl_csv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dl_csv_btn.setProperty("accent", "secondary")
        self._dl_csv_btn.setEnabled(False)
        self._dl_csv_btn.clicked.connect(lambda: self._download_cleaned("csv"))
        dl_row.addWidget(self._dl_csv_btn)

        self._dl_xlsx_btn = QPushButton("📥  Download Cleaned Excel")
        self._dl_xlsx_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dl_xlsx_btn.setProperty("accent", "secondary")
        self._dl_xlsx_btn.setEnabled(False)
        self._dl_xlsx_btn.clicked.connect(lambda: self._download_cleaned("xlsx"))
        dl_row.addWidget(self._dl_xlsx_btn)

        dl_row.addStretch()
        inner_layout.addLayout(dl_row)

        inner_layout.addStretch()
        scroll.setWidget(inner)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(scroll)
        layout.addWidget(self._content)

    def set_dataframe(self, df: pd.DataFrame):
        self._empty.hide()
        self._content.show()
        rows, cols = df.shape
        missing = df.isnull().sum().sum()
        dupes = df.duplicated().sum()
        self._dataset_info.setText(
            f"{rows:,} rows × {cols} columns  |  "
            f"{missing:,} missing values  |  {dupes:,} duplicates"
        )

    def _run_cleaning(self):
        df = self._state.get("current_df")
        if df is None:
            return

        options = {k: cb.isChecked() for k, cb in self._checks.items()}
        if not any(options.values()):
            return

        self._run_btn.setEnabled(False)
        self._progress.show()
        self._progress_label.show()
        self._progress.setValue(0)

        self._worker = CleaningWorker(df, options)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._progress_label.setText(msg)

    def _on_finished(self, cleaned_df: pd.DataFrame, report: dict):
        self._run_btn.setEnabled(True)
        self._progress.setValue(100)
        self._progress_label.setText("Cleaning complete!")
        self._progress_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")

        before = report["before"]
        after = report["after"]
        steps_text = "\n".join(f"  ✓  {s}" for s in report["steps"])
        self._report_text.setText(
            f"Before: {before[0]:,} rows × {before[1]} columns\n"
            f"After:  {after[0]:,} rows × {after[1]} columns\n\n"
            f"Steps performed:\n{steps_text}"
        )
        self._report_text.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")

        self._state["current_df"] = cleaned_df
        self.set_dataframe(cleaned_df)
        self.data_cleaned.emit(cleaned_df)

        self._dl_csv_btn.setEnabled(True)
        self._dl_xlsx_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress.hide()
        self._progress_label.setText(f"Error: {msg}")
        self._progress_label.setStyleSheet(f"color: {Colors.ERROR};")

    def _download_cleaned(self, fmt: str):
        """Download the cleaned DataFrame as CSV or Excel."""
        df = self._state.get("current_df")
        if df is None:
            return
        ext_map = {"csv": "CSV Files (*.csv)", "xlsx": "Excel Files (*.xlsx)"}
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned Data", f"cleaned_data.{fmt}", ext_map.get(fmt, "All (*)")
        )
        if path:
            if fmt == "xlsx":
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)
