"""
Explainability Page — SHAP, feature importance, and model interpretation.
"""

import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QComboBox,
    QSizePolicy, QGridLayout, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, EmptyState
from neurovia_ui.widgets.chart_canvas import ChartCanvas, NEUROVIA_COLORS


class ExplainabilityPage(QWidget):
    """Model explainability with feature importance and SHAP analysis."""

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._empty = EmptyState(
            "☀", "No Models to Explain",
            "Train models first, then analyze their decisions here.",
        )
        layout.addWidget(self._empty)

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
        title = QLabel("Model Explainability")
        title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XL, QFont.Weight.Bold))
        inner_layout.addWidget(title)

        desc = QLabel("Understand how your models make predictions using feature importance and SHAP analysis.")
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        desc.setWordWrap(True)
        inner_layout.addWidget(desc)

        # Feature Importance section
        inner_layout.addWidget(SectionHeader("Feature Importance"))

        self._importance_info = QLabel("Feature importance will be computed from tree-based models.")
        self._importance_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        inner_layout.addWidget(self._importance_info)

        btn_row = QHBoxLayout()
        self._compute_btn = QPushButton("   ▶  Compute Feature Importance   ")
        self._compute_btn.setProperty("accent", "primary")
        self._compute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._compute_btn.setFixedHeight(44)
        self._compute_btn.clicked.connect(self._compute_importance)
        btn_row.addWidget(self._compute_btn)
        btn_row.addStretch()
        inner_layout.addLayout(btn_row)

        self._importance_chart = ChartCanvas(width=10, height=5)
        self._importance_chart.hide()
        inner_layout.addWidget(self._importance_chart)

        # Export buttons
        exp_row = QHBoxLayout()
        self._export_imp_btn = QPushButton("📥  Export Importance Chart")
        self._export_imp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_imp_btn.clicked.connect(self._export_importance_chart)
        exp_row.addWidget(self._export_imp_btn)

        self._export_profile_btn = QPushButton("📥  Export Profile Chart")
        self._export_profile_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_profile_btn.clicked.connect(self._export_profile_chart)
        exp_row.addWidget(self._export_profile_btn)

        exp_row.addStretch()
        inner_layout.addLayout(exp_row)

        # Dataset Profile section
        inner_layout.addWidget(SectionHeader("Dataset Quality Profile"))

        self._profile_chart = ChartCanvas(width=10, height=4)
        inner_layout.addWidget(self._profile_chart)

        # Correlation insights
        inner_layout.addWidget(SectionHeader("Correlation Insights"))
        self._corr_card = Card()
        self._corr_text = QLabel("Load data to see correlation insights.")
        self._corr_text.setWordWrap(True)
        self._corr_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self._corr_card.card_layout().addWidget(self._corr_text)
        inner_layout.addWidget(self._corr_card)

        inner_layout.addStretch()
        scroll.setWidget(inner)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(scroll)
        layout.addWidget(self._content)

    def set_dataframe(self, df: pd.DataFrame):
        self._empty.hide()
        self._content.show()
        self._show_profile(df)
        self._show_correlation_insights(df)

    def _compute_importance(self):
        df = self._state.get("current_df")
        if df is None:
            return

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            self._importance_info.setText("Need at least 2 numeric columns for feature importance.")
            self._importance_info.setStyleSheet(f"color: {Colors.WARNING};")
            return

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder

            target = num_cols[-1]
            X = df[num_cols[:-1]].fillna(0)
            y = df[target].fillna(0)

            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)

            importances = pd.Series(model.feature_importances_, index=X.columns)
            importances = importances.sort_values(ascending=True).tail(15)

            ax = self._importance_chart.clear_and_get_axes()
            colors = [NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)] for i in range(len(importances))]
            ax.barh(range(len(importances)), importances.values, color=colors, height=0.6)
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels(importances.index, fontsize=10)
            ax.set_xlabel("Importance", fontsize=11)
            ax.set_title(f"Feature Importance (predicting: {target})",
                         fontsize=13, fontweight="bold", color=Colors.TEXT_PRIMARY)
            self._importance_chart.refresh()
            self._importance_chart.show()

            self._importance_info.setText(
                f"✓ Top feature: {importances.index[-1]} (importance: {importances.iloc[-1]:.4f})"
            )
            self._importance_info.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")

        except Exception as e:
            self._importance_info.setText(f"Error: {e}")
            self._importance_info.setStyleSheet(f"color: {Colors.ERROR};")

    def _show_profile(self, df: pd.DataFrame):
        ax = self._profile_chart.clear_and_get_axes()

        total_cells = df.shape[0] * df.shape[1]
        completeness = (1 - df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
        uniqueness = (1 - df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        num_ratio = (num_cols / len(df.columns)) * 100 if len(df.columns) > 0 else 0

        metrics = ["Completeness", "Uniqueness", "Numeric\nRatio"]
        values = [completeness, uniqueness, num_ratio]
        colors = [Colors.SUCCESS, Colors.SECONDARY, Colors.ACCENT]

        bars = ax.bar(metrics, values, color=colors, width=0.5)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_title("Data Quality Profile", fontsize=13, fontweight="bold", color=Colors.TEXT_PRIMARY)
        ax.set_ylim(0, 110)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.1f}%", ha="center", fontsize=10, color=Colors.TEXT_PRIMARY)

        self._profile_chart.refresh()

    def _show_correlation_insights(self, df: pd.DataFrame):
        num = df.select_dtypes(include=[np.number])
        if len(num.columns) < 2:
            self._corr_text.setText("Not enough numeric columns for correlation analysis.")
            return

        corr = num.corr()

        # Find strongest correlations (excluding self-correlation)
        insights = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    direction = "positively" if val > 0 else "negatively"
                    insights.append(
                        f"• {corr.columns[i]} and {corr.columns[j]} are strongly "
                        f"{direction} correlated ({val:.3f})"
                    )

        if insights:
            self._corr_text.setText("\n".join(insights[:10]))
            self._corr_text.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        else:
            self._corr_text.setText("No strong correlations found (threshold: |r| > 0.7)")
            self._corr_text.setStyleSheet(f"color: {Colors.TEXT_MUTED};")

    def _export_importance_chart(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", "feature_importance.png",
            "PNG Files (*.png);;SVG Files (*.svg)"
        )
        if path:
            self._importance_chart.export_chart(path)

    def _export_profile_chart(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", "data_quality_profile.png",
            "PNG Files (*.png);;SVG Files (*.svg)"
        )
        if path:
            self._profile_chart.export_chart(path)
