"""
NeuroviaI Main Window
Central application window with sidebar navigation and stacked page content.
"""

import pandas as pd

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QStatusBar, QFrame,
    QSizePolicy, QApplication,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont, QIcon

from neurovia_ui.theme import Colors, Fonts, build_stylesheet
from neurovia_ui.widgets.sidebar import SideBar
from neurovia_ui.pages.dashboard import DashboardPage
from neurovia_ui.pages.data_import import DataImportPage
from neurovia_ui.pages.data_explorer import DataExplorerPage
from neurovia_ui.pages.cleaning import CleaningPage
from neurovia_ui.pages.visualization import VisualizationPage
from neurovia_ui.pages.ml_training import MLTrainingPage
from neurovia_ui.pages.explainability_page import ExplainabilityPage
from neurovia_ui.pages.agent_page import AgentPage
from neurovia_ui.pages.settings import SettingsPage


PAGE_TITLES = {
    "dashboard":      "Dashboard",
    "import":         "Import Data",
    "explorer":       "Data Explorer",
    "agents":         "AI Agents",
    "cleaning":       "Data Cleaning",
    "visualization":  "Visualization",
    "ml_training":    "ML Training",
    "explainability": "Explainability",
    "settings":       "Settings",
}


class NeuroViaMainWindow(QMainWindow):
    """Main application window for NeuroviaI Data Analytics Platform."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroviaI — Data Analytics Platform")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        # Shared application state
        self._state = {
            "current_df": None,
            "current_file": None,
            "sessions": 0,
            "models_trained": 0,
            "rows_processed": 0,
            "avg_quality": None,
        }

        # Apply theme
        self.setStyleSheet(build_stylesheet())

        # ── Central Widget ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ──
        self._sidebar = SideBar()
        self._sidebar.page_changed.connect(self._navigate_to)
        main_layout.addWidget(self._sidebar)

        # ── Content area ──
        content_area = QWidget()
        content_area.setObjectName("content_area")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top bar with page title
        top_bar = QFrame()
        top_bar.setObjectName("top_bar")
        top_bar.setFixedHeight(56)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(28, 0, 28, 0)

        self._page_title = QLabel("Dashboard")
        self._page_title.setObjectName("page_title")
        top_bar_layout.addWidget(self._page_title)

        top_bar_layout.addStretch()

        self._data_indicator = QLabel("")
        self._data_indicator.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_XS}px;"
        )
        top_bar_layout.addWidget(self._data_indicator)

        content_layout.addWidget(top_bar)

        # Stacked pages
        self._stack = QStackedWidget()
        self._pages = {}

        self._add_page("dashboard",      DashboardPage(self._state))
        self._add_page("import",         DataImportPage(self._state))
        self._add_page("explorer",       DataExplorerPage(self._state))
        self._add_page("agents",         AgentPage(self._state))
        self._add_page("cleaning",       CleaningPage(self._state))
        self._add_page("visualization",  VisualizationPage(self._state))
        self._add_page("ml_training",    MLTrainingPage(self._state))
        self._add_page("explainability", ExplainabilityPage(self._state))
        self._add_page("settings",       SettingsPage(self._state))

        content_layout.addWidget(self._stack)
        main_layout.addWidget(content_area)

        # ── Status Bar ──
        status = QStatusBar()
        status.showMessage("  ◆ NeuroviaI v2.0.0  |  Ready")
        self.setStatusBar(status)

        # ── Wire signals ──
        self._connect_signals()

        # Start on dashboard
        self._navigate_to("dashboard")

    def _add_page(self, key: str, widget: QWidget):
        self._pages[key] = widget
        self._stack.addWidget(widget)

    def _navigate_to(self, key: str):
        if key in self._pages:
            self._stack.setCurrentWidget(self._pages[key])
            self._page_title.setText(PAGE_TITLES.get(key, key.title()))

            # Refresh dashboard metrics when navigating to it
            if key == "dashboard":
                self._pages["dashboard"].refresh_metrics()

    def _connect_signals(self):
        imp = self._pages["import"]
        dash = self._pages["dashboard"]
        explorer = self._pages["explorer"]
        clean = self._pages["cleaning"]
        viz = self._pages["visualization"]
        ml = self._pages["ml_training"]
        explain = self._pages["explainability"]

        # Data loaded → propagate to all pages
        imp.data_loaded.connect(self._on_data_loaded)

        # Cleaning done → update downstream
        clean.data_cleaned.connect(self._on_data_cleaned)

        # Dashboard quick action buttons
        dash.btn_import.clicked.connect(lambda: self._sidebar.navigate_to("import"))
        dash.btn_analyze.clicked.connect(lambda: self._sidebar.navigate_to("explorer"))
        dash.btn_clean.clicked.connect(lambda: self._sidebar.navigate_to("cleaning"))
        dash.btn_train.clicked.connect(lambda: self._sidebar.navigate_to("ml_training"))

        # Explorer empty state button
        if hasattr(explorer._empty, "action_button"):
            explorer._empty.action_button.clicked.connect(
                lambda: self._sidebar.navigate_to("import")
            )

    def _on_data_loaded(self, df: pd.DataFrame, filepath: str):
        self._state["rows_processed"] = (
            self._state.get("rows_processed", 0) + len(df)
        )
        total_cells = df.shape[0] * df.shape[1]
        quality = (1 - df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 100
        self._state["avg_quality"] = quality

        # Update data indicator
        self._data_indicator.setText(
            f"📊 {df.shape[0]:,} rows × {df.shape[1]} cols"
        )

        # Propagate to all data-aware pages
        self._pages["explorer"].set_dataframe(df)
        self._pages["agents"].set_dataframe(df)
        self._pages["cleaning"].set_dataframe(df)
        self._pages["visualization"].set_dataframe(df)
        self._pages["ml_training"].set_dataframe(df)
        self._pages["explainability"].set_dataframe(df)

        # Auto-navigate to explorer
        self._sidebar.navigate_to("explorer")

    def _on_data_cleaned(self, df: pd.DataFrame):
        self._state["current_df"] = df

        total_cells = df.shape[0] * df.shape[1]
        quality = (1 - df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 100
        self._state["avg_quality"] = quality

        self._data_indicator.setText(
            f"📊 {df.shape[0]:,} rows × {df.shape[1]} cols (cleaned)"
        )

        self._pages["explorer"].set_dataframe(df)
        self._pages["agents"].set_dataframe(df)
        self._pages["visualization"].set_dataframe(df)
        self._pages["ml_training"].set_dataframe(df)
        self._pages["explainability"].set_dataframe(df)
