"""
Dashboard Page — Landing screen with metrics, quick actions, and system overview.
"""

import psutil
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import MetricCard, SectionHeader, Card, ActionButton


class DashboardPage(QWidget):
    """Main dashboard with metrics overview and quick actions."""

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state

        # Scroll wrapper
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(32, 28, 32, 28)
        self._layout.setSpacing(24)

        self._build_header()
        self._build_metrics()
        self._build_quick_actions()
        self._build_system_status()

        self._layout.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Auto-refresh system stats every 3 seconds
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_system_stats)
        self._timer.start(3000)

    # ── Sections ──

    def _build_header(self):
        welcome = QLabel("Welcome back")
        welcome.setProperty("subheading", True)
        welcome.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self._layout.addWidget(welcome)

        title = QLabel("Analytics Dashboard")
        title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_HERO, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        self._layout.addWidget(title)

        desc = QLabel("Monitor your data analysis sessions, model performance, and system health.")
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SM}px;")
        self._layout.addWidget(desc)

    def _build_metrics(self):
        self._layout.addWidget(SectionHeader("Overview"))

        grid = QHBoxLayout()
        grid.setSpacing(16)

        self._card_sessions = MetricCard("Sessions", "0", "", Colors.PRIMARY)
        self._card_models = MetricCard("Models Trained", "0", "", Colors.SECONDARY)
        self._card_rows = MetricCard("Rows Processed", "0", "", Colors.ACCENT)
        self._card_quality = MetricCard("Avg Quality", "—", "", Colors.WARNING)

        for card in [self._card_sessions, self._card_models,
                     self._card_rows, self._card_quality]:
            grid.addWidget(card)

        self._layout.addLayout(grid)

    def _build_quick_actions(self):
        self._layout.addWidget(SectionHeader("Quick Actions"))

        row = QHBoxLayout()
        row.setSpacing(16)

        self.btn_import = ActionButton("⊕", "Import Dataset", "primary")
        self.btn_analyze = ActionButton("⊙", "Auto Analyze", "secondary")
        self.btn_clean = ActionButton("✦", "Clean Data")
        self.btn_train = ActionButton("◈", "Train Models")

        for btn in [self.btn_import, self.btn_analyze, self.btn_clean, self.btn_train]:
            row.addWidget(btn)

        row.addStretch()
        self._layout.addLayout(row)

    def _build_system_status(self):
        self._layout.addWidget(SectionHeader("System Resources"))

        card = Card()
        grid = QGridLayout()
        grid.setSpacing(16)

        self._cpu_bar = self._make_stat_row("CPU", grid, 0)
        self._ram_bar = self._make_stat_row("Memory", grid, 1)
        self._disk_bar = self._make_stat_row("Disk", grid, 2)

        card.card_layout().addLayout(grid)
        self._layout.addWidget(card)

        self._update_system_stats()

    def _make_stat_row(self, label: str, grid: QGridLayout, row: int):
        lbl = QLabel(label)
        lbl.setFixedWidth(80)
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-weight: bold;")
        grid.addWidget(lbl, row, 0)

        bar_bg = QFrame()
        bar_bg.setFixedHeight(10)
        bar_bg.setStyleSheet(
            f"background: {Colors.BG_ELEVATED}; border-radius: 5px;"
        )
        bar_bg.setMinimumWidth(200)

        bar_fill = QFrame(bar_bg)
        bar_fill.setFixedHeight(10)
        bar_fill.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {Colors.PRIMARY}, stop:1 {Colors.SECONDARY});"
            f"border-radius: 5px;"
        )
        bar_fill.setFixedWidth(0)

        grid.addWidget(bar_bg, row, 1)

        pct_label = QLabel("0%")
        pct_label.setFixedWidth(50)
        pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pct_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_XS}px;")
        grid.addWidget(pct_label, row, 2)

        return (bar_bg, bar_fill, pct_label)

    # ── Updates ──

    def _update_system_stats(self):
        try:
            cpu = psutil.cpu_percent(interval=0)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
        except Exception:
            cpu = ram = disk = 0

        for pct, (bar_bg, bar_fill, pct_label) in [
            (cpu, self._cpu_bar), (ram, self._ram_bar), (disk, self._disk_bar)
        ]:
            width = max(1, int(bar_bg.width() * pct / 100))
            bar_fill.setFixedWidth(width)
            pct_label.setText(f"{pct:.0f}%")

    def refresh_metrics(self):
        """Update metric cards from app state."""
        s = self._state
        self._card_sessions.set_value(str(s.get("sessions", 0)))
        self._card_models.set_value(str(s.get("models_trained", 0)))
        rows = s.get("rows_processed", 0)
        if rows > 1_000_000:
            self._card_rows.set_value(f"{rows / 1_000_000:.1f}M")
        elif rows > 1_000:
            self._card_rows.set_value(f"{rows / 1_000:.1f}K")
        else:
            self._card_rows.set_value(str(rows))
        quality = s.get("avg_quality")
        if quality is not None:
            self._card_quality.set_value(f"{quality:.0f}%")
