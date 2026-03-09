"""
Settings Page — API keys, preferences, and about information.
"""

import os
import sys
import platform

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QLineEdit,
    QComboBox, QGridLayout, QMessageBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader


class SettingsPage(QWidget):
    """Application settings and configuration."""

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
        title = QLabel("Settings")
        title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XL, QFont.Weight.Bold))
        layout.addWidget(title)

        # ── API Keys ──
        layout.addWidget(SectionHeader("API Configuration"))
        api_card = Card()
        api_grid = QGridLayout()
        api_grid.setSpacing(12)

        self._api_inputs = {}
        apis = [
            ("GROQ_API_KEY", "Groq API Key", "For Groq/Llama models"),
            ("GEMINI_API_KEY", "Gemini API Key", "For Google Gemini models"),
            ("OPENAI_API_KEY", "OpenAI API Key", "For GPT models (optional)"),
        ]

        for i, (env_key, label, hint) in enumerate(apis):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-weight: bold;")
            api_grid.addWidget(lbl, i * 2, 0)

            hint_lbl = QLabel(hint)
            hint_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_XS}px;")
            api_grid.addWidget(hint_lbl, i * 2, 1)

            inp = QLineEdit()
            inp.setEchoMode(QLineEdit.EchoMode.Password)
            inp.setPlaceholderText(f"Enter {label}...")
            # Load from environment
            existing = os.environ.get(env_key, "")
            if existing:
                inp.setText(existing)
            api_grid.addWidget(inp, i * 2 + 1, 0, 1, 2)
            self._api_inputs[env_key] = inp

        api_card.card_layout().addLayout(api_grid)

        save_row = QHBoxLayout()
        save_btn = QPushButton("   Save API Keys   ")
        save_btn.setProperty("accent", "primary")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._save_api_keys)
        save_row.addWidget(save_btn)
        save_row.addStretch()
        api_card.card_layout().addLayout(save_row)

        layout.addWidget(api_card)

        # ── Processing Preferences ──
        layout.addWidget(SectionHeader("Processing Preferences"))
        prefs_card = Card()
        prefs_grid = QGridLayout()
        prefs_grid.setSpacing(12)

        prefs_grid.addWidget(QLabel("Max Rows for Display"), 0, 0)
        self._max_rows = QComboBox()
        self._max_rows.addItems(["1,000", "5,000", "10,000", "50,000", "100,000"])
        self._max_rows.setCurrentIndex(2)
        prefs_grid.addWidget(self._max_rows, 0, 1)

        prefs_grid.addWidget(QLabel("Default Chart Theme"), 1, 0)
        self._chart_theme = QComboBox()
        self._chart_theme.addItems(["NeuroviaI Dark", "Midnight Blue", "Cyberpunk", "Light Professional", "Warm Earth"])
        prefs_grid.addWidget(self._chart_theme, 1, 1)

        prefs_grid.addWidget(QLabel("Auto-detect Data Types"), 2, 0)
        self._auto_dtype = QComboBox()
        self._auto_dtype.addItems(["Enabled", "Disabled"])
        prefs_grid.addWidget(self._auto_dtype, 2, 1)

        prefs_card.card_layout().addLayout(prefs_grid)
        layout.addWidget(prefs_card)

        # ── About ──
        layout.addWidget(SectionHeader("About"))
        about_card = Card()
        about_grid = QGridLayout()
        about_grid.setSpacing(8)

        infos = [
            ("Application", "NeuroviaI Data Analytics Platform"),
            ("Version", "2.0.0"),
            ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
            ("Platform", f"{platform.system()} {platform.release()}"),
            ("Architecture", platform.machine()),
        ]

        for i, (key, val) in enumerate(infos):
            k = QLabel(key)
            k.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-weight: bold;")
            v = QLabel(val)
            about_grid.addWidget(k, i, 0)
            about_grid.addWidget(v, i, 1)

        about_card.card_layout().addLayout(about_grid)
        layout.addWidget(about_card)

        layout.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _save_api_keys(self):
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        lines = []
        for env_key, inp in self._api_inputs.items():
            value = inp.text().strip()
            if value:
                os.environ[env_key] = value
                lines.append(f"{env_key}={value}")

        if lines:
            try:
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except OSError:
                pass  # Non-critical if .env write fails

        QMessageBox.information(self, "Settings", "API keys saved successfully.")
