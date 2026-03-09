"""
NeuroviaI Sidebar Navigation Widget
Vertical sidebar with brand logo, navigation items, and settings.
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QPushButton, QLabel,
    QSpacerItem, QSizePolicy, QWidget, QHBoxLayout,
)
from PySide6.QtCore import Signal, Qt, QPointF
from PySide6.QtGui import QFont, QPainter, QPainterPath, QLinearGradient, QColor, QPen, QRadialGradient

import math

from neurovia_ui.theme import Colors, Fonts


class LogoWidget(QWidget):
    """Small hexagonal neural logo painted via QPainter."""

    def __init__(self, size: int = 32, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._size = size

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = cy = self._size / 2
        r = self._size * 0.4

        # Glow
        glow = QRadialGradient(QPointF(cx, cy), r * 1.6)
        glow.setColorAt(0, QColor(16, 185, 129, 35))
        glow.setColorAt(1, QColor(16, 185, 129, 0))
        p.setBrush(glow)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r * 1.6, r * 1.6)

        # Hexagon
        path = QPainterPath()
        for i in range(6):
            angle = math.radians(60 * i - 90)
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        path.closeSubpath()

        grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
        grad.setColorAt(0, QColor("#10B981"))
        grad.setColorAt(1, QColor("#06B6D4"))
        p.setPen(QPen(QColor("#10B981"), 1.5))
        p.setBrush(grad)
        p.drawPath(path)

        # Center dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#060A13"))
        p.drawEllipse(QPointF(cx, cy), r * 0.22, r * 0.22)
        p.setBrush(QColor("#10B981"))
        p.drawEllipse(QPointF(cx, cy), r * 0.12, r * 0.12)
        p.end()


NAV_ITEMS = [
    ("dashboard",      "◆  Dashboard"),
    ("import",         "⊕  Import Data"),
    ("explorer",       "⊙  Data Explorer"),
    ("agents",         "⬡  AI Agents"),
    ("cleaning",       "✦  Cleaning"),
    ("visualization",  "▣  Visualization"),
    ("ml_training",    "◈  ML Training"),
    ("explainability", "☀  Explainability"),
]

BOTTOM_ITEMS = [
    ("settings",       "⚙  Settings"),
]


class SideBar(QFrame):
    """Left-hand navigation sidebar."""

    page_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(250)

        self._buttons: dict[str, QPushButton] = {}
        self._active_key: str = "dashboard"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Brand header ──
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 28, 24, 20)
        header_layout.setSpacing(4)

        # Logo + brand text row
        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        logo = LogoWidget(32)
        brand_row.addWidget(logo)

        brand = QLabel("NEUROVIA·I")
        brand_font = QFont(Fonts.FAMILY, Fonts.SIZE_LG, QFont.Weight.Bold)
        brand.setFont(brand_font)
        brand.setStyleSheet(f"color: {Colors.PRIMARY}; background: transparent;")
        brand_row.addWidget(brand)
        brand_row.addStretch()
        header_layout.addLayout(brand_row)

        subtitle = QLabel("Data Analytics Platform")
        sub_font = QFont(Fonts.FAMILY, Fonts.SIZE_XS)
        subtitle.setFont(sub_font)
        subtitle.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
        header_layout.addWidget(subtitle)

        layout.addWidget(header)

        # ── Separator ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {Colors.SIDEBAR_BORDER}; max-height: 1px;")
        layout.addWidget(sep)

        layout.addSpacing(8)

        # ── Navigation items ──
        for key, label in NAV_ITEMS:
            btn = self._make_nav_button(key, label)
            layout.addWidget(btn)

        layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        # ── Bottom separator ──
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {Colors.SIDEBAR_BORDER}; max-height: 1px;")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        for key, label in BOTTOM_ITEMS:
            btn = self._make_nav_button(key, label)
            layout.addWidget(btn)

        layout.addSpacing(16)

        # ── Version label ──
        version = QLabel("  v2.0.0")
        version.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: 9px; background: transparent; padding-left: 20px;"
        )
        layout.addWidget(version)
        layout.addSpacing(8)

        # Set initial active state
        self._set_active("dashboard")

    def _make_nav_button(self, key: str, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("active", False)
        btn.setFixedHeight(44)
        btn.clicked.connect(lambda checked, k=key: self._on_click(k))
        self._buttons[key] = btn
        return btn

    def _on_click(self, key: str):
        self._set_active(key)
        self.page_changed.emit(key)

    def _set_active(self, key: str):
        # Deactivate previous
        if self._active_key in self._buttons:
            old = self._buttons[self._active_key]
            old.setProperty("active", False)
            old.style().polish(old)

        # Activate new
        self._active_key = key
        if key in self._buttons:
            new = self._buttons[key]
            new.setProperty("active", True)
            new.style().polish(new)

    def navigate_to(self, key: str):
        """Programmatically navigate to a page."""
        self._on_click(key)
