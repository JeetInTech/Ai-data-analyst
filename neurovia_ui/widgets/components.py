"""
NeuroVia Reusable UI Components
Metric cards, section panels, action buttons, loading spinners, etc.
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QGraphicsDropShadowEffect, QWidget, QPushButton,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property, QSize
from PySide6.QtGui import QFont, QColor, QPainter, QPen

from neurovia_ui.theme import Colors, Fonts


class MetricCard(QFrame):
    """A dashboard metric card showing value, label, and optional trend."""

    def __init__(self, label: str, value: str = "0", trend: str = "",
                 accent: str = Colors.PRIMARY, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(6)

        # Accent bar at top
        accent_bar = QFrame()
        accent_bar.setFixedHeight(3)
        accent_bar.setStyleSheet(
            f"background: {accent}; border-radius: 1px; border: none;"
        )
        layout.addWidget(accent_bar)
        layout.addSpacing(8)

        # Value
        self._value_label = QLabel(value)
        self._value_label.setObjectName("metric_value")
        val_font = QFont(Fonts.FAMILY, Fonts.SIZE_XXL, QFont.Weight.Bold)
        self._value_label.setFont(val_font)
        layout.addWidget(self._value_label)

        # Bottom row: label + trend
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self._label = QLabel(label.upper())
        self._label.setObjectName("metric_label")
        bottom.addWidget(self._label)

        bottom.addStretch()

        if trend:
            is_up = trend.startswith("+") or trend.startswith("↑")
            self._trend = QLabel(trend)
            self._trend.setObjectName("metric_trend_up" if is_up else "metric_trend_down")
            bottom.addWidget(self._trend)

        layout.addLayout(bottom)

    def set_value(self, value: str):
        self._value_label.setText(value)

    def set_trend(self, trend: str):
        if hasattr(self, "_trend"):
            self._trend.setText(trend)


class SectionHeader(QWidget):
    """Section title with optional action button."""

    def __init__(self, title: str, action_text: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 8)

        lbl = QLabel(title)
        lbl.setObjectName("section_header")
        layout.addWidget(lbl)

        layout.addStretch()

        if action_text:
            btn = QPushButton(action_text)
            btn.setProperty("accent", "ghost")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(btn)
            self.action_button = btn


class Card(QFrame):
    """Generic card container with rounded corners and border."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setSpacing(12)

    def card_layout(self) -> QVBoxLayout:
        return self._layout


class ActionButton(QPushButton):
    """Large action button for dashboards and quick actions."""

    def __init__(self, icon_text: str, label: str, accent: str = "primary", parent=None):
        super().__init__(parent)
        self.setText(f"{icon_text}\n{label}")
        self.setProperty("accent", accent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(160, 80)
        self.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_SM, QFont.Weight.DemiBold))


class Chip(QLabel):
    """Small tag / chip label."""

    def __init__(self, text: str, variant: str = "default", parent=None):
        super().__init__(text, parent)
        names = {
            "default": "chip",
            "primary": "chip_primary",
            "blue": "chip_blue",
        }
        self.setObjectName(names.get(variant, "chip"))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(26)


class LoadingSpinner(QWidget):
    """Animated loading spinner using QPainter."""

    def __init__(self, size: int = 40, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._size = size

        self._animation = QPropertyAnimation(self, b"rotation_angle")
        self._animation.setDuration(1000)
        self._animation.setStartValue(0)
        self._animation.setEndValue(360)
        self._animation.setLoopCount(-1)  # Infinite
        self._animation.setEasingCurve(QEasingCurve.Type.Linear)

    def _get_angle(self):
        return self._angle

    def _set_angle(self, val):
        self._angle = val
        self.update()

    rotation_angle = Property(int, _get_angle, _set_angle)

    def start(self):
        self._animation.start()
        self.show()

    def stop(self):
        self._animation.stop()
        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(Colors.BG_CARD), 3)
        painter.setPen(pen)
        painter.drawEllipse(4, 4, self._size - 8, self._size - 8)

        pen_active = QPen(QColor(Colors.PRIMARY), 3)
        pen_active.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_active)
        painter.drawArc(4, 4, self._size - 8, self._size - 8,
                        self._angle * 16, 90 * 16)
        painter.end()


class EmptyState(QWidget):
    """Empty state placeholder with icon, message, and optional action."""

    def __init__(self, icon: str = "📊", message: str = "No data yet",
                 description: str = "", action_text: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon_lbl = QLabel(icon)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(f"font-size: 48px; background: transparent;")
        layout.addWidget(icon_lbl)

        msg = QLabel(message)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_LG, QFont.Weight.DemiBold))
        msg.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(msg)

        if description:
            desc = QLabel(description)
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            desc.setWordWrap(True)
            layout.addWidget(desc)

        if action_text:
            btn = QPushButton(action_text)
            btn.setProperty("accent", "primary")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedWidth(200)
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
            self.action_button = btn
