"""
NeuroviaI Data Analytics Platform — Entry Point
Launch the modern PySide6 desktop application.
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QLinearGradient

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    from PySide6.QtGui import QPainterPath, QRadialGradient, QPen
    from PySide6.QtCore import QPointF

    # ── Splash Screen ──
    splash_pix = QPixmap(560, 340)
    splash_pix.fill(QColor("#060A13"))

    painter = QPainter(splash_pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # ── Logo: geometric hexagon with neural nodes ──
    cx, cy = 80, 90
    # Outer glow
    glow = QRadialGradient(QPointF(cx, cy), 50)
    glow.setColorAt(0, QColor(16, 185, 129, 40))
    glow.setColorAt(1, QColor(16, 185, 129, 0))
    painter.setBrush(glow)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QPointF(cx, cy), 50, 50)

    # Hexagon
    import math
    hex_path = QPainterPath()
    r = 28
    for i in range(6):
        angle = math.radians(60 * i - 90)
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        if i == 0:
            hex_path.moveTo(px, py)
        else:
            hex_path.lineTo(px, py)
    hex_path.closeSubpath()

    hex_grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
    hex_grad.setColorAt(0, QColor("#10B981"))
    hex_grad.setColorAt(1, QColor("#06B6D4"))
    painter.setPen(QPen(QColor("#10B981"), 2))
    painter.setBrush(hex_grad)
    painter.drawPath(hex_path)

    # Inner neural nodes
    painter.setPen(Qt.PenStyle.NoPen)
    nodes = [(cx, cy), (cx - 10, cy - 12), (cx + 10, cy - 12),
             (cx - 10, cy + 12), (cx + 10, cy + 12)]
    for i, (nx, ny) in enumerate(nodes):
        painter.setBrush(QColor("#060A13") if i == 0 else QColor("#0F172A"))
        painter.drawEllipse(QPointF(nx, ny), 5 if i == 0 else 3, 5 if i == 0 else 3)

    # Node connections
    painter.setPen(QPen(QColor("#10B981"), 1))
    for nx, ny in nodes[1:]:
        painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    # Center dot
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#10B981"))
    painter.drawEllipse(QPointF(cx, cy), 3, 3)

    # Brand text
    painter.setPen(QColor("#10B981"))
    painter.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
    painter.drawText(125, 100, "NEUROVIA·I")

    # Gradient accent line
    grad = QLinearGradient(60, 130, 500, 130)
    grad.setColorAt(0, QColor("#10B981"))
    grad.setColorAt(0.5, QColor("#06B6D4"))
    grad.setColorAt(1, QColor("#3B82F6"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(grad)
    painter.drawRect(60, 120, 440, 2)

    painter.setPen(QColor("#94A3B8"))
    painter.setFont(QFont("Segoe UI", 13))
    painter.drawText(125, 150, "AI-Powered Data Analytics Platform")

    painter.setPen(QColor("#64748B"))
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(60, 220, "Initializing neural agents...")
    painter.drawText(60, 300, "v2.0.0")

    painter.end()

    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    # ── Load main window (imports may take a moment) ──
    from neurovia_ui.main_window import NeuroViaMainWindow

    window = NeuroViaMainWindow()

    # Close splash and show main window
    QTimer.singleShot(1500, lambda: (splash.close(), window.show()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
