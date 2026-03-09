"""
NeuroVia Theme System
Luxury dark theme with emerald green, electric blue, and deep black tones.
"""


class Colors:
    """Neurovia brand color palette"""

    # Background hierarchy (darkest to lightest)
    BG_DARKEST    = "#060A13"
    BG_DARK       = "#0B1120"
    BG_BASE       = "#0F172A"
    BG_CARD       = "#1E293B"
    BG_ELEVATED   = "#273548"
    BG_HOVER      = "#334155"
    BG_INPUT      = "#151E2E"

    # Brand primaries
    PRIMARY       = "#10B981"   # Emerald green
    PRIMARY_HOVER = "#059669"
    PRIMARY_LIGHT = "#34D399"
    PRIMARY_DIM   = "#064E3B"

    SECONDARY       = "#3B82F6"  # Electric blue
    SECONDARY_HOVER = "#2563EB"
    SECONDARY_LIGHT = "#60A5FA"
    SECONDARY_DIM   = "#1E3A5F"

    ACCENT       = "#06B6D4"   # Cyan
    ACCENT_HOVER = "#0891B2"

    # Text
    TEXT_PRIMARY   = "#F1F5F9"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED     = "#64748B"
    TEXT_ON_PRIMARY = "#FFFFFF"

    # Borders
    BORDER         = "#1E293B"
    BORDER_LIGHT   = "#334155"
    BORDER_FOCUS   = "#3B82F6"

    # Semantic
    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    ERROR   = "#EF4444"
    INFO    = "#3B82F6"

    # Sidebar
    SIDEBAR_BG     = "#070D19"
    SIDEBAR_HOVER  = "#111C2E"
    SIDEBAR_ACTIVE = "#0C2A3D"
    SIDEBAR_BORDER = "#152035"


class Fonts:
    FAMILY      = "Segoe UI"
    SIZE_XS     = 10
    SIZE_SM     = 11
    SIZE_MD     = 13
    SIZE_LG     = 16
    SIZE_XL     = 20
    SIZE_XXL    = 28
    SIZE_HERO   = 36


def build_stylesheet() -> str:
    """Generate the complete QSS stylesheet for the application."""
    c = Colors
    return f"""
    /* ===== GLOBAL ===== */
    QMainWindow {{
        background-color: {c.BG_DARK};
    }}
    QWidget {{
        color: {c.TEXT_PRIMARY};
        font-family: "{Fonts.FAMILY}";
        font-size: {Fonts.SIZE_SM}px;
    }}
    QWidget:disabled {{
        color: {c.TEXT_MUTED};
    }}

    /* ===== SCROLL AREA ===== */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}

    /* ===== LABELS ===== */
    QLabel {{
        background: transparent;
        border: none;
        padding: 0px;
    }}
    QLabel[heading="true"] {{
        font-size: {Fonts.SIZE_XL}px;
        font-weight: bold;
        color: {c.TEXT_PRIMARY};
    }}
    QLabel[subheading="true"] {{
        font-size: {Fonts.SIZE_MD}px;
        color: {c.TEXT_SECONDARY};
    }}
    QLabel[muted="true"] {{
        color: {c.TEXT_MUTED};
        font-size: {Fonts.SIZE_XS}px;
    }}

    /* ===== PUSH BUTTONS ===== */
    QPushButton {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 10px 20px;
        font-size: {Fonts.SIZE_SM}px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {c.BG_ELEVATED};
        border-color: {c.BORDER_LIGHT};
    }}
    QPushButton:pressed {{
        background-color: {c.BG_HOVER};
    }}
    QPushButton[accent="primary"] {{
        background-color: {c.PRIMARY};
        color: {c.TEXT_ON_PRIMARY};
        border: none;
        font-weight: bold;
    }}
    QPushButton[accent="primary"]:hover {{
        background-color: {c.PRIMARY_HOVER};
    }}
    QPushButton[accent="secondary"] {{
        background-color: {c.SECONDARY};
        color: {c.TEXT_ON_PRIMARY};
        border: none;
        font-weight: bold;
    }}
    QPushButton[accent="secondary"]:hover {{
        background-color: {c.SECONDARY_HOVER};
    }}
    QPushButton[accent="danger"] {{
        background-color: {c.ERROR};
        color: {c.TEXT_ON_PRIMARY};
        border: none;
        font-weight: bold;
    }}
    QPushButton[accent="ghost"] {{
        background-color: transparent;
        border: none;
        color: {c.TEXT_SECONDARY};
    }}
    QPushButton[accent="ghost"]:hover {{
        color: {c.TEXT_PRIMARY};
        background-color: {c.BG_CARD};
    }}

    /* ===== LINE EDIT / TEXT EDIT ===== */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 10px 14px;
        color: {c.TEXT_PRIMARY};
        font-size: {Fonts.SIZE_SM}px;
        selection-background-color: {c.SECONDARY_DIM};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {c.BORDER_FOCUS};
    }}
    QLineEdit:hover, QTextEdit:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QLineEdit[error="true"] {{
        border-color: {c.ERROR};
    }}

    /* ===== COMBO BOX ===== */
    QComboBox {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 10px 14px;
        color: {c.TEXT_PRIMARY};
        min-width: 120px;
    }}
    QComboBox:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QComboBox:focus {{
        border-color: {c.BORDER_FOCUS};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {c.TEXT_SECONDARY};
        margin-right: 10px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 4px;
        selection-background-color: {c.SECONDARY_DIM};
        outline: none;
    }}

    /* ===== SPIN BOX ===== */
    QSpinBox, QDoubleSpinBox {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 8px 12px;
        color: {c.TEXT_PRIMARY};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {c.BORDER_FOCUS};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background: transparent;
        border: none;
        width: 20px;
    }}

    /* ===== CHECK BOX / RADIO ===== */
    QCheckBox {{
        spacing: 10px;
        color: {c.TEXT_PRIMARY};
    }}
    QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        border-radius: 4px;
        border: 2px solid {c.BORDER_LIGHT};
        background-color: {c.BG_INPUT};
    }}
    QCheckBox::indicator:hover {{
        border-color: {c.PRIMARY};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c.PRIMARY};
        border-color: {c.PRIMARY};
    }}
    QRadioButton {{
        spacing: 10px;
    }}
    QRadioButton::indicator {{
        width: 20px;
        height: 20px;
        border-radius: 10px;
        border: 2px solid {c.BORDER_LIGHT};
        background-color: {c.BG_INPUT};
    }}
    QRadioButton::indicator:checked {{
        background-color: {c.PRIMARY};
        border-color: {c.PRIMARY};
    }}

    /* ===== TABLE VIEW ===== */
    QTableView {{
        background-color: {c.BG_DARK};
        alternate-background-color: {c.BG_BASE};
        gridline-color: {c.BORDER};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        selection-background-color: {c.SECONDARY_DIM};
        selection-color: {c.TEXT_PRIMARY};
    }}
    QTableView::item {{
        padding: 8px 12px;
        border: none;
    }}
    QTableView::item:selected {{
        background-color: {c.SECONDARY_DIM};
    }}
    QTableView::item:hover {{
        background-color: {c.BG_CARD};
    }}
    QHeaderView {{
        background-color: transparent;
    }}
    QHeaderView::section {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_SECONDARY};
        font-weight: bold;
        font-size: {Fonts.SIZE_XS}px;
        text-transform: uppercase;
        border: none;
        border-bottom: 2px solid {c.BORDER};
        border-right: 1px solid {c.BORDER};
        padding: 10px 12px;
    }}
    QHeaderView::section:hover {{
        color: {c.TEXT_PRIMARY};
        background-color: {c.BG_ELEVATED};
    }}

    /* ===== TAB WIDGET ===== */
    QTabWidget::pane {{
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        background: {c.BG_DARK};
        top: -1px;
    }}
    QTabBar {{
        background: transparent;
    }}
    QTabBar::tab {{
        background: {c.BG_CARD};
        color: {c.TEXT_SECONDARY};
        padding: 10px 24px;
        border: 1px solid {c.BORDER};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 2px;
        font-weight: 500;
    }}
    QTabBar::tab:hover {{
        color: {c.TEXT_PRIMARY};
        background: {c.BG_ELEVATED};
    }}
    QTabBar::tab:selected {{
        background: {c.BG_DARK};
        color: {c.PRIMARY};
        border-bottom: 2px solid {c.PRIMARY};
    }}

    /* ===== PROGRESS BAR ===== */
    QProgressBar {{
        background-color: {c.BG_CARD};
        border: none;
        border-radius: 6px;
        text-align: center;
        color: {c.TEXT_PRIMARY};
        font-size: {Fonts.SIZE_XS}px;
        font-weight: bold;
        min-height: 12px;
        max-height: 12px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {c.PRIMARY},
            stop:1 {c.SECONDARY}
        );
        border-radius: 6px;
    }}

    /* ===== SCROLLBAR ===== */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {c.BG_HOVER};
        border-radius: 4px;
        min-height: 40px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.TEXT_MUTED};
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        border: none;
        background: transparent;
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c.BG_HOVER};
        border-radius: 4px;
        min-width: 40px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c.TEXT_MUTED};
    }}
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        border: none;
        background: transparent;
        width: 0px;
    }}

    /* ===== SPLITTER ===== */
    QSplitter::handle {{
        background-color: {c.BORDER};
        margin: 2px;
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}

    /* ===== GROUP BOX ===== */
    QGroupBox {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER};
        border-radius: 10px;
        margin-top: 14px;
        padding: 20px 16px 16px 16px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 8px;
        color: {c.TEXT_SECONDARY};
    }}

    /* ===== LIST WIDGET ===== */
    QListWidget {{
        background-color: {c.BG_DARK};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 4px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 10px 14px;
        border-radius: 6px;
        margin: 2px 0;
    }}
    QListWidget::item:hover {{
        background-color: {c.BG_CARD};
    }}
    QListWidget::item:selected {{
        background-color: {c.SECONDARY_DIM};
        color: {c.TEXT_PRIMARY};
    }}

    /* ===== TREE WIDGET ===== */
    QTreeWidget, QTreeView {{
        background-color: {c.BG_DARK};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        outline: none;
    }}
    QTreeWidget::item, QTreeView::item {{
        padding: 6px 8px;
        border-radius: 4px;
    }}
    QTreeWidget::item:hover, QTreeView::item:hover {{
        background-color: {c.BG_CARD};
    }}
    QTreeWidget::item:selected, QTreeView::item:selected {{
        background-color: {c.SECONDARY_DIM};
    }}

    /* ===== TOOL TIP ===== */
    QToolTip {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: {Fonts.SIZE_XS}px;
    }}

    /* ===== STATUS BAR ===== */
    QStatusBar {{
        background-color: {c.BG_DARKEST};
        color: {c.TEXT_MUTED};
        border-top: 1px solid {c.BORDER};
        font-size: {Fonts.SIZE_XS}px;
        padding: 4px 12px;
    }}
    QStatusBar::item {{
        border: none;
    }}

    /* ===== MENU BAR ===== */
    QMenuBar {{
        background-color: {c.BG_DARKEST};
        color: {c.TEXT_SECONDARY};
        border-bottom: 1px solid {c.BORDER};
        padding: 2px;
    }}
    QMenuBar::item {{
        padding: 6px 12px;
        border-radius: 4px;
    }}
    QMenuBar::item:selected {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
    }}
    QMenu {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 6px;
    }}
    QMenu::item {{
        padding: 8px 32px 8px 16px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {c.SECONDARY_DIM};
    }}
    QMenu::separator {{
        height: 1px;
        background: {c.BORDER};
        margin: 4px 8px;
    }}

    /* ===== SLIDER ===== */
    QSlider::groove:horizontal {{
        height: 6px;
        background: {c.BG_CARD};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c.PRIMARY};
        width: 18px;
        height: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {c.PRIMARY_LIGHT};
    }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {c.PRIMARY}, stop:1 {c.SECONDARY}
        );
        border-radius: 3px;
    }}

    /* ===== FRAME ===== */
    QFrame[frameShape="4"] {{
        color: {c.BORDER};
        max-height: 1px;
    }}
    QFrame[frameShape="5"] {{
        color: {c.BORDER};
        max-width: 1px;
    }}

    /* ===== SIDEBAR SPECIFIC ===== */
    #sidebar {{
        background-color: {c.SIDEBAR_BG};
        border-right: 1px solid {c.SIDEBAR_BORDER};
    }}
    #sidebar QPushButton {{
        text-align: left;
        padding: 12px 20px;
        border: none;
        border-radius: 0px;
        color: {c.TEXT_MUTED};
        font-size: {Fonts.SIZE_SM}px;
        font-weight: 500;
        background: transparent;
    }}
    #sidebar QPushButton:hover {{
        background-color: {c.SIDEBAR_HOVER};
        color: {c.TEXT_SECONDARY};
    }}
    #sidebar QPushButton[active="true"] {{
        background-color: {c.SIDEBAR_ACTIVE};
        color: {c.PRIMARY};
        border-left: 3px solid {c.PRIMARY};
        font-weight: bold;
    }}

    /* ===== CONTENT AREA ===== */
    #content_area {{
        background-color: {c.BG_DARK};
    }}
    #top_bar {{
        background-color: {c.BG_DARKEST};
        border-bottom: 1px solid {c.BORDER};
    }}
    #page_title {{
        font-size: {Fonts.SIZE_LG}px;
        font-weight: bold;
        color: {c.TEXT_PRIMARY};
    }}

    /* ===== CARD PANELS ===== */
    #card {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER};
        border-radius: 12px;
    }}
    #card:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    #metric_value {{
        font-size: {Fonts.SIZE_XXL}px;
        font-weight: bold;
        color: {c.TEXT_PRIMARY};
    }}
    #metric_label {{
        font-size: {Fonts.SIZE_XS}px;
        color: {c.TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    #metric_trend_up {{
        color: {c.SUCCESS};
        font-size: {Fonts.SIZE_XS}px;
        font-weight: bold;
    }}
    #metric_trend_down {{
        color: {c.ERROR};
        font-size: {Fonts.SIZE_XS}px;
        font-weight: bold;
    }}

    /* ===== DRAG DROP ZONE ===== */
    #drop_zone {{
        background-color: {c.BG_BASE};
        border: 2px dashed {c.BORDER_LIGHT};
        border-radius: 16px;
    }}
    #drop_zone:hover {{
        border-color: {c.PRIMARY};
        background-color: {c.PRIMARY_DIM};
    }}
    #drop_zone_icon {{
        font-size: 48px;
        color: {c.TEXT_MUTED};
    }}
    #drop_zone_text {{
        font-size: {Fonts.SIZE_MD}px;
        color: {c.TEXT_SECONDARY};
    }}
    #drop_zone_hint {{
        font-size: {Fonts.SIZE_XS}px;
        color: {c.TEXT_MUTED};
    }}

    /* ===== SECTION HEADERS ===== */
    #section_header {{
        font-size: {Fonts.SIZE_MD}px;
        font-weight: bold;
        color: {c.TEXT_PRIMARY};
        padding-bottom: 4px;
    }}

    /* ===== CHIP / TAG ===== */
    #chip {{
        background-color: {c.BG_ELEVATED};
        border: 1px solid {c.BORDER};
        border-radius: 14px;
        padding: 4px 12px;
        font-size: {Fonts.SIZE_XS}px;
        color: {c.TEXT_SECONDARY};
    }}
    #chip_primary {{
        background-color: {c.PRIMARY_DIM};
        border-color: {c.PRIMARY};
        color: {c.PRIMARY_LIGHT};
    }}
    #chip_blue {{
        background-color: {c.SECONDARY_DIM};
        border-color: {c.SECONDARY};
        color: {c.SECONDARY_LIGHT};
    }}
    """
