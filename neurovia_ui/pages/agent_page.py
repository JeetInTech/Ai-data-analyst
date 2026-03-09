"""
NeuroviaI Agent Page — Chat-based interface for the multi-agent intelligence system.
Users can run the full analysis pipeline or ask specific questions.
"""

import base64
import json
import traceback
from io import BytesIO

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QScrollArea, QFrame, QComboBox,
    QSizePolicy, QProgressBar, QApplication,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QTimer
from PySide6.QtGui import QFont, QPixmap, QImage

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, ActionButton, EmptyState


# ─────────────────────── Worker Thread ───────────────────────

class AgentWorkerSignals(QObject):
    status_update = Signal(str, str, str, str)   # stage, agent, status, detail
    finished = Signal(object)                      # PipelineResult or AgentResult
    error = Signal(str)


class AgentWorker(QThread):
    """Runs agent pipeline in a background thread."""

    def __init__(self, mode: str, df: pd.DataFrame, target: str = "",
                 question: str = "", stages: list = None):
        super().__init__()
        self.signals = AgentWorkerSignals()
        self.mode = mode       # "pipeline" | "ask" | "single"
        self.df = df
        self.target = target
        self.question = question
        self.stages = stages

    def run(self):
        try:
            from neurovia_agents.orchestrator import Orchestrator
            orch = Orchestrator()
            orch.on_status_change(
                lambda stage, agent, status, detail: self.signals.status_update.emit(
                    stage, agent, str(status), detail
                )
            )

            if self.mode == "pipeline":
                result = orch.run_full_pipeline(self.df, self.target, self.stages)
            elif self.mode == "ask":
                result = orch.ask(self.df, self.question, self.target)
            elif self.mode == "single" and self.stages:
                result = orch.run_single_stage(self.stages[0], self.df, self.target)
            else:
                result = orch.run_full_pipeline(self.df, self.target)

            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ─────────────────────── Chat Bubble ───────────────────────

import re as _re

def _md_to_html(text: str) -> str:
    """Convert simple markdown to HTML for QLabel RichText."""
    # Escape HTML entities already in text (but keep existing HTML tags)
    # Convert **bold** to <b>bold</b>
    text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Convert *italic* to <i>italic</i>
    text = _re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Convert `code` to styled code
    text = _re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Convert _italic_ (but not __double__)
    text = _re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'<i>\1</i>', text)
    # Convert newlines to <br>
    text = text.replace('\n', '<br>')
    # Convert bullet points
    text = text.replace('<br>  • ', '<br>&nbsp;&nbsp;• ')
    return text


class ChatBubble(QFrame):
    """A single message bubble in the agent chat."""

    def __init__(self, sender: str, text: str, is_agent: bool = True,
                 accent: str = Colors.PRIMARY, parent=None):
        super().__init__(parent)
        self.setObjectName("chat_bubble")

        align = Qt.AlignmentFlag.AlignLeft if is_agent else Qt.AlignmentFlag.AlignRight

        bg = Colors.BG_CARD if is_agent else Colors.PRIMARY + "22"
        border_color = Colors.BORDER if is_agent else Colors.PRIMARY
        self.setStyleSheet(f"""
            QFrame#chat_bubble {{
                background: {bg};
                border: 1px solid {border_color};
                border-radius: 12px;
                padding: 0px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # Sender label
        sender_lbl = QLabel(sender)
        sender_lbl.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XS, QFont.Weight.Bold))
        sender_lbl.setStyleSheet(f"color: {accent}; background: transparent;")
        layout.addWidget(sender_lbl, alignment=align)

        # Message text — convert markdown to HTML
        html_text = _md_to_html(text)
        msg = QLabel(html_text)
        msg.setWordWrap(True)
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; font-size: {Fonts.SIZE_MD}px; line-height: 1.6;")
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg.setMinimumWidth(200)
        layout.addWidget(msg, alignment=align)

    def add_image(self, base64_png: str):
        """Attach a chart image to this bubble."""
        raw = base64.b64decode(base64_png)
        pixmap = QPixmap()
        pixmap.loadFromData(raw)
        if pixmap.width() > 600:
            pixmap = pixmap.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setStyleSheet("background: transparent; border-radius: 6px;")
        self.layout().addWidget(img_label)


# ─────────────────────── Agent Page ───────────────────────

class AgentPage(QWidget):
    """Interactive multi-agent intelligence page."""

    data_loaded = Signal(pd.DataFrame, str)

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self._state = state
        self._df: pd.DataFrame | None = None
        self._worker: AgentWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(16)

        # ── Header ──
        header = SectionHeader("AI Agent System", "Multi-agent intelligence pipeline powered by LLMs")
        layout.addWidget(header)

        # ── Controls row ──
        controls = QHBoxLayout()
        controls.setSpacing(12)

        # Pipeline mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Full Pipeline",
            "Profile Only",
            "Clean Only",
            "Feature Engineering",
            "Model Selection",
            "Visualization",
        ])
        self._mode_combo.setFixedHeight(38)
        self._mode_combo.setFixedWidth(200)
        controls.addWidget(self._mode_combo)

        # Target column selector
        target_label = QLabel("Target:")
        target_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;")
        controls.addWidget(target_label)
        self._target_combo = QComboBox()
        self._target_combo.setFixedHeight(38)
        self._target_combo.setFixedWidth(200)
        self._target_combo.addItem("(auto-detect)")
        controls.addWidget(self._target_combo)

        controls.addStretch()

        # Run button
        self._run_btn = ActionButton("▶  Run Agent Pipeline", Colors.PRIMARY)
        self._run_btn.setFixedHeight(40)
        self._run_btn.setFixedWidth(220)
        self._run_btn.clicked.connect(self._on_run_pipeline)
        controls.addWidget(self._run_btn)

        layout.addLayout(controls)

        # ── Progress bar ──
        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_DARK};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {Colors.PRIMARY}, stop:1 {Colors.ACCENT});
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self._progress)

        # ── Status label ──
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-size: {Fonts.SIZE_SM}px; font-weight: 600;"
        )
        layout.addWidget(self._status_label)

        # ── Chat area (scrollable) ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
        """)

        self._chat_container = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(12)
        self._chat_layout.addStretch()

        scroll.setWidget(self._chat_container)
        self._scroll = scroll
        layout.addWidget(scroll)

        # ── Input row ──
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask a question about your data...")
        self._input.setFixedHeight(44)
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 10px;
                padding: 0 16px;
                color: {Colors.TEXT_PRIMARY};
                font-size: {Fonts.SIZE_SM}px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.PRIMARY};
            }}
        """)
        self._input.returnPressed.connect(self._on_ask)
        input_row.addWidget(self._input)

        self._ask_btn = ActionButton("Ask ↵", Colors.SECONDARY)
        self._ask_btn.setFixedHeight(44)
        self._ask_btn.setFixedWidth(90)
        self._ask_btn.clicked.connect(self._on_ask)
        input_row.addWidget(self._ask_btn)

        layout.addLayout(input_row)

        # ── Empty state ──
        self._empty = EmptyState(
            icon="◇",
            message="Import data first, then run the AI agent pipeline to automatically\n"
                    "profile, clean, engineer features, train models, and visualize your data.",
            action_text="Import Data",
        )
        layout.addWidget(self._empty)
        scroll.setVisible(False)

    def set_dataframe(self, df: pd.DataFrame):
        """Called when data is loaded or cleaned."""
        self._df = df
        self._empty.setVisible(False)
        self._scroll.setVisible(True)

        # Update target column combo
        self._target_combo.clear()
        self._target_combo.addItem("(auto-detect)")
        for col in df.columns:
            self._target_combo.addItem(col)

        self._add_system_message(
            f"Data loaded: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns. "
            f"Ready for analysis."
        )

    def _add_chat_bubble(self, sender: str, text: str, is_agent: bool = True,
                         accent: str = Colors.PRIMARY) -> ChatBubble:
        # Insert before the stretch
        idx = self._chat_layout.count() - 1
        bubble = ChatBubble(sender, text, is_agent, accent)
        self._chat_layout.insertWidget(idx, bubble)
        # Scroll to bottom
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))
        return bubble

    def _add_system_message(self, text: str):
        self._add_chat_bubble("◆ NeuroviaI System", text, True, Colors.ACCENT)

    def _on_run_pipeline(self):
        if self._df is None:
            self._add_system_message("⚠ No data loaded. Please import data first.")
            return
        if self._worker and self._worker.isRunning():
            self._add_system_message("⚠ Agent pipeline is already running.")
            return

        mode_text = self._mode_combo.currentText()
        mode_map = {
            "Full Pipeline": ("pipeline", None),
            "Profile Only": ("single", ["profile"]),
            "Clean Only": ("single", ["clean"]),
            "Feature Engineering": ("single", ["feature_engineer"]),
            "Model Selection": ("single", ["model_select"]),
            "Visualization": ("single", ["visualize"]),
        }
        mode, stages = mode_map.get(mode_text, ("pipeline", None))

        target = self._target_combo.currentText()
        if target == "(auto-detect)":
            target = ""

        self._add_chat_bubble("You", f"▶ Run **{mode_text}**" + (f" · Target: `{target}`" if target else ""),
                              is_agent=False, accent=Colors.SECONDARY)

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status_label.setText("Starting agent pipeline...")

        self._worker = AgentWorker(mode, self._df, target, stages=stages)
        self._worker.signals.status_update.connect(self._on_status_update)
        self._worker.signals.finished.connect(self._on_pipeline_finished)
        self._worker.signals.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_ask(self):
        question = self._input.text().strip()
        if not question:
            return
        if self._df is None:
            self._add_system_message("⚠ No data loaded. Please import data first.")
            return
        if self._worker and self._worker.isRunning():
            self._add_system_message("⚠ Agent is busy. Please wait.")
            return

        self._input.clear()
        self._add_chat_bubble("You", question, is_agent=False, accent=Colors.SECONDARY)

        target = self._target_combo.currentText()
        if target == "(auto-detect)":
            target = ""

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status_label.setText("Thinking...")

        self._worker = AgentWorker("ask", self._df, target, question=question)
        self._worker.signals.status_update.connect(self._on_status_update)
        self._worker.signals.finished.connect(self._on_pipeline_finished)
        self._worker.signals.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_status_update(self, stage: str, agent: str, status: str, detail: str):
        self._status_label.setText(f"⟐ {agent} — {detail}")

    def _on_pipeline_finished(self, result):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("")

        from neurovia_agents.orchestrator import PipelineResult
        from neurovia_agents.base_agent import AgentResult

        if isinstance(result, PipelineResult):
            self._display_pipeline_result(result)
        elif isinstance(result, AgentResult):
            self._display_agent_result(result)

    def _on_pipeline_error(self, error_msg: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("")
        self._add_chat_bubble("⚠ Error", error_msg, True, "#EF4444")

    def _display_pipeline_result(self, result):
        from neurovia_agents.orchestrator import PipelineResult

        # Show summary for each stage
        for stage_name, stage_result in result.stages.items():
            title = stage_name.replace("_", " ").title()
            summary = stage_result.summary or "Completed"
            provider = stage_result.llm_provider
            duration = f"{stage_result.duration:.1f}s"
            text = (
                f"<b>{title}</b> — {summary}<br>"
                f"<span style='color:{Colors.TEXT_MUTED}; font-size:11px;'>"
                f"via {provider} · {duration}</span>"
            )

            colors = {
                "profile": Colors.ACCENT,
                "clean": Colors.PRIMARY,
                "feature_engineer": "#8B5CF6",
                "model_select": Colors.SECONDARY,
                "visualize": "#F59E0B",
            }
            self._add_chat_bubble(
                f"Agent: {stage_result.agent_name}",
                text, True,
                colors.get(stage_name, Colors.PRIMARY),
            )

        # Show charts
        if result.charts:
            chart_bubble = self._add_chat_bubble(
                "Visualizer Agent",
                f"Generated **{len(result.charts)}** charts:",
                True, "#F59E0B",
            )
            for chart in result.charts:
                if "image" in chart:
                    chart_bubble.add_image(chart["image"])

        # Overall summary
        if result.best_model:
            self._add_system_message(f"✓ Best model: **{result.best_model}** · Total time: {result.total_duration:.1f}s")
        else:
            self._add_system_message(f"✓ Pipeline complete · {result.total_duration:.1f}s")

    def _display_agent_result(self, result):
        summary = result.summary or "Completed"
        provider = result.llm_provider
        duration = f"{result.duration:.1f}s"
        text = (
            f"{summary}<br>"
            f"<span style='color:{Colors.TEXT_MUTED}; font-size:11px;'>"
            f"via {provider} · {duration}</span>"
        )
        self._add_chat_bubble(
            f"Agent: {result.agent_name}",
            text, True, Colors.PRIMARY,
        )

        # If there are charts in the output
        if isinstance(result.output, dict) and "charts" in result.output:
            for chart in result.output["charts"]:
                if "image" in chart:
                    bubble = self._add_chat_bubble("Chart", "", True, "#F59E0B")
                    bubble.add_image(chart["image"])
