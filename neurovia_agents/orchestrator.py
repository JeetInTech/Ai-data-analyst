"""
Orchestrator Agent — Central coordinator that routes tasks across specialist agents.
Manages the full analysis pipeline: Profile → Clean → Feature → Model → Visualize.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

from neurovia_agents.base_agent import AgentResult, AgentStatus
from neurovia_agents.profiler_agent import ProfilerAgent
from neurovia_agents.cleaner_agent import CleanerAgent
from neurovia_agents.feature_engineer_agent import FeatureEngineerAgent
from neurovia_agents.model_selector_agent import ModelSelectorAgent
from neurovia_agents.visualizer_agent import VisualizerAgent

log = logging.getLogger("neurovia.orchestrator")


@dataclass
class PipelineResult:
    """Full pipeline outcome."""
    success: bool
    stages: dict = field(default_factory=dict)  # stage_name → AgentResult
    summary: str = ""
    total_duration: float = 0.0
    charts: list = field(default_factory=list)
    best_model: str = ""
    context: dict = field(default_factory=dict)


class Orchestrator:
    """
    Coordinates the multi-agent analysis pipeline.
    
    Supports:
      - Full pipeline (profile → clean → features → model → visualize)
      - Individual stage runs
      - Custom pipelines (any subset of stages)
    """

    STAGES = ["profile", "clean", "feature_engineer", "model_select", "visualize"]

    def __init__(self):
        self._agents = {
            "profile": ProfilerAgent(),
            "clean": CleanerAgent(),
            "feature_engineer": FeatureEngineerAgent(),
            "model_select": ModelSelectorAgent(),
            "visualize": VisualizerAgent(),
        }
        self._status_callback: Optional[callable] = None

    def on_status_change(self, callback: callable):
        """
        Register a callback for pipeline status updates.
        callback(stage_name, agent_name, status, detail)
        """
        self._status_callback = callback
        for name, agent in self._agents.items():
            agent.on_status_change(
                lambda a_name, status, detail, s=name: self._emit(s, a_name, status, detail)
            )

    def _emit(self, stage: str, agent_name: str, status: AgentStatus, detail: str):
        if self._status_callback:
            self._status_callback(stage, agent_name, status, detail)

    def run_full_pipeline(self, df: pd.DataFrame, target_column: str = "",
                          stages: Optional[list[str]] = None) -> PipelineResult:
        """
        Run the full or partial analysis pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Target column for ML (optional)
            stages: List of stage names to run. None = all stages.
        
        Returns:
            PipelineResult with all outcomes
        """
        start = time.time()
        stages = stages or self.STAGES

        context = {
            "dataframe": df.copy(),
            "target_column": target_column,
            "original_shape": df.shape,
        }

        result = PipelineResult(success=True, context=context)

        for stage_name in stages:
            if stage_name not in self._agents:
                log.warning(f"Unknown stage: {stage_name}")
                continue

            log.info(f"Starting stage: {stage_name}")
            self._emit(stage_name, self._agents[stage_name].name, AgentStatus.THINKING, "Starting...")

            agent = self._agents[stage_name]
            stage_result = agent.run(context)
            result.stages[stage_name] = stage_result

            if not stage_result.success:
                log.error(f"Stage {stage_name} failed: {stage_result.summary}")
                result.success = False
                result.summary = f"Pipeline failed at stage '{stage_name}': {stage_result.summary}"
                break

            # Pass results forward in context
            if stage_name == "profile":
                context["profile"] = stage_result.output
            elif stage_name == "model_select":
                result.best_model = (stage_result.output or {}).get("best_model", "")

            log.info(f"Stage {stage_name} completed in {stage_result.duration:.1f}s via {stage_result.llm_provider}")

        result.charts = context.get("charts", [])
        result.total_duration = time.time() - start

        if result.success:
            summaries = []
            for s, r in result.stages.items():
                summaries.append(f"**{s.replace('_', ' ').title()}**: {r.summary}")
            result.summary = "\n".join(summaries)

        return result

    def run_single_stage(self, stage_name: str, df: pd.DataFrame,
                         target_column: str = "", extra_context: dict = None) -> AgentResult:
        """Run a single agent stage."""
        if stage_name not in self._agents:
            return AgentResult(success=False, output=None,
                               summary=f"Unknown stage: {stage_name}")

        context = {
            "dataframe": df.copy(),
            "target_column": target_column,
            **(extra_context or {}),
        }
        return self._agents[stage_name].run(context)

    def ask(self, df: pd.DataFrame, question: str, target_column: str = "") -> AgentResult:
        """
        Handle a natural-language question about the data.
        Uses direct LLM call for conversational Q&A, falls back to local analysis.
        Routes specific task keywords to specialist agents.
        """
        q = question.lower().strip()
        start = time.time()

        # Route specific task keywords to specialist agents
        if any(w in q for w in ["clean my", "clean the", "clean data", "run clean",
                                 "impute", "remove duplicates", "fix missing"]):
            return self._run_with_fallback("clean", df, question, target_column)
        elif any(w in q for w in ["run feature", "engineer feature", "encode columns",
                                   "scale data", "normalize", "one-hot"]):
            return self._run_with_fallback("feature_engineer", df, question, target_column)
        elif any(w in q for w in ["train model", "run model", "fit model", "run prediction",
                                   "train and compare", "best model"]):
            return self._run_with_fallback("model_select", df, question, target_column)
        elif any(w in q for w in ["run pipeline", "full pipeline", "run all"]):
            return self._run_with_fallback("profile", df, question, target_column)

        # For all other questions: direct LLM Q&A
        return self._ask_direct(df, question, target_column)

    def _run_with_fallback(self, stage: str, df: pd.DataFrame,
                           question: str, target_column: str) -> AgentResult:
        """Run a specialist agent with fallback to local analysis."""
        result = self.run_single_stage(stage, df, target_column)
        if result.success and "max iterations" not in (result.summary or "").lower():
            return result
        log.warning(f"Agent stage '{stage}' failed or unhelpful, falling back to local")
        return self._answer_locally(df, question, target_column)

    def _ask_direct(self, df: pd.DataFrame, question: str,
                    target_column: str = "") -> AgentResult:
        """
        Answer questions with a direct LLM call — no complex agent tool loop.
        Builds a data summary as context and lets the LLM respond naturally.
        """
        start = time.time()

        # Build concise data context
        data_summary = self._build_data_summary(df, target_column)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are NeuroviaI, an AI data analytics assistant. "
                    "You help users understand and analyze their data. "
                    "Answer clearly and concisely. Use plain language, not markdown. "
                    "If the question is about the dataset, base your answer on the data summary provided. "
                    "If asked a general question, answer it helpfully. "
                    "Keep responses focused and under 200 words unless more detail is needed."
                ),
            },
            {
                "role": "user",
                "content": f"Dataset Summary:\n{data_summary}\n\nUser Question: {question}",
            },
        ]

        try:
            from neurovia_agents.llm_client import get_llm_client
            client = get_llm_client()
            response = client.complete(messages=messages, temperature=0.4, max_tokens=1024)
            return AgentResult(
                success=True,
                output={"direct_answer": True},
                summary=response.content,
                agent_name="NeuroviaI",
                duration=time.time() - start,
                llm_provider=response.provider,
            )
        except RuntimeError:
            log.warning("Direct LLM call failed, falling back to local analysis")
            return self._answer_locally(df, question, target_column)

    def _build_data_summary(self, df: pd.DataFrame, target_column: str = "") -> str:
        """Build a concise text summary of the dataset for LLM context."""
        lines = []
        lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        lines.append(f"Columns: {', '.join(df.columns[:20])}")
        if target_column:
            lines.append(f"Target column: {target_column}")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        lines.append(f"Numeric columns ({len(num_cols)}): {', '.join(num_cols[:10])}")
        lines.append(f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}")

        # Missing values
        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        lines.append(f"Missing values: {total_missing} total ({total_missing/df.size*100:.1f}%)")

        # Key stats for numeric columns
        if num_cols:
            desc = df[num_cols[:6]].describe().round(2).to_string()
            lines.append(f"Numeric Stats:\n{desc}")

        # Categorical value counts
        for col in cat_cols[:3]:
            vc = df[col].value_counts().head(5)
            vals = ", ".join(f"{k}: {v}" for k, v in vc.items())
            lines.append(f"{col} top values: {vals}")

        return "\n".join(lines)

    def _answer_locally(self, df: pd.DataFrame, question: str,
                        target_column: str = "") -> AgentResult:
        """
        Answer questions using local pandas analysis when LLM providers are unavailable.
        Returns HTML-formatted output.
        """
        start = time.time()
        q = question.lower()
        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        parts = []
        parts.append(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

        # Missing data
        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        if total_missing > 0:
            pct = total_missing / df.size * 100
            parts.append(f"Missing Values: {total_missing:,} cells ({pct:.1f}%)")
            worst = missing[missing > 0].sort_values(ascending=False).head(5)
            for col, cnt in worst.items():
                parts.append(f"  {col}: {cnt:,} ({cnt / len(df) * 100:.1f}%)")
        else:
            parts.append("Missing Values: None")

        # Duplicates
        dup_count = int(df.duplicated().sum())
        parts.append(f"Duplicate Rows: {dup_count:,}")

        # Numeric summary
        if num_cols:
            parts.append(f"\nNumeric Columns ({len(num_cols)}):")
            for col in num_cols[:8]:
                s = df[col].dropna()
                if len(s) > 0:
                    parts.append(
                        f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
                        f"range=[{s.min():.2f}, {s.max():.2f}]"
                    )

        # Categorical summary
        if cat_cols:
            parts.append(f"\nCategorical Columns ({len(cat_cols)}):")
            for col in cat_cols[:8]:
                n_unique = df[col].nunique()
                top_val = df[col].mode().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
                parts.append(f"  {col}: {n_unique} unique, top: '{top_val}'")

        # Correlations
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    r = corr.iloc[i, j]
                    if abs(r) > 0.5:
                        pairs.append((num_cols[i], num_cols[j], r))
            if pairs:
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                parts.append("\nStrong Correlations (|r| > 0.5):")
                for c1, c2, r in pairs[:5]:
                    parts.append(f"  {c1} <> {c2}: r={r:.3f}")

        parts.append("\n(LLM providers temporarily unavailable - showing local analysis)")

        return AgentResult(
            success=True,
            output={"local_analysis": True},
            summary="\n".join(parts),
            agent_name="NeuroviaI",
            duration=time.time() - start,
            llm_provider="local",
        )
