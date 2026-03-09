"""
Profiler Agent — Analyzes dataset structure, quality, and statistical properties.
First agent in the pipeline: produces a comprehensive data profile.
"""

import pandas as pd
import numpy as np

from neurovia_agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are the **Data Profiler Agent** for NeuroVia, an advanced analytics platform.

Your expertise:
- Statistical profiling of datasets (distributions, central tendency, dispersion)
- Data quality assessment (missing values, duplicates, type consistency)
- Column-level analysis (cardinality, uniqueness, patterns)  
- Detecting potential issues (high cardinality, skewed distributions, constant columns)

You receive a dataset summary and use your tools to build a comprehensive profile.
Be precise with numbers and percentages. Focus on actionable insights."""


def _tool_get_shape(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    return {"rows": len(df), "columns": len(df.columns), "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2)}


def _tool_get_dtypes(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    dtype_map = df.dtypes.astype(str).to_dict()
    return {"dtypes": dtype_map, "numeric_count": len(df.select_dtypes(include="number").columns),
            "categorical_count": len(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_count": len(df.select_dtypes(include="datetime").columns)}


def _tool_get_missing(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    cols_with_missing = {col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                         for col in df.columns if missing[col] > 0}
    return {"total_missing_cells": int(missing.sum()),
            "total_cells": int(df.size),
            "overall_missing_pct": round(missing.sum() / df.size * 100, 2),
            "columns_with_missing": cols_with_missing}


def _tool_get_statistics(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        return {"numeric_stats": {}, "message": "No numeric columns found"}

    stats = {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        stats[col] = {
            "mean": round(float(s.mean()), 4) if len(s) else None,
            "median": round(float(s.median()), 4) if len(s) else None,
            "std": round(float(s.std()), 4) if len(s) > 1 else None,
            "min": round(float(s.min()), 4) if len(s) else None,
            "max": round(float(s.max()), 4) if len(s) else None,
            "skewness": round(float(s.skew()), 4) if len(s) > 2 else None,
            "kurtosis": round(float(s.kurtosis()), 4) if len(s) > 3 else None,
            "zeros_pct": round(float((s == 0).sum() / len(s) * 100), 2) if len(s) else 0,
        }
    return {"numeric_stats": stats}


def _tool_get_categorical_stats(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    cat_df = df.select_dtypes(include=["object", "category"])
    if cat_df.empty:
        return {"categorical_stats": {}, "message": "No categorical columns found"}

    stats = {}
    for col in cat_df.columns:
        s = cat_df[col].dropna()
        vc = s.value_counts()
        stats[col] = {
            "unique_count": int(s.nunique()),
            "cardinality_pct": round(s.nunique() / max(len(s), 1) * 100, 2),
            "top_values": vc.head(5).to_dict(),
            "avg_length": round(float(s.astype(str).str.len().mean()), 1) if len(s) else 0,
        }
    return {"categorical_stats": stats}


def _tool_get_duplicates(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    dup_count = int(df.duplicated().sum())
    return {"duplicate_rows": dup_count,
            "duplicate_pct": round(dup_count / max(len(df), 1) * 100, 2),
            "total_rows": len(df)}


def _tool_get_correlations(context: dict, top_n: int = 10, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return {"correlations": [], "message": "Need at least 2 numeric columns"}

    corr = num_df.corr()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append({"col_a": corr.columns[i], "col_b": corr.columns[j],
                              "correlation": round(float(r), 4)})
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {"top_correlations": pairs[:top_n]}


def _tool_get_sample(context: dict, n: int = 5, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    sample = df.head(min(n, len(df)))
    return {"sample": sample.to_dict(orient="records"), "columns": list(df.columns)}


class ProfilerAgent(BaseAgent):
    """Analyzes dataset structure, quality, and statistical properties."""

    def __init__(self):
        super().__init__(
            name="Profiler",
            role="Data profiling and quality assessment specialist",
            system_prompt=SYSTEM_PROMPT,
        )

    def _register_tools(self):
        self.register_tool("get_shape", _tool_get_shape,
                           "Get dataset dimensions and memory usage",
                           {"type": "object", "properties": {}})
        self.register_tool("get_dtypes", _tool_get_dtypes,
                           "Get column data types and type counts",
                           {"type": "object", "properties": {}})
        self.register_tool("get_missing", _tool_get_missing,
                           "Get missing value analysis per column",
                           {"type": "object", "properties": {}})
        self.register_tool("get_statistics", _tool_get_statistics,
                           "Get descriptive statistics for numeric columns",
                           {"type": "object", "properties": {}})
        self.register_tool("get_categorical_stats", _tool_get_categorical_stats,
                           "Get statistics for categorical columns",
                           {"type": "object", "properties": {}})
        self.register_tool("get_duplicates", _tool_get_duplicates,
                           "Check for duplicate rows",
                           {"type": "object", "properties": {}})
        self.register_tool("get_correlations", _tool_get_correlations,
                           "Get top correlations between numeric columns",
                           {"type": "object", "properties": {"top_n": {"type": "integer", "default": 10}}})
        self.register_tool("get_sample", _tool_get_sample,
                           "Get sample rows from the dataset",
                           {"type": "object", "properties": {"n": {"type": "integer", "default": 5}}})

    def _build_task_prompt(self, context: dict) -> str:
        df: pd.DataFrame = context["dataframe"]
        return f"""Analyze this dataset comprehensively:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}

Use your tools to gather information, then produce a complete data profile.
Start with get_shape and get_dtypes, then investigate further based on what you find.
When finished, use FINISH with a structured result containing:
- shape, dtypes, missing_summary, quality_score (0-100), issues, recommendations"""
