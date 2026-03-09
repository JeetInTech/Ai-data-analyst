"""
Cleaner Agent — Intelligently cleans data based on the Profiler's findings.
Applies cleaning operations via tools and explains each decision.
"""

import pandas as pd
import numpy as np

from neurovia_agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are the **Data Cleaner Agent** for NeuroVia, an advanced analytics platform.

Your expertise:
- Missing value imputation (mean/median/mode/drop based on context)
- Duplicate removal
- Outlier detection and handling (IQR, z-score)
- Data type correction (strings to numbers, date parsing)
- Text normalization (strip, lowercase, consistent encoding)
- Removing constant/near-constant columns

You receive a data profile from the Profiler agent and decide which cleaning steps to apply.
Always explain WHY you chose each operation. Be conservative — don't remove data unnecessarily.
Prefer imputation over deletion when possible."""


def _tool_remove_duplicates(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    before = len(df)
    df_clean = df.drop_duplicates()
    removed = before - len(df_clean)
    context["dataframe"] = df_clean
    return {"removed": removed, "remaining_rows": len(df_clean)}


def _tool_drop_missing_columns(context: dict, threshold: float = 0.7, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        context["dataframe"] = df
    return {"dropped_columns": cols_to_drop, "threshold": threshold, "remaining_columns": len(df.columns)}


def _tool_impute_numeric(context: dict, strategy: str = "median", **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_cols = df.select_dtypes(include="number").columns
    imputed = {}
    for col in num_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            if strategy == "median":
                val = df[col].median()
            elif strategy == "mean":
                val = df[col].mean()
            elif strategy == "zero":
                val = 0
            else:
                val = df[col].median()
            df[col] = df[col].fillna(val)
            imputed[col] = {"filled": n_missing, "value": round(float(val), 4)}
    context["dataframe"] = df
    return {"strategy": strategy, "imputed_columns": imputed}


def _tool_impute_categorical(context: dict, strategy: str = "mode", **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    imputed = {}
    for col in cat_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            if strategy == "mode":
                mode_val = df[col].mode()
                val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            elif strategy == "unknown":
                val = "Unknown"
            else:
                val = "Unknown"
            df[col] = df[col].fillna(val)
            imputed[col] = {"filled": n_missing, "value": val}
    context["dataframe"] = df
    return {"strategy": strategy, "imputed_columns": imputed}


def _tool_handle_outliers(context: dict, method: str = "iqr", factor: float = 1.5, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_cols = df.select_dtypes(include="number").columns
    capped = {}
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
        else:  # zscore
            mean, std = s.mean(), s.std()
            lower, upper = mean - factor * std, mean + factor * std
        outliers_below = int((df[col] < lower).sum())
        outliers_above = int((df[col] > upper).sum())
        if outliers_below + outliers_above > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            capped[col] = {"below": outliers_below, "above": outliers_above,
                           "lower_bound": round(float(lower), 4), "upper_bound": round(float(upper), 4)}
    context["dataframe"] = df
    return {"method": method, "capped_columns": capped}


def _tool_normalize_text(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    cat_cols = df.select_dtypes(include=["object"]).columns
    normalized = []
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        normalized.append(col)
    context["dataframe"] = df
    return {"normalized_columns": normalized}


def _tool_drop_constant_columns(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        context["dataframe"] = df
    return {"dropped_columns": constant_cols, "remaining_columns": len(df.columns)}


def _tool_fix_dtypes(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    fixed = {}
    for col in df.select_dtypes(include=["object"]).columns:
        # Try numeric conversion
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0.5 * df[col].notna().sum():
            df[col] = converted
            fixed[col] = "numeric"
            continue
        # Try datetime conversion
        try:
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if dt.notna().sum() > 0.5 * df[col].notna().sum():
                df[col] = dt
                fixed[col] = "datetime"
        except Exception:
            pass
    context["dataframe"] = df
    return {"fixed_columns": fixed}


def _tool_get_current_state(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / max(df.size, 1) * 100, 2),
        "duplicates": int(df.duplicated().sum()),
    }


class CleanerAgent(BaseAgent):
    """Cleans data intelligently based on profiler findings."""

    def __init__(self):
        super().__init__(
            name="Cleaner",
            role="Data cleaning and preprocessing specialist",
            system_prompt=SYSTEM_PROMPT,
            max_iterations=8,
        )

    def _register_tools(self):
        self.register_tool("remove_duplicates", _tool_remove_duplicates,
                           "Remove duplicate rows", {"type": "object", "properties": {}})
        self.register_tool("drop_missing_columns", _tool_drop_missing_columns,
                           "Drop columns with missing values above threshold",
                           {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.7}}})
        self.register_tool("impute_numeric", _tool_impute_numeric,
                           "Impute missing numeric values",
                           {"type": "object", "properties": {"strategy": {"type": "string", "enum": ["median", "mean", "zero"]}}})
        self.register_tool("impute_categorical", _tool_impute_categorical,
                           "Impute missing categorical values",
                           {"type": "object", "properties": {"strategy": {"type": "string", "enum": ["mode", "unknown"]}}})
        self.register_tool("handle_outliers", _tool_handle_outliers,
                           "Cap outliers using IQR or z-score method",
                           {"type": "object", "properties": {"method": {"type": "string", "enum": ["iqr", "zscore"]}, "factor": {"type": "number", "default": 1.5}}})
        self.register_tool("normalize_text", _tool_normalize_text,
                           "Strip whitespace and lowercase all text columns",
                           {"type": "object", "properties": {}})
        self.register_tool("drop_constant_columns", _tool_drop_constant_columns,
                           "Remove columns with only one unique value",
                           {"type": "object", "properties": {}})
        self.register_tool("fix_dtypes", _tool_fix_dtypes,
                           "Auto-detect and fix column data types",
                           {"type": "object", "properties": {}})
        self.register_tool("get_current_state", _tool_get_current_state,
                           "Get current state of the dataframe after cleaning",
                           {"type": "object", "properties": {}})

    def _build_task_prompt(self, context: dict) -> str:
        df: pd.DataFrame = context["dataframe"]
        profile = context.get("profile", {})
        profile_str = str(profile)[:2000] if profile else "No profile available"

        return f"""Clean this dataset based on the profiler's findings.

## Current State
- Shape: {df.shape[0]} rows × {df.shape[1]} columns  
- Missing cells: {df.isnull().sum().sum()} ({round(df.isnull().sum().sum() / max(df.size, 1) * 100, 1)}%)
- Duplicates: {df.duplicated().sum()}

## Profiler Report
{profile_str}

Apply cleaning operations in a logical order:
1. Remove duplicates first
2. Drop columns with too much missing data
3. Fix data types
4. Impute remaining missing values
5. Handle outliers if needed
6. Remove constant columns

Use get_current_state between steps to verify progress.
When done, FINISH with a summary of all changes made."""
