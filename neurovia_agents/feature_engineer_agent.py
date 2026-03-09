"""
Feature Engineer Agent — Creates and selects features for ML modelling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from neurovia_agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are the **Feature Engineer Agent** for NeuroVia, an advanced analytics platform.

Your expertise:
- Feature creation (interactions, polynomials, aggregations, date parts)
- Encoding categorical variables (label, one-hot, target encoding)
- Feature selection (variance threshold, correlation filtering)
- Scaling and transformation (log, sqrt, standard, minmax)
- Creating train-ready feature matrices

You receive cleaned data and a target column hint. Your job is to create the best possible
feature set for ML modelling. Be smart about which features to create — don't explode dimensionality
unnecessarily. Focus on features that will have predictive power."""


def _tool_encode_categoricals(context: dict, method: str = "label", max_cardinality: int = 20, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded = {}

    for col in cat_cols:
        nunique = df[col].nunique()
        if method == "onehot" and nunique <= max_cardinality:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoded[col] = {"method": "onehot", "new_columns": list(dummies.columns)}
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoded[col] = {"method": "label", "classes": len(le.classes_)}

    context["dataframe"] = df
    return {"encoded_columns": encoded, "total_columns": len(df.columns)}


def _tool_create_interactions(context: dict, top_n: int = 3, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return {"message": "Not enough numeric columns for interactions"}

    # Only create interactions between top correlated pairs
    corr = df[num_cols].corr().abs()
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if pd.notna(corr.iloc[i, j]):
                pairs.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    created = []
    for a, b, _ in pairs[:top_n]:
        name = f"{a}_x_{b}"
        df[name] = df[a] * df[b]
        created.append(name)

    context["dataframe"] = df
    return {"created_features": created}


def _tool_create_date_features(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    created = {}
    for col in dt_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df = df.drop(columns=[col])
        created[col] = [f"{col}_year", f"{col}_month", f"{col}_dayofweek"]
    context["dataframe"] = df
    return {"date_features": created}


def _tool_log_transform(context: dict, columns: list = None, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    if columns is None:
        # Auto-detect skewed columns
        num_cols = df.select_dtypes(include="number").columns
        columns = [c for c in num_cols if df[c].skew() > 2 and (df[c] > 0).all()]

    transformed = []
    for col in columns:
        if col in df.columns and (df[col] > 0).all():
            df[f"{col}_log"] = np.log1p(df[col])
            transformed.append(f"{col}_log")
    context["dataframe"] = df
    return {"transformed": transformed}


def _tool_drop_low_variance(context: dict, threshold: float = 0.01, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_cols = df.select_dtypes(include="number").columns
    variances = df[num_cols].var()
    low_var = variances[variances < threshold].index.tolist()
    if low_var:
        df = df.drop(columns=low_var)
        context["dataframe"] = df
    return {"dropped": low_var, "threshold": threshold}


def _tool_drop_high_correlation(context: dict, threshold: float = 0.95, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 2:
        return {"dropped": []}
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        df = df.drop(columns=to_drop)
        context["dataframe"] = df
    return {"dropped": to_drop, "threshold": threshold}


def _tool_get_feature_summary(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    return {
        "total_features": len(df.columns),
        "numeric": len(df.select_dtypes(include="number").columns),
        "non_numeric": len(df.select_dtypes(exclude="number").columns),
        "columns": list(df.columns),
        "shape": list(df.shape),
    }


class FeatureEngineerAgent(BaseAgent):
    """Creates and selects features for ML modelling."""

    def __init__(self):
        super().__init__(
            name="FeatureEngineer",
            role="Feature engineering and selection specialist",
            system_prompt=SYSTEM_PROMPT,
            max_iterations=6,
        )

    def _register_tools(self):
        self.register_tool("encode_categoricals", _tool_encode_categoricals,
                           "Encode categorical columns (label or onehot)",
                           {"type": "object", "properties": {
                               "method": {"type": "string", "enum": ["label", "onehot"]},
                               "max_cardinality": {"type": "integer", "default": 20}}})
        self.register_tool("create_interactions", _tool_create_interactions,
                           "Create interaction features between top correlated numeric pairs",
                           {"type": "object", "properties": {"top_n": {"type": "integer", "default": 3}}})
        self.register_tool("create_date_features", _tool_create_date_features,
                           "Extract year/month/dayofweek from datetime columns",
                           {"type": "object", "properties": {}})
        self.register_tool("log_transform", _tool_log_transform,
                           "Apply log1p transform to skewed numeric columns",
                           {"type": "object", "properties": {"columns": {"type": "array", "items": {"type": "string"}}}})
        self.register_tool("drop_low_variance", _tool_drop_low_variance,
                           "Drop numeric columns with variance below threshold",
                           {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.01}}})
        self.register_tool("drop_high_correlation", _tool_drop_high_correlation,
                           "Drop one column from highly correlated pairs",
                           {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.95}}})
        self.register_tool("get_feature_summary", _tool_get_feature_summary,
                           "Get current feature set summary",
                           {"type": "object", "properties": {}})

    def _build_task_prompt(self, context: dict) -> str:
        df: pd.DataFrame = context["dataframe"]
        target = context.get("target_column", "")
        profile = context.get("profile", {})

        return f"""Engineer features for ML modelling.

## Dataset
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Target column: {target or 'Not specified'}
- Numeric: {len(df.select_dtypes(include='number').columns)}
- Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}
- Datetime: {len(df.select_dtypes(include='datetime').columns)}

## Instructions
1. First use get_feature_summary to understand current state
2. Encode categorical variables appropriately
3. Extract date features if datetime columns exist
4. Create interaction features if beneficial
5. Apply log transform to highly skewed columns
6. Remove low variance and highly correlated features
7. FINISH with the final feature list and what was done"""
