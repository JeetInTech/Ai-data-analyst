"""
Model Selector Agent — Chooses, trains, and evaluates ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from neurovia_agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are the **Model Selector Agent** for NeuroVia, an advanced analytics platform.

Your expertise:
- Detecting task type (classification vs regression)
- Selecting appropriate models based on dataset characteristics
- Training and evaluating models with cross-validation
- Comparing model performance
- Recommending the best model with explanation

You'll decide which models to try based on dataset size, feature count, and task type.
Be efficient — don't try all models if the dataset characteristics clearly favor certain algorithms."""


CLASSIFIERS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "knn": lambda: KNeighborsClassifier(),
    "svm": lambda: SVC(random_state=42),
}

REGRESSORS = {
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(random_state=42),
    "lasso": lambda: Lasso(random_state=42),
    "knn": lambda: KNeighborsRegressor(),
    "svr": lambda: SVR(),
}


def _detect_task_type(y: pd.Series) -> str:
    if y.dtype in ["object", "category", "bool"]:
        return "classification"
    if y.nunique() <= 20 and y.nunique() / max(len(y), 1) < 0.05:
        return "classification"
    return "regression"


def _tool_detect_task(context: dict, target: str = "", **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    target = target or context.get("target_column", "")
    if not target or target not in df.columns:
        return {"error": f"Target column '{target}' not found. Available: {list(df.columns)}"}

    y = df[target]
    task = _detect_task_type(y)
    return {
        "target": target,
        "task_type": task,
        "unique_values": int(y.nunique()),
        "dtype": str(y.dtype),
        "sample_values": y.value_counts().head(5).to_dict() if task == "classification" else {
            "mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4),
            "min": round(float(y.min()), 4), "max": round(float(y.max()), 4)
        },
    }


def _tool_train_model(context: dict, model_name: str = "random_forest",
                      target: str = "", cv_folds: int = 5, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    target = target or context.get("target_column", "")
    if not target or target not in df.columns:
        return {"error": f"Target '{target}' not found"}

    y = df[target]
    X = df.drop(columns=[target])

    # Only numeric features
    X = X.select_dtypes(include="number")
    if X.empty:
        return {"error": "No numeric features available for training"}

    # Drop any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    task = _detect_task_type(y)
    models = CLASSIFIERS if task == "classification" else REGRESSORS

    if model_name not in models:
        return {"error": f"Unknown model '{model_name}'. Available: {list(models.keys())}"}

    model = models[model_name]()
    scoring = "f1_weighted" if task == "classification" else "r2"
    folds = min(cv_folds, max(2, len(X) // 10))

    try:
        scores = cross_val_score(model, X, y, cv=folds, scoring=scoring, n_jobs=-1)
        # Train final model on full data
        model.fit(X, y)
        context.setdefault("trained_models", {})[model_name] = model
        context["feature_columns"] = list(X.columns)

        return {
            "model": model_name,
            "task_type": task,
            "metric": scoring,
            "cv_scores": [round(float(s), 4) for s in scores],
            "mean_score": round(float(scores.mean()), 4),
            "std_score": round(float(scores.std()), 4),
            "features_used": len(X.columns),
            "samples_used": len(X),
        }
    except Exception as e:
        return {"error": f"Training {model_name} failed: {str(e)}"}


def _tool_compare_models(context: dict, **kwargs) -> dict:
    results = context.get("model_results", [])
    if not results:
        return {"error": "No model results to compare. Train models first."}
    sorted_results = sorted(results, key=lambda x: x.get("mean_score", 0), reverse=True)
    return {"ranking": sorted_results, "best_model": sorted_results[0]["model"]}


def _tool_get_feature_importance(context: dict, model_name: str = "", **kwargs) -> dict:
    trained = context.get("trained_models", {})
    if model_name and model_name in trained:
        model = trained[model_name]
    elif trained:
        model_name = list(trained.keys())[0]
        model = trained[model_name]
    else:
        return {"error": "No trained models available"}

    features = context.get("feature_columns", [])
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return {"model": model_name,
                "feature_importance": [{"feature": f, "importance": round(float(i), 4)} for f, i in feat_imp[:20]]}
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).ravel() if model.coef_.ndim > 1 else np.abs(model.coef_)
        feat_imp = sorted(zip(features, coefs), key=lambda x: x[1], reverse=True)
        return {"model": model_name,
                "feature_importance": [{"feature": f, "importance": round(float(i), 4)} for f, i in feat_imp[:20]]}
    return {"error": f"Model {model_name} doesn't support feature importance"}


def _tool_list_available_models(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    target = context.get("target_column", "")
    if target and target in df.columns:
        task = _detect_task_type(df[target])
    else:
        task = "unknown"
    models = CLASSIFIERS if task == "classification" else REGRESSORS
    return {"task_type": task, "available_models": list(models.keys())}


class ModelSelectorAgent(BaseAgent):
    """Selects, trains, and compares ML models."""

    def __init__(self):
        super().__init__(
            name="ModelSelector",
            role="ML model selection, training, and evaluation specialist",
            system_prompt=SYSTEM_PROMPT,
            max_iterations=10,
        )

    def _register_tools(self):
        self.register_tool("detect_task", _tool_detect_task,
                           "Detect whether the task is classification or regression",
                           {"type": "object", "properties": {"target": {"type": "string"}}})
        self.register_tool("list_available_models", _tool_list_available_models,
                           "List available models for the detected task type",
                           {"type": "object", "properties": {}})
        self.register_tool("train_model", _tool_train_model,
                           "Train and evaluate a model with cross-validation",
                           {"type": "object", "properties": {
                               "model_name": {"type": "string"},
                               "target": {"type": "string"},
                               "cv_folds": {"type": "integer", "default": 5}}})
        self.register_tool("compare_models", _tool_compare_models,
                           "Compare all trained models and rank them",
                           {"type": "object", "properties": {}})
        self.register_tool("get_feature_importance", _tool_get_feature_importance,
                           "Get feature importance from a trained model",
                           {"type": "object", "properties": {"model_name": {"type": "string"}}})

    def _build_task_prompt(self, context: dict) -> str:
        df: pd.DataFrame = context["dataframe"]
        target = context.get("target_column", "")

        return f"""Select and train the best ML model for this dataset.

## Dataset
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Target: {target or 'Not specified — detect the most likely target column'}
- Numeric features: {len(df.select_dtypes(include='number').columns)}

## Instructions
1. Detect the task type (classification/regression)
2. List available models
3. Train at least 3 different models
4. Store each result — after training each model, remember the scores
5. Compare and rank results
6. Get feature importance from the best model
7. FINISH with: best model name, performance metrics, feature importance, and recommendations"""

    def run(self, context: dict):
        # Initialize model results storage
        context.setdefault("model_results", [])

        # Wrap train_model to auto-collect results
        original_train = self._tools["train_model"]["function"]
        def _train_and_store(ctx, **kw):
            result = original_train(ctx, **kw)
            if "error" not in result:
                ctx.setdefault("model_results", []).append(result)
            return result
        self._tools["train_model"]["function"] = _train_and_store

        return super().run(context)
