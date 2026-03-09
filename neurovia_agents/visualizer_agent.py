"""
Visualizer Agent — Creates intelligent visualizations based on data characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

from neurovia_agents.base_agent import BaseAgent

# NeuroVia chart palette
COLORS = ["#10B981", "#3B82F6", "#06B6D4", "#8B5CF6", "#F59E0B",
          "#EF4444", "#EC4899", "#14B8A6", "#6366F1", "#84CC16"]

SYSTEM_PROMPT = """You are the **Visualization Agent** for NeuroVia, an advanced analytics platform.

Your expertise:
- Choosing the right chart type for data characteristics
- Creating publication-quality visualizations
- Distribution analysis, relationship plots, comparison charts
- Correlation heatmaps, feature importance plots
- Auto-detecting the most insightful visualizations

Select chart types based on data types and column relationships.
Numeric vs numeric → scatter. Categorical counts → bar. Distribution → histogram. Time series → line.
Always create visualizations that tell a clear story about the data."""


def _apply_dark_theme(fig, ax):
    """Apply NeuroVia dark theme to matplotlib figure."""
    bg = "#0A0F1C"
    text = "#E2E8F0"
    grid = "#1E293B"
    fig.set_facecolor(bg)
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            a.set_facecolor(bg)
            a.tick_params(colors=text)
            a.xaxis.label.set_color(text)
            a.yaxis.label.set_color(text)
            a.title.set_color(text)
            a.spines["bottom"].set_color(grid)
            a.spines["left"].set_color(grid)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.grid(True, alpha=0.15, color=grid)
    else:
        ax.set_facecolor(bg)
        ax.tick_params(colors=text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)
        ax.spines["bottom"].set_color(grid)
        ax.spines["left"].set_color(grid)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color=grid)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _tool_histogram(context: dict, column: str = "", bins: int = 30, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    if not column or column not in df.columns:
        num_cols = df.select_dtypes(include="number").columns
        column = num_cols[0] if len(num_cols) else ""
    if not column:
        return {"error": "No numeric column found"}

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_theme(fig, ax)
    ax.hist(df[column].dropna(), bins=bins, color=COLORS[0], alpha=0.8, edgecolor=COLORS[0])
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {column}")

    img = _fig_to_base64(fig)
    context.setdefault("charts", []).append({"type": "histogram", "column": column, "image": img})
    return {"chart": "histogram", "column": column, "generated": True}


def _tool_scatter(context: dict, x: str = "", y: str = "", **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not x and len(num_cols) >= 2:
        x, y = num_cols[0], num_cols[1]
    if not x or not y:
        return {"error": "Need 2 numeric columns"}

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_theme(fig, ax)
    ax.scatter(df[x], df[y], c=COLORS[1], alpha=0.5, s=20)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")

    img = _fig_to_base64(fig)
    context.setdefault("charts", []).append({"type": "scatter", "x": x, "y": y, "image": img})
    return {"chart": "scatter", "x": x, "y": y, "generated": True}


def _tool_bar_chart(context: dict, column: str = "", top_n: int = 15, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    if not column:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        column = cat_cols[0] if len(cat_cols) else ""
    if not column:
        return {"error": "No categorical column found"}

    vc = df[column].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_theme(fig, ax)
    bars = ax.barh(range(len(vc)), vc.values, color=COLORS[:len(vc)])
    ax.set_yticks(range(len(vc)))
    ax.set_yticklabels(vc.index, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(vc)} values in {column}")
    ax.invert_yaxis()

    img = _fig_to_base64(fig)
    context.setdefault("charts", []).append({"type": "bar", "column": column, "image": img})
    return {"chart": "bar", "column": column, "generated": True}


def _tool_correlation_heatmap(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns"}

    # Limit to top 15 columns for readability
    if num_df.shape[1] > 15:
        num_df = num_df.iloc[:, :15]

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    _apply_dark_theme(fig, ax)
    im = ax.imshow(corr, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Heatmap")

    img = _fig_to_base64(fig)
    context.setdefault("charts", []).append({"type": "heatmap", "image": img})
    return {"chart": "heatmap", "columns": len(corr.columns), "generated": True}


def _tool_box_plot(context: dict, column: str = "", **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    if not column:
        num_cols = df.select_dtypes(include="number").columns
        column = num_cols[0] if len(num_cols) else ""
    if not column:
        return {"error": "No numeric column found"}

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_theme(fig, ax)
    bp = ax.boxplot(df[column].dropna(), patch_artist=True, vert=True)
    bp["boxes"][0].set_facecolor(COLORS[2])
    bp["boxes"][0].set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color="#E2E8F0")
    ax.set_ylabel(column)
    ax.set_title(f"Box Plot — {column}")

    img = _fig_to_base64(fig)
    context.setdefault("charts", []).append({"type": "boxplot", "column": column, "image": img})
    return {"chart": "boxplot", "column": column, "generated": True}


def _tool_auto_visualize(context: dict, max_charts: int = 4, **kwargs) -> dict:
    """Create the most insightful charts automatically."""
    df: pd.DataFrame = context["dataframe"]
    created = []

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Distribution of first numeric column
    if num_cols:
        _tool_histogram(context, column=num_cols[0])
        created.append(f"histogram:{num_cols[0]}")

    # 2. Scatter of top correlated pair
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_idx = np.unravel_index(corr.values.argmax(), corr.shape)
        x, y = num_cols[max_idx[0]], num_cols[max_idx[1]]
        _tool_scatter(context, x=x, y=y)
        created.append(f"scatter:{x}vs{y}")

    # 3. Bar chart of first categorical
    if cat_cols:
        _tool_bar_chart(context, column=cat_cols[0])
        created.append(f"bar:{cat_cols[0]}")

    # 4. Correlation heatmap
    if len(num_cols) >= 3:
        _tool_correlation_heatmap(context)
        created.append("heatmap")

    return {"auto_charts_created": created, "total": len(created)}


def _tool_list_columns(context: dict, **kwargs) -> dict:
    df: pd.DataFrame = context["dataframe"]
    return {
        "numeric": df.select_dtypes(include="number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include="datetime").columns.tolist(),
    }


class VisualizerAgent(BaseAgent):
    """Creates intelligent, themed visualizations."""

    def __init__(self):
        super().__init__(
            name="Visualizer",
            role="Data visualization specialist",
            system_prompt=SYSTEM_PROMPT,
            max_iterations=6,
        )

    def _register_tools(self):
        self.register_tool("list_columns", _tool_list_columns,
                           "List all columns by type", {"type": "object", "properties": {}})
        self.register_tool("histogram", _tool_histogram,
                           "Create histogram for a numeric column",
                           {"type": "object", "properties": {"column": {"type": "string"}, "bins": {"type": "integer", "default": 30}}})
        self.register_tool("scatter", _tool_scatter,
                           "Create scatter plot for two numeric columns",
                           {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "string"}}})
        self.register_tool("bar_chart", _tool_bar_chart,
                           "Create bar chart for a categorical column",
                           {"type": "object", "properties": {"column": {"type": "string"}, "top_n": {"type": "integer", "default": 15}}})
        self.register_tool("correlation_heatmap", _tool_correlation_heatmap,
                           "Create correlation heatmap for numeric columns",
                           {"type": "object", "properties": {}})
        self.register_tool("box_plot", _tool_box_plot,
                           "Create box plot for a numeric column",
                           {"type": "object", "properties": {"column": {"type": "string"}}})
        self.register_tool("auto_visualize", _tool_auto_visualize,
                           "Automatically create the most insightful charts",
                           {"type": "object", "properties": {"max_charts": {"type": "integer", "default": 4}}})

    def _build_task_prompt(self, context: dict) -> str:
        df: pd.DataFrame = context["dataframe"]
        target = context.get("target_column", "")

        return f"""Create insightful visualizations for this dataset.

## Dataset
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Numeric columns: {df.select_dtypes(include='number').columns.tolist()}
- Categorical columns: {df.select_dtypes(include=['object', 'category']).columns.tolist()}
- Target: {target or 'Not specified'}

## Instructions
1. List columns to understand what's available
2. Use auto_visualize for a quick overview
3. Create additional targeted charts based on interesting patterns
4. FINISH with a summary of charts created and key visual insights"""
