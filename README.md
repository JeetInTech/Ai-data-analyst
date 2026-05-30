# ◆ NeuroviaI — AI-Powered Data Analytics Platform

A desktop data analytics platform powered by a multi-agent LLM system. Import any dataset, and NeuroviaI's AI agents will automatically profile, clean, engineer features, train ML models, and generate visualizations — all through an intelligent pipeline that reasons about your data.

Built with PySide6 (Qt6) for a premium dark-themed desktop experience. No browser, no server — runs entirely on your machine.

---

## Features

### Multi-Agent Intelligence System
NeuroviaI uses a pipeline of specialist AI agents, each backed by real LLM reasoning (Groq, Gemini, or local Ollama), that collaborate to analyze your data end-to-end:

| Agent | Role |
|-------|------|
| **Profiler** | Statistical profiling, quality assessment, missing value analysis, correlation detection |
| **Cleaner** | Duplicate removal, imputation, outlier handling, type correction, text normalization |
| **Feature Engineer** | Categorical encoding, interaction features, date extraction, log transforms, feature selection |
| **Model Selector** | Task detection (classification/regression), trains & compares 12 models, feature importance |
| **Visualizer** | Auto-generates charts with 20+ chart types including 3D plots with dark theme |
| **Orchestrator** | Coordinates the full pipeline, routes natural-language questions to the right agent |

Each agent follows a **think → act → observe** loop — it reasons about what to do, calls tools on your data, observes the results, and continues until done.

### Desktop UI
- **Dashboard** — Session metrics, system resources, quick actions
- **Data Import** — Drag-and-drop or browse for CSV, TSV, Excel, JSON, Parquet, Feather files
- **Data Explorer** — Full data table with search/sort, per-column profiling with mini charts, **CSV/Excel export**
- **AI Agents** — Chat-based interface to run the full pipeline or ask questions about your data
- **Cleaning** — 6-step cleaning pipeline with progress tracking, **download cleaned data as CSV/Excel**
- **Visualization** — 20+ chart types (2D & 3D) with theme control, X/Y/Z/Color selectors, **export charts as PNG/SVG**
- **ML Training** — Train 11 models with cross-validation, **export results as CSV/Excel**
- **Explainability** — Feature importance, data quality profile, correlation insights, **export charts**
- **Settings** — API key management, processing preferences

### Visualization Engine
**15 2D chart types:** Histogram, Scatter, Line, Bar, Box, Heatmap, Pie, Violin, Pair Plot, Area, Radar, Waterfall, Treemap, Bubble, Density

**5 3D chart types (matplotlib):** 3D Scatter, 3D Bar, 3D Surface, 3D Wireframe, 3D Contour

**5 chart themes:** NeuroviaI Dark, Midnight Blue, Cyberpunk, Light Professional, Warm Earth

**Export:** PNG, SVG, CSV, Excel from every page

### Supported Models
**Classification:** Random Forest, Extra Trees, Logistic Regression, KNN, SVM, XGBoost, LightGBM, MLP Neural Net  
**Regression:** Random Forest, Extra Trees, Linear Regression, Ridge, Lasso, KNN, SVR, XGBoost, LightGBM

### LLM Provider Failover
Agents try providers in priority order with automatic fallback:
1. **Groq** (Llama 3.3 70B) — fastest inference
2. **Gemini** (2.0 Flash) — strong reasoning
3. **Ollama** (local) — no API key needed, fully offline

---

## Quick Start

### Prerequisites
- Python 3.11+
- At least one LLM provider configured (or Ollama running locally)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

You need at least one provider. Get free API keys:
- **Groq:** https://console.groq.com/keys
- **Gemini:** https://aistudio.google.com/apikey
- **Ollama:** Install from https://ollama.com then run `ollama pull llama3.1`

### Launch

```bash
python run_neurovia.py
```

---

## Project Structure

```
neuroviai/
├── run_neurovia.py              # Entry point with splash screen & logo
├── requirements.txt             # Dependencies
├── .env                         # API keys (gitignored)
│
├── neurovia_ui/                 # PySide6 desktop UI
│   ├── theme.py                 # Colors, fonts, QSS stylesheet
│   ├── main_window.py           # Main window + page routing
│   ├── widgets/
│   │   ├── sidebar.py           # Navigation sidebar with logo
│   │   ├── components.py        # MetricCard, ActionButton, EmptyState, etc.
│   │   ├── data_table.py        # Pandas-backed table with search/sort
│   │   └── chart_canvas.py      # Matplotlib canvas with theme support
│   └── pages/
│       ├── dashboard.py         # Metrics + system resources
│       ├── data_import.py       # Drag-and-drop file import
│       ├── data_explorer.py     # Data table + column profiler + export
│       ├── agent_page.py        # AI agent chat interface
│       ├── cleaning.py          # Cleaning pipeline + download cleaned data
│       ├── visualization.py     # 20+ chart types (2D/3D) + theme control
│       ├── ml_training.py       # Model training + comparison + export
│       ├── explainability_page.py # Feature importance + quality + export
│       └── settings.py          # API keys + preferences
│
└── neurovia_agents/             # Multi-agent LLM system
    ├── llm_client.py            # Unified LLM client (Groq/Gemini/Ollama)
    ├── base_agent.py            # BaseAgent ABC + think→act→observe loop
    ├── orchestrator.py          # Pipeline coordinator
    ├── profiler_agent.py        # Data profiling (8 tools)
    ├── cleaner_agent.py         # Data cleaning (9 tools)
    ├── feature_engineer_agent.py # Feature engineering (7 tools)
    ├── model_selector_agent.py  # ML model selection (5 tools)
    └── visualizer_agent.py      # Chart generation (7 tools)
```

---

## How the Agent System Works

1. **You import data** — CSV, Excel, JSON, Parquet, or Feather
2. **You click "Run Agent Pipeline"** on the AI Agents page
3. **The Orchestrator** coordinates each specialist agent in sequence:

```
Profile → Clean → Feature Engineer → Model Select → Visualize
```

4. **Each agent** receives the data + context from previous agents, calls the LLM to reason about what tools to use, executes those tools on the actual DataFrame, and passes results forward
5. **Results appear** as chat bubbles with inline charts, model comparisons, and actionable summaries

You can also type natural-language questions like:
- *"What are the most correlated features?"*
- *"Clean this dataset"*
- *"Train models to predict the price column"*
- *"Show me the distribution of all columns"*

The Orchestrator automatically routes your question to the right specialist agent.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | PySide6 (Qt6) |
| ML | scikit-learn, XGBoost, LightGBM |
| Data | pandas, NumPy |
| Charts | matplotlib (2D + 3D), scipy |
| LLM | httpx (direct API calls to Groq, Gemini, Ollama) |
| System | psutil |

---

## License

MIT

---

<p align="center">
  <strong>◆ NEUROVIA·I</strong> — AI-Powered Data Analytics Platform<br>
  <em>Built with intelligence, designed with precision.</em>
</p>

## 2026-05-27 15:59:18 IST

- Daily progress note: Recorded a small documentation checkpoint for project continuity.
- Streak maintained: meaningful documentation checkpoint recorded.

## 2026-05-27 16:18:14 IST

- Daily progress note: Recorded a small documentation checkpoint for project continuity.
- Streak maintained: meaningful documentation checkpoint recorded.

## 2026-05-30 19:01:07 IST

- Daily progress note: Reviewed project notes and refreshed the daily development log.
- Streak maintained: meaningful documentation checkpoint recorded.
