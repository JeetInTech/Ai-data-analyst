"""
ML Training Page — Model selection, training, and results comparison.
"""

import pandas as pd
import numpy as np
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QComboBox,
    QCheckBox, QProgressBar, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QGroupBox,
    QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor

from neurovia_ui.theme import Colors, Fonts
from neurovia_ui.widgets.components import Card, SectionHeader, EmptyState, MetricCard
from neurovia_ui.widgets.chart_canvas import ChartCanvas, NEUROVIA_COLORS


AVAILABLE_MODELS = {
    "Random Forest":      {"type": "ensemble",   "sklearn": "RandomForestClassifier"},
    "XGBoost":            {"type": "boosting",   "lib": "xgboost"},
    "LightGBM":           {"type": "boosting",   "lib": "lightgbm"},
    "Logistic Regression": {"type": "linear",    "sklearn": "LogisticRegression"},
    "Linear Regression":  {"type": "linear",     "sklearn": "LinearRegression"},
    "Ridge":              {"type": "linear",     "sklearn": "Ridge"},
    "Lasso":              {"type": "linear",     "sklearn": "Lasso"},
    "SVM":                {"type": "svm",        "sklearn": "SVC"},
    "Extra Trees":        {"type": "ensemble",   "sklearn": "ExtraTreesClassifier"},
    "KNN":                {"type": "neighbors",  "sklearn": "KNeighborsClassifier"},
    "MLP Neural Net":     {"type": "neural",     "sklearn": "MLPClassifier"},
}


class TrainingWorker(QThread):
    """Background worker for model training."""
    progress = Signal(int, str)
    model_done = Signal(str, dict)  # (model_name, metrics)
    finished = Signal(dict)  # full results
    error = Signal(str)

    def __init__(self, df: pd.DataFrame, target: str, models: list, task_type: str):
        super().__init__()
        self._df = df
        self._target = target
        self._models = models
        self._task_type = task_type

    def run(self):
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

            df = self._df.copy()
            target = self._target

            # Prepare features
            X = df.drop(columns=[target])
            y = df[target]

            # Encode categoricals
            le_dict = {}
            for col in X.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le

            # Handle target encoding for classification
            is_classification = self._task_type == "classification"
            if is_classification and y.dtype == "object":
                le_target = LabelEncoder()
                y = pd.Series(le_target.fit_transform(y), name=target)

            # Drop non-numeric
            X = X.select_dtypes(include=[np.number])
            X = X.fillna(X.median())

            # Scale
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            results = {}
            total = len(self._models)

            for i, name in enumerate(self._models):
                self.progress.emit(int((i / total) * 100), f"Training {name}...")
                try:
                    model = self._create_model(name, is_classification)
                    start = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start

                    y_pred = model.predict(X_test)

                    if is_classification:
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                        cv = cross_val_score(model, X_scaled, y, cv=min(5, len(X_scaled)), scoring="accuracy")
                        metrics = {
                            "accuracy": acc,
                            "f1_score": f1,
                            "cv_mean": cv.mean(),
                            "cv_std": cv.std(),
                            "train_time": train_time,
                        }
                    else:
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        cv = cross_val_score(model, X_scaled, y, cv=min(5, len(X_scaled)), scoring="r2")
                        metrics = {
                            "r2": r2,
                            "rmse": rmse,
                            "cv_mean": cv.mean(),
                            "cv_std": cv.std(),
                            "train_time": train_time,
                        }

                    results[name] = metrics
                    self.model_done.emit(name, metrics)

                except Exception as e:
                    results[name] = {"error": str(e)}
                    self.model_done.emit(name, {"error": str(e)})

            self.progress.emit(100, "Training complete!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _create_model(self, name: str, is_classification: bool):
        if name == "Random Forest":
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        elif name == "XGBoost":
            import xgboost as xgb
            if is_classification:
                return xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
            return xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

        elif name == "LightGBM":
            import lightgbm as lgb
            if is_classification:
                return lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            return lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        elif name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)

        elif name == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

        elif name == "Ridge":
            from sklearn.linear_model import Ridge
            return Ridge(random_state=42)

        elif name == "Lasso":
            from sklearn.linear_model import Lasso
            return Lasso(random_state=42)

        elif name == "SVM":
            if is_classification:
                from sklearn.svm import SVC
                return SVC(kernel="rbf", random_state=42)
            from sklearn.svm import SVR
            return SVR(kernel="rbf")

        elif name == "Extra Trees":
            if is_classification:
                from sklearn.ensemble import ExtraTreesClassifier
                return ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        elif name == "KNN":
            if is_classification:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier()
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor()

        elif name == "MLP Neural Net":
            if is_classification:
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier(max_iter=300, random_state=42)
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(max_iter=300, random_state=42)

        raise ValueError(f"Unknown model: {name}")


class MLTrainingPage(QWidget):
    """ML model training with configuration, progress, and results."""

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._worker = None
        self._results = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._empty = EmptyState(
            "◈", "No Data for Training",
            "Import and clean a dataset first.",
        )
        layout.addWidget(self._empty)

        self._content = QWidget()
        self._content.hide()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(32, 28, 32, 28)
        inner_layout.setSpacing(20)

        # Header
        title = QLabel("ML Model Training")
        title.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XL, QFont.Weight.Bold))
        inner_layout.addWidget(title)

        # Configuration
        config_row = QHBoxLayout()
        config_row.setSpacing(20)

        # Target column selector
        t_group = QVBoxLayout()
        t_group.addWidget(QLabel("Target Column"))
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(200)
        self._target_combo.currentTextChanged.connect(self._detect_task_type)
        t_group.addWidget(self._target_combo)
        config_row.addLayout(t_group)

        # Task type
        tt_group = QVBoxLayout()
        tt_group.addWidget(QLabel("Task Type"))
        self._task_combo = QComboBox()
        self._task_combo.addItems(["Auto Detect", "Classification", "Regression"])
        self._task_combo.setMinimumWidth(160)
        tt_group.addWidget(self._task_combo)
        config_row.addLayout(tt_group)

        config_row.addStretch()
        inner_layout.addLayout(config_row)

        # Model selection
        inner_layout.addWidget(SectionHeader("Select Models"))
        models_card = Card()
        self._model_checks = {}
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, name in enumerate(AVAILABLE_MODELS):
            cb = QCheckBox(name)
            cb.setChecked(name in ("Random Forest", "XGBoost", "LightGBM", "Logistic Regression"))
            grid.addWidget(cb, i // 3, i % 3)
            self._model_checks[name] = cb
        models_card.card_layout().addLayout(grid)

        # Quick select buttons
        qs_row = QHBoxLayout()
        for label, keys in [
            ("All", list(AVAILABLE_MODELS.keys())),
            ("Fast Only", ["Logistic Regression", "Linear Regression", "Ridge", "KNN"]),
            ("Boosting", ["XGBoost", "LightGBM"]),
            ("None", []),
        ]:
            btn = QPushButton(label)
            btn.setProperty("accent", "ghost")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, k=keys: self._quick_select(k))
            qs_row.addWidget(btn)
        qs_row.addStretch()
        models_card.card_layout().addLayout(qs_row)
        inner_layout.addWidget(models_card)

        # Train button + progress
        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("   ▶  Start Training   ")
        self._train_btn.setProperty("accent", "primary")
        self._train_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._train_btn.setFixedHeight(48)
        self._train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._train_btn)
        btn_row.addStretch()
        inner_layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(12)
        self._progress.hide()
        inner_layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        inner_layout.addWidget(self._progress_label)

        # Results table
        inner_layout.addWidget(SectionHeader("Results"))
        self._results_table = QTableWidget()
        self._results_table.setMinimumHeight(200)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        inner_layout.addWidget(self._results_table)

        # Results chart
        self._results_chart = ChartCanvas(width=10, height=4)
        self._results_chart.hide()
        inner_layout.addWidget(self._results_chart)

        # Export buttons
        export_row = QHBoxLayout()
        self._export_results_btn = QPushButton("📥  Export Results CSV")
        self._export_results_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_results_btn.setEnabled(False)
        self._export_results_btn.clicked.connect(self._export_results)
        export_row.addWidget(self._export_results_btn)

        self._export_chart_btn = QPushButton("📥  Export Chart PNG")
        self._export_chart_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_chart_btn.setEnabled(False)
        self._export_chart_btn.clicked.connect(self._export_chart_image)
        export_row.addWidget(self._export_chart_btn)

        export_row.addStretch()
        inner_layout.addLayout(export_row)

        inner_layout.addStretch()
        scroll.setWidget(inner)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(scroll)
        layout.addWidget(self._content)

    def set_dataframe(self, df: pd.DataFrame):
        self._empty.hide()
        self._content.show()

        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        for col in df.columns:
            self._target_combo.addItem(col)
        # Default: last column as target
        if len(df.columns) > 0:
            self._target_combo.setCurrentIndex(len(df.columns) - 1)
        self._target_combo.blockSignals(False)
        self._detect_task_type()

    def _detect_task_type(self):
        if self._task_combo.currentText() != "Auto Detect":
            return
        df = self._state.get("current_df")
        target = self._target_combo.currentText()
        if df is None or target not in df.columns:
            return
        col = df[target]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 20:
            self._task_combo.setCurrentText("Regression")
        else:
            self._task_combo.setCurrentText("Classification")

    def _quick_select(self, keys: list):
        for name, cb in self._model_checks.items():
            cb.setChecked(name in keys)

    def _start_training(self):
        df = self._state.get("current_df")
        if df is None:
            return

        target = self._target_combo.currentText()
        if not target or target not in df.columns:
            return

        models = [n for n, cb in self._model_checks.items() if cb.isChecked()]
        if not models:
            return

        task = self._task_combo.currentText().lower()
        if task == "auto detect":
            task = "classification"

        self._results = {}
        self._train_btn.setEnabled(False)
        self._progress.show()
        self._progress.setValue(0)

        self._worker = TrainingWorker(df, target, models, task)
        self._worker.progress.connect(self._on_progress)
        self._worker.model_done.connect(self._on_model_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._progress_label.setText(msg)

    def _on_model_done(self, name: str, metrics: dict):
        self._results[name] = metrics
        self._update_results_table()

    def _on_finished(self, results: dict):
        self._train_btn.setEnabled(True)
        self._progress.setValue(100)
        self._progress_label.setText("✓ Training complete!")
        self._progress_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")
        self._state["models_trained"] = self._state.get("models_trained", 0) + len(results)
        self._export_results_btn.setEnabled(True)
        self._export_chart_btn.setEnabled(True)
        self._show_results_chart()

    def _on_error(self, msg: str):
        self._train_btn.setEnabled(True)
        self._progress.hide()
        self._progress_label.setText(f"Error: {msg}")
        self._progress_label.setStyleSheet(f"color: {Colors.ERROR};")

    def _update_results_table(self):
        if not self._results:
            return

        # Determine columns based on task type
        sample = next(iter(self._results.values()))
        if "accuracy" in sample:
            headers = ["Model", "Accuracy", "F1 Score", "CV Mean ± Std", "Time (s)"]
        elif "r2" in sample:
            headers = ["Model", "R² Score", "RMSE", "CV Mean ± Std", "Time (s)"]
        else:
            headers = ["Model", "Status"]

        self._results_table.setRowCount(len(self._results))
        self._results_table.setColumnCount(len(headers))
        self._results_table.setHorizontalHeaderLabels(headers)

        for row, (name, metrics) in enumerate(self._results.items()):
            self._results_table.setItem(row, 0, QTableWidgetItem(name))

            if "error" in metrics:
                item = QTableWidgetItem(f"Error: {metrics['error'][:50]}")
                item.setForeground(QColor(Colors.ERROR))
                self._results_table.setItem(row, 1, QTableWidgetItem(item))
            elif "accuracy" in metrics:
                self._results_table.setItem(row, 1, QTableWidgetItem(f"{metrics['accuracy']:.4f}"))
                self._results_table.setItem(row, 2, QTableWidgetItem(f"{metrics['f1_score']:.4f}"))
                self._results_table.setItem(row, 3, QTableWidgetItem(
                    f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"))
                self._results_table.setItem(row, 4, QTableWidgetItem(f"{metrics['train_time']:.2f}"))
            elif "r2" in metrics:
                self._results_table.setItem(row, 1, QTableWidgetItem(f"{metrics['r2']:.4f}"))
                self._results_table.setItem(row, 2, QTableWidgetItem(f"{metrics['rmse']:.4f}"))
                self._results_table.setItem(row, 3, QTableWidgetItem(
                    f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"))
                self._results_table.setItem(row, 4, QTableWidgetItem(f"{metrics['train_time']:.2f}"))

    def _show_results_chart(self):
        if not self._results:
            return

        valid = {k: v for k, v in self._results.items() if "error" not in v}
        if not valid:
            return

        self._results_chart.show()
        ax = self._results_chart.clear_and_get_axes()

        names = list(valid.keys())
        metric_key = "accuracy" if "accuracy" in next(iter(valid.values())) else "r2"
        values = [v[metric_key] for v in valid.values()]
        colors = [NEUROVIA_COLORS[i % len(NEUROVIA_COLORS)] for i in range(len(names))]

        bars = ax.barh(range(len(names)), values, color=colors, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel(metric_key.replace("_", " ").title(), fontsize=11)
        ax.set_title("Model Comparison", fontsize=13, fontweight="bold", color=Colors.TEXT_PRIMARY)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color=Colors.TEXT_SECONDARY)

        ax.invert_yaxis()
        self._results_chart.refresh()

    def _export_results(self):
        """Export ML training results as CSV."""
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "ml_results.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            rows = []
            for name, metrics in self._results.items():
                row = {"Model": name}
                row.update({k: v for k, v in metrics.items() if k != "error"})
                rows.append(row)
            df = pd.DataFrame(rows)
            if path.endswith(".xlsx"):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)

    def _export_chart_image(self):
        """Export results comparison chart as PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", "model_comparison.png",
            "PNG Files (*.png);;SVG Files (*.svg)"
        )
        if path:
            self._results_chart.export_chart(path)
