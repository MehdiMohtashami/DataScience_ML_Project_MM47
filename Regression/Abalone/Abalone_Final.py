import os
import sys
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QComboBox, QMessageBox, QTabWidget, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

sns.set_style("whitegrid")
DATA_PATH = "abalone.data.csv"
MODEL_PATH = "abalone_best_model.joblib"
RANDOM_STATE = 42

RAW_COLS = ["Sex", "Length", "Diameter", "Height",
            "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

# -------------------- Reading the data set --------------------
def load_raw_df(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 9:
        df.columns = RAW_COLS
    else:
        df = pd.read_csv(path)
        df.columns = [c.strip().replace("_", " ") for c in df.columns]
        if "Sex" not in df.columns:
            raise ValueError("Unexpected CSV format. Expected 9 columns (Sex..Rings).")
    # Filter to remove invalid rows such as headers if they were read incorrectly
    df = df[df['Sex'].isin(['M', 'F', 'I'])]
    numeric_cols = ["Length", "Diameter", "Height", "Whole weight",
                    "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    return df

# --------------------Feature extraction --------------------
def prepare_features(df):
    d = df.copy()
    d["Volume"] = d["Length"] * d["Diameter"] * d["Height"]
    d["Shucked_Whole_ratio"] = d["Shucked weight"] / (d["Whole weight"] + 1e-9)
    d["Viscera_Whole_ratio"] = d["Viscera weight"] / (d["Whole weight"] + 1e-9)
    d["Shell_Whole_ratio"] = d["Shell weight"] / (d["Whole weight"] + 1e-9)
    d["Length_Diameter"] = d["Length"] * d["Diameter"]
    d["Length_sq"] = d["Length"] ** 2
    d["Diameter_sq"] = d["Diameter"] ** 2
    d["log_Whole_weight"] = np.log1p(d["Whole weight"])
    d["log_Shucked_weight"] = np.log1p(d["Shucked weight"])
    return d

# -------------------- App --------------------
class AbaloneApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Abalone")
        self.setWindowTitle("Abalone Age Predictor")
        self.resize(1100, 760)

        # dataset
        try:
            self.df_raw = load_raw_df()
            self.df_feat = prepare_features(self.df_raw)
            self.X = self.df_feat.drop('Rings', axis=1)
            self.y = self.df_feat['Rings']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=RANDOM_STATE
            )
        except Exception as e:
            QMessageBox.critical(self, "Dataset Error", str(e))
            raise

        # Load existing Pipeline model
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
        else:
            QMessageBox.critical(self, "Model Error", f"{MODEL_PATH} not found!")
            raise SystemExit("Model required. Exiting.")

        # Calculating Model Accuracy (R² Score)
        self.r2 = r2_score(self.y_test, self.pipeline.predict(self.X_test))
        # Model name
        self.model_name = self.pipeline.steps[-1][1].__class__.__name__

        self.pred_rings = None
        self.input_data = None

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_input = QWidget()
        self.tab_visual = QWidget()
        self.tab_imp = QWidget()
        self.tabs.addTab(self.tab_input, "Input & Predict")
        self.tabs.addTab(self.tab_visual, "Visualizations")
        self.tabs.addTab(self.tab_imp, "Feature Importance")

        self._build_input_tab()
        self._build_visual_tab()
        self._build_importance_tab()

    # ---------- Input tab----------
    def _build_input_tab(self):
        grid = QGridLayout()
        self.tab_input.setLayout(grid)

        # Sex
        grid.addWidget(QLabel("Sex (M/F/I):"), 0, 0)
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["M", "F", "I"])
        grid.addWidget(self.sex_combo, 0, 1)

        # Numeric features
        self.input_spins = {}
        ranges = {
            "Length": (0.05, 1.5),
            "Diameter": (0.05, 1.5),
            "Height": (0.0, 0.6),
            "Whole weight": (0.0, 3.5),
            "Shucked weight": (0.0, 2.0),
            "Viscera weight": (0.0, 1.0),
            "Shell weight": (0.0, 2.0),
        }
        row = 1
        for feat, (low, high) in ranges.items():
            grid.addWidget(QLabel(feat + ":"), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(4)
            spin.setSingleStep((high-low)/100.0)
            spin.setValue((low+high)/2.0)
            grid.addWidget(spin, row, 1)
            self.input_spins[feat] = spin
            row += 1

        # Buttons
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predict 🔮")
        self.predict_btn.clicked.connect(self.on_predict)
        btn_layout.addWidget(self.predict_btn)
        grid.addLayout(btn_layout, row, 0, 1, 2)


        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)
        btn_layout.addWidget(self.back_button)
        grid.addLayout(btn_layout, row, 2, 1, 2)

        # Result
        self.result_label = QLabel("Predicted Rings: —    Predicted Age (yrs): —")
        self.result_label.setStyleSheet("font-size:16px; font-weight:bold; color:#1f77b4;")
        grid.addWidget(self.result_label, row+1, 0, 1, 2)

        # Model info
        self.model_label = QLabel(f"Model: {self.model_name} (R² Score: %{self.r2*100:.4f})")
        self.model_label.setStyleSheet("font-size:17px; color:#1f77b4;font-weight:bold;")
        grid.addWidget(self.model_label, row+2, 0, 1, 2)

    # ---------- Predict ----------
    def on_predict(self):
        try:
            data = {"Sex": self.sex_combo.currentText()}
            for feat, spin in self.input_spins.items():
                data[feat] = float(spin.value())
            df_in = pd.DataFrame([data])
            df_in_feat = prepare_features(df_in)
            pred_rings = float(self.pipeline.predict(df_in_feat)[0])
            pred_age = pred_rings + 1.5
            self.result_label.setText(f"Predicted Rings: {pred_rings:.2f}    Predicted Age (yrs): {pred_age:.2f}")
            self.pred_rings = pred_rings
            self.input_data = df_in_feat.iloc[0]
            self.update_visuals()
        except Exception as e:
            QMessageBox.critical(self, "Prediction error", str(e))

    # ----------  Visual ----------
    def _build_visual_tab(self):
        v = QVBoxLayout()
        self.tab_visual.setLayout(v)
        self.fig_vis = Figure(figsize=(9,6))
        self.canvas_vis = FigureCanvas(self.fig_vis)
        v.addWidget(self.canvas_vis)

    def update_visuals(self):
        self.fig_vis.clear()
        ax1 = self.fig_vis.add_subplot(221)
        ax2 = self.fig_vis.add_subplot(222)
        ax3 = self.fig_vis.add_subplot(223)
        ax4 = self.fig_vis.add_subplot(224)

        sns.histplot(self.df_feat["Rings"], bins=25, kde=True, ax=ax1, color="#4c72b0")
        ax1.set_title("Rings Distribution")
        if self.pred_rings is not None:
            ax1.axvline(self.pred_rings, color='red', linestyle='--', label='Predicted')
            ax1.legend()

        sns.scatterplot(x="Length", y="Diameter", hue="Rings", data=self.df_feat, ax=ax2, palette='viridis', alpha=0.7)
        ax2.set_title("Length vs Diameter")
        if self.pred_rings is not None and self.input_data is not None:
            ax2.scatter(self.input_data['Length'], self.input_data['Diameter'], color='red', s=100, label='Predicted')
            ax2.legend()

        sns.boxplot(x='Sex', y='Rings', data=self.df_feat, ax=ax3)
        ax3.set_title("Rings by Sex")

        sns.scatterplot(x="Shell weight", y="Rings", data=self.df_feat, ax=ax4, alpha=0.7)
        ax4.set_title("Shell weight vs Rings")
        if self.pred_rings is not None and self.input_data is not None:
            ax4.scatter(self.input_data['Shell weight'], self.pred_rings, color='red', s=100, label='Predicted')
            ax4.legend()

        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

    # ---------- Importance ----------
    def _build_importance_tab(self):
        v = QVBoxLayout()
        self.tab_imp.setLayout(v)
        self.fig_imp = Figure(figsize=(8,4))
        self.canvas_imp = FigureCanvas(self.fig_imp)
        v.addWidget(self.canvas_imp)

        self.imp_btn = QPushButton("Compute permutation importance")
        self.imp_btn.clicked.connect(self.on_compute_importance)
        v.addWidget(self.imp_btn)

    def on_compute_importance(self):
        self.fig_imp.clear()
        ax = self.fig_imp.add_subplot(111)
        try:
            imp = permutation_importance(
                self.pipeline, self.X_test, self.y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            sorted_idx = imp.importances_mean.argsort()
            ax.boxplot(imp.importances[sorted_idx].T, vert=False, tick_labels=self.X_test.columns[sorted_idx])
            ax.set_title("Permutation Importance")
            self.fig_imp.tight_layout()
            self.canvas_imp.draw()
        except Exception as e:
            ax.text(0.5, 0.5, f"Error computing importance:\n{str(e)}", color='red', ha='center')
            self.canvas_imp.draw()

    def close_and_go_back(self):
        self.close()
# --------------------Run --------------------
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 10, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window =AbaloneApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()