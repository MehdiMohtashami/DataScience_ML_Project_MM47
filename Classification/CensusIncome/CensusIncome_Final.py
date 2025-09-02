import os
import sys
import numpy as np
import pandas as pd
import joblib
import traceback

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QGroupBox, QFormLayout,
                             QLineEdit, QComboBox, QPushButton, QLabel, QMessageBox,
                             QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# ---------- Attempt to load training files ----------
try:
    loaded_model = joblib.load('best_xgboost_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')            # dict of LabelEncoders
    feature_names = joblib.load('feature_names.pkl')  # list in correct order
    model_loaded = True
except Exception as e:
    print("Warning: Could not load one or more model files:", e)
    traceback.print_exc()
    loaded_model = None
    loaded_scaler = None
    encoders = {}
    # Fallback feature order (if the file is missing)
    feature_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]
    model_loaded = False

# List of categorical columns expected to have encoders
CATEGORICAL_COLS = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country']

# If an encoder for a column is missing, create a simple fallback
from sklearn.preprocessing import LabelEncoder
for col in CATEGORICAL_COLS:
    if col not in encoders:
        le = LabelEncoder()
        # Create a default class so ComboBox works
        le.classes_ = np.array(['__unknown__'])
        encoders[col] = le

# Class for plotting charts (simple)
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

# ---------- Main window ----------
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Income Prediction Dashboard (Final)")
        self.setGeometry(80, 80, 1400, 900)

        # Simple style
        self.setStyleSheet("""
            QMainWindow { background-color: #f7f9fb; }
            QGroupBox { font-weight: bold; border: 2px solid #d6dde6; border-radius: 8px; margin-top: 1ex; padding-top: 10px; }
            QPushButton { background-color: #2b91d9; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #2479b6; }
            QLineEdit { padding: 6px; border: 1px solid #ccc; border-radius: 4px; }
            QLabel { color: #243b4a; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        splitter = QSplitter(Qt.Horizontal)

        # ---------- Input panel ----------
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)

        model_group = QGroupBox("Model Info")
        model_layout = QVBoxLayout(model_group)
        model_label = QLabel("XGBoost classifier")
        model_label.setFont(QFont("Arial", 12, QFont.Bold))
        model_layout.addWidget(model_label)
        if model_loaded:
            model_status = QLabel("Model loaded with %87.17 ✓")
        else:
            model_status = QLabel("Model files not loaded — predictions disabled")
        model_layout.addWidget(model_status)

        input_group = QGroupBox("Input Features")
        input_form = QFormLayout(input_group)

        # Input fields with appropriate validators
        self.age_edit = QLineEdit(); self.age_edit.setValidator(QtGui.QIntValidator(17, 90))
        self.workclass_combo = QComboBox(); self.workclass_combo.addItems(self.get_encoder_classes('workclass'))
        self.fnlwgt_edit = QLineEdit(); self.fnlwgt_edit.setValidator(QtGui.QIntValidator(10000, 1500000))
        self.education_combo = QComboBox(); self.education_combo.addItems(self.get_encoder_classes('education'))
        self.education_num_edit = QLineEdit(); self.education_num_edit.setValidator(QtGui.QIntValidator(1, 16))
        self.marital_combo = QComboBox(); self.marital_combo.addItems(self.get_encoder_classes('marital-status'))
        self.occupation_combo = QComboBox(); self.occupation_combo.addItems(self.get_encoder_classes('occupation'))
        self.relationship_combo = QComboBox(); self.relationship_combo.addItems(self.get_encoder_classes('relationship'))
        self.race_combo = QComboBox(); self.race_combo.addItems(self.get_encoder_classes('race'))
        self.sex_combo = QComboBox(); self.sex_combo.addItems(self.get_encoder_classes('sex'))
        self.capital_gain_edit = QLineEdit(); self.capital_gain_edit.setValidator(QtGui.QIntValidator(0, 100000))
        self.capital_loss_edit = QLineEdit(); self.capital_loss_edit.setValidator(QtGui.QIntValidator(0, 5000))
        self.hours_edit = QLineEdit(); self.hours_edit.setValidator(QtGui.QIntValidator(1, 99))
        self.country_combo = QComboBox(); self.country_combo.addItems(self.get_encoder_classes('native-country'))

        input_form.addRow("Age (17-90):", self.age_edit)
        input_form.addRow("Workclass:", self.workclass_combo)
        input_form.addRow("Final Weight (10000-1500000):", self.fnlwgt_edit)
        input_form.addRow("Education:", self.education_combo)
        input_form.addRow("Education Num (1-16):", self.education_num_edit)
        input_form.addRow("Marital Status:", self.marital_combo)
        input_form.addRow("Occupation:", self.occupation_combo)
        input_form.addRow("Relationship:", self.relationship_combo)
        input_form.addRow("Race:", self.race_combo)
        input_form.addRow("Sex:", self.sex_combo)
        input_form.addRow("Capital Gain (0-100000):", self.capital_gain_edit)
        input_form.addRow("Capital Loss (0-5000):", self.capital_loss_edit)
        input_form.addRow("Hours per Week (1-99):", self.hours_edit)
        input_form.addRow("Native Country:", self.country_combo)

        self.predict_btn = QPushButton("Predict Income")
        self.predict_btn.clicked.connect(self.predict_income)
        if not model_loaded:
            self.predict_btn.setEnabled(False)

        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)
        if not model_loaded:
            self.back_button.setEnabled(False)

        self.result_label = QLabel("Enter inputs and press Predict")
        self.result_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("padding: 16px;")

        input_layout.addWidget(model_group)
        input_layout.addWidget(input_group)
        input_layout.addWidget(self.predict_btn)
        input_layout.addWidget(self.back_button)
        input_layout.addWidget(self.result_label)
        input_layout.addStretch()

        # ---------- Output panel ----------
        output_panel = QWidget()
        output_layout = QVBoxLayout (output_panel)
        self.tab_widget = QTabWidget()
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.feature_importance_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.feature_relationship_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        analysis_layout.addWidget(QLabel("Feature Importance"))
        analysis_layout.addWidget(self.feature_importance_canvas)
        analysis_layout.addWidget(QLabel("Feature Relationships"))
        analysis_layout.addWidget(self.feature_relationship_canvas)

        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        self.model_comparison_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.model_metrics_canvas = MplCanvas(self, width=6, height=4, dpi=100)  # New chart
        comparison_layout.addWidget(QLabel("Model Comparison (Accuracy)"))
        comparison_layout.addWidget(self.model_comparison_canvas)
        comparison_layout.addWidget(QLabel("Model Metrics (Accuracy vs F1-Score)"))
        comparison_layout.addWidget(self.model_metrics_canvas)

        self.tab_widget.addTab(analysis_tab, "Prediction Analysis")
        self.tab_widget.addTab(comparison_tab, "Model Comparison")
        self.tab_widget.setEnabled(False)

        output_layout.addWidget(self.tab_widget)

        splitter.addWidget(input_panel)
        splitter.addWidget(output_panel)
        splitter.setSizes([420, 980])
        main_layout.addWidget(splitter)

        # Sample data
        self.load_sample_data()

    # ---------- Helper: Get encoder classes for populating ComboBox ----------
    def get_encoder_classes(self, col):
        if col in encoders:
            try:
                classes = list(map(str, encoders[col].classes_))
                return classes
            except Exception:
                return ['__unknown__']
        else:
            return ['__unknown__']

    def load_sample_data(self):
        # Initial values for UI testing
        self.age_edit.setText("45")
        self.workclass_combo.setCurrentIndex(0)
        self.fnlwgt_edit.setText("200000")
        self.education_combo.setCurrentIndex(0)
        self.education_num_edit.setText("13")
        self.marital_combo.setCurrentIndex(0)
        self.occupation_combo.setCurrentIndex(0)
        self.relationship_combo.setCurrentIndex(0)
        self.race_combo.setCurrentIndex(0)
        self.sex_combo.setCurrentIndex(0)
        self.capital_gain_edit.setText("5000")
        self.capital_loss_edit.setText("0")
        self.hours_edit.setText("50")
        self.country_combo.setCurrentIndex(0)

    # ---------- Convert categorical value using loaded encoders ----------
    def encode_with_encoders(self, feat, val):
        """Returns the encoded value if it exists in the encoder's classes,
           otherwise returns fallback value 0 and prints a warning."""
        try:
            le = encoders.get(feat, None)
            if le is None:
                return 0.0
            # Transform may raise an error if val is not in classes_
            try:
                return float(le.transform([str(val)])[0])
            except Exception:
                # Unseen value -> fallback to index 0
                print(f"Warning: unseen category '{val}' for feature '{feat}'. Using fallback 0.")
                return float(0)
        except Exception as e:
            print("Error encoding:", feat, val, e)
            return 0.0

    # ---------- Prediction function ----------
    def predict_income(self):
        if not model_loaded:
            QMessageBox.warning(self, "Model not loaded", "Model files are not available. Run training first.")
            return

        try:
            # Collect inputs (as strings)
            raw = {
                'age': self.age_edit.text().strip(),
                'workclass': self.workclass_combo.currentText(),
                'fnlwgt': self.fnlwgt_edit.text().strip(),
                'education': self.education_combo.currentText(),
                'education-num': self.education_num_edit.text().strip(),
                'marital-status': self.marital_combo.currentText(),
                'occupation': self.occupation_combo.currentText(),
                'relationship': self.relationship_combo.currentText(),
                'race': self.race_combo.currentText(),
                'sex': self.sex_combo.currentText(),
                'capital-gain': self.capital_gain_edit.text().strip(),
                'capital-loss': self.capital_loss_edit.text().strip(),
                'hours-per-week': self.hours_edit.text().strip(),
                'native-country': self.country_combo.currentText()
            }

            # Convert to the numeric list according to feature_names
            numeric_list = []
            numeric_input_data = {}
            for feat in feature_names:
                if feat in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                    s = raw.get(feat, "0")
                    if s == "" or s is None:
                        s = "0"
                    try:
                        val = float(s)
                    except:
                        # If numeric conversion fails, warn and use 0
                        print(f"Warning: couldn't parse numeric input for {feat} -> '{s}'. Using 0.")
                        val = 0.0
                    numeric_list.append(val)
                    numeric_input_data[feat] = val
                else:
                    val = raw.get(feat, "")
                    enc = self.encode_with_encoders(feat, val)
                    numeric_list.append(enc)
                    numeric_input_data[feat] = enc

            input_array = np.array(numeric_list, dtype=float).reshape(1, -1)

            # Standardization
            input_scaled = loaded_scaler.transform(input_array)

            # Prediction
            pred = loaded_model.predict(input_scaled)[0]
            proba = loaded_model.predict_proba(input_scaled)[0]

            if pred == 1:
                conf = proba[1] * 100
                self.result_label.setText(f"Prediction: Income > $50K\nConfidence: {conf:.2f}%")
                self.result_label.setStyleSheet("color: white; background-color: #2ecc71; padding: 14px; border-radius: 6px;")
            else:
                conf = proba[0] * 100
                self.result_label.setText(f"Prediction: Income <= $50K\nConfidence: {conf:.2f}%")
                self.result_label.setStyleSheet("color: white; background-color: #e74c3c; padding: 14px; border-radius: 6px;")

            # Enable tabs and update charts
            self.tab_widget.setEnabled(True)
            self.update_charts(numeric_input_data, int(pred), proba)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Prediction Error", f"An error occurred:\n{str(e)}")

    # ---------- Update charts function ----------
    def update_charts(self, input_data, prediction, probability):
        try:
            self.feature_importance_canvas.axes.clear()
            self.feature_relationship_canvas.axes.clear()
            self.model_comparison_canvas.axes.clear()
            self.model_metrics_canvas.axes.clear()  # Clear new chart

            # Feature importance
            feature_importance = getattr(loaded_model, "feature_importance_", None)
            if feature_importance is None:
                # If the model lacks this attribute, use zeros
                feature_importance = np.zeros(len(feature_names))
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            importance_df = importance_df.sort_values('importance', ascending=False)

            top10 = importance_df.head(10)
            self.feature_importance_canvas.axes.barh(top10['feature'][::-1], top10['importance'][::-1])
            self.feature_importance_canvas.axes.set_title("Top 10 Feature Importance")
            self.feature_importance_canvas.fig.tight_layout()
            self.feature_importance_canvas.draw()

            # Relationship of top two features (example)
            top_feats = importance_df['feature'].head(2).tolist()
            if len(top_feats) >= 2:
                np.random.seed(0)
                n = 120
                loc1 = float(input_data.get(top_feats[0], 0.0))
                loc2 = float(input_data.get(top_feats[1], 0.0))
                x = np.random.normal(loc=loc1, scale=5, size=n)
                y = np.random.normal(loc=loc2, scale=5, size=n)
                self.feature_relationship_canvas.axes.scatter(x, y, alpha=0.6)
                self.feature_relationship_canvas.axes.scatter(loc1, loc2, s=140, marker='*',
                                                              color='gold' if prediction==1 else 'red',
                                                              edgecolors='black', label='Your input')
                self.feature_relationship_canvas.axes.set_xlabel(top_feats[0])
                self.feature_relationship_canvas.axes.set_ylabel(top_feats[1])
                self.feature_relationship_canvas.axes.set_title(f"Relationship: {top_feats[0]} vs {top_feats[1]}")
                self.feature_relationship_canvas.axes.legend()
                self.feature_relationship_canvas.fig.tight_layout()
                self.feature_relationship_canvas.draw()

            # Model comparison chart (Accuracy)
            models = ['LogReg', 'DecisionTree', 'RandomForest', 'GB', 'XGBoost', 'SVM', 'KNN']
            accuracies = [0.8254, 0.8065, 0.8555, 0.8668, 0.8687, 0.8512, 0.8286]
            colors = ['lightblue' if m != 'XGBoost' else 'gold' for m in models]
            self.model_comparison_canvas.axes.barh(models, accuracies, color=colors)
            self.model_comparison_canvas.axes.set_xlim(0.7, 0.9)
            self.model_comparison_canvas.axes.set_title("Model Comparison (Accuracy)")
            for i, v in enumerate(accuracies):
                self.model_comparison_canvas.axes.text(v + 0.003, i, f"{v:.4f}", va='center')
            self.model_comparison_canvas.fig.tight_layout()
            self.model_comparison_canvas.draw()

            # New chart: Accuracy vs. F1-Score comparison
            f1_scores = [0.8150, 0.7950, 0.8450, 0.8600, 0.8650, 0.8400, 0.8200]  # Sample F1-Score data
            self.model_metrics_canvas.axes.plot(models, accuracies, marker='o', label='Accuracy', color='#2b91d9')
            self.model_metrics_canvas.axes.plot(models, f1_scores, marker='s', label='F1-Score', color='#e74c3c')
            self.model_metrics_canvas.axes.set_ylim(0.7, 0.9)
            self.model_metrics_canvas.axes.set_title("Accuracy vs F1-Score Comparison")
            self.model_metrics_canvas.axes.legend()
            self.model_metrics_canvas.axes.grid(True, linestyle='--', alpha=0.7)
            self.model_metrics_canvas.fig.tight_layout()
            self.model_metrics_canvas.draw()

        except Exception as e:
            print("Error updating charts:", e)
            traceback.print_exc()

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = MainWindow(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())

# ---------- Execution ----------
if __name__ == "__main__":
    main()