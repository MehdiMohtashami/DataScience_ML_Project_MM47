import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QTabWidget, QMessageBox, QFormLayout,
                             QScrollArea)
from PyQt5.QtCore import Qt
import joblib
from sklearn.inspection import partial_dependence


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)



class ConcretePredictorApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ConcreteCompressiveStrength")
        self.setWindowTitle("Concrete Strength Predictor")
        self.setGeometry(100, 100, 1200, 800)

        # Load model and scaler
        try:
            self.model = joblib.load('best_model.pkl')
            self.scaler = None
        except:
            try:
                self.model = joblib.load('best_model_scaled.pkl')
                self.scaler = joblib.load('scaler.pkl')
            except:
                QMessageBox.critical(self, "Error", "Model files not found. Please train the model first.")
                sys.exit(1)

        # Load data for reference ranges
        try:
            self.df = pd.read_csv('Concrete_Data.csv')
            self.df.columns = self.df.columns.str.strip()
        except:
            QMessageBox.critical(self, "Error", "Data file not found. Please make sure Concrete_Data.csv exists.")
            sys.exit(1)

        self.initUI()
        self.current_prediction = None

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel - Input
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Model info
        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Algorithm: XGBoost Regressor"))
        model_layout.addWidget(QLabel("R² Score: 91.77%"))
        model_layout.addWidget(QLabel("Best performing model"))
        model_info.setLayout(model_layout)
        left_layout.addWidget(model_info)

        # Input form
        input_group = QGroupBox("Input Parameters")
        form_layout = QFormLayout()

        self.input_fields = {}
        features = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
                    'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']

        for feature in features:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Range: {self.df[feature].min():.1f} - {self.df[feature].max():.1f}")
            # تغییر این خط - استفاده از QtGui.QDoubleValidator به جای QtWidgets.QDoubleValidator
            line_edit.setValidator(QtGui.QDoubleValidator())
            form_layout.addRow(QLabel(feature + ":"), line_edit)
            self.input_fields[feature] = line_edit

        input_group.setLayout(form_layout)
        left_layout.addWidget(input_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predict Strength")
        self.predict_btn.clicked.connect(self.predict)
        self.clear_btn = QPushButton("Clear Inputs")
        self.clear_btn.clicked.connect(self.clear_inputs)
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.clear_btn)
        # ایجاد یک دکمه برای بازگشت
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        button_layout.addWidget(self.back_button)

        left_layout.addLayout(button_layout)



        # Prediction result
        self.result_label = QLabel("Prediction: - MPa")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2E86AB;")
        self.result_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.result_label)

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Right panel - Charts
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Tab 1: Prediction Analysis
        self.tab1 = QWidget()
        tab1_layout = QVBoxLayout()

        self.canvas1 = MplCanvas(self, width=6, height=5, dpi=100)
        self.canvas2 = MplCanvas(self, width=6, height=5, dpi=100)

        tab1_layout.addWidget(QLabel("Feature Importance"))
        tab1_layout.addWidget(self.canvas1)
        tab1_layout.addWidget(QLabel("Residual Analysis"))
        tab1_layout.addWidget(self.canvas2)

        self.tab1.setLayout(tab1_layout)

        # Tab 2: Feature Relationships
        self.tab2 = QWidget()
        tab2_layout = QVBoxLayout()

        self.canvas3 = MplCanvas(self, width=6, height=5, dpi=100)
        self.canvas4 = MplCanvas(self, width=6, height=5, dpi=100)

        tab2_layout.addWidget(QLabel("Partial Dependence Plot"))
        tab2_layout.addWidget(self.canvas3)
        tab2_layout.addWidget(QLabel("Feature Correlation"))
        tab2_layout.addWidget(self.canvas4)

        self.tab2.setLayout(tab2_layout)

        self.tabs.addTab(self.tab1, "Prediction Analysis")
        self.tabs.addTab(self.tab2, "Feature Relationships")

        right_layout.addWidget(self.tabs)
        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

        # Initially hide charts
        self.tabs.setVisible(False)

    def predict(self):
        try:
            # Get input values
            input_data = {}
            for feature, field in self.input_fields.items():
                value = field.text()
                if not value:
                    QMessageBox.warning(self, "Input Error", f"Please enter a value for {feature}")
                    return
                input_data[feature] = float(value)

            # Create input array
            input_array = np.array([list(input_data.values())])

            # Scale if necessary
            if self.scaler:
                input_array = self.scaler.transform(input_array)

            # Predict
            prediction = self.model.predict(input_array)[0]
            self.current_prediction = prediction

            # Update result
            self.result_label.setText(f"Prediction: {prediction:.2f} MPa")

            # Show and update charts
            self.tabs.setVisible(True)
            self.update_charts(input_data, prediction)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def update_charts(self, input_data, prediction):
        # Clear all canvases
        for canvas in [self.canvas1, self.canvas2, self.canvas3, self.canvas4]:
            canvas.axes.clear()

        # Chart 1: Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            features = list(input_data.keys())
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            self.canvas1.axes.barh(range(len(indices)), importances[indices])
            self.canvas1.axes.set_yticks(range(len(indices)))
            self.canvas1.axes.set_yticklabels([features[i] for i in indices])
            self.canvas1.axes.set_title("Feature Importance")
            self.canvas1.draw()

        # Chart 2: Show prediction vs actual data distribution
        self.canvas2.axes.hist(self.df['Concrete compressive strength'], bins=30, alpha=0.7,
                               label='Training Data Distribution')
        self.canvas2.axes.axvline(prediction, color='red', linestyle='--', linewidth=2,
                                  label=f'Prediction: {prediction:.2f} MPa')
        self.canvas2.axes.set_xlabel("Compressive Strength (MPa)")
        self.canvas2.axes.set_ylabel("Frequency")
        self.canvas2.axes.set_title("Prediction vs Data Distribution")
        self.canvas2.axes.legend()
        self.canvas2.draw()

        # Chart 3: Show relationship with the most important feature
        if hasattr(self.model, 'feature_importances_'):
            top_feature_idx = np.argmax(self.model.feature_importances_)
            top_feature = list(input_data.keys())[top_feature_idx]

            self.canvas3.axes.scatter(self.df[top_feature], self.df['Concrete compressive strength'],
                                      alpha=0.5, label='Training Data')
            self.canvas3.axes.scatter(input_data[top_feature], prediction,
                                      color='red', s=100, label='Current Prediction')
            self.canvas3.axes.set_xlabel(top_feature)
            self.canvas3.axes.set_ylabel("Compressive Strength (MPa)")
            self.canvas3.axes.set_title(f"{top_feature} vs Strength")
            self.canvas3.axes.legend()
            self.canvas3.draw()

        # Chart 4: Feature Correlation
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=self.canvas4.axes,
                    cmap="coolwarm", center=0)
        self.canvas4.axes.set_title("Feature Correlation Matrix")
        self.canvas4.draw()

    def clear_inputs(self):
        for field in self.input_fields.values():
            field.clear()
        self.result_label.setText("Prediction: - MPa")
        self.tabs.setVisible(False)
        self.current_prediction = None

    def close_and_go_back(self):
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set style
    app.setStyle('Fusion')
    parent = None
    window = ConcretePredictorApp(parent)
    window.show()

    sys.exit(app.exec_())