import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTabWidget, QMessageBox, QComboBox)
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class EnergyEfficiencyUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EnergyEfficiency")
        self.setWindowTitle("EnergyEfficiency Prediction")
        self.setGeometry(100, 100, 1200, 800)

        # بارگذاری مدل
        self.model_path = 'best_rf_model.joblib'
        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "Error", f"Model file {self.model_path} not found!")
            sys.exit()
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading model: {e}")
            sys.exit()

        # ویجت اصلی
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # بخش ورودی‌ها (چپ)
        self.input_widget = QWidget()
        self.input_layout = QVBoxLayout(self.input_widget)
        self.main_layout.addWidget(self.input_widget, 1)

        # اطلاعات مدل
        self.model_info = QLabel(
            "Model: Random Forest\nR²: 0.9977\nMAE: 0.3563 kWh/m²",
            alignment=Qt.AlignCenter
        )
        self.model_info.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        self.input_layout.addWidget(self.model_info)

        # ورودی‌ها با بازه‌های دقیق
        self.inputs = {}
        input_ranges = {
            'X1': (0.62, 0.98, "Relative Compactness (0.62-0.98)"),
            'X2': (514.5, 808.5, "Surface Area (514.5-808.5)"),
            'X3': (245.0, 416.5, "Wall Area (245.0-416.5)"),
            'X4': (110.25, 220.5, "Roof Area (110.25-220.5)"),
            'X5': ([3.5, 7.0], "Overall Height (3.5 or 7.0)"),
            'X6': (2, 5, "Orientation (2, 3, 4, 5)"),
            'X7': (0.0, 0.4, "Glazing Area (0.0-0.4)"),
            'X8': (0, 5, "Glazing Area Distribution (0-5)")
        }

        for key, value in input_ranges.items():
            lbl = QLabel(value[-1])
            lbl.setStyleSheet("font-size: 14px; color: #34495e;")
            self.input_layout.addWidget(lbl)

            if key == 'X5':  # برای X5 فقط 3.5 یا 7.0
                combo = QComboBox()
                combo.addItems(['3.5', '7.0'])
                combo.setStyleSheet("font-size: 14px; padding: 5px;")
                self.inputs[key] = combo
            else:
                line_edit = QLineEdit()
                line_edit.setStyleSheet("font-size: 14px; padding: 5px;")
                if key in ['X6', 'X8']:  # اعداد صحیح
                    validator = QIntValidator(int(value[0]), int(value[1]))
                else:  # اعداد اعشاری
                    validator = QDoubleValidator(value[0], value[1], 2)
                    validator.setNotation(QDoubleValidator.StandardNotation)
                line_edit.setValidator(validator)
                self.inputs[key] = line_edit
            self.input_layout.addWidget(self.inputs[key])

        # دکمه پیش‌بینی
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.predict_btn.clicked.connect(self.predict)
        self.input_layout.addWidget(self.predict_btn)

        # ایجاد یک دکمه برای بازگشت
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        self.back_button.setStyleSheet("""
                    QPushButton {
                        background-color: gray;
                        color: white;
                        font-size: 16px;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: gray;
                    }
                """)
        self.input_layout.addWidget(self.back_button)

        # نتیجه پیش‌بینی
        self.result_label = QLabel("Prediction: N/A", alignment=Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; color: #e74c3c; font-weight: bold;")
        self.input_layout.addWidget(self.result_label)
        self.input_layout.addStretch()

        # بخش چارت‌ها (راست)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget, 2)

        # تب Prediction Analysis
        self.pred_analysis_tab = QWidget()
        self.pred_analysis_layout = QVBoxLayout(self.pred_analysis_tab)
        self.tab_widget.addTab(self.pred_analysis_tab, "Prediction Analysis")

        # تب Feature Importance
        self.feat_importance_tab = QWidget()
        self.feat_importance_layout = QVBoxLayout(self.feat_importance_tab)
        self.tab_widget.addTab(self.feat_importance_tab, "Feature Importance")

        # تب Feature Relationship
        self.feat_relation_tab = QWidget()
        self.feat_relation_layout = QVBoxLayout(self.feat_relation_tab)
        self.tab_widget.addTab(self.feat_relation_tab, "Feature Relationship")

        # ذخیره پیش‌بینی‌ها
        self.prediction = None
        self.input_data = None

    def predict(self):
        # پاکسازی چارت‌های قبلی
        for i in reversed(range(self.pred_analysis_layout.count())):
            self.pred_analysis_layout.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.feat_importance_layout.count())):
            self.feat_importance_layout.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.feat_relation_layout.count())):
            self.feat_relation_layout.itemAt(i).widget().deleteLater()

        # گرفتن ورودی‌ها
        try:
            input_values = []
            for key in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
                if key == 'X5':
                    value = float(self.inputs[key].currentText())
                else:
                    value = self.inputs[key].text()
                    if not value:
                        QMessageBox.warning(self, "Input Error", f"Please enter a value for {key}")
                        return
                    value = float(value)
                    # چک بازه
                    if key in ['X1', 'X2', 'X3', 'X4', 'X7']:
                        min_val, max_val = {
                            'X1': (0.62, 0.98), 'X2': (514.5, 808.5), 'X3': (245.0, 416.5),
                            'X4': (110.25, 220.5), 'X7': (0.0, 0.4)
                        }[key]
                        if not (min_val <= value <= max_val):
                            QMessageBox.warning(self, "Input Error", f"{key} must be between {min_val} and {max_val}")
                            return
                    elif key in ['X6', 'X8']:
                        min_val, max_val = {'X6': (2, 5), 'X8': (0, 5)}[key]
                        if not (min_val <= value <= max_val and value.is_integer()):
                            QMessageBox.warning(self, "Input Error",
                                                f"{key} must be an integer between {min_val} and {max_val}")
                            return
                input_values.append(value)
            self.input_data = pd.DataFrame([input_values],
                                           columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])

            # پیش‌بینی
            self.prediction = self.model.predict(self.input_data)[0]
            self.result_label.setText(f"Predicted Heating Load: {self.prediction:.2f} kWh/m²")

            # نمایش چارت‌ها
            self.show_prediction_analysis()
            self.show_feature_importance()
            self.show_feature_relationship()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction error: {e}")

    def show_prediction_analysis(self):
        # چارت ۱: Scatter Plot (پیش‌بینی در مقابل مقادیر نمونه)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sample_data = pd.read_csv('ENB2012_data.csv').sample(20, random_state=42)
        sns.scatterplot(x=sample_data.index, y=sample_data['Y1'], ax=ax1, label='Sample Data', color='blue')
        ax1.scatter([0], [self.prediction], color='red', s=100, label='Prediction')
        ax1.set_title('Prediction vs Sample Data')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Heating Load (kWh/m²)')
        ax1.legend()
        canvas1 = FigureCanvas(fig1)
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.addWidget(canvas1)
        self.pred_analysis_layout.addWidget(canvas_widget)
        plt.close(fig1)

        # چارت ۲: Violin Plot (توزیع پیش‌بینی در مقابل داده‌ها)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        data = pd.read_csv('ENB2012_data.csv')['Y1'].values
        sns.violinplot(y=data, ax=ax2, color='lightblue')
        ax2.axhline(self.prediction, color='red', linestyle='--', label='Prediction')
        ax2.set_title('Prediction in Data Distribution')
        ax2.set_ylabel('Heating Load (kWh/m²)')
        ax2.legend()
        canvas2 = FigureCanvas(fig2)
        canvas_widget2 = QWidget()
        canvas_layout2 = QVBoxLayout(canvas_widget2)
        canvas_layout2.addWidget(canvas2)
        self.pred_analysis_layout.addWidget(canvas_widget2)
        plt.close(fig2)

    def show_feature_importance(self):
        # چارت ۱: Bar Plot
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        importances = self.model.feature_importances_
        features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
        sns.barplot(x=importances, y=features, hue=features, ax=ax1, palette='viridis', legend=False)
        ax1.set_title('Feature Importance (Bar)')
        ax1.set_xlabel('Importance')
        canvas1 = FigureCanvas(fig1)
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.addWidget(canvas1)
        self.feat_importance_layout.addWidget(canvas_widget)
        plt.close(fig1)

        # چارت ۲: Pie Chart
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(importances, labels=features, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax2.set_title('Feature Importance (Pie)')
        canvas2 = FigureCanvas(fig2)
        canvas_widget2 = QWidget()
        canvas_layout2 = QVBoxLayout(canvas_widget2)
        canvas_layout2.addWidget(canvas2)
        self.feat_importance_layout.addWidget(canvas_widget2)
        plt.close(fig2)

    def show_feature_relationship(self):
        # چارت ۱: Scatter Plot (X1 vs Y1 با پیش‌بینی)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        data = pd.read_csv('ENB2012_data.csv')
        sns.scatterplot(x='X1', y='Y1', data=data, ax=ax1, label='Data', color='blue')
        ax1.scatter(self.input_data['X1'], self.prediction, color='red', s=100, label='Prediction')
        ax1.set_title('Relative Compactness vs Heating Load')
        ax1.set_xlabel('Relative Compactness (X1)')
        ax1.set_ylabel('Heating Load (kWh/m²)')
        ax1.legend()
        canvas1 = FigureCanvas(fig1)
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.addWidget(canvas1)
        self.feat_relation_layout.addWidget(canvas_widget)
        plt.close(fig1)

        # چارت ۲: Heatmap (همبستگی ویژگی‌ها)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        corr = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Feature Correlation Heatmap')
        canvas2 = FigureCanvas(fig2)
        canvas_widget2 = QWidget()
        canvas_layout2 = QVBoxLayout(canvas_widget2)
        canvas_layout2.addWidget(canvas2)
        self.feat_relation_layout.addWidget(canvas_widget2)
        plt.close(fig2)

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = EnergyEfficiencyUI(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()