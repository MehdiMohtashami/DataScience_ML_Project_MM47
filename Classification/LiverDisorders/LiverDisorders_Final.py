
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTabWidget, QGridLayout, QMessageBox)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib


class LiverDisordersUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LiverDisorders")
        self.setWindowTitle("BUPA LiverDisorders Prediction")
        self.setGeometry(100, 100, 1200, 800)

        # لود مدل و پیش‌پردازش
        try:
            self.model = joblib.load('liver_disorders_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.selector = joblib.load('selector.pkl')
            self.le = joblib.load('label_encoder.pkl')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model files: {str(e)}")
            sys.exit(1)

        self.features = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks',
                         'Sgpt_Sgot_ratio', 'Gammagt_Alkphos_ratio', 'Sgpt_Gammagt']
        self.feature_ranges = {
            'Mcv': (80.0, 100.0), 'Alkphos': (20.0, 140.0), 'Sgpt': (10.0, 100.0),
            'Sgot': (10.0, 80.0), 'Gammagt': (10.0, 200.0), 'Drinks': (0.0, 20.0)
        }

        # لود دیتاست برای چارت‌ها
        try:
            self.df = pd.read_csv('bupa.data.csv')
        except FileNotFoundError:
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'
            self.df = pd.read_csv(url, header=None)
            self.df.columns = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks', 'Selector']

        # Feature Engineering برای دیتاست
        self.df['Sgpt_Sgot_ratio'] = self.df['Sgpt'] / (self.df['Sgot'] + 1e-5)
        self.df['Gammagt_Alkphos_ratio'] = self.df['Gammagt'] / (self.df['Alkphos'] + 1e-5)
        self.df['Sgpt_Gammagt'] = self.df['Sgpt'] * self.df['Gammagt']

        # رابط کاربری
        self.init_ui()

        # متغیر برای ذخیره پیش‌بینی‌ها
        self.last_prediction = None
        self.canvases = []  # لیست برای مدیریت چارت‌ها

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # عنوان و اطلاعات مدل
        model_info = QLabel("Model: Random Forest\nAccuracy: 0.70 | F1 Score: 0.70 | CV Score: 0.78")
        model_info.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        main_layout.addWidget(model_info, alignment=Qt.AlignCenter)

        # تب‌ها
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # تب ورودی و پیش‌بینی
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        self.tabs.addTab(input_widget, "Input & Prediction")

        # فیلدهای ورودی با محدوده
        input_grid = QGridLayout()
        self.input_fields = {}
        for i, feature in enumerate(['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']):
            label = QLabel(f"{feature} ({self.feature_ranges[feature][0]}-{self.feature_ranges[feature][1]}):")
            input_field = QLineEdit()
            validator = QDoubleValidator(self.feature_ranges[feature][0], self.feature_ranges[feature][1], 2)
            input_field.setValidator(validator)
            input_grid.addWidget(label, i, 0)
            input_grid.addWidget(input_field, i, 1)
            self.input_fields[feature] = input_field
        input_layout.addLayout(input_grid)

        # دکمه پیش‌بینی
        predict_button = QPushButton("Make Prediction")
        predict_button.setStyleSheet(
            "background-color: #3498db; color: white; padding: 15px; min-width: 150px; font-size: 16px; font-weight: bold;")
        predict_button.clicked.connect(self.predict)
        input_layout.addWidget(predict_button, alignment=Qt.AlignCenter)

        back_button = QPushButton("Back to Main", self)
        back_button.setStyleSheet(
            "background-color: gray; color: white; padding: 15px; min-width: 150px; font-size: 16px; font-weight: bold;")
        back_button.clicked.connect(self.close_and_go_back)
        input_layout.addWidget(back_button, alignment=Qt.AlignCenter)


        # نتیجه پیش‌بینی
        self.result_label = QLabel("Prediction: None")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #27ae60;")
        input_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        # تب Prediction Analysis
        self.prediction_tab = QWidget()
        self.prediction_layout = QHBoxLayout(self.prediction_tab)
        self.tabs.addTab(self.prediction_tab, "Prediction Analysis")

        # تب Feature Importance
        self.importance_tab = QWidget()
        self.importance_layout = QHBoxLayout(self.importance_tab)
        self.tabs.addTab(self.importance_tab, "Feature Importance")

        # تب Feature Relationship
        self.relationship_tab = QWidget()
        self.relationship_layout = QHBoxLayout(self.relationship_tab)
        self.tabs.addTab(self.relationship_tab, "Feature Relationship")

    def predict(self):
        try:
            # گرفتن مقادیر ورودی و چک بازه
            inputs = {}
            for feature in ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']:
                value = self.input_fields[feature].text()
                if not value or float(value) < self.feature_ranges[feature][0] or float(value) > \
                        self.feature_ranges[feature][1]:
                    QMessageBox.warning(self, "Error",
                                        f"Please enter a value for {feature} between {self.feature_ranges[feature][0]} and {self.feature_ranges[feature][1]}")
                    self.clear_charts()
                    return
                inputs[feature] = float(value)

            # Feature Engineering
            inputs['Sgpt_Sgot_ratio'] = inputs['Sgpt'] / (inputs['Sgot'] + 1e-5)
            inputs['Gammagt_Alkphos_ratio'] = inputs['Gammagt'] / (inputs['Alkphos'] + 1e-5)
            inputs['Sgpt_Gammagt'] = inputs['Sgpt'] * inputs['Gammagt']

            # آماده‌سازی داده و پیش‌بینی
            input_df = pd.DataFrame([inputs], columns=self.features)
            X_new = self.selector.transform(input_df)
            X_new_scaled = self.scaler.transform(X_new)
            pred = self.model.predict(X_new_scaled)
            pred_label = self.le.inverse_transform(pred)[0]
            self.last_prediction = {'inputs': inputs, 'label': pred_label}

            # نمایش نتیجه
            self.result_label.setText(f"Prediction: Selector = {pred_label}")

            # پاک کردن و به‌روزرسانی چارت‌ها
            self.clear_charts()
            self.update_charts()

        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"Error at {pd.Timestamp.now()}: {str(e)}\n")
            self.result_label.setText(f"Error: {str(e)}")
            self.clear_charts()

    def clear_charts(self):
        # پاک کردن چارت‌های قبلی
        for canvas in self.canvases:
            if canvas:
                canvas.figure.clear()
                canvas.deleteLater()
        self.canvases = []

        # به‌روزرسانی layout
        for layout in [self.prediction_layout, self.importance_layout, self.relationship_layout]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def update_charts(self):
        if not self.last_prediction:
            return

        # Prediction Analysis
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sns.scatterplot(data=self.df, x='Sgpt', y='Gammagt', hue='Selector', style='Selector', palette='deep', ax=ax1)
        ax1.scatter(self.last_prediction['inputs']['Sgpt'], self.last_prediction['inputs']['Gammagt'],
                    c='red', marker='*', s=200, label='Prediction')
        ax1.set_title('Prediction vs Dataset (Sgpt vs Gammagt)')
        ax1.legend()

        means = self.df[self.features[:6]].mean()
        user_inputs = [self.last_prediction['inputs'][f] for f in self.features[:6]]
        x = np.arange(len(self.features[:6]))
        width = 0.35
        ax2.bar(x - width / 2, means, width, label='Dataset Mean', color='skyblue')
        ax2.bar(x + width / 2, user_inputs, width, label='User Input', color='salmon')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.features[:6], rotation=45)
        ax2.set_title('User Input vs Dataset Mean')
        ax2.legend()
        canvas1 = FigureCanvas(fig1)
        self.prediction_layout.addWidget(canvas1)
        self.canvases.append(canvas1)

        # Feature Importance
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
        importances = self.model.feature_importances_
        feature_names = [self.features[i] for i in range(len(importances))]
        sns.barplot(x=importances, y=feature_names, hue=feature_names, palette='viridis', ax=ax3, legend=False)
        ax3.set_title('Feature Importance (Bar)')

        ax4.pie(importances, labels=feature_names, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax4.set_title('Feature Importance (Pie)')
        canvas2 = FigureCanvas(fig2)
        self.importance_layout.addWidget(canvas2)
        self.canvases.append(canvas2)

        # Feature Relationship
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 4))
        corr_matrix = self.df[self.features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5)
        ax5.set_title('Feature Correlation Heatmap')

        sns.scatterplot(data=self.df, x='Sgpt', y='Sgot', hue='Selector', style='Selector', palette='dark', ax=ax6)
        ax6.scatter(self.last_prediction['inputs']['Sgpt'], self.last_prediction['inputs']['Sgot'],
                    c='red', marker='*', s=200, label='Prediction')
        ax6.set_title('Feature Relationship (Sgpt vs Sgot)')
        ax6.legend()
        canvas3 = FigureCanvas(fig3)
        self.relationship_layout.addWidget(canvas3)
        self.canvases.append(canvas3)

    def close_and_go_back(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    parent = None
    window = LiverDisordersUI(parent)
    window.show()
    sys.exit(app.exec_())