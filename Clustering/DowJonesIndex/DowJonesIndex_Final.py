import sys
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTabWidget, QComboBox, QMessageBox
)
from PyQt5.QtGui import QDoubleValidator, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns


class StockPredictionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Price Prediction")
        self.setGeometry(100, 100, 1200, 800)

        self.model = load_model('lstm_model.h5')
        self.scaler = joblib.load('scaler.joblib')
        self.label_encoder = joblib.load('label_encoder.joblib')

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.model_info = QLabel("Model: LSTM | Accuracy: 0.520")
        self.model_info.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.main_layout.addWidget(self.model_info)

        self.features = [
            'stock_encoded', 'close', 'volume_log', 'percent_change_price',
            'price_range', 'volume_ratio', 'volatility',
            'lag_percent_change_price', 'lag_2_percent_change_price',
            'ma_3_close', 'ema_3_close', 'lag_volatility', 'lag_volume_ratio'
        ]
        self.input_fields = {}
        self.input_layout = QVBoxLayout()
        self.main_layout.addLayout(self.input_layout)

        self.ranges = {
            'stock_encoded': (0, 4),
            'close': (0, 1),
            'volume_log': (0, 1),
            'percent_change_price': (-100, 100),
            'price_range': (0, 1),
            'volume_ratio': (0.5, 2),
            'volatility': (0, 1),
            'lag_percent_change_price': (-100, 100),
            'lag_2_percent_change_price': (-100, 100),
            'ma_3_close': (0, 1),
            'ema_3_close': (0, 1),
            'lag_volatility': (0, 1),
            'lag_volume_ratio': (0.5, 2)
        }

        instructions = {
            'stock_encoded': 'Select stock: 0 (^DJI), 1 (AAPL), 2 (MSFT), 3 (GOOGL), 4 (AMZN)',
            'close': 'Enter normalized close price (0 to 1)',
            'volume_log': 'Enter log-transformed volume (0 to 1)',
            'percent_change_price': 'Enter price change % (-100 to 100)',
            'price_range': 'Enter price range (0 to 1)',
            'volume_ratio': 'Enter volume ratio (0.5 to 2)',
            'volatility': 'Enter volatility (0 to 1)',
            'lag_percent_change_price': 'Enter lagged price change % (-100 to 100)',
            'lag_2_percent_change_price': 'Enter 2nd lagged price change % (-100 to 100)',
            'ma_3_close': 'Enter 3-day moving average of close (0 to 1)',
            'ema_3_close': 'Enter 3-day EMA of close (0 to 1)',
            'lag_volatility': 'Enter lagged volatility (0 to 1)',
            'lag_volume_ratio': 'Enter lagged volume ratio (0.5 to 2)'
        }

        for feature in self.features:
            layout = QHBoxLayout()
            label = QLabel(f"{feature}:")
            if feature == 'stock_encoded':
                input_field = QComboBox()
                input_field.addItems(['0', '1', '2', '3', '4'])
                input_field.setCurrentText('0')
            else:
                input_field = QLineEdit()
                input_field.setPlaceholderText(instructions[feature])
                validator = QDoubleValidator(self.ranges[feature][0], self.ranges[feature][1], 6)
                input_field.setValidator(validator)
            layout.addWidget(label)
            layout.addWidget(input_field)
            self.input_fields[feature] = input_field
            self.input_layout.addLayout(layout)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        self.main_layout.addWidget(self.predict_button)

        self.back_btn = QPushButton("Back to Main")
        self.back_btn.clicked.connect(self.go_back)
        self.main_layout.addWidget(self.back_btn)

        self.result_label = QLabel("Prediction: Not yet predicted")
        self.main_layout.addWidget(self.result_label)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.pred_analysis_tab = QWidget()
        self.pred_analysis_layout = QVBoxLayout(self.pred_analysis_tab)
        self.tabs.addTab(self.pred_analysis_tab, "Prediction Analysis")
        self.pred_canvas1 = None
        self.pred_canvas2 = None

        self.feat_relation_tab = QWidget()
        self.feat_relation_layout = QVBoxLayout(self.feat_relation_tab)
        self.tabs.addTab(self.feat_relation_tab, "Feature Relationship")
        self.rel_canvas1 = None
        self.rel_canvas2 = None

        self.predictions = []
        self.inputs = []

    def predict(self):
        try:
            input_data = []
            for feature in self.features:
                value = (
                    self.input_fields[feature].currentText()
                    if feature == 'stock_encoded'
                    else self.input_fields[feature].text()
                )
                if not value:
                    QMessageBox.critical(self, "Input Error", "All fields must be filled")
                    return
                val = float(value)
                if feature != 'stock_encoded' and (val < self.ranges[feature][0] or val > self.ranges[feature][1]):
                    QMessageBox.critical(self, "Input Error",
                                         f"{feature} must be between {self.ranges[feature][0]} and {self.ranges[feature][1]}")
                    return
                input_data.append(val)
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numbers")
            return

        arr = np.array([input_data] * 5, dtype=np.float32).reshape(1, 5, len(self.features))

        try:
            probs = self.model.predict(arr)
            pred = (probs > 0.5).astype(int)[0][0]
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Model prediction failed:\n{str(e)}")
            return

        self.predictions.append(pred)
        self.inputs.append(input_data)
        self.result_label.setText(f"Prediction: {'Up' if pred == 1 else 'Down'}")

        self.clear_charts()
        self.plot_prediction_analysis()
        self.plot_feature_relationship()

    def clear_charts(self):
        for canvas in [self.pred_canvas1, self.pred_canvas2, self.rel_canvas1, self.rel_canvas2]:
            if canvas:
                canvas.setParent(None)

        self.pred_canvas1 = self.pred_canvas2 = None
        self.rel_canvas1 = self.rel_canvas2 = None

    def plot_prediction_analysis(self):
        with plt.ioff():
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            counts = [self.predictions.count(0), self.predictions.count(1)]
            ax1.bar(['Down', 'Up'], counts, color=['red', 'green'], alpha=0.7)
            ax1.set_title('Prediction Distribution')
            self.pred_canvas1 = FigureCanvas(fig1)
            self.pred_analysis_layout.addWidget(self.pred_canvas1)

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            labels = ['Down', 'Up']
            sizes = [1 if self.predictions[-1] == 0 else 0, 1 if self.predictions[-1] == 1 else 0]
            ax2.pie(sizes, labels=labels, colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
            ax2.set_title('Latest Prediction')
            self.pred_canvas2 = FigureCanvas(fig2)
            self.pred_analysis_layout.addWidget(self.pred_canvas2)

    def plot_feature_relationship(self):
        if not self.inputs:
            return
        input_df = pd.DataFrame(self.inputs, columns=self.features)

        with plt.ioff():
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            corr = input_df.corr()
            sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax1)
            ax1.set_title('Feature Correlation Heatmap')
            self.rel_canvas1 = FigureCanvas(fig1)
            self.feat_relation_layout.addWidget(self.rel_canvas1)

            if len(input_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                top_features = ['close', 'percent_change_price']
                ax2.scatter(input_df[top_features[0]], input_df[top_features[1]], c=self.predictions, cmap='RdYlGn', alpha=0.7)
                ax2.set_title(f'{top_features[0]} vs {top_features[1]}')
                ax2.set_xlabel(top_features[0])
                ax2.set_ylabel(top_features[1])
                self.rel_canvas2 = FigureCanvas(fig2)
                self.feat_relation_layout.addWidget(self.rel_canvas2)

    def go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = StockPredictionUI()
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()