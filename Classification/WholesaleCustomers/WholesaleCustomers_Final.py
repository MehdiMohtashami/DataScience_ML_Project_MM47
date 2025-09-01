import sys
import pandas as pd
import numpy as np
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler

class WholesalePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model_and_data()
        self.charts_visible = False
        self.user_input = None

    def initUI(self):
        self.setWindowTitle('Wholesale Customer Channel Predictor')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QHBoxLayout()

        # Left side: Inputs and Prediction
        left_layout = QVBoxLayout()

        # Model info at top
        self.model_label = QLabel('Model: Logistic Regression\nAccuracy: 92% (Test Set)')
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(self.model_label)

        # Inputs with placeholders for typical ranges
        validator = QDoubleValidator(0, 1e6, 0)  # Only positive numbers

        self.fresh_input = QLineEdit()
        self.fresh_input.setPlaceholderText('Enter Fresh (e.g., 3 to 112151)')
        self.fresh_input.setValidator(validator)
        left_layout.addWidget(QLabel('Fresh:'))
        left_layout.addWidget(self.fresh_input)

        self.milk_input = QLineEdit()
        self.milk_input.setPlaceholderText('Enter Milk (e.g., 55 to 73498)')
        self.milk_input.setValidator(validator)
        left_layout.addWidget(QLabel('Milk:'))
        left_layout.addWidget(self.milk_input)

        self.grocery_input = QLineEdit()
        self.grocery_input.setPlaceholderText('Enter Grocery (e.g., 3 to 92780)')
        self.grocery_input.setValidator(validator)
        left_layout.addWidget(QLabel('Grocery:'))
        left_layout.addWidget(self.grocery_input)

        self.frozen_input = QLineEdit()
        self.frozen_input.setPlaceholderText('Enter Frozen (e.g., 25 to 60869)')
        self.frozen_input.setValidator(validator)
        left_layout.addWidget(QLabel('Frozen:'))
        left_layout.addWidget(self.frozen_input)

        self.detergents_input = QLineEdit()
        self.detergents_input.setPlaceholderText('Enter Detergents_Paper (e.g., 3 to 40827)')
        self.detergents_input.setValidator(validator)
        left_layout.addWidget(QLabel('Detergents_Paper:'))
        left_layout.addWidget(self.detergents_input)

        self.delicassen_input = QLineEdit()
        self.delicassen_input.setPlaceholderText('Enter Delicassen (e.g., 3 to 47943)')
        self.delicassen_input.setValidator(validator)
        left_layout.addWidget(QLabel('Delicassen:'))
        left_layout.addWidget(self.delicassen_input)

        # Predict button
        predict_btn = QPushButton('Predict Channel')
        predict_btn.clicked.connect(self.predict)
        left_layout.addWidget(predict_btn)

        # Prediction result
        self.result_label = QLabel('Prediction: ')
        self.result_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        left_layout.addWidget(self.result_label)

        main_layout.addLayout(left_layout)

        # Right side: Tabs for charts (initially hidden)
        self.tab_widget = QTabWidget()
        self.tab_widget.setVisible(False)

        # Tab 1: Prediction Analysis
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout()

        # Chart 1: Bar plot for Feature Importance
        self.fig_importance = Figure(figsize=(5, 4))
        self.canvas_importance = FigureCanvas(self.fig_importance)
        analysis_layout.addWidget(self.canvas_importance)

        # Chart 2: Radar chart for User Input vs Means
        self.fig_radar = Figure(figsize=(5, 4))
        self.canvas_radar = FigureCanvas(self.fig_radar)
        analysis_layout.addWidget(self.canvas_radar)

        analysis_widget.setLayout(analysis_layout)
        self.tab_widget.addTab(analysis_widget, "Prediction Analysis (Importance & Comparison)")

        # Tab 2: Feature Relationships
        relations_widget = QWidget()
        relations_layout = QVBoxLayout()

        # Chart 3: Heatmap for Correlations
        self.fig_heatmap = Figure(figsize=(5, 4))
        self.canvas_heatmap = FigureCanvas(self.fig_heatmap)
        relations_layout.addWidget(self.canvas_heatmap)

        # Chart 4: Scatter plot with user point
        self.fig_scatter = Figure(figsize=(5, 4))
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        relations_layout.addWidget(self.canvas_scatter)

        relations_widget.setLayout(relations_layout)
        self.tab_widget.addTab(relations_widget, "Feature Relationships (Heatmap & Scatter)")

        main_layout.addWidget(self.tab_widget)

        self.setLayout(main_layout)

    def load_model_and_data(self):
        try:
            self.model = joblib.load('logistic_regression_model.joblib')
            self.scaler = joblib.load('scaler_class.joblib')
            # Load data for means and correlations
            self.df = pd.read_csv('Wholesale customers data.csv')
            self.features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
            # Compute means for radar
            self.mean_channel1 = self.df[self.df['Channel'] == 1][self.features].mean()
            self.mean_channel2 = self.df[self.df['Channel'] == 2][self.features].mean()
            # Correlation matrix
            self.corr_matrix = self.df[self.features].corr()
        except FileNotFoundError:
            QMessageBox.warning(self, 'Error', 'Model, scaler, or data not found! Ensure files are present.')

    def predict(self):
        try:
            inputs = [
                float(self.fresh_input.text()),
                float(self.milk_input.text()),
                float(self.grocery_input.text()),
                float(self.frozen_input.text()),
                float(self.detergents_input.text()),
                float(self.delicassen_input.text())
            ]
            self.user_input = np.array([inputs])
            scaled_input = self.scaler.transform(self.user_input)

            prediction = self.model.predict(scaled_input)[0]
            channel = 'Horeca (1)' if prediction == 1 else 'Retail (2)'

            self.result_label.setText(f'Prediction: {channel}')

            # Show charts
            self.tab_widget.setVisible(True)
            self.update_charts(prediction)

        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter valid positive numbers for all fields!')

    def update_charts(self, prediction):
        # Clear previous plots
        self.fig_importance.clf()
        self.fig_radar.clf()
        self.fig_heatmap.clf()
        self.fig_scatter.clf()

        # Chart 1: Feature Importance (Bar Plot)
        ax_importance = self.fig_importance.add_subplot(111)
        coef = np.abs(self.model.coef_[0])
        ax_importance.bar(self.features, coef, color='skyblue')
        ax_importance.set_title('Feature Importance (Absolute Coefficients)')
        ax_importance.set_ylabel('Importance')
        ax_importance.set_xticklabels(self.features, rotation=45, ha='right')
        self.canvas_importance.draw()

        # Chart 2: Radar Chart (User vs Means)
        ax_radar = self.fig_radar.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(self.features), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        user_values = np.concatenate((self.user_input[0], [self.user_input[0][0]]))
        mean1_values = np.concatenate((self.mean_channel1.values, [self.mean_channel1[0]]))
        mean2_values = np.concatenate((self.mean_channel2.values, [self.mean_channel2[0]]))

        ax_radar.plot(angles, user_values, label='User Input', linewidth=2)
        ax_radar.fill(angles, user_values, alpha=0.25)
        ax_radar.plot(angles, mean1_values, label='Mean Horeca (1)', linewidth=1, linestyle='--')
        ax_radar.plot(angles, mean2_values, label='Mean Retail (2)', linewidth=1, linestyle='--')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(self.features)
        ax_radar.set_title('User Input vs Channel Means (Radar Chart)')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        self.canvas_radar.draw()

        # Chart 3: Correlation Heatmap
        ax_heatmap = self.fig_heatmap.add_subplot(111)
        im = ax_heatmap.imshow(self.corr_matrix, cmap='coolwarm')
        ax_heatmap.set_xticks(np.arange(len(self.features)))
        ax_heatmap.set_yticks(np.arange(len(self.features)))
        ax_heatmap.set_xticklabels(self.features, rotation=45, ha='right')
        ax_heatmap.set_yticklabels(self.features)
        for i in range(len(self.features)):
            for j in range(len(self.features)):
                ax_heatmap.text(j, i, f'{self.corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')
        self.fig_heatmap.colorbar(im, ax=ax_heatmap)
        ax_heatmap.set_title('Feature Correlation Heatmap')
        self.canvas_heatmap.draw()

        # Chart 4: Scatter Plot (Grocery vs Detergents_Paper) with user point
        ax_scatter = self.fig_scatter.add_subplot(111)
        ax_scatter.scatter(self.df['Grocery'], self.df['Detergents_Paper'], c=self.df['Channel'], cmap='viridis', alpha=0.5, label='Data Points')
        ax_scatter.scatter(self.user_input[0][2], self.user_input[0][4], color='red', marker='x', s=100, label='User Prediction')
        ax_scatter.set_xlabel('Grocery')
        ax_scatter.set_ylabel('Detergents_Paper')
        ax_scatter.set_title('Scatter: Grocery vs Detergents_Paper (User Point in Red)')
        ax_scatter.legend()
        self.canvas_scatter.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WholesalePredictor()
    ex.show()
    sys.exit(app.exec_())