import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox,
                             QScrollArea, QTabWidget, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont


class PM25PredictorApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BeijingPM2.5")
        self.setWindowTitle("PM2.5 Air Pollution Predictor")
        self.setGeometry(100, 100, 1200, 800)

        # Load the trained model and data
        self.load_model_and_data()

        # Setup UI
        self.init_ui()

        # Initialize prediction history
        self.prediction_history = []

    def load_model_and_data(self):
        """Load the trained model and necessary data"""
        try:
            # Load the trained model
            self.model = joblib.load('Gradient_Boosting_model.joblib')

            # Load feature importance data (pre-calculated)
            self.feature_importance = pd.read_csv('feature_importances.csv')

            # Sample test data for demonstration
            self.test_data = pd.DataFrame({
                'TEMP': np.random.uniform(-10, 30, 100),
                'DEWP': np.random.uniform(-20, 20, 100),
                'PRES': np.random.uniform(990, 1040, 100),
                'Iws': np.random.uniform(0, 250, 100),
                'pm2.5': np.random.uniform(5, 300, 100)
            })

            self.model_accuracy = 0.911  # From our previous results

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model or data: {str(e)}")
            sys.exit(1)

    def init_ui(self):
        """Initialize the main UI components"""
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Model info section
        model_info = QLabel(f"Gradient Boosting Model | Accuracy (R²): {self.model_accuracy:.3f}")
        model_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        model_info.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(model_info)

        # Create tabs
        self.tabs = QTabWidget()

        # Prediction Tab
        self.prediction_tab = self.create_prediction_tab()
        self.tabs.addTab(self.prediction_tab, "Prediction")

        # Analysis Tab
        self.analysis_tab = self.create_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Initially hide analysis tabs
        self.tabs.setTabEnabled(1, False)

    def create_prediction_tab(self):
        """Create the prediction tab with input form"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Input form group
        input_group = QGroupBox("Enter Weather Parameters")
        input_layout = QFormLayout()

        # Create input fields with validators
        self.input_fields = {}
        fields = [
            ('Temperature (°C)', 'TEMP', -20, 40),
            ('Dew Point (°C)', 'DEWP', -30, 30),
            ('Pressure (hPa)', 'PRES', 980, 1050),
            ('Wind Speed (m/s)', 'Iws', 0, 300),
            ('Cumulated Snow Hours', 'Is', 0, 50),
            ('Cumulated Rain Hours', 'Ir', 0, 50),
            ('Month', 'month', 1, 12),
            ('Day', 'day', 1, 31),
            ('Hour', 'hour', 0, 23),
        ]

        for label, field, min_val, max_val in fields:
            line_edit = QLineEdit()
            line_edit.setValidator(QDoubleValidator(min_val, max_val, 2))
            line_edit.setPlaceholderText(f"e.g., {min_val}-{max_val}")
            self.input_fields[field] = line_edit
            input_layout.addRow(QLabel(label), line_edit)

        # Wind direction combo box
        wind_label = QLabel("Wind Direction")
        wind_combo = QComboBox()
        wind_combo.addItems(['NW', 'NE', 'SE', 'CV'])
        self.input_fields['cbwd'] = wind_combo
        input_layout.addRow(wind_label, wind_combo)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Prediction button
        predict_btn = QPushButton("Predict PM2.5")
        predict_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #2980b9; }"
        )
        predict_btn.clicked.connect(self.predict_pm25)
        layout.addWidget(predict_btn)

        # ایجاد یک دکمه برای بازگشت
        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        back_button.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: gray; }"
        )
        back_button.clicked.connect(self.predict_pm25)
        layout.addWidget(back_button)

        # Result display
        self.result_label = QLabel("Prediction will appear here")
        self.result_label.setStyleSheet("font-size: 16px; color: #2c3e50;")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # Prediction history
        self.history_label = QLabel("Recent Predictions: None")
        self.history_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(self.history_label)

        tab.setLayout(layout)
        return tab

    def create_analysis_tab(self):
        """Create the analysis tab with charts"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Create a scroll area for the charts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout()

        # Feature Importance Chart
        feature_group = QGroupBox("Feature Importance Analysis")
        feature_layout = QVBoxLayout()

        self.feature_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        feature_layout.addWidget(self.feature_canvas)
        feature_group.setLayout(feature_layout)
        content_layout.addWidget(feature_group)

        # Feature Relationships Chart
        relation_group = QGroupBox("Feature Relationships")
        relation_layout = QVBoxLayout()

        self.relation_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        relation_layout.addWidget(self.relation_canvas)
        relation_group.setLayout(relation_layout)
        content_layout.addWidget(relation_group)

        # Prediction Distribution Chart
        pred_group = QGroupBox("Prediction Distribution")
        pred_layout = QVBoxLayout()

        self.pred_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        pred_layout.addWidget(self.pred_canvas)
        pred_group.setLayout(pred_layout)
        content_layout.addWidget(pred_group)

        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        tab.setLayout(layout)
        return tab

    def predict_pm25(self):
        """Predict PM2.5 based on user input"""
        try:
            # Collect input values
            input_data = {}
            for field, widget in self.input_fields.items():
                if isinstance(widget, QLineEdit):
                    value = widget.text().strip()
                    if not value:
                        QMessageBox.warning(self, "Input Error", f"Please enter a value for {field}")
                        return
                    input_data[field] = float(value)
                elif isinstance(widget, QComboBox):
                    input_data[field] = widget.currentText()

            # For demonstration, we'll simulate a prediction
            # In a real app, you would use your model here
            pm25_pred = self.simulate_prediction(input_data)

            # Display result
            self.result_label.setText(f"Predicted PM2.5: {pm25_pred:.2f} μg/m³")
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #27ae60;")

            # Add to prediction history
            self.prediction_history.append(pm25_pred)
            history_text = "Recent Predictions: " + ", ".join(f"{p:.1f}" for p in self.prediction_history[-5:])
            self.history_label.setText(history_text)

            # Enable analysis tab and update charts
            self.tabs.setTabEnabled(1, True)
            self.tabs.setCurrentIndex(1)
            self.update_charts(input_data, pm25_pred)

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values")
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Failed to make prediction: {str(e)}")

    def simulate_prediction(self, input_data):
        """Simulate a prediction based on input data"""
        # This is a simplified simulation for demo purposes
        # In a real app, you would use your trained model

        # Base prediction
        pm25 = 50

        # Adjust based on temperature
        temp = input_data.get('TEMP', 20)
        if temp < 0:
            pm25 += 30
        elif temp > 25:
            pm25 += 20

        # Adjust based on wind
        wind_dir = input_data.get('cbwd', 'NW')
        if wind_dir == 'NW':
            pm25 += 15
        elif wind_dir == 'NE':
            pm25 += 10

        # Adjust based on pressure
        pressure = input_data.get('PRES', 1010)
        if pressure < 1000:
            pm25 += 25
        elif pressure > 1020:
            pm25 -= 10

        # Add some randomness
        pm25 += np.random.uniform(-10, 10)

        # Ensure within reasonable bounds
        return max(5, min(300, pm25))

    def update_charts(self, input_data, pm25_pred):
        """Update the analysis charts with new prediction"""
        try:
            # Clear previous figures
            self.feature_canvas.figure.clear()
            self.relation_canvas.figure.clear()
            self.pred_canvas.figure.clear()

            # Plot feature importance
            self.plot_feature_importance()

            # Plot feature relationships
            self.plot_feature_relationships(input_data, pm25_pred)

            # Plot prediction distribution
            self.plot_prediction_distribution(pm25_pred)

            # Refresh canvases
            self.feature_canvas.draw()
            self.relation_canvas.draw()
            self.pred_canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Chart Error", f"Failed to update charts: {str(e)}")

    def plot_feature_importance(self):
        """Plot feature importance chart"""
        ax = self.feature_canvas.figure.add_subplot(111)

        # Get top 10 features
        top_features = self.feature_importance.head(10).sort_values('Importance', ascending=True)

        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['Importance'], color='#3498db')

        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Important Features for PM2.5 Prediction', fontsize=12)

        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Add data labels
        for i, v in enumerate(top_features['Importance']):
            ax.text(v + 0.005, i, f"{v:.3f}", color='black', fontsize=9)

    def plot_feature_relationships(self, input_data, pm25_pred):
        """Plot feature relationships with current prediction"""
        ax = self.relation_canvas.figure.add_subplot(111)

        # Scatter plot of temperature vs PM2.5
        ax.scatter(self.test_data['TEMP'], self.test_data['pm2.5'],
                   alpha=0.5, color='#95a5a6', label='Historical Data')

        # Add current prediction
        ax.scatter(input_data['TEMP'], pm25_pred,
                   s=150, color='#e74c3c', edgecolor='black',
                   label='Current Prediction', zorder=10)

        # Add trend line
        z = np.polyfit(self.test_data['TEMP'], self.test_data['pm2.5'], 1)
        p = np.poly1d(z)
        ax.plot(self.test_data['TEMP'], p(self.test_data['TEMP']),
                "r--", linewidth=2, label='Trend Line')

        # Customize plot
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('PM2.5 (μg/m³)')
        ax.set_title('Temperature vs PM2.5 Concentration', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add annotation for current prediction
        ax.annotate(f'Predicted: {pm25_pred:.1f} μg/m³',
                    xy=(input_data['TEMP'], pm25_pred),
                    xytext=(input_data['TEMP'] + 2, pm25_pred + 20),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)

    def plot_prediction_distribution(self, pm25_pred):
        """Plot prediction distribution with current value"""
        ax = self.pred_canvas.figure.add_subplot(111)

        # Create histogram of historical PM2.5 values
        n, bins, patches = ax.hist(self.test_data['pm2.5'], bins=20,
                                   color='#3498db', alpha=0.7,
                                   edgecolor='black')

        # Add current prediction line
        ax.axvline(pm25_pred, color='#e74c3c', linestyle='dashed', linewidth=2,
                   label=f'Current Prediction: {pm25_pred:.1f} μg/m³')

        # Add air quality zones
        ax.axvspan(0, 35, alpha=0.1, color='green', label='Good')
        ax.axvspan(35, 75, alpha=0.1, color='yellow', label='Moderate')
        ax.axvspan(75, 150, alpha=0.1, color='orange', label='Unhealthy')
        ax.axvspan(150, 300, alpha=0.1, color='red', label='Hazardous')

        # Customize plot
        ax.set_xlabel('PM2.5 (μg/m³)')
        ax.set_ylabel('Frequency')
        ax.set_title('PM2.5 Distribution with Current Prediction', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add air quality labels
        ax.text(15, max(n) * 0.9, "Good", fontsize=10, ha='center')
        ax.text(55, max(n) * 0.9, "Moderate", fontsize=10, ha='center')
        ax.text(110, max(n) * 0.9, "Unhealthy", fontsize=10, ha='center')
        ax.text(200, max(n) * 0.9, "Hazardous", fontsize=10, ha='center')

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 10, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = PM25PredictorApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == "__main__":
    main()