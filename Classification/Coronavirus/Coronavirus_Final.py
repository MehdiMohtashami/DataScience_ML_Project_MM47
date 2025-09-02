import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QComboBox, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont, QIntValidator
import warnings

# Filtering alerts
warnings.filterwarnings("ignore")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setWindowTitle("Coronavirus")

    def clear_axes(self):
        """Clear all existing axes"""
        for ax in self.fig.axes:
            self.fig.delaxes(ax)

    def create_axes(self):
        """Creating new axes"""
        self.clear_axes()
        return self.fig.add_subplot(111)


class CovidPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.le_country = None
        self.le_province = None
        self.features = None
        self.current_prediction = None
        self.current_colorbar = None
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle('COVID-19 Prediction Dashboard')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info section
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout(model_info_group)

        model_name = QLabel("Random Forest Classifier")
        model_name.setFont(QFont("Arial", 12, QFont.Bold))
        model_accuracy = QLabel("Accuracy: 91.08%")
        model_accuracy.setFont(QFont("Arial", 10))
        model_desc = QLabel("Predicts if COVID-19 cases will double in next 7 days based on current statistics.")
        model_desc.setWordWrap(True)

        model_info_layout.addWidget(model_name)
        model_info_layout.addWidget(model_accuracy)
        model_info_layout.addWidget(model_desc)

        # Input section
        input_group = QGroupBox("Input Parameters")
        input_layout = QGridLayout(input_group)

        self.input_fields = {}
        labels = [
            ('Latitude', 'Lat', 'e.g., 31.8 (-90 to 90)'),
            ('Longitude', 'Long', 'e.g., 117.2 (-180 to 180)'),
            ('Confirmed', 'Confirmed', 'Confirmed cases (≥ 0)'),
            ('Deaths', 'Deaths', 'Death count (≥ 0)'),
            ('Recovered', 'Recovered', 'Recovered cases (≥ 0)'),
            ('Day of Week', 'Day_of_Week', '0-6 (0=Mon)'),
            ('Month', 'Month', '1-12 (1=Jan)')
        ]

        row = 0
        for label_text, field_name, placeholder in labels:
            label = QLabel(label_text)
            input_field = QLineEdit()
            input_field.setPlaceholderText(placeholder)

            # تنظیم validator مناسب برای هر فیلد
            if field_name in ['Lat', 'Long']:
                input_field.setValidator(
                    QDoubleValidator(-90, 90, 6) if field_name == 'Lat' else QDoubleValidator(-180, 180, 6))
            elif field_name in ['Confirmed', 'Deaths', 'Recovered']:
                input_field.setValidator(QIntValidator(0, 1000000))
            elif field_name == 'Day_of_Week':
                input_field.setValidator(QIntValidator(0, 6))
            elif field_name == 'Month':
                input_field.setValidator(QIntValidator(1, 12))

            self.input_fields[field_name] = input_field

            input_layout.addWidget(label, row, 0)
            input_layout.addWidget(input_field, row, 1)
            row += 1

        # Country and Province selection
        country_label = QLabel("Country/Region")
        self.country_combo = QComboBox()
        province_label = QLabel("Province/State")
        self.province_combo = QComboBox()

        input_layout.addWidget(country_label, row, 0)
        input_layout.addWidget(self.country_combo, row, 1)
        row += 1
        input_layout.addWidget(province_label, row, 0)
        input_layout.addWidget(self.province_combo, row, 1)

        # Predict button
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.predict)
        predict_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)
        back_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-weight: bold; }")

        # Add sections to left layout
        left_layout.addWidget(model_info_group)
        left_layout.addWidget(input_group)
        left_layout.addWidget(predict_btn)
        left_layout.addWidget(back_button)
        left_layout.addStretch()

        # Right panel for charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget for different chart types
        self.tab_widget = QTabWidget()

        # Prediction Analysis tab
        self.prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(self.prediction_tab)

        self.prediction_result = QLabel("Please make a prediction to see results")
        self.prediction_result.setAlignment(Qt.AlignCenter)
        self.prediction_result.setFont(QFont("Arial", 14, QFont.Bold))

        self.probability_canvas = MplCanvas(self, width=5, height=4, dpi=100)

        prediction_layout.addWidget(self.prediction_result)
        prediction_layout.addWidget(self.probability_canvas)

        # Feature Importance tab
        self.importance_tab = QWidget()
        importance_layout = QVBoxLayout(self.importance_tab)

        self.importance_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        importance_layout.addWidget(self.importance_canvas)

        # Feature Relationships tab
        self.relationships_tab = QWidget()
        relationships_layout = QVBoxLayout(self.relationships_tab)

        # Combo boxes for feature selection
        relationship_controls = QWidget()
        controls_layout = QHBoxLayout(relationship_controls)

        controls_layout.addWidget(QLabel("X-axis:"))
        self.x_feature_combo = QComboBox()
        self.x_feature_combo.currentIndexChanged.connect(self.update_relationship_chart)
        controls_layout.addWidget(self.x_feature_combo)

        controls_layout.addWidget(QLabel("Y-axis:"))
        self.y_feature_combo = QComboBox()
        self.y_feature_combo.currentIndexChanged.connect(self.update_relationship_chart)
        controls_layout.addWidget(self.y_feature_combo)

        controls_layout.addWidget(QLabel("Color by:"))
        self.color_feature_combo = QComboBox()
        self.color_feature_combo.currentIndexChanged.connect(self.update_relationship_chart)
        controls_layout.addWidget(self.color_feature_combo)

        self.relationship_canvas = MplCanvas(self, width=5, height=4, dpi=100)

        relationships_layout.addWidget(relationship_controls)
        relationships_layout.addWidget(self.relationship_canvas)

        # Add tabs to tab widget
        self.tab_widget.addTab(self.prediction_tab, "Prediction Analysis")
        self.tab_widget.addTab(self.importance_tab, "Feature Importance")
        self.tab_widget.addTab(self.relationships_tab, "Feature Relationships")

        # Initially hide the tab widget
        self.tab_widget.setVisible(False)

        right_layout.addWidget(self.tab_widget)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Apply styles
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                padding: 2px;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                background: #e0e0e0;
                padding: 8px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #4CAF50;
            }
        """)

    def load_model(self):
        try:
            self.model = joblib.load('random_forest_covid_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.le_country = joblib.load('label_encoder_country.pkl')
            self.le_province = joblib.load('label_encoder_province.pkl')

            # Load feature names
            with open('feature_names.txt', 'r') as f:
                self.features = [line.strip() for line in f.readlines()]

            # Populate country and province combos
            self.country_combo.addItems(self.le_country.classes_)
            self.province_combo.addItems(self.le_province.classes_)

            # Populate feature combos for relationship tab
            numeric_features = ['Lat', 'Long', 'Confirmed', 'Deaths', 'Recovered',
                                'Mortality_Rate', 'Recovery_Rate', 'Active_Cases']
            self.x_feature_combo.addItems(numeric_features)
            self.y_feature_combo.addItems(numeric_features)
            self.color_feature_combo.addItems(['None'] + numeric_features)

            # Set default selections
            self.x_feature_combo.setCurrentText('Confirmed')
            self.y_feature_combo.setCurrentText('Deaths')
            self.color_feature_combo.setCurrentText('Recovery_Rate')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def validate_inputs(self):
        required_fields = ['Lat', 'Long', 'Confirmed', 'Deaths', 'Recovered', 'Day_of_Week', 'Month']

        for field in required_fields:
            if not self.input_fields[field].text():
                QMessageBox.warning(self, "Missing Input", f"Please enter a value for {field}")
                return False

            try:
                value = float(self.input_fields[field].text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Please enter a valid number for {field}")
                return False

            # Validate ranges
            if field == 'Lat' and (value < -90 or value > 90):
                QMessageBox.warning(self, "Invalid Input", "Latitude must be between -90 and 90")
                return False

            if field == 'Long' and (value < -180 or value > 180):
                QMessageBox.warning(self, "Invalid Input", "Longitude must be between -180 and 180")
                return False

            if field in ['Confirmed', 'Deaths', 'Recovered'] and value < 0:
                QMessageBox.warning(self, "Invalid Input", f"{field} must be a non-negative number")
                return False

            if field == 'Day_of_Week' and (value < 0 or value > 6):
                QMessageBox.warning(self, "Invalid Input", "Day of Week must be between 0 and 6")
                return False

            if field == 'Month' and (value < 1 or value > 12):
                QMessageBox.warning(self, "Invalid Input", "Month must be between 1 and 12")
                return False

        return True

    def predict(self):
        if not self.validate_inputs():
            return

        try:
            # Prepare input data
            input_data = {}
            for field in self.input_fields:
                input_data[field] = float(self.input_fields[field].text())

            # Calculate derived features
            confirmed = input_data['Confirmed']
            deaths = input_data['Deaths']
            recovered = input_data['Recovered']

            input_data['Mortality_Rate'] = deaths / confirmed if confirmed > 0 else 0
            input_data['Recovery_Rate'] = recovered / confirmed if confirmed > 0 else 0
            input_data['Active_Cases'] = confirmed - deaths - recovered

            # Encode country and province
            country = self.country_combo.currentText()
            province = self.province_combo.currentText()

            input_data['Country_Encoded'] = self.le_country.transform([country])[0]
            input_data['Province_Encoded'] = self.le_province.transform([province])[0]

            # Create feature vector in correct order
            feature_vector = [input_data[feature] for feature in self.features]
            feature_vector = np.array(feature_vector).reshape(1, -1)

            # Scale features - suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_features = self.scaler.transform(feature_vector)

            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # Store current prediction for chart updates
            self.current_prediction = {
                'prediction': prediction,
                'probabilities': probabilities,
                'input_data': input_data
            }

            # Update UI
            self.update_prediction_display()
            self.update_charts()

            # Show the tab widget if it was hidden
            if not self.tab_widget.isVisible():
                self.tab_widget.setVisible(True)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")

    def update_prediction_display(self):
        if not self.current_prediction:
            return

        prediction = self.current_prediction['prediction']
        probabilities = self.current_prediction['probabilities']

        result_text = "Cases will DOUBLE in the next 7 days" if prediction == 1 else "Cases will NOT double in the next 7 days"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        self.prediction_result.setText(
            f"{result_text}\n"
            f"Confidence: {confidence * 100:.2f}%\n"
            f"Probability of doubling: {probabilities[1] * 100:.2f}%\n"
            f"Probability of not doubling: {probabilities[0] * 100:.2f}%"
        )

        # Update probability chart
        self.update_probability_chart()

    def update_probability_chart(self):
        if not self.current_prediction:
            return

        probabilities = self.current_prediction['probabilities']
        labels = ['Not Doubling', 'Doubling']
        colors = ['#4CAF50', '#F44336']

        # Creating new axes
        ax = self.probability_canvas.create_axes()

        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            probabilities,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )

        # Draw a circle in the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        ax.set_title('Prediction Probability Distribution')

        # Update the canvas
        self.probability_canvas.draw()

    def update_feature_importance_chart(self):
        if not self.model:
            return

        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Creating new axes
        ax = self.importance_canvas.create_axes()

        # Create horizontal bar chart
        features_sorted = [self.features[i] for i in indices]
        importances_sorted = importances[indices]

        y_pos = np.arange(len(features_sorted))

        bars = ax.barh(y_pos, importances_sorted, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features_sorted)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')

        # Add value labels on bars
        for i, v in enumerate(importances_sorted):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')

        # Update the canvas
        self.importance_canvas.draw()

    def update_relationship_chart(self):
        """Update the feature relationship chart when combo boxes change"""
        # Creating new axes
        ax = self.relationship_canvas.create_axes()

        # Check if we have a prediction to display
        if not self.current_prediction:
            ax.text(0.5, 0.5, 'Please make a prediction first',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            self.relationship_canvas.draw()
            return

        # Get selected features
        x_feature = self.x_feature_combo.currentText()
        y_feature = self.y_feature_combo.currentText()
        color_feature = self.color_feature_combo.currentText()

        # Create scatter plot
        if color_feature != 'None' and self.current_prediction:
            scatter = ax.scatter(
                self.current_prediction['input_data'][x_feature],
                self.current_prediction['input_data'][y_feature],
                c=self.current_prediction['input_data'][color_feature],
                cmap='viridis',
                s=100,
                edgecolors='black'
            )

            # Add colorbar
            self.current_colorbar = self.relationship_canvas.fig.colorbar(scatter, ax=ax)
            self.current_colorbar.set_label(color_feature)
        else:
            scatter = ax.scatter(
                self.current_prediction['input_data'][x_feature],
                self.current_prediction['input_data'][y_feature],
                s=100,
                color='red',
                edgecolors='black'
            )

        # Highlight the current prediction point
        ax.scatter(
            self.current_prediction['input_data'][x_feature],
            self.current_prediction['input_data'][y_feature],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidth=2
        )

        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'{y_feature} vs {x_feature}')
        ax.grid(True, alpha=0.3)

        # Update the canvas
        self.relationship_canvas.draw()

    def update_charts(self):
        self.update_probability_chart()
        self.update_feature_importance_chart()
        self.update_relationship_chart()

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = CovidPredictorApp()
    window.show()
    if parent is None:
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()