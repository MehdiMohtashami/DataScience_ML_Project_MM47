import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QFormLayout, QTabWidget, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont, QPalette, QColor


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OccupancyDetection")
        self.setWindowTitle("OccupancyDetection - XGBoost Model (99.29% Accuracy)")
        self.setGeometry(100, 100, 1400, 900)

        # Define valid ranges for each input
        self.valid_ranges = {
            'Temperature': (19.0, 25.0),
            'Humidity': (16.0, 40.0),
            'Light': (0.0, 1700.0),
            'CO2': (400.0, 2100.0),
            'HumidityRatio': (0.0025, 0.0065),
            'hour': (0, 23),
            'day_of_week': (0, 6),
            'month': (1, 12)
        }

        # Load model and scaler
        try:
            self.model = joblib.load('XGBoost_model.pkl')
            self.scaler = joblib.load('standard_scaler.pkl')
            self.df_combined = self.load_and_preprocess_data()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model or data: {str(e)}")
            sys.exit(1)

        # Set dark theme
        self.set_dark_theme()

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for inputs
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info
        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_info)
        model_label = QLabel(
            "XGBoost Classifier\nAccuracy: 99.29%\n\nThis model predicts room occupancy based on sensor data.")
        model_label.setAlignment(Qt.AlignCenter)
        model_label.setWordWrap(True)
        model_layout.addWidget(model_label)
        left_layout.addWidget(model_info)

        # Input form
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout(input_group)

        # Create input fields with validators
        self.input_fields = {}
        fields = [
            ('Temperature:', '20-24°C (e.g., 23.15)'),
            ('Humidity:', '25-35% (e.g., 27.27)'),
            ('Light:', '0-1600 Lux (e.g., 426.0)'),
            ('CO2:', '400-2000 ppm (e.g., 721.25)'),
            ('HumidityRatio:', '0.003-0.006 (e.g., 0.00479)'),
            ('Hour:', '0-23 (e.g., 14)'),
            ('Day of week;', '0-6 (0=Monday, 6=Sunday)'),
            ('Month:', '1-12 (e.g., 2)')
        ]

        for field, placeholder in fields:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            line_edit.setValidator(QDoubleValidator())
            input_layout.addRow(QLabel(field), line_edit)
            self.input_fields[field] = line_edit

        left_layout.addWidget(input_group)

        # Predict button
        predict_btn = QPushButton("Predict Occupancy")
        predict_btn.clicked.connect(self.predict)
        predict_btn.setStyleSheet(
            "QPushButton { background-color: #2ecc71; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(predict_btn)

        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)
        back_button.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(back_button)

        # Result display
        self.result_label = QLabel("Please enter values and click Predict")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "QLabel { background-color: #34495e; color: white; padding: 15px; border-radius: 5px; }")
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label)

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualizations (initially hidden)
        self.visualization_panel = QWidget()
        self.visualization_panel.setVisible(False)
        right_layout = QVBoxLayout(self.visualization_panel)

        # Create tab widget for different visualizations
        self.tab_widget = QTabWidget()

        # Feature Importance Tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        self.feature_canvas_container = QWidget()
        feature_layout.addWidget(self.feature_canvas_container)
        self.tab_widget.addTab(feature_tab, "Feature Importance")

        # Prediction Analysis Tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.analysis_canvas_container = QWidget()
        analysis_layout.addWidget(self.analysis_canvas_container)
        self.tab_widget.addTab(analysis_tab, "Prediction Analysis")

        # Feature Relationship Tab
        relationship_tab = QWidget()
        relationship_layout = QVBoxLayout(relationship_tab)
        self.relationship_canvas_container = QWidget()
        relationship_layout.addWidget(self.relationship_canvas_container)
        self.tab_widget.addTab(relationship_tab, "Feature Relationships")

        # Add tabs to right layout
        right_layout.addWidget(self.tab_widget)

        # Add right panel to main layout
        main_layout.addWidget(self.visualization_panel)

        # Store current prediction for visualization
        self.current_prediction = None
        self.current_features = None

        # Store feature names for reference
        self.feature_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week',
                              'month']

        # Initialize canvas variables
        self.feature_canvas = None
        self.analysis_canvas = None
        self.relationship_canvas = None

    def set_dark_theme(self):
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d8e0e9;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #d8e0e9;
                color: Black;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: Black;
            }
            QLineEdit {
                background-color: #7997b5;
                color: white;
                border: 1px solid #d8e0e9;
                border-radius: 3px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #d8e0e9;
                background: #d8e0e9;
            }
            QTabBar::tab {
                background: #7997b5;
                color: white;
                padding: 8px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #f2f5f8;
            }
        """)

    def load_and_preprocess_data(self):
        # Load and preprocess data (same as training)
        df1 = pd.read_csv('datatraining.csv')
        df2 = pd.read_csv('datatest.csv')
        df3 = pd.read_csv('datatest2.csv')

        df1['source'] = 'training'
        df2['source'] = 'test1'
        df3['source'] = 'test2'

        df_combined = pd.concat([df1, df2, df3], ignore_index=True)

        # Preprocess date column
        df_combined['date'] = pd.to_datetime(df_combined['date'], format='%m/%d/%Y %H:%M')
        df_combined['hour'] = df_combined['date'].dt.hour
        df_combined['day_of_week'] = df_combined['date'].dt.dayofweek
        df_combined['month'] = df_combined['date'].dt.month
        df_combined.drop('date', axis=1, inplace=True)

        return df_combined

    def validate_inputs(self):
        """Validate that all inputs are within acceptable ranges"""
        errors = []

        for field in self.feature_names:
            value = self.input_fields[field].text()
            if not value:
                errors.append(f"Please enter a value for {field}")
                continue

            try:
                num_value = float(value)
                min_val, max_val = self.valid_ranges[field]

                if num_value < min_val or num_value > max_val:
                    errors.append(f"{field} must be between {min_val} and {max_val}")
            except ValueError:
                errors.append(f"{field} must be a valid number")

        return errors

    def create_canvas(self, container):
        """Create a new canvas and replace the old one"""
        # Clear the container
        layout = container.layout()
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            layout = QVBoxLayout(container)
            container.setLayout(layout)

        # Create new canvas
        canvas = MplCanvas(self, width=6, height=5)
        layout.addWidget(canvas)
        return canvas

    def predict(self):
        # Validate inputs first
        validation_errors = self.validate_inputs()
        if validation_errors:
            error_msg = "\n".join(validation_errors)
            QMessageBox.warning(self, "Input Validation Error", error_msg)
            return

        # Get input values
        try:
            input_values = []
            for field in self.feature_names:
                value = self.input_fields[field].text()
                input_values.append(float(value))

            # Store features for visualization
            self.current_features = input_values

            # Create a DataFrame with proper feature names for scaling
            input_df = pd.DataFrame([input_values], columns=self.feature_names)

            # Scale and predict
            input_scaled = self.scaler.transform(input_df)
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]

            # Store prediction for visualization
            self.current_prediction = prediction

            # Update result label
            status = "OCCUPIED" if prediction == 1 else "NOT OCCUPIED"
            confidence = probability[prediction] * 100
            self.result_label.setText(f"Prediction: {status}\nConfidence: {confidence:.2f}%")

            # Show visualizations
            self.visualization_panel.setVisible(True)
            self.update_visualizations()

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")

    def update_visualizations(self):
        # Create new canvases
        self.feature_canvas = self.create_canvas(self.feature_canvas_container)
        self.analysis_canvas = self.create_canvas(self.analysis_canvas_container)
        self.relationship_canvas = self.create_canvas(self.relationship_canvas_container)

        # 1. Feature Importance Plot
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Create a fresh bar plot
            bars = self.feature_canvas.axes.bar(range(len(importances)), importances[indices])

            # Customize the plot
            self.feature_canvas.axes.set_xticks(range(len(importances)))
            self.feature_canvas.axes.set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
            self.feature_canvas.axes.set_title('Feature Importance', fontweight='bold')
            self.feature_canvas.axes.set_ylabel('Importance', fontweight='bold')
            self.feature_canvas.axes.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(importances[indices]):
                self.feature_canvas.axes.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')

        # 2. Prediction Analysis (Confidence Distribution)
        # Generate some sample predictions to show distribution
        sample_data = self.df_combined.sample(1000)
        X_sample = sample_data[self.feature_names]
        y_sample = sample_data['Occupancy']

        X_sample_scaled = self.scaler.transform(X_sample)
        probabilities = self.model.predict_proba(X_sample_scaled)

        # Plot confidence distribution for each class
        occupied_probs = probabilities[sample_data['Occupancy'] == 1, 1]
        not_occupied_probs = probabilities[sample_data['Occupancy'] == 0, 0]

        # Create fresh histograms
        self.analysis_canvas.axes.hist(not_occupied_probs, alpha=0.7, label='Not Occupied', bins=20, color='skyblue',
                                       edgecolor='black')
        self.analysis_canvas.axes.hist(occupied_probs, alpha=0.7, label='Occupied', bins=20, color='salmon',
                                       edgecolor='black')

        # Add current prediction to the plot
        current_input_df = pd.DataFrame([self.current_features], columns=self.feature_names)
        current_scaled = self.scaler.transform(current_input_df)
        current_confidence = self.model.predict_proba(current_scaled)[0][self.current_prediction]

        self.analysis_canvas.axes.axvline(x=current_confidence, color='green', linestyle='--',
                                          linewidth=2, label=f'Current Prediction: {current_confidence:.2f}')

        self.analysis_canvas.axes.set_xlabel('Prediction Confidence', fontweight='bold')
        self.analysis_canvas.axes.set_ylabel('Frequency', fontweight='bold')
        self.analysis_canvas.axes.set_title('Prediction Confidence Distribution', fontweight='bold')
        self.analysis_canvas.axes.legend()
        self.analysis_canvas.axes.grid(alpha=0.3)

        # 3. Feature Relationships (Scatter plot of two most important features)
        if hasattr(self.model, 'feature_importances_'):
            # Get two most important features
            top_two_idx = np.argsort(self.model.feature_importances_)[-2:]
            top_two_features = [self.feature_names[i] for i in top_two_idx]

            # Create a fresh scatter plot
            scatter = self.relationship_canvas.axes.scatter(
                self.df_combined[top_two_features[0]],
                self.df_combined[top_two_features[1]],
                c=self.df_combined['Occupancy'],
                alpha=0.6,
                cmap='coolwarm',
                s=30
            )

            # Add current prediction point
            self.relationship_canvas.axes.scatter(
                self.current_features[top_two_idx[0]],
                self.current_features[top_two_idx[1]],
                c='lime' if self.current_prediction == 1 else 'gold',
                s=200,
                marker='*',
                edgecolors='black',
                linewidth=2,
                label='Current Prediction'
            )

            self.relationship_canvas.axes.set_xlabel(top_two_features[0], fontweight='bold')
            self.relationship_canvas.axes.set_ylabel(top_two_features[1], fontweight='bold')
            self.relationship_canvas.axes.set_title(
                f'Relationship between {top_two_features[0]} and {top_two_features[1]}', fontweight='bold')
            self.relationship_canvas.axes.legend()
            self.relationship_canvas.axes.grid(alpha=0.3)

            # Add colorbar
            cbar = self.relationship_canvas.fig.colorbar(scatter)
            cbar.set_label('Occupancy (0=No, 1=Yes)', fontweight='bold')

        # Update all canvases
        self.feature_canvas.draw()
        self.analysis_canvas.draw()
        self.relationship_canvas.draw()

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

if __name__ == "__main__":
    main()