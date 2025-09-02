import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QMessageBox, QFormLayout, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class BloodDonationApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BloodTransfusionServiceCenter")
        self.setWindowTitle("Blood Donation Prediction System")
        self.setGeometry(100, 100, 1200, 800)

        # Load model and scaler
        try:
            self.model = joblib.load('logistic_regression_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            with open('feature_info.json', 'r') as f:
                self.feature_info = json.load(f)
            self.accuracy = 0.7458  # From our CV results
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model files: {str(e)}")
            sys.exit(1)

        # Initialize current prediction
        self.current_prediction = None

        self.init_ui()

    def init_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Model info
        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_info)
        model_layout.addWidget(QLabel(f"Model: Logistic Regression"))
        model_layout.addWidget(QLabel(f"CV ROC AUC: {self.accuracy:.4f}"))
        left_layout.addWidget(model_info)

        # Input form
        input_group = QGroupBox("Donor Information")
        form_layout = QFormLayout(input_group)

        # Create input fields with validation
        self.recency_input = QLineEdit()
        self.recency_input.setValidator(QIntValidator(0, 100))
        self.recency_input.setPlaceholderText("0-100")

        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QIntValidator(1, 100))
        self.frequency_input.setPlaceholderText("1-100")

        self.monetary_input = QLineEdit()
        self.monetary_input.setValidator(QIntValidator(250, 20000))
        self.monetary_input.setPlaceholderText("250-20000")

        self.time_input = QLineEdit()
        self.time_input.setValidator(QIntValidator(1, 100))
        self.time_input.setPlaceholderText("1-100")

        form_layout.addRow("Recency (months):", self.recency_input)
        form_layout.addRow("Frequency (times):", self.frequency_input)
        form_layout.addRow("Monetary (cc blood):", self.monetary_input)
        form_layout.addRow("Time (months):", self.time_input)

        left_layout.addWidget(input_group)

        # Prediction button
        self.predict_btn = QPushButton("Predict Donation")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.predict_btn)

        #Back to Main
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)
        self.back_button.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-weight: bold; }")
        left_layout.addWidget(self.back_button)

        # Result display
        self.result_group = QGroupBox("Prediction Result")
        self.result_layout = QVBoxLayout(self.result_group)
        self.result_label = QLabel("Please enter donor information and click 'Predict Donation'")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("QLabel { padding: 10px; }")
        self.result_layout.addWidget(self.result_label)
        self.result_group.setVisible(False)
        left_layout.addWidget(self.result_group)

        left_layout.addStretch()

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualizations (initially hidden)
        self.viz_panel = QTabWidget()
        self.viz_panel.setVisible(False)

        # Feature Importance Tab
        self.feature_importance_tab = QWidget()
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_tab)
        self.feature_figure = Figure(figsize=(10, 8))
        self.feature_canvas = FigureCanvas(self.feature_figure)
        self.feature_importance_layout.addWidget(self.feature_canvas)
        self.viz_panel.addTab(self.feature_importance_tab, "Feature Importance")

        # Prediction Analysis Tab
        self.prediction_analysis_tab = QWidget()
        self.prediction_analysis_layout = QVBoxLayout(self.prediction_analysis_tab)
        self.prediction_figure = Figure(figsize=(10, 8))
        self.prediction_canvas = FigureCanvas(self.prediction_figure)
        self.prediction_analysis_layout.addWidget(self.prediction_canvas)
        self.viz_panel.addTab(self.prediction_analysis_tab, "Prediction Analysis")

        # Feature Relationship Tab
        self.feature_relationship_tab = QWidget()
        self.feature_relationship_layout = QVBoxLayout(self.feature_relationship_tab)
        self.relationship_figure = Figure(figsize=(10, 8))
        self.relationship_canvas = FigureCanvas(self.relationship_figure)
        self.feature_relationship_layout.addWidget(self.relationship_canvas)
        self.viz_panel.addTab(self.feature_relationship_tab, "Feature Relationships")

        # Donor Comparison Tab
        self.donor_comparison_tab = QWidget()
        self.donor_comparison_layout = QVBoxLayout(self.donor_comparison_tab)
        self.comparison_figure = Figure(figsize=(10, 8))
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        self.donor_comparison_layout.addWidget(self.comparison_canvas)
        self.viz_panel.addTab(self.donor_comparison_tab, "Donor Comparison")

        main_layout.addWidget(self.viz_panel)

    def predict(self):
        # Validate inputs
        try:
            recency = int(self.recency_input.text())
            frequency = int(self.frequency_input.text())
            monetary = int(self.monetary_input.text())
            time = int(self.time_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for all fields.")
            return

        # Calculate derived features
        donation_ratio = frequency / (time + 1)
        avg_donation_amount = monetary / (frequency + 1)
        recency_ratio = recency / (time + 1)

        # Create feature array in the correct order (same as training)
        features = np.array([[recency, frequency, monetary, time,
                              donation_ratio, avg_donation_amount, recency_ratio]])

        try:
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            probability = self.model.predict_proba(features_scaled)[0, 1]
            prediction = self.model.predict(features_scaled)[0]
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error during prediction: {str(e)}")
            return

        # Store current prediction for visualization
        self.current_prediction = {
            'features': {
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary,
                'Time': time,
                'Donation Ratio': donation_ratio,
                'Avg Donation Amount': avg_donation_amount,
                'Recency Ratio': recency_ratio
            },
            'probability': probability,
            'prediction': prediction
        }

        # Update result display
        result_text = f"""
        <h3>Prediction Result:</h3>
        <b>Will donate blood:</b> {'Yes' if prediction == 1 else 'No'}<br>
        <b>Probability:</b> {probability:.4f}<br><br>

        <b>Input Values:</b><br>
        - Recency: {recency} months<br>
        - Frequency: {frequency} donations<br>
        - Monetary: {monetary} cc<br>
        - Time: {time} months<br>
        """
        self.result_label.setText(result_text)
        self.result_group.setVisible(True)

        # Show and update visualizations
        self.viz_panel.setVisible(True)
        self.update_visualizations()

    def update_visualizations(self):
        # Update all visualizations
        self.plot_feature_importance()
        self.plot_prediction_analysis()
        self.plot_feature_relationships()
        self.plot_donor_comparison()

    def load_model_and_data(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Current directory: {current_dir}")
            print(f"Files in directory: {os.listdir(current_dir)}")
            # Get the current directory path (the folder where this file is located)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Using absolute paths to download files
            # self.model = joblib.load(os.path.join(current_dir, 'logistic_regression_model.pkl'))
            self.model = joblib.load(os.path.join(current_dir, 'logistic_regression_model.pkl'))
            self.scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
            self.feature_names = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))

            # Load original data for visualization
            self.df = pd.read_csv(os.path.join(current_dir, 'transfusion.data.csv'))
            self.preprocess_data()

            self.model_accuracy = 0.9858  # From our previous evaluation

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            sys.exit(1)
    def plot_feature_importance(self):
        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)

        # Get feature importance from model coefficients
        if hasattr(self.model, 'coef_'):
            feature_names = ['Recency', 'Frequency', 'Monetary', 'Time',
                             'Donation Ratio', 'Avg Donation', 'Recency Ratio']
            importance = np.abs(self.model.coef_[0])

            # Create DataFrame for sorting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)

            # Plot horizontal bar chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

            # Add value labels
            for i, (value, feature) in enumerate(zip(importance_df['Importance'], importance_df['Feature'])):
                ax.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold')

            ax.set_xlabel('Importance (Absolute Coefficient Value)')
            ax.set_title('Feature Importance in Prediction Model')
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        self.feature_canvas.draw()

    def plot_prediction_analysis(self):
        self.prediction_figure.clear()
        ax = self.prediction_figure.add_subplot(111)

        # Create probability distribution
        probabilities = [1 - self.current_prediction['probability'],
                         self.current_prediction['probability']]
        labels = ['Will Not Donate', 'Will Donate']
        colors = ['#ff9999', '#66b3ff']

        # Create pie chart
        wedges, texts, autotexts = ax.pie(probabilities, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)

        # Style the chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Donation Probability Analysis')

        # Add explanation text
        explanation = f"Probability: {self.current_prediction['probability']:.4f}\n"
        explanation += "Prediction: Will Donate" if self.current_prediction[
                                                        'prediction'] == 1 else "Prediction: Will Not Donate"
        ax.text(0, -1.5, explanation, ha='center', fontsize=12,
                bbox=dict(facecolor='lightgray', alpha=0.7))

        self.prediction_canvas.draw()

    def plot_feature_relationships(self):
        self.relationship_figure.clear()

        # Create a 2x2 grid of subplots
        ax1 = self.relationship_figure.add_subplot(221)
        ax2 = self.relationship_figure.add_subplot(222)
        ax3 = self.relationship_figure.add_subplot(223)
        ax4 = self.relationship_figure.add_subplot(224)

        # Sample data for demonstration
        np.random.seed(42)

        # Plot 1: Recency vs Frequency
        x_vals = np.random.randint(0, 75, 100)
        y_vals = np.random.randint(1, 50, 100)
        ax1.scatter(x_vals, y_vals, alpha=0.6, label='Other donors')
        ax1.scatter(self.current_prediction['features']['Recency'],
                    self.current_prediction['features']['Frequency'],
                    color='red', s=100, label='Current donor', edgecolors='black')
        ax1.set_xlabel('Recency (months)')
        ax1.set_ylabel('Frequency (times)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Monetary vs Time
        x_vals = np.random.randint(250, 12500, 100)
        y_vals = np.random.randint(2, 98, 100)
        ax2.scatter(x_vals, y_vals, alpha=0.6, label='Other donors')
        ax2.scatter(self.current_prediction['features']['Monetary'],
                    self.current_prediction['features']['Time'],
                    color='red', s=100, label='Current donor', edgecolors='black')
        ax2.set_xlabel('Monetary (cc blood)')
        ax2.set_ylabel('Time (months)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Plot 3: Recency vs Donation Ratio
        x_vals = np.random.randint(0, 75, 100)
        y_vals = np.random.random(100) * 2
        ax3.scatter(x_vals, y_vals, alpha=0.6, label='Other donors')
        ax3.scatter(self.current_prediction['features']['Recency'],
                    self.current_prediction['features']['Donation Ratio'],
                    color='red', s=100, label='Current donor', edgecolors='black')
        ax3.set_xlabel('Recency (months)')
        ax3.set_ylabel('Donation Ratio')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Plot 4: Time vs Avg Donation Amount
        x_vals = np.random.randint(2, 98, 100)
        y_vals = np.random.randint(200, 1000, 100)
        ax4.scatter(x_vals, y_vals, alpha=0.6, label='Other donors')
        ax4.scatter(self.current_prediction['features']['Time'],
                    self.current_prediction['features']['Avg Donation Amount'],
                    color='red', s=100, label='Current donor', edgecolors='black')
        ax4.set_xlabel('Time (months)')
        ax4.set_ylabel('Avg Donation Amount (cc)')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)

        self.relationship_figure.suptitle('Feature Relationships and Current Donor Position', fontsize=16)
        self.relationship_figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.relationship_canvas.draw()

    def plot_donor_comparison(self):
        self.comparison_figure.clear()
        ax = self.comparison_figure.add_subplot(111, polar=True)

        # Create radar chart for donor comparison
        categories = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donation Ratio']

        # Normalize values for radar chart (0-1 scale)
        normalized_values = []
        for category in categories:
            value = self.current_prediction['features'].get(category, 0)

            # Normalize based on reasonable ranges
            if category == 'Recency':
                normalized = 1 - min(value / 74, 1)  # Inverse as lower recency is better
            elif category == 'Frequency':
                normalized = min(value / 50, 1)
            elif category == 'Monetary':
                normalized = min(value / 12500, 1)
            elif category == 'Time':
                normalized = min(value / 98, 1)
            else:  # Donation Ratio
                normalized = min(value * 2, 1)  # Arbitrary scaling

            normalized_values.append(normalized)

        # Complete the circle
        normalized_values += normalized_values[:1]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Plot current donor
        ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', label='Current Donor')
        ax.fill(angles, normalized_values, alpha=0.25)

        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Add grid and legend
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_title('Donor Profile Radar Chart', size=16, y=1.05)

        self.comparison_canvas.draw()

    def close_and_go_back(self):
        self.close()


def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 9, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = BloodDonationApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()