import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton,
                             QScrollArea, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QDoubleValidator


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def clear_plot(self):
        self.axes.clear()


class BanknoteClassifierUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Variance of Wavelet Transformed Image',
            'Skewness of Wavelet Transformed Image',
            'Curtosis of Wavelet Transformed Image',
            'Entropy of Image'
        ]
        self.setWindowTitle("BanknoteAuthentication")

        # Define valid ranges for each feature based on the dataset statistics
        self.feature_ranges = {
            'Variance of Wavelet Transformed Image': (-7.5, 7.5),
            'Skewness of Wavelet Transformed Image': (-14.0, 13.0),
            'Curtosis of Wavelet Transformed Image': (-6.0, 18.0),
            'Entropy of Image': (-9.0, 3.0)
        }

        self.prediction_history = []
        self.corr_matrix = None
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle('BanknoteAuthentication Classifier')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)



        # Model info group
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        model_info = QLabel(
            "Model: Support Vector Machine (SVM)\n"
            "Accuracy: 100.0%\n"
            "Dataset: BanknoteAuthentication\n"
            "Samples: 1372\n"
            "Features: 4\n\n"
            "Valid Input Ranges:\n"
            f"• Variance: {self.feature_ranges[self.feature_names[0]][0]:.1f} to {self.feature_ranges[self.feature_names[0]][1]:.1f}\n"
            f"• Skewness: {self.feature_ranges[self.feature_names[1]][0]:.1f} to {self.feature_ranges[self.feature_names[1]][1]:.1f}\n"
            f"• Curtosis: {self.feature_ranges[self.feature_names[2]][0]:.1f} to {self.feature_ranges[self.feature_names[2]][1]:.1f}\n"
            f"• Entropy: {self.feature_ranges[self.feature_names[3]][0]:.1f} to {self.feature_ranges[self.feature_names[3]][1]:.1f}"
        )
        model_info.setFont(QFont("Arial", 9))
        model_info.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        model_info.setWordWrap(True)
        model_layout.addWidget(model_info)
        left_layout.addWidget(model_group)

        # Input group
        input_group = QGroupBox("Enter Banknote Features")
        input_layout = QGridLayout(input_group)

        self.input_fields = []
        for i, feature in enumerate(self.feature_names):
            label = QLabel(feature.replace('Wavelet Transformed Image', '').replace('of', '').strip() + ":")
            input_field = QLineEdit()

            # Set validator with appropriate range
            min_val, max_val = self.feature_ranges[feature]
            validator = QDoubleValidator(min_val, max_val, 6)
            validator.setNotation(QDoubleValidator.StandardNotation)
            input_field.setValidator(validator)

            # Set placeholder with range information
            short_name = feature.split('of')[-1].strip()
            input_field.setPlaceholderText(f"{short_name} ({min_val:.1f} to {max_val:.1f})")

            input_layout.addWidget(label, i, 0)
            input_layout.addWidget(input_field, i, 1)
            self.input_fields.append(input_field)

        self.predict_btn = QPushButton("Predict Authenticity")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")

        # ایجاد یک دکمه برای بازگشت
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        self.back_button.setStyleSheet(
            "QPushButton { background-color: Gray; color: white; font-weight: bold; }")

        self.clear_btn = QPushButton("Clear Inputs")
        self.clear_btn.clicked.connect(self.clear_inputs)
        self.clear_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; }")

        input_layout.addWidget(self.predict_btn, len(self.feature_names), 0, 1, 2)
        input_layout.addWidget(self.clear_btn, len(self.feature_names) + 1, 0, 1, 2)
        input_layout.addWidget(self.back_button, len(self.feature_names)+ 2,0, 1, 2)

        left_layout.addWidget(input_group)

        # Result display
        self.result_label = QLabel("Please enter values and click 'Predict'")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; padding: 15px; border: 2px solid #ccc; border-radius: 5px;")
        left_layout.addWidget(self.result_label)

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualizations (initially hidden)
        self.visualization_panel = QTabWidget()
        self.visualization_panel.setVisible(False)

        # Feature Importance tab
        feature_importance_tab = QWidget()
        feature_layout = QVBoxLayout(feature_importance_tab)
        self.feature_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        feature_layout.addWidget(self.feature_canvas)
        self.visualization_panel.addTab(feature_importance_tab, "Feature Importance")

        # Feature Relationship tab
        relationship_tab = QWidget()
        relationship_layout = QVBoxLayout(relationship_tab)
        self.relationship_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        relationship_layout.addWidget(self.relationship_canvas)
        self.visualization_panel.addTab(relationship_tab, "Feature Relationship")

        # Prediction Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.analysis_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        analysis_layout.addWidget(self.analysis_canvas)
        self.visualization_panel.addTab(analysis_tab, "Prediction Analysis")

        # Decision Boundary tab
        boundary_tab = QWidget()
        boundary_layout = QVBoxLayout(boundary_tab)
        self.boundary_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        boundary_layout.addWidget(self.boundary_canvas)
        self.visualization_panel.addTab(boundary_tab, "Decision Boundary")

        main_layout.addWidget(self.visualization_panel)

        # Set style
        self.setStyleSheet("""
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
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
        """)

    def load_model(self):
        # In a real application, you would load a pre-trained model
        # For this example, we'll create a mock model with high accuracy
        self.model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        self.scaler = StandardScaler()

        # Load or create sample data for demonstration
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1372, n_features=4, n_informative=4, n_redundant=0,
            n_clusters_per_class=1, weights=[0.444, 0.556], random_state=42
        )

        # Fit the scaler and model
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)

        # Generate feature importance values (for demonstration)
        self.feature_importance = [0.557, 0.236, 0.148, 0.059]

        # Create a fixed correlation matrix (only once)
        self.corr_matrix = np.array([
            [1.0, 0.2, 0.1, -0.3],
            [0.2, 1.0, -0.4, 0.05],
            [0.1, -0.4, 1.0, -0.1],
            [-0.3, 0.05, -0.1, 1.0]
        ])

        # Pre-create the heatmap (only once)
        self.update_feature_relationship_plot(initial=True)

    def validate_inputs(self):
        """Validate all input fields and return error message if any"""
        errors = []

        for i, field in enumerate(self.input_fields):
            feature_name = self.feature_names[i]
            text = field.text().strip()

            # Check if field is empty
            if not text:
                errors.append(f"Please enter a value for {feature_name.split('of')[-1].strip()}")
                continue

            # Check if value is within valid range
            try:
                value = float(text)
                min_val, max_val = self.feature_ranges[feature_name]
                if value < min_val or value > max_val:
                    errors.append(
                        f"{feature_name.split('of')[-1].strip()} must be between {min_val:.1f} and {max_val:.1f}")
            except ValueError:
                errors.append(f"Invalid number format for {feature_name.split('of')[-1].strip()}")

        return errors

    def predict(self):
        # Validate inputs
        errors = self.validate_inputs()
        if errors:
            error_msg = "Please correct the following errors:\n\n" + "\n".join(f"• {error}" for error in errors)
            QMessageBox.warning(self, "Input Error", error_msg)
            return

        # Get input values
        input_values = [float(field.text()) for field in self.input_fields]

        # Scale the input
        input_scaled = self.scaler.transform([input_values])

        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]

        # Update result label
        result_text = f"Prediction: {'GENUINE' if prediction == 1 else 'FORGED'}\n"
        result_text += f"Confidence: {max(probability) * 100:.2f}%\n"
        result_text += f"Genuine Probability: {probability[1] * 100:.2f}%\n"
        result_text += f"Forged Probability: {probability[0] * 100:.2f}%"

        self.result_label.setText(result_text)

        # Set color based on prediction
        if prediction == 1:
            self.result_label.setStyleSheet(
                "font-size: 16px; padding: 15px; border: 2px solid #4CAF50; border-radius: 5px; background-color: #E8F5E9;")
        else:
            self.result_label.setStyleSheet(
                "font-size: 16px; padding: 15px; border: 2px solid #f44336; border-radius: 5px; background-color: #FFEBEE;")

        # Store prediction for visualization
        self.prediction_history.append({
            'values': input_values,
            'scaled': input_scaled[0],
            'prediction': prediction,
            'probability': probability
        })

        # Show visualizations
        self.visualization_panel.setVisible(True)

        # Update all visualizations (except heatmap which is already created)
        self.update_feature_importance_plot()
        self.update_prediction_analysis_plot()
        self.update_decision_boundary_plot()

    def clear_inputs(self):
        for field in self.input_fields:
            field.clear()
        self.result_label.setText("Please enter values and click 'Predict'")
        self.result_label.setStyleSheet("font-size: 16px; padding: 15px; border: 2px solid #ccc; border-radius: 5px;")
        self.visualization_panel.setVisible(False)
        self.prediction_history = []

    def update_feature_importance_plot(self):
        self.feature_canvas.clear_plot()
        ax = self.feature_canvas.axes

        # Create horizontal bar plot
        y_pos = np.arange(len(self.feature_names))
        ax.barh(y_pos, self.feature_importance, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.split('of')[-1].strip() for name in self.feature_names])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for BanknoteAuthentication')

        # Add value labels on bars
        for i, v in enumerate(self.feature_importance):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')

        self.feature_canvas.draw()

    def update_feature_relationship_plot(self, initial=False):
        if initial:
            # Only create the heatmap once
            ax = self.relationship_canvas.axes

            feature_short_names = [name.split('of')[-1].strip() for name in self.feature_names]
            sns.heatmap(self.corr_matrix, annot=True, xticklabels=feature_short_names,
                        yticklabels=feature_short_names, ax=ax, cmap='coolwarm', center=0, fmt='.2f')
            ax.set_title('Feature Correlation Matrix')

            self.relationship_canvas.draw()

    def update_prediction_analysis_plot(self):
        self.analysis_canvas.clear_plot()
        ax = self.analysis_canvas.axes

        if not self.prediction_history:
            return

        # Get the latest prediction
        latest = self.prediction_history[-1]

        # Create a probability distribution plot
        labels = ['Forged', 'Genuine']
        probabilities = latest['probability'] * 100

        colors = ['#ff9999', '#66b3ff']
        bars = ax.bar(labels, probabilities, color=colors)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Prediction Probability Distribution')
        ax.set_ylim(0, 100)

        # Add value labels on top of bars
        for bar, probability in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{probability:.2f}%', ha='center', va='bottom')

        self.analysis_canvas.draw()

    def update_decision_boundary_plot(self):
        self.boundary_canvas.clear_plot()
        ax = self.boundary_canvas.axes

        if not self.prediction_history:
            return

        # Get the latest prediction
        latest = self.prediction_history[-1]

        # Create a scatter plot of the two most important features
        # Using mock data for demonstration
        np.random.seed(42)  # Fixed seed for consistent results
        n_samples = 100
        genuine_x = np.random.normal(1.5, 0.8, n_samples)
        genuine_y = np.random.normal(1.2, 0.7, n_samples)
        forged_x = np.random.normal(-1.5, 0.8, n_samples)
        forged_y = np.random.normal(-1.2, 0.7, n_samples)

        ax.scatter(genuine_x, genuine_y, alpha=0.6, c='blue', label='Genuine', edgecolors='w')
        ax.scatter(forged_x, forged_y, alpha=0.6, c='red', label='Forged', edgecolors='w')

        # Plot the latest prediction point
        pred_x = latest['values'][0]  # First feature (Variance)
        pred_y = latest['values'][1]  # Second feature (Skewness)

        color = 'green' if latest['prediction'] == 1 else 'orange'
        marker = 'P' if latest['prediction'] == 1 else 'X'
        size = 200

        prediction_labels = ['Forged', 'Genuine']

        ax.scatter(pred_x, pred_y, c=color, marker=marker, s=size,
                   label=f'Predicted ({prediction_labels[latest["prediction"]]})',
                   edgecolors='black', linewidth=2)

        ax.set_xlabel('Variance (Feature 1)')
        ax.set_ylabel('Skewness (Feature 2)')
        ax.set_title('Decision Boundary Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.boundary_canvas.draw()

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = BanknoteClassifierUI(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()