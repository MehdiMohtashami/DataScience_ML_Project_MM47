import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QTabWidget, QFormLayout, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont, QPalette, QColor
import seaborn as sns

# تنظیم سبک نمودارها
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class StockPredictionApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IstanbulStockExchange")
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None
        self.current_prediction = None
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle('IstanbulStockExchange Prediction')
        self.setGeometry(100, 100, 1400, 900)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info group
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout(model_info_group)

        model_label = QLabel("Ensemble Model (Logistic Regression + SVC + Gradient Boosting)")
        model_label.setWordWrap(True)
        model_label.setStyleSheet("QLabel { background-color: #e8f4f8; padding: 10px; border-radius: 5px; }")

        accuracy_label = QLabel("Accuracy: 84.11%")
        accuracy_label.setStyleSheet("QLabel { background-color: #e8f8e8; padding: 10px; border-radius: 5px; }")

        model_info_layout.addWidget(model_label)
        model_info_layout.addWidget(accuracy_label)

        # Input group
        input_group = QGroupBox("Enter Feature Values")
        input_layout = QFormLayout(input_group)

        # Create input fields for features with validation
        self.input_fields = {}
        features_info = {
            'EM': {'desc': 'MSCI Emerging Markets Index', 'range': '-0.04 to 0.05', 'default': '0.0088'},
            'EU': {'desc': 'MSCI European Index', 'range': '-0.05 to 0.07', 'default': '0.0113'},
            'EM_lag1': {'desc': 'EM Lag 1 (previous day)', 'range': '-0.04 to 0.05', 'default': '0.0085'},
            'EM_MA5': {'desc': 'EM 5-day Moving Average', 'range': '-0.03 to 0.04', 'default': '0.0072'},
            'EU_MA5': {'desc': 'EU 5-day Moving Average', 'range': '-0.04 to 0.05', 'default': '0.0091'},
            'EM_trend': {'desc': 'EM Trend (current - previous)', 'range': '-0.02 to 0.02', 'default': '0.0003'},
            'EU_trend': {'desc': 'EU Trend (current - previous)', 'range': '-0.02 to 0.02', 'default': '0.0005'},
            'EM_lag2': {'desc': 'EM Lag 2 (2 days ago)', 'range': '-0.04 to 0.05', 'default': '0.0082'},
            'EU_lag2': {'desc': 'EU Lag 2 (2 days ago)', 'range': '-0.05 to 0.07', 'default': '0.0108'},
            'EM_EU_interaction': {'desc': 'EM × EU Interaction', 'range': '-0.003 to 0.004', 'default': '0.0001'}
        }

        for feature, info in features_info.items():
            # Add input field with validation
            line_edit = QLineEdit()
            line_edit.setValidator(QDoubleValidator())
            line_edit.setPlaceholderText(f"Range: {info['range']}")
            line_edit.setText(info['default'])
            self.input_fields[feature] = line_edit

            # Add label with description
            label = QLabel(f"{feature}: {info['desc']}")
            label.setWordWrap(True)
            input_layout.addRow(label, line_edit)

        # Predict button
        predict_btn = QPushButton("Predict Stock Direction")
        predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        predict_btn.clicked.connect(self.predict)

        # ایجاد یک دکمه برای بازگشت
        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back) # متد close رو صدا میزنه
        back_button.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-weight: bold; padding: 10px; }")

        # Result display
        self.result_label = QLabel("Please enter values and click Predict")
        self.result_label.setStyleSheet("QLabel { padding: 15px; font-weight: bold; }")
        self.result_label.setAlignment(Qt.AlignCenter)

        # Add widgets to left layout
        left_layout.addWidget(model_info_group)
        left_layout.addWidget(input_group)
        left_layout.addWidget(predict_btn)
        left_layout.addWidget(back_button)
        left_layout.addWidget(self.result_label)
        left_layout.addStretch()

        # Right panel for charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget for different chart types
        self.tab_widget = QTabWidget()

        # Prediction Analysis tab
        self.prediction_tab = QWidget()
        self.prediction_layout = QVBoxLayout(self.prediction_tab)

        # Feature Importance tab
        self.feature_importance_tab = QWidget()
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_tab)

        # Feature Relationship tab
        self.relationship_tab = QWidget()
        self.relationship_layout = QVBoxLayout(self.relationship_tab)

        # Add tabs
        self.tab_widget.addTab(self.prediction_tab, "Prediction Analysis")
        self.tab_widget.addTab(self.feature_importance_tab, "Feature Importance")
        self.tab_widget.addTab(self.relationship_tab, "Feature Relationships")

        right_layout.addWidget(self.tab_widget)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Initially disable tabs until prediction is made
        self.tab_widget.setEnabled(False)

    def load_model(self):
        try:
            self.model = joblib.load('ensemble_model.pkl')
            self.scaler = joblib.load('ensemble_scaler.pkl')
            self.feature_names = joblib.load('ensemble_features.pkl')

            # Define feature importance based on previous analysis
            self.feature_importance = {
                'EM': 0.224299,
                'EU': 0.062617,
                'EM_lag1': 0.055140,
                'EM_trend': 0.038318,
                'EU_MA5': 0.037383,
                'EU_trend': 0.032710,
                'EM_MA5': 0.014019,
                'EM_EU_interaction': 0.011215,
                'EM_lag2': 0.009346,
                'EU_lag2': 0.001869
            }

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def validate_inputs(self):
        # Define reasonable ranges for each feature
        valid_ranges = {
            'EM': (-0.04, 0.05),
            'EU': (-0.05, 0.07),
            'EM_lag1': (-0.04, 0.05),
            'EM_MA5': (-0.03, 0.04),
            'EU_MA5': (-0.04, 0.05),
            'EM_trend': (-0.02, 0.02),
            'EU_trend': (-0.02, 0.02),
            'EM_lag2': (-0.04, 0.05),
            'EU_lag2': (-0.05, 0.07),
            'EM_EU_interaction': (-0.003, 0.004)
        }

        for feature, line_edit in self.input_fields.items():
            value = line_edit.text()
            if not value:
                QMessageBox.warning(self, "Missing Value", f"Please enter a value for {feature}")
                return False

            try:
                num_value = float(value)
                min_val, max_val = valid_ranges[feature]
                if not (min_val <= num_value <= max_val):
                    QMessageBox.warning(self, "Invalid Range",
                                        f"Value for {feature} should be between {min_val} and {max_val}")
                    return False
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Please enter a valid number for {feature}")
                return False

        return True

    def predict(self):
        if not self.validate_inputs():
            return

        try:
            # Get values from input fields in the correct order
            input_values = []
            for feature in self.feature_names:
                value = self.input_fields[feature].text()
                input_values.append(float(value))

            # Convert to DataFrame with feature names to avoid warnings
            input_df = pd.DataFrame([input_values], columns=self.feature_names)

            # Scale the input
            scaled_input = self.scaler.transform(input_df)

            # Make prediction
            prediction = self.model.predict(scaled_input)[0]
            probabilities = self.model.predict_proba(scaled_input)[0]

            # Store current prediction
            self.current_prediction = {
                'input': input_values,
                'prediction': prediction,
                'probabilities': probabilities,
                'scaled_input': scaled_input[0]
            }

            # Update result label
            result_text = f"Prediction: {'UP' if prediction == 1 else 'DOWN'}\n"
            result_text += f"Probability UP: {probabilities[1] * 100:.2f}%\n"
            result_text += f"Probability DOWN: {probabilities[0] * 100:.2f}%"

            self.result_label.setText(result_text)
            if prediction == 1:
                self.result_label.setStyleSheet(
                    "QLabel { background-color: #d4edda; color: #155724; padding: 15px; font-weight: bold; border: 2px solid #c3e6cb; }")
            else:
                self.result_label.setStyleSheet(
                    "QLabel { background-color: #f8d7da; color: #721c24; padding: 15px; font-weight: bold; border: 2px solid #f5c6cb; }")

            # Enable tabs and update charts
            self.tab_widget.setEnabled(True)
            self.update_charts()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def update_charts(self):
        # Clear previous charts
        self.clear_layout(self.prediction_layout)
        self.clear_layout(self.feature_importance_layout)
        self.clear_layout(self.relationship_layout)

        # Create new charts
        self.create_prediction_charts()
        self.create_feature_importance_charts()
        self.create_relationship_charts()

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def create_prediction_charts(self):
        # Chart 1: Probability Distribution
        canvas1 = MplCanvas(self, width=6, height=4)
        ax1 = canvas1.fig.add_subplot(111)

        labels = ['DOWN', 'UP']
        probs = self.current_prediction['probabilities']

        colors = ['#ff9999' if i == 0 else '#66b3ff' for i in range(len(labels))]
        wedges, texts, autotexts = ax1.pie(probs, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax1.set_title('Prediction Probability Distribution', fontweight='bold')
        self.prediction_layout.addWidget(canvas1)

        # Chart 2: Confidence Indicator
        canvas2 = MplCanvas(self, width=6, height=3)
        ax2 = canvas2.fig.add_subplot(111)

        confidence = max(self.current_prediction['probabilities'])
        ax2.barh(['Confidence'], [confidence],
                 color=['#4CAF50' if confidence > 0.7 else '#FFC107' if confidence > 0.6 else '#F44336'])
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Confidence Level')
        ax2.set_title('Prediction Confidence', fontweight='bold')

        # Add value text on bars
        for i, v in enumerate([confidence]):
            ax2.text(v + 0.01, i, f'{v:.3f}', color='black', fontweight='bold')

        self.prediction_layout.addWidget(canvas2)

    def create_feature_importance_charts(self):
        # Chart 1: Horizontal Bar Chart
        canvas1 = MplCanvas(self, width=7, height=5)
        ax1 = canvas1.fig.add_subplot(111)

        features = list(self.feature_importance.keys())
        importance = list(self.feature_importance.values())

        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

        bars = ax1.barh(y_pos, importance, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Importance')
        ax1.set_title('Feature Importance Ranking', fontweight='bold')

        # Add value labels
        for i, v in enumerate(importance):
            ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')

        self.feature_importance_layout.addWidget(canvas1)

        # Chart 2: Polar Chart
        canvas2 = MplCanvas(self, width=6, height=6)
        ax2 = canvas2.fig.add_subplot(111, polar=True)

        features = list(self.feature_importance.keys())
        importance = list(self.feature_importance.values())

        theta = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
        radii = importance
        width = 2 * np.pi / len(features)

        bars = ax2.bar(theta, radii, width=width, bottom=0.0, alpha=0.7)

        # Use colors from previous chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
            bar.set_alpha(0.7)

        # Set feature labels
        ax2.set_xticks(theta)
        ax2.set_xticklabels(features, fontsize=8)
        ax2.set_title('Feature Importance (Polar View)', fontweight='bold', va='bottom')

        self.feature_importance_layout.addWidget(canvas2)

    def create_relationship_charts(self):
        # Get current input values
        input_values = self.current_prediction['input']

        # Chart 1: Top Features Correlation
        canvas1 = MplCanvas(self, width=7, height=5)
        ax1 = canvas1.fig.add_subplot(111)

        top_features = ['EM', 'EU', 'EM_lag1', 'EM_trend', 'EU_MA5']
        top_importance = [self.feature_importance[f] for f in top_features]
        # Use the feature_names list directly (no need for tolist())
        top_values = [input_values[i] for i, f in enumerate(self.feature_names) if f in top_features]

        # Create scatter plot
        scatter = ax1.scatter(top_importance, top_values, s=100, c=top_importance,
                              cmap='viridis', alpha=0.7)

        # Add labels for each point
        for i, feature in enumerate(top_features):
            ax1.annotate(feature, (top_importance[i], top_values[i]),
                         xytext=(5, 5), textcoords='offset points', fontweight='bold')

        ax1.set_xlabel('Feature Importance')
        ax1.set_ylabel('Input Value')
        ax1.set_title('Top Features: Importance vs Input Value', fontweight='bold')

        # Add colorbar to the same figure
        cbar = canvas1.fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Importance')

        self.relationship_layout.addWidget(canvas1)

        # Chart 2: Radar Chart for Top Features
        canvas2 = MplCanvas(self, width=6, height=6)
        ax2 = canvas2.fig.add_subplot(111, polar=True)

        # Create radar chart
        categories = top_features
        N = len(categories)

        # Normalize values for radar chart
        normalized_values = np.array(top_values) / max(np.abs(top_values))

        # Repeat first value to close the circle
        values = np.concatenate((normalized_values, [normalized_values[0]]))

        # Calculate angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Create radar chart
        ax2.plot(angles, values, 'o-', linewidth=2, color='b', alpha=0.7)
        ax2.fill(angles, values, alpha=0.25, color='b')

        # Add labels
        ax2.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)
        ax2.set_title('Top Features Radar Chart (Normalized Values)', fontweight='bold', va='bottom')
        ax2.grid(True)

        self.relationship_layout.addWidget(canvas2)

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show the main window
    mainWindow = StockPredictionApp(parent)
    mainWindow.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()