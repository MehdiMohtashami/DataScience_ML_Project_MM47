import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QComboBox, QPushButton, QGridLayout,
                             QMessageBox, QScrollArea, QTabWidget, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import traceback
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
# Set style for plots
plt.style.use('default')
sns.set_palette("viridis")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class CarEvaluationApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_artifacts = None
        self.current_prediction = None
        self.feature_mappings = {}  # برای ذخیره mapping ویژگی‌ها
        self.initUI()
        self.load_model()
        self.setWindowTitle("CarEvaluation")


    def go_back(self):
        """بستن این UI و نمایش دوباره MainForm"""
        self.close()
        if self.parent:
            self.parent.show()

    def initUI(self):
        self.setWindowTitle('Car Acceptability Predictor - Tuned Random Forest (97.98% Accuracy)')
        self.setGeometry(300, 100, 1400, 900)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        v = QVBoxLayout(self)
        v.addWidget(QLabel("CarEvaluation - Prediction UI"))
        v.addWidget(QTextEdit("اینجا فرم مدل CarEvaluation خواهد بود..."))
        back = QPushButton("⬅ بازگشت به MainForm")
        back.clicked.connect(self.go_back)  # وصلش کردیم به تابع go_back
        v.addWidget(back)

        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Prediction tab
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)

        # Model info section
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout()
        self.model_info_label = QLabel(
            "Model: Tuned Random Forest\n"
            "Accuracy: 97.98%\n"
            "Best Parameters: max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200"
        )
        self.model_info_label.setStyleSheet("QLabel { background-color: #e8f5e9; padding: 10px; border-radius: 5px; }")
        self.model_info_label.setAlignment(Qt.AlignCenter)
        model_info_layout.addWidget(self.model_info_label)
        model_info_group.setLayout(model_info_layout)
        prediction_layout.addWidget(model_info_group)

        # Input section
        input_group = QGroupBox("Car Features Input")
        input_layout = QGridLayout()
        input_layout.setHorizontalSpacing(10)  # کاهش فاصله افقی

        # Feature options
        self.feature_options = {
            'buying': ['vhigh', 'high', 'med', 'low'],
            'maint': ['vhigh', 'high', 'med', 'low'],
            'doors': ['2', '3', '4', '5more'],
            'persons': ['2', '4', 'more'],
            'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high']
        }

        # Help texts for each feature
        help_texts = {
            'buying': 'Select buying price (vhigh, high, med, low)',
            'maint': 'Select maintenance price (vhigh, high, med, low)',
            'doors': 'Select number of doors (2, 3, 4, 5more)',
            'persons': 'Select capacity (2, 4, more)',
            'lug_boot': 'Select luggage boot size (small, med, big)',
            'safety': 'Select safety level (low, med, high)'
        }

        self.comboboxes = {}
        row = 0
        for feature, options in self.feature_options.items():
            # ایجاد Label اول
            label = QLabel(f"{feature.capitalize()}:")
            label.setToolTip(help_texts[feature])
            label.setMinimumWidth(70)  # عرض ثابت برای labelها

            # ایجاد ComboBox بعد از Label
            combo = QComboBox()
            combo.addItems(options)
            combo.setCurrentIndex(2 if feature in ['buying', 'maint'] else 1)  # Set reasonable defaults
            combo.setToolTip(help_texts[feature])
            combo.setMinimumWidth(100)  # عرض ثابت برای comboboxها
            self.comboboxes[feature] = combo

            # اضافه کردن به Layout (ابتدا Label سپس ComboBox)
            input_layout.addWidget(label, row, 0)
            input_layout.addWidget(combo, row, 1)
            row += 1

        # تنظیم ستون‌ها برای چیدمان بهتر
        input_layout.setColumnStretch(0, 1)  # ستون labelها
        input_layout.setColumnStretch(1, 2)  # ستون comboboxها

        input_group.setLayout(input_layout)
        prediction_layout.addWidget(input_group)

        # Predict button
        self.predict_btn = QPushButton("Predict Acceptability")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        prediction_layout.addWidget(self.predict_btn)

        #back_button
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        self.back_button.setStyleSheet("""
                    QPushButton { 
                        background-color: gray; 
                        color: white; 
                        font-weight: bold; 
                        padding: 15px;
                        border-radius: 5px;
                        font-size: 14px;
                    }
                    QPushButton:hover {
                        background-color: #A9A9A9;
                    }
                """)
        prediction_layout.addWidget(self.back_button)

        # Result section
        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("Please make a prediction to see results")
        self.result_label.setStyleSheet("""
            QLabel { 
                background-color: #f5f5f5; 
                padding: 20px; 
                font-weight: bold; 
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        prediction_layout.addWidget(result_group)

        # Add some stretch to push everything up
        prediction_layout.addStretch()

        # Add prediction tab to tabs
        tabs.addTab(prediction_tab, "Prediction")

        # Analysis tab (initially empty, will be populated after prediction)
        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        tabs.addTab(self.analysis_tab, "Prediction Analysis")

        # Set initial tab index
        tabs.setCurrentIndex(0)

    def load_model(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'car_acceptability_tuned_rf_model.pkl')

            # بررسی وجود فایل مدل
            if os.path.exists(model_path):
                self.model_artifacts = joblib.load(model_path)
                print("Model loaded successfully")
            else:
                # نمایش پیام خطا و ایجاد یک مدل ساده برای تست
                error_msg = f"Model file not found: {model_path}\n\nCreating a simple model for demonstration."
                print(error_msg)
                QMessageBox.warning(self, "Warning", error_msg)
                self.create_simple_model()

            # ایجاد mapping دستی برای ویژگی‌ها
            self.feature_mappings = {}
            for feature, options in self.feature_options.items():
                self.feature_mappings[feature] = {option: idx for idx, option in enumerate(options)}

            print("Feature mappings created:", self.feature_mappings)

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}\n\nPlease make sure the file 'car_acceptability_tuned_rf_model.pkl' exists in the same directory."
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def manual_encode_features(self, input_data):
        """تبدیل دستی مقادیر ویژگی‌ها به اعداد بر اساس mapping از پیش تعریف شده"""
        encoded_data = []
        for feature in self.model_artifacts['feature_names']:
            value = input_data[feature]
            encoded_value = self.feature_mappings[feature][value]
            encoded_data.append(encoded_value)
        return np.array([encoded_data])

    def predict(self):
        try:
            if self.model_artifacts is None:
                QMessageBox.warning(self, "Warning", "Model not loaded. Please check if the model file exists.")
                return

            # Get input values
            input_data = {}
            for feature, combo in self.comboboxes.items():
                input_data[feature] = combo.currentText()

            print("Input data:", input_data)

            # Encode features manually
            encoded_input = self.manual_encode_features(input_data)
            print("Encoded input:", encoded_input)

            # Create a DataFrame with proper feature names to avoid warnings
            encoded_df = pd.DataFrame(encoded_input, columns=self.model_artifacts['feature_names'])

            # Make prediction
            model = self.model_artifacts['model']
            prediction_encoded = model.predict(encoded_df)
            prediction_proba = model.predict_proba(encoded_df)

            print("Prediction encoded:", prediction_encoded)
            print("Prediction probabilities:", prediction_proba)

            # Decode prediction
            prediction = self.model_artifacts['target_encoder'].inverse_transform(prediction_encoded)

            print("Decoded prediction:", prediction)

            # Store current prediction for visualization
            self.current_prediction = {
                'input': input_data,
                'prediction': prediction[0],
                'probabilities': prediction_proba[0],
                'encoded_input': encoded_input[0]
            }

            # Update result label
            confidence = max(prediction_proba[0]) * 100
            self.result_label.setText(
                f"Prediction: {prediction[0]}\n"
                f"Confidence: {confidence:.2f}%\n\n"
                f"All probabilities:\n"
                f"{', '.join([f'{cls}: {prob * 100:.2f}%' for cls, prob in zip(self.model_artifacts['target_encoder'].classes_, prediction_proba[0])])}"
            )

            # Update analysis tab with visualizations
            self.update_analysis_tab()

        except Exception as e:
            error_msg = f"An error occurred during prediction: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Error", f"An error occurred during prediction: {str(e)}")

    def update_analysis_tab(self):
        try:
            # Clear previous content
            for i in reversed(range(self.analysis_layout.count())):
                widget = self.analysis_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)

            # Create scroll area for analysis tab
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)

            # Feature Importance Visualization
            importance_group = QGroupBox("Feature Importance Analysis")
            importance_layout = QVBoxLayout()

            # Get feature importances
            model = self.model_artifacts['model']
            feature_importances = model.feature_importances_
            feature_names = self.model_artifacts['feature_names']

            # Create horizontal bar chart for feature importance
            importance_fig = Figure(figsize=(10, 6))
            importance_ax = importance_fig.add_subplot(111)

            # Sort features by importance
            sorted_idx = np.argsort(feature_importances)
            pos = np.arange(sorted_idx.shape[0])

            bars = importance_ax.barh(pos, feature_importances[sorted_idx])
            importance_ax.set_yticks(pos)
            importance_ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            importance_ax.set_xlabel('Importance')
            importance_ax.set_title('Feature Importance (Random Forest)')

            # Highlight the current input's most important feature
            most_important_feature_idx = sorted_idx[-1]
            bars[len(bars) - 1].set_color('orange')

            importance_canvas = FigureCanvas(importance_fig)
            importance_layout.addWidget(importance_canvas)

            # Add interpretation text
            interpretation_text = (
                f"The most important feature for this model is '{feature_names[most_important_feature_idx]}'.\n"
                f"Your input for this feature: '{self.current_prediction['input'][feature_names[most_important_feature_idx]]}'"
            )
            interpretation_label = QLabel(interpretation_text)
            interpretation_label.setWordWrap(True)
            interpretation_label.setStyleSheet(
                "QLabel { padding: 10px; background-color: #f9f9f9; border-radius: 5px; }")
            importance_layout.addWidget(interpretation_label)

            importance_group.setLayout(importance_layout)
            scroll_layout.addWidget(importance_group)

            # Prediction Probability Visualization
            probability_group = QGroupBox("Prediction Probability Distribution")
            probability_layout = QVBoxLayout()

            # Create bar chart for prediction probabilities
            prob_fig = Figure(figsize=(10, 6))
            prob_ax = prob_fig.add_subplot(111)

            classes = self.model_artifacts['target_encoder'].classes_
            probabilities = self.current_prediction['probabilities']

            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            bars = prob_ax.bar(classes, probabilities, color=colors)
            prob_ax.set_xlabel('Car Acceptability Classes')
            prob_ax.set_ylabel('Probability')
            prob_ax.set_title('Prediction Probability Distribution')
            prob_ax.set_ylim(0, 1)

            # Add value labels on bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                height = bar.get_height()
                prob_ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{prob:.3f}', ha='center', va='bottom')

            # Highlight the predicted class
            predicted_idx = np.where(classes == self.current_prediction['prediction'])[0][0]
            bars[predicted_idx].set_color('orange')

            probability_canvas = FigureCanvas(prob_fig)
            probability_layout.addWidget(probability_canvas)

            probability_group.setLayout(probability_layout)
            scroll_layout.addWidget(probability_group)

            # Input Feature Comparison Radar Chart
            radar_group = QGroupBox("Input Feature Comparison (Radar Chart)")
            radar_layout = QVBoxLayout()

            # Create radar chart
            radar_fig = Figure(figsize=(8, 8))
            radar_ax = radar_fig.add_subplot(111, polar=True)

            # Categories and values
            categories = list(self.feature_options.keys())
            N = len(categories)

            # Normalize values for radar chart
            values = []
            for feature in categories:
                options = self.feature_options[feature]
                current_value = self.current_prediction['input'][feature]
                normalized_value = options.index(current_value) / max(1, len(options) - 1)
                values.append(normalized_value)

            # Complete the circle
            values += values[:1]

            # Calculate angles
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            # Draw radar chart
            radar_ax.plot(angles, values, 'o-', linewidth=2, label='Your Input')
            radar_ax.fill(angles, values, alpha=0.25)

            # Draw average line (midpoint for all features)
            avg_values = [0.5] * (N + 1)
            radar_ax.plot(angles, avg_values, 'r--', linewidth=1, label='Average')

            # Add labels
            radar_ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            radar_ax.set_title('Input Feature Comparison')
            radar_ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            radar_canvas = FigureCanvas(radar_fig)
            radar_layout.addWidget(radar_canvas)

            radar_group.setLayout(radar_layout)
            scroll_layout.addWidget(radar_group)

            # Set up the scroll area
            scroll.setWidget(scroll_content)
            self.analysis_layout.addWidget(scroll)

            # Switch to analysis tab
            self.centralWidget().findChild(QTabWidget).setCurrentIndex(1)

        except Exception as e:
            error_msg = f"An error occurred while creating visualizations: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Error", f"An error occurred while creating visualizations: {str(e)}")

    def close_and_go_back(self):
        self.close()


def main(parent=None):
    try:
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle('Fusion')

        window = CarEvaluationApp(parent)
        window.show()
        #
        sys.exit(app.exec_())
        win = CarEvaluationApp()
        win.show()
        return win
    except Exception as e:
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())
        QMessageBox.critical(None, "Fatal Error", f"The application encountered a fatal error: {str(e)}")


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = CarEvaluationApp()
#     window.show()
#     sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CarEvaluationApp()
    window.show()
    sys.exit(app.exec_())
