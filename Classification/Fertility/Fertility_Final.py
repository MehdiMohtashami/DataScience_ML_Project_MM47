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
                             QComboBox, QScrollArea, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont
from scipy.special import expit

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class FertilityApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fertility")
        self.title = "Fertility Diagnosis Predictor"
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.model = None
        self.scaler = None
        self.le = None
        self.accuracy = 0.0
        self.df = None
        self.feature_names = [
            'Season', 'Age', 'Childish Diseases', 'Accident or serious trauma',
            'Surgical intervention', 'High fevers in last year',
            'Frequency of alcohol consumption', 'Smoking Habit',
            'Number of hours spent sitting per day'
        ]

        # Load model and scaler
        self.load_model()

        # Load data for visualization
        self.load_data()

        self.initUI()
        self.prediction_history = []

    def load_model(self):
        try:
            self.model = joblib.load('best_fertility_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.le = joblib.load('label_encoder.pkl')
            self.accuracy = 0.875  # From your previous results
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.warning(self, "Error", f"Model files not found or corrupted: {e}")

    def load_data(self):
        try:
            self.df = pd.read_csv('fertility_Diagnosis.csv')
            if self.le:
                self.df['Output'] = self.le.transform(self.df['Output'])
            print("Data loaded successfully")
        except Exception as e:
            print(f"Error loading data: {e}")
            QMessageBox.warning(self, "Error", f"Data file not found or corrupted: {e}")

    def initUI(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info
        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_info)
        model_layout.addWidget(QLabel(f"Algorithm: Support Vector Machine (SVM)"))
        model_layout.addWidget(QLabel(f"Accuracy: {self.accuracy * 100:.2f}%"))
        model_layout.addWidget(QLabel("Target: Fertility Diagnosis"))
        model_layout.addWidget(QLabel("Output: N (Normal), O (Altered)"))
        left_layout.addWidget(model_info)

        # Input fields
        input_group = QGroupBox("Patient Information")
        input_layout = QVBoxLayout(input_group)

        # Create input fields with validators and range information
        self.input_fields = {}
        fields = [
            ('Season', 'Season in which analysis was performed:\n(-1: winter, -0.33: spring, 0.33: summer, 1: fall)',
             -1, 1),
            ('Age', 'Age at the time of analysis (scaled 0-1):\nActual age is between 18-36', 0, 1),
            ('Childish Diseases', 'Childish diseases (0: no, 1: yes)', 0, 1),
            ('Accident or serious trauma', 'Accident or serious trauma (0: no, 1: yes)', 0, 1),
            ('Surgical intervention', 'Surgical intervention (0: no, 1: yes)', 0, 1),
            ('High fevers in last year',
             'High fevers in last year:\n(-1: <3 months ago, 0: >3 months ago, 1: no fever)', -1, 1),
            ('Frequency of alcohol consumption',
             'Frequency of alcohol consumption (scaled 0-1):\n1: several times a day, 0.8: every day, 0.6: several times a week,\n0.4: once a week, 0.2: hardly ever or never',
             0, 1),
            ('Smoking Habit', 'Smoking habit (-1: never, 0: occasional, 1: daily)', -1, 1),
            ('Number of hours spent sitting per day',
             'Hours spent sitting per day (scaled 0-1):\nActual hours between 0-16', 0, 1)
        ]

        for field, tooltip, min_val, max_val in fields:
            layout = QHBoxLayout()
            label = QLabel(field + ":")
            label.setToolTip(tooltip)
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"{min_val} to {max_val}")
            validator = QDoubleValidator(min_val, max_val, 4)
            validator.setNotation(QDoubleValidator.StandardNotation)
            input_field.setValidator(validator)
            input_field.setToolTip(tooltip)
            layout.addWidget(label)
            layout.addWidget(input_field)
            input_layout.addLayout(layout)
            self.input_fields[field] = input_field

        # Predict button and result label
        predict_btn = QPushButton("Predict Diagnosis")
        predict_btn.clicked.connect(self.predict)
        input_layout.addWidget(predict_btn)
        self.result_label = QLabel("Prediction: N/A", alignment=Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; color: #e74c3c; font-weight: bold;")
        input_layout.addWidget(self.result_label)

        # ایجاد یک دکمه برای بازگشت
        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        input_layout.addWidget(back_button)

        left_layout.addWidget(input_group)

        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create tab widget for different visualizations
        self.tab_widget = QTabWidget()

        # Tab 1: Prediction Analysis
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout(self.tab1)
        self.tab1_canvas = MplCanvas(self.tab1, width=8, height=6)
        self.tab1_layout.addWidget(self.tab1_canvas)
        self.tab_widget.addTab(self.tab1, "Prediction Analysis")

        # Tab 2: Feature Importance
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout(self.tab2)
        self.tab2_canvas = MplCanvas(self.tab2, width=8, height=6)
        self.tab2_layout.addWidget(self.tab2_canvas)
        self.tab_widget.addTab(self.tab2, "Feature Importance")

        # Tab 3: Feature Relationships
        self.tab3 = QWidget()
        self.tab3_layout = QVBoxLayout(self.tab3)
        self.tab3_canvas = MplCanvas(self.tab3, width=8, height=6)
        self.tab3_layout.addWidget(self.tab3_canvas)
        self.tab_widget.addTab(self.tab3, "Feature Relationships")

        # Tab 4: Distribution Analysis
        self.tab4 = QWidget()
        self.tab4_layout = QVBoxLayout(self.tab4)
        self.tab4_canvas = MplCanvas(self.tab4, width=8, height=6)
        self.tab4_layout.addWidget(self.tab4_canvas)
        self.tab_widget.addTab(self.tab4, "Distribution Analysis")

        right_layout.addWidget(self.tab_widget)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Initially hide tabs until prediction is made
        self.tab_widget.setVisible(False)

    def validate_inputs(self):
        errors = []
        for field_name, field in self.input_fields.items():
            try:
                value = float(field.text())
                # Get min and max from validator
                validator = field.validator()
                min_val = validator.bottom()
                max_val = validator.top()

                if value < min_val or value > max_val:
                    errors.append(f"{field_name} must be between {min_val} and {max_val}")
            except ValueError:
                errors.append(f"Please enter a valid number for {field_name}")

        return errors

    def predict(self):
        # Validate inputs
        errors = self.validate_inputs()
        if errors:
            error_msg = "\n".join(errors)
            QMessageBox.warning(self, "Input Error", error_msg)
            return

        # Get input data
        input_data = []
        for field_name, field in self.input_fields.items():
            value = float(field.text())
            input_data.append(value)

        # Make prediction
        if self.model and self.scaler:
            try:
                # Create a DataFrame with feature names to avoid the warning
                input_df = pd.DataFrame([input_data], columns=self.feature_names)

                # Scale input data
                input_scaled = self.scaler.transform(input_df)

                # Predict
                prediction = self.model.predict(input_scaled)

                # Get prediction confidence
                if hasattr(self.model, 'predict_proba'):
                    probability = self.model.predict_proba(input_scaled)
                    confidence = np.max(probability) * 100
                elif hasattr(self.model, 'decision_function'):
                    # Use decision function as a proxy for confidence
                    decision_values = self.model.decision_function(input_scaled)
                    # Convert decision function to probability using sigmoid
                    confidence = expit(np.abs(decision_values[0])) * 100
                else:
                    # Fallback to a default confidence
                    confidence = 85.0  # Based on model accuracy

                # Convert prediction to original label
                diagnosis = self.le.inverse_transform(prediction)[0]

                # Store prediction
                self.prediction_history.append({
                    'input': input_data,
                    'diagnosis': diagnosis,
                    'confidence': confidence
                })

                # Show result in UI
                self.result_label.setText(f"Diagnosis: {diagnosis}\nConfidence: {confidence:.2f}%")

                # Show visualization tabs
                self.tab_widget.setVisible(True)

                # Update visualizations
                self.update_visualizations(input_data, diagnosis)
            except Exception as e:
                QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {e}")
                print(f"Prediction error: {e}")
        else:
            QMessageBox.warning(self, "Error", "Model not loaded. Please check model files.")

    def update_visualizations(self, input_data, diagnosis):
        # Clear previous plots
        for canvas in [self.tab1_canvas, self.tab2_canvas, self.tab3_canvas, self.tab4_canvas]:
            canvas.figure.clf()

        # Tab 1: Prediction Analysis - Radar chart
        self.create_radar_chart(input_data, diagnosis)

        # Tab 2: Feature Importance - Bar chart (using coefficients from SVM)
        self.create_feature_importance_chart()

        # Tab 3: Feature Relationships - Scatter plot with user input highlighted
        self.create_feature_relationship_chart(input_data, diagnosis)

        # Tab 4: Distribution Analysis - Violin plots
        self.create_distribution_chart(input_data, diagnosis)

        # Refresh all canvases
        for canvas in [self.tab1_canvas, self.tab2_canvas, self.tab3_canvas, self.tab4_canvas]:
            canvas.draw()

    def create_radar_chart(self, input_data, diagnosis):
        if self.df is None:
            self.tab1_canvas.axes.text(0.5, 0.5, 'No data available for visualization',
                                       ha='center', va='center', transform=self.tab1_canvas.axes.transAxes)
            return

        # Normalize data for radar chart
        normalized_data = []
        for i, col in enumerate(self.feature_names):
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            normalized_val = (input_data[i] - min_val) / (max_val - min_val + 1e-8)
            normalized_data.append(normalized_val)

        # Number of variables
        categories = self.feature_names
        N = len(categories)

        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Use polar projection
        self.tab1_canvas.axes.clear()
        ax = self.tab1_canvas.figure.add_subplot(111, projection='polar')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], color="grey", size=7)
        ax.set_ylim(0, 1)

        # Plot data
        values = normalized_data
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f"Patient: {diagnosis}")
        ax.fill(angles, values, 'b', alpha=0.1)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.set_title('Patient Profile Radar Chart', size=14, color='blue', pad=20)

    def create_feature_importance_chart(self):
        # Get feature importance from SVM coefficients
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            # If no coefficients, use equal importance
            importance = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        # Clear and set up the figure
        self.tab2_canvas.figure.clf()
        fig = self.tab2_canvas.figure
        ax = fig.add_subplot(121)  # Left subplot for bar chart
        table_ax = fig.add_subplot(122)  # Right subplot for table

        # Plot horizontal bar chart
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['Importance'], align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (SVM Coefficients)')

        # Add value labels on bars
        for i, v in enumerate(importance_df['Importance']):
            ax.text(v + 0.01, i, f'{v:.3f}', color='blue', fontweight='bold')

        # Create table
        table_data = [[f] for f in importance_df['Feature']]
        table_data = [[f, f'{i:.3f}'] for f, i in zip(importance_df['Feature'], importance_df['Importance'])]
        table = table_ax.table(cellText=table_data, colLabels=['Feature', 'Importance'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4CAF50')
            else:
                cell.set_facecolor('#f0f0f0')
        table_ax.axis('off')

        # Adjust layout
        plt.tight_layout()

    def create_feature_relationship_chart(self, input_data, diagnosis):
        if self.df is None:
            self.tab3_canvas.axes.text(0.5, 0.5, 'No data available for visualization',
                                       ha='center', va='center', transform=self.tab3_canvas.axes.transAxes)
            return

        # Select two most important features for scatter plot
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
            top_features = np.argsort(importance)[-2:]
        else:
            top_features = [0, 1]  # Default to first two features

        feature1_idx, feature2_idx = top_features
        feature1 = self.feature_names[feature1_idx]
        feature2 = self.feature_names[feature2_idx]

        # Clear and set up the axes
        self.tab3_canvas.figure.clf()
        ax = self.tab3_canvas.figure.add_subplot(111)
        ax.clear()

        # Scatter plot of all data points
        scatter = ax.scatter(
            self.df[feature1],
            self.df[feature2],
            c=self.df['Output'],
            cmap='viridis',
            alpha=0.6,
            s=50
        )

        # Highlight the current prediction
        ax.scatter(
            input_data[feature1_idx],
            input_data[feature2_idx],
            c='red' if diagnosis == 'O' else 'blue',
            s=200,
            marker='X',
            edgecolors='black',
            linewidth=2,
            label=f'Patient: {diagnosis}'
        )

        # Set axis limits based on data range
        ax.set_xlim(self.df[feature1].min() - 0.1, self.df[feature1].max() + 0.1)
        ax.set_ylim(self.df[feature2].min() - 0.1, self.df[feature2].max() + 0.1)

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title('Feature Relationship Scatter Plot')
        ax.legend()

        # Add colorbar using figure
        self.tab3_canvas.figure.colorbar(scatter, ax=ax, label='Diagnosis (0: N, 1: O)')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    def create_distribution_chart(self, input_data, diagnosis):
        if self.df is None:
            self.tab4_canvas.axes.text(0.5, 0.5, 'No data available for visualization',
                                       ha='center', va='center', transform=self.tab4_canvas.axes.transAxes)
            return

        # Create violin plots for each feature
        self.tab4_canvas.figure.clf()
        ax = self.tab4_canvas.figure.add_subplot(111)
        ax.clear()

        # Prepare data
        plot_data = []
        for feature in self.feature_names:
            plot_data.append(self.df[feature])

        # Create violin plot
        violin_parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)

        # Customize colors
        for pc in violin_parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)

        # Add current patient's data as points
        for i, value in enumerate(input_data):
            ax.scatter(i + 1, value, color='red' if diagnosis == 'O' else 'blue', s=50, zorder=3)

        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title('Distribution of Features with Patient Highlighted')

        # Set x-axis labels
        ax.set_xticks(range(1, len(self.feature_names) + 1))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3)

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = FertilityApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()