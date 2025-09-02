import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QScrollArea,
                             QFormLayout, QSizePolicy, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont, QPalette
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Knowledge Level Predictor")
        self.setGeometry(100, 100, 1600, 1000)

        # Load model and data
        self.load_model_and_data()

        # Initialize user input
        self.user_input = np.zeros((1, 5))

        # Initialize charts
        self.init_charts()

        # Setup UI
        self.setup_ui()

    def load_model_and_data(self):
        """Load the trained model, scaler, label encoder and original data"""
        try:
            self.model = joblib.load('user_knowledge_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.le = joblib.load('label_encoder.pkl')
            self.accuracy = 0.9615  # From our previous results

            # Load or create original data for visualization
            # In a real scenario, you would load the original dataset
            self.original_data = pd.DataFrame({
                'STG': np.random.uniform(0, 1, 100),
                'SCG': np.random.uniform(0, 1, 100),
                'STR': np.random.uniform(0, 1, 100),
                'LPR': np.random.uniform(0, 1, 100),
                'PEG': np.random.uniform(0, 1, 100),
                'UNS': np.random.choice(['Very Low', 'Low', 'Middle', 'High'], 100)
            })

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Model files not found. Please train the model first.")
            sys.exit(1)

    def init_charts(self):
        """Initialize the chart canvases"""
        self.feature_importance_canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.pca_canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.parallel_canvas = MplCanvas(self, width=6, height=5, dpi=100)

        # Create a larger canvas for radar chart
        self.radar_fig = Figure(figsize=(8, 6), dpi=100)
        self.radar_canvas = FigureCanvas(self.radar_fig)
        self.radar_canvas.setParent(self)
        self.radar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.radar_canvas.updateGeometry()

        self.prediction_analysis_canvas = MplCanvas(self, width=6, height=5, dpi=100)

    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(500)
        left_layout = QVBoxLayout(left_panel)

        # Model info group
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        model_layout.addWidget(QLabel(f"Model: Linear SVM"))
        model_layout.addWidget(QLabel(f"Accuracy: {self.accuracy * 100:.2f}%"))
        model_layout.addWidget(QLabel("Trained to predict student knowledge level"))
        left_layout.addWidget(model_group)

        # Input group
        input_group = QGroupBox("Student Metrics Input")
        input_layout = QGridLayout(input_group)

        # Create input fields with validators and tooltips
        self.input_fields = {}
        features = ['STG', 'SCG', 'STR', 'LPR', 'PEG']
        descriptions = [
            "Study time for goal object materials (0.0-1.0)",
            "Repetition number for goal object materials (0.0-1.0)",
            "Study time for related objects (0.0-1.0)",
            "Exam performance for related objects (0.0-1.0)",
            "Exam performance for goal objects (0.0-1.0)"
        ]

        for i, (feature, desc) in enumerate(zip(features, descriptions)):
            # Create label with full description
            label = QLabel(f"{feature}: {desc}")
            label.setWordWrap(True)

            # Create input field with placeholder text
            line_edit = QLineEdit()
            line_edit.setValidator(QDoubleValidator(0.0, 1.0, 4))
            line_edit.setPlaceholderText("0.0-1.0")
            line_edit.textChanged.connect(self.update_user_input)

            input_layout.addWidget(label, i * 2, 0)
            input_layout.addWidget(line_edit, i * 2 + 1, 0)
            self.input_fields[feature] = line_edit

        left_layout.addWidget(input_group)

        # Predict button
        self.predict_btn = QPushButton("Predict Knowledge Level")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.predict_btn)

        self.back_btn = QPushButton("Back to Main")
        self.back_btn.clicked.connect(self.go_back)
        self.back_btn.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-size: 16px; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.back_btn)

        # Result display
        self.result_label = QLabel("Please enter values and click Predict")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 15px; border: 1px solid #ccc; font-weight: bold; min-height: 80px; }")
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(self.result_label)

        # Add stretch to push everything to the top
        left_layout.addStretch()

        # Right panel for visualizations
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create tab widget for charts
        self.tab_widget = QTabWidget()

        # Feature Importance tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        feature_layout.addWidget(QLabel("Feature Importance Analysis"))
        feature_layout.addWidget(self.feature_importance_canvas)
        self.tab_widget.addTab(feature_tab, "Feature Importance")

        # Data Projection tab
        pca_tab = QWidget()
        pca_layout = QVBoxLayout(pca_tab)
        pca_layout.addWidget(QLabel("Data Projection (PCA)"))
        pca_layout.addWidget(self.pca_canvas)
        self.tab_widget.addTab(pca_tab, "Data Projection")

        # Parallel Coordinates tab
        parallel_tab = QWidget()
        parallel_layout = QVBoxLayout(parallel_tab)
        parallel_layout.addWidget(QLabel("Parallel Coordinates"))
        parallel_layout.addWidget(self.parallel_canvas)
        self.tab_widget.addTab(parallel_tab, "Parallel Coordinates")

        # Radar Chart tab
        radar_tab = QWidget()
        radar_layout = QVBoxLayout(radar_tab)
        radar_layout.addWidget(QLabel("Radar Chart Comparison"))
        radar_layout.addWidget(self.radar_canvas)
        self.tab_widget.addTab(radar_tab, "Radar Chart")

        # Prediction Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.addWidget(QLabel("Prediction Confidence Analysis"))
        analysis_layout.addWidget(self.prediction_analysis_canvas)
        self.tab_widget.addTab(analysis_tab, "Prediction Analysis")

        right_layout.addWidget(self.tab_widget)

        # Initially hide the tab widget until prediction is made
        self.tab_widget.setVisible(False)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def update_user_input(self):
        """Update the user input array when any field changes"""
        try:
            for i, feature in enumerate(['STG', 'SCG', 'STR', 'LPR', 'PEG']):
                text = self.input_fields[feature].text()
                if text:
                    self.user_input[0, i] = float(text)
        except ValueError:
            pass

    def predict(self):
        """Make prediction based on user input"""
        # Check if all fields are filled
        for field in self.input_fields.values():
            if not field.text():
                QMessageBox.warning(self, "Input Error", "Please fill all fields with values between 0.0 and 1.0")
                return

        try:
            # Convert to DataFrame with feature names to avoid warnings
            feature_names = ['STG', 'SCG', 'STR', 'LPR', 'PEG']
            input_df = pd.DataFrame(self.user_input, columns=feature_names)

            # Scale the input
            scaled_input = self.scaler.transform(input_df)

            # Make prediction
            prediction_encoded = self.model.predict(scaled_input)
            prediction = self.le.inverse_transform(prediction_encoded)[0]

            # Get decision function values for confidence analysis
            if hasattr(self.model, 'decision_function'):
                decision_values = self.model.decision_function(scaled_input)[0]
            else:
                decision_values = None

            # Update result label
            self.result_label.setText(f"Predicted Knowledge Level: {prediction}")

            # Update charts
            self.update_charts(scaled_input, prediction, decision_values)

            # Show the tab widget
            self.tab_widget.setVisible(True)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")

    def update_charts(self, scaled_input, prediction, decision_values):
        """Update all charts with the new prediction"""
        # 1. Feature Importance Chart
        self.plot_feature_importance()

        # 2. PCA Projection Chart
        self.plot_pca_projection(scaled_input, prediction)

        # 3. Parallel Coordinates Chart
        self.plot_parallel_coordinates(scaled_input, prediction)

        # 4. Radar Chart
        self.plot_radar_chart(scaled_input, prediction)

        # 5. Prediction Analysis Chart
        self.plot_prediction_analysis(decision_values, prediction)

    def plot_feature_importance(self):
        """Plot feature importance based on model coefficients"""
        ax = self.feature_importance_canvas.axes
        ax.clear()

        # Get feature importance from model coefficients
        if hasattr(self.model, 'coef_'):
            # For linear models
            if len(self.model.coef_.shape) > 1:
                # Multi-class: take average absolute coefficient across classes
                importance = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                # Binary classification
                importance = np.abs(self.model.coef_[0])
        else:
            # For non-linear models, use a placeholder
            importance = np.array([0.2, 0.15, 0.25, 0.3, 0.1])

        features = ['STG', 'SCG', 'STR', 'LPR', 'PEG']

        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, align='center', color=sns.color_palette("husl", len(features)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')

        self.feature_importance_canvas.draw()

    def plot_pca_projection(self, scaled_input, prediction):
        """Plot PCA projection of data with user point highlighted"""
        ax = self.pca_canvas.axes
        ax.clear()

        # Create some sample data for visualization (in real app, use your actual data)
        np.random.seed(42)
        n_samples = 200
        sample_data = np.random.randn(n_samples, 5)
        sample_labels = np.random.choice(['Very Low', 'Low', 'Middle', 'High'], n_samples)

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(sample_data)

        # Plot all points
        for label in np.unique(sample_labels):
            idx = sample_labels == label
            ax.scatter(pca_result[idx, 0], pca_result[idx, 1], label=label, alpha=0.6)

        # Project user input and plot it
        user_pca = pca.transform(scaled_input)
        ax.scatter(user_pca[0, 0], user_pca[0, 1], s=200, c='red',
                   marker='X', label=f'You: {prediction}', edgecolors='black')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('PCA Projection of Knowledge Data')
        ax.legend()

        self.pca_canvas.draw()

    def plot_parallel_coordinates(self, scaled_input, prediction):
        """Plot parallel coordinates with user point highlighted"""
        ax = self.parallel_canvas.axes
        ax.clear()

        # Create sample data
        np.random.seed(42)
        n_samples = 50  # Fewer samples for clearer visualization
        sample_data = np.random.rand(n_samples, 5)
        sample_labels = np.random.choice(['Very Low', 'Low', 'Middle', 'High'], n_samples)

        features = ['STG', 'SCG', 'STR', 'LPR', 'PEG']

        # Plot parallel coordinates for sample data
        for i in range(n_samples):
            color = 'lightgray'
            alpha = 0.3
            linewidth = 1
            ax.plot(range(5), sample_data[i], color=color, alpha=alpha, linewidth=linewidth)

        # Plot user data with highlight
        user_data = scaled_input[0]
        ax.plot(range(5), user_data, color='red', linewidth=3, marker='o',
                label=f'You: {prediction}')

        ax.set_xticks(range(5))
        ax.set_xticklabels(features)
        ax.set_ylabel('Scaled Value')
        ax.set_title('Parallel Coordinates Plot')
        ax.legend()

        self.parallel_canvas.draw()

    def plot_radar_chart(self, scaled_input, prediction):
        """Plot radar chart comparing user to average values by class"""
        # Clear the previous figure
        self.radar_fig.clear()

        # Create sample class averages
        categories = ['STG', 'SCG', 'STR', 'LPR', 'PEG']
        N = len(categories)

        # Sample average values for each class
        class_avgs = {
            'Very Low': [0.1, 0.05, 0.15, 0.1, 0.2],
            'Low': [0.3, 0.25, 0.35, 0.3, 0.4],
            'Middle': [0.5, 0.45, 0.55, 0.6, 0.5],
            'High': [0.7, 0.65, 0.75, 0.8, 0.8]
        }

        # What each angle will be at
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the circle

        # Initialise the spider plot
        ax = self.radar_fig.add_subplot(111, polar=True)

        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], categories)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)

        # Plot each class average
        colors = ['blue', 'green', 'orange', 'purple']
        lines = []  # Store line objects for legend

        for i, (cls, values) in enumerate(class_avgs.items()):
            values += values[:1]  # Close the circle
            line, = ax.plot(angles, values, linewidth=1, linestyle='solid', label=cls, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
            lines.append(line)

        # Plot user values
        user_values = scaled_input[0].tolist()
        user_values += user_values[:1]  # Close the circle
        user_line, = ax.plot(angles, user_values, linewidth=2, linestyle='solid', label=f'You: {prediction}',
                             color='red')
        ax.fill(angles, user_values, alpha=0.3, color='red')
        lines.append(user_line)

        # Add legend only if we have lines
        if lines:
            ax.legend(lines, [line.get_label() for line in lines], loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Add a title
        plt.title('Comparison with Class Averages')

        self.radar_canvas.draw()

    def plot_prediction_analysis(self, decision_values, prediction):
        """Plot prediction confidence analysis"""
        ax = self.prediction_analysis_canvas.axes
        ax.clear()

        if decision_values is not None:
            # For models with decision function
            classes = self.le.classes_
            ax.bar(range(len(classes)), decision_values, color=sns.color_palette("husl", len(classes)))
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_ylabel('Decision Function Value')
            ax.set_title('Model Confidence for Each Class')

            # Highlight the predicted class
            predicted_idx = np.where(classes == prediction)[0][0]
            ax.patches[predicted_idx].set_edgecolor('red')
            ax.patches[predicted_idx].set_linewidth(3)
        else:
            # For models without decision function
            ax.text(0.5, 0.5, 'Confidence analysis not available\nfor this model type',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Prediction Confidence Analysis')

        self.prediction_analysis_canvas.draw()
    def go_back(self):
        self.close()


def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == "__main__":
    main()