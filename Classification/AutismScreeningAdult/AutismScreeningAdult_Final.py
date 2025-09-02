import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QFormLayout, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class AutismScreeningAdult_Final(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Autism Screening Prediction Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Load model and data
        self.load_model_and_data()

        # Setup UI
        self.setup_ui()

        # Initialize charts
        self.init_charts()

    def load_model_and_data(self):
        try:
            # Get the path to the current directory.
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Using absolute paths to download files
            self.model = joblib.load(os.path.join(current_dir, 'best_model_real.joblib'))
            self.scaler = joblib.load(os.path.join(current_dir, 'scaler_real.joblib'))
            self.feature_names = joblib.load(os.path.join(current_dir, 'feature_names_real.joblib'))

            # Load original data for visualization
            self.df = pd.read_csv(os.path.join(current_dir, 'Autism-Adult-Data.csv'))
            self.preprocess_data()

            self.model_accuracy = 0.9858  # From our previous evaluation

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            sys.exit(1)

    def preprocess_data(self):
        # Basic preprocessing for visualization
        self.df['age'].fillna(self.df['age'].median(), inplace=True)
        self.df['ethnicity'].fillna('Unknown', inplace=True)
        self.df['relation'].fillna('Unknown', inplace=True)

        # Convert target to numerical
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.df['Class/ASD_num'] = le.fit_transform(self.df['Class/ASD'])

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info
        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel(f"Model: XGBoost"))
        model_layout.addWidget(QLabel(f"Accuracy: {100*self.model_accuracy:.4f}"))
        model_layout.addWidget(QLabel("Trained on AutismScreeningAdult Dataset"))
        model_info.setLayout(model_layout)
        left_layout.addWidget(model_info)

        # Input form
        input_group = QGroupBox("Patient Information")
        form_layout = QFormLayout()

        # Age input
        self.age_input = QLineEdit()
        self.age_input.setValidator(QIntValidator(1, 120))
        self.age_input.setPlaceholderText("17-100")
        form_layout.addRow("Age:", self.age_input)
        # Gender input
        self.gender_input = QComboBox()
        self.gender_input.addItems(["Male", "Female"])
        form_layout.addRow("Gender:", self.gender_input)

        # A1-A10 scores
        self.score_inputs = []
        for i in range(1, 11):
            score_input = QComboBox()
            score_input.addItems(["0", "1"])
            score_input.setCurrentIndex(1)  # Default to 1
            form_layout.addRow(f"A{i} Score:", score_input)
            self.score_inputs.append(score_input)

        # Additional binary features
        self.jundice_input = QComboBox()
        self.jundice_input.addItems(["No", "Yes"])
        form_layout.addRow("Born with Jaundice:", self.jundice_input)

        self.austim_input = QComboBox()
        self.austim_input.addItems(["No", "Yes"])
        form_layout.addRow("Family History of Autism:", self.austim_input)

        self.used_app_input = QComboBox()
        self.used_app_input.addItems(["No", "Yes"])
        form_layout.addRow("Used Screening App Before:", self.used_app_input)

        input_group.setLayout(form_layout)

        # Scroll area for input form
        scroll = QScrollArea()
        scroll.setWidget(input_group)
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll)

        # Predict button
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 16px; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.predict_btn)

        # Prediction result
        self.result_label = QLabel("Please fill the form and click 'Predict'")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; font-size: 16px; font-weight: bold; }")
        left_layout.addWidget(self.result_label)

        # Back button
        self.back_btn = QPushButton("Back to Main")
        self.back_btn.clicked.connect(self.go_back)
        self.back_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 16px; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.back_btn)

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualizations
        self.right_panel = QTabWidget()
        main_layout.addWidget(self.right_panel)

    def init_charts(self):
        # Create tabs for different visualizations
        self.importance_tab = QWidget()
        self.relationship_tab = QWidget()
        self.analysis_tab = QWidget()

        self.importance_layout = QVBoxLayout(self.importance_tab)
        self.relationship_layout = QVBoxLayout(self.relationship_tab)
        self.analysis_layout = QVBoxLayout(self.analysis_tab)

        # Create chart canvases
        self.importance_canvas = MplCanvas(self, width=6, height=5)
        self.relationship_canvas = MplCanvas(self, width=6, height=5)
        self.analysis_canvas1 = MplCanvas(self, width=6, height=4)
        self.analysis_canvas2 = MplCanvas(self, width=6, height=4)

        # Add to layouts
        self.importance_layout.addWidget(self.importance_canvas)
        self.relationship_layout.addWidget(self.relationship_canvas)

        analysis_layout_h = QHBoxLayout()
        analysis_layout_h.addWidget(self.analysis_canvas1)
        analysis_layout_h.addWidget(self.analysis_canvas2)
        self.analysis_layout.addLayout(analysis_layout_h)

        # Add tabs to right panel
        self.right_panel.addTab(self.importance_tab, "Feature Importance")
        self.right_panel.addTab(self.relationship_tab, "Feature Relationships")
        self.right_panel.addTab(self.analysis_tab, "Prediction Analysis")

        # Initially hide the right panel
        self.right_panel.setVisible(False)

    def predict(self):
        try:
            # Get input values
            input_data = self.get_input_data()

            # Create feature vector
            features = self.create_feature_vector(input_data)

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]

            # Update result
            result_text = "Autism Detected: YES" if prediction == 1 else "Autism Detected: NO"
            result_text += f"\nConfidence: {probability[prediction] * 100:.2f}%"
            self.result_label.setText(result_text)

            # Update visualizations
            self.update_visualizations(input_data, prediction, probability)

            # Show the right panel
            self.right_panel.setVisible(True)

        except Exception as e:
            QMessageBox.warning(self, "Prediction Error", f"Error during prediction: {str(e)}")

    def get_input_data(self):
        # Validate age
        if not self.age_input.text():
            raise ValueError("Please enter age")

        age = int(self.age_input.text())
        if age < 17 or age > 100:
            raise ValueError("Age must be between 17 and 100")

        # Get other values
        gender = 1 if self.gender_input.currentText() == "Male" else 0
        scores = [int(input.currentText()) for input in self.score_inputs]
        jundice = 1 if self.jundice_input.currentText() == "Yes" else 0
        austim = 1 if self.austim_input.currentText() == "Yes" else 0
        used_app = 1 if self.used_app_input.currentText() == "Yes" else 0

        return {
            'age': age,
            'gender': gender,
            'scores': scores,
            'jundice': jundice,
            'austim': austim,
            'used_app': used_app
        }

    def create_feature_vector(self, input_data):
        # Create a zero vector with the same length as feature names
        features = np.zeros(len(self.feature_names))

        # Map input data to feature vector
        feature_index_map = {name: idx for idx, name in enumerate(self.feature_names)}

        # Set values for known features
        if 'age' in feature_index_map:
            features[feature_index_map['age']] = input_data['age']

        if 'gender' in feature_index_map:
            features[feature_index_map['gender']] = input_data['gender']

        for i, score in enumerate(input_data['scores'], 1):
            feature_name = f'A{i}_Score'
            if feature_name in feature_index_map:
                features[feature_index_map[feature_name]] = score

        if 'jundice' in feature_index_map:
            features[feature_index_map['jundice']] = input_data['jundice']

        if 'austim' in feature_index_map:
            features[feature_index_map['austim']] = input_data['austim']

        if 'used_app_before' in feature_index_map:
            features[feature_index_map['used_app_before']] = input_data['used_app']

        # Set default values for one-hot encoded features
        self.set_default_one_hot_values(features, feature_index_map)

        return features

    def set_default_one_hot_values(self, features, feature_index_map):
        # Set default values for one-hot encoded features
        # Based on most common values in training data
        defaults = {
            'ethnicity': 'White-European',
            'contry_of_res': 'United States',
            'relation': 'Self'
        }

        for feature, value in defaults.items():
            default_col = f"{feature}_{value.replace(' ', '_')}"
            if default_col in feature_index_map:
                features[feature_index_map[default_col]] = 1

    def update_visualizations(self, input_data, prediction, probability):
        # Clear previous plots
        self.importance_canvas.axes.clear()
        self.relationship_canvas.axes.clear()
        self.analysis_canvas1.axes.clear()
        self.analysis_canvas2.axes.clear()

        # Plot 1: Feature Importance
        self.plot_feature_importance()

        # Plot 2: Feature Relationships
        self.plot_feature_relationships(input_data, prediction)

        # Plot 3-4: Prediction Analysis
        self.plot_prediction_analysis(input_data, prediction, probability)

        # Refresh canvases
        self.importance_canvas.draw()
        self.relationship_canvas.draw()
        self.analysis_canvas1.draw()
        self.analysis_canvas2.draw()

    def plot_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            # Get feature importance
            importance = self.model.feature_importances_
            feature_names = self.feature_names

            # Create a DataFrame for sorting
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })

            # Sort by importance
            feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).head(10)

            # Plot
            self.importance_canvas.axes.barh(
                feat_imp_df['feature'],
                feat_imp_df['importance'],
                color=sns.color_palette("viridis", len(feat_imp_df))
            )
            self.importance_canvas.axes.set_xlabel('Importance')
            self.importance_canvas.axes.set_title('Top 10 Feature Importance')
            self.importance_canvas.fig.tight_layout()

    def plot_feature_relationships(self, input_data, prediction):
        # Select two most important numerical features for scatter plot
        numerical_features = ['age', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score',
                              'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']

        # Find which numerical features are available in our dataset
        available_features = [f for f in numerical_features if f in self.df.columns]

        if len(available_features) >= 2:
            # Use age and A9_Score as default
            x_feature = 'age'
            y_feature = 'A9_Score'

            # Create scatter plot
            sns.scatterplot(
                data=self.df,
                x=x_feature,
                y=y_feature,
                hue='Class/ASD_num',
                palette={0: 'blue', 1: 'red'},
                alpha=0.6,
                ax=self.relationship_canvas.axes
            )

            # Plot the current prediction point
            self.relationship_canvas.axes.scatter(
                input_data['age'],
                input_data['scores'][8],  # A9_Score is the 9th item (index 8)
                color='green' if prediction == 0 else 'orange',
                s=200,
                edgecolors='black',
                linewidth=2,
                label='Current Prediction'
            )

            self.relationship_canvas.axes.set_xlabel(x_feature)
            self.relationship_canvas.axes.set_ylabel(y_feature)
            self.relationship_canvas.axes.set_title(f'Relationship between {x_feature} and {y_feature}')
            self.relationship_canvas.axes.legend()
            self.relationship_canvas.fig.tight_layout()

    def plot_prediction_analysis(self, input_data, prediction, probability):
        # Plot 1: Score distribution comparison
        self.plot_score_comparison(input_data, prediction)

        # Plot 2: Probability visualization
        self.plot_probability(probability)

    def plot_score_comparison(self, input_data, prediction):
        # Prepare data for radar chart
        categories = [f'A{i}' for i in range(1, 11)]
        values = input_data['scores']

        # Calculate average scores for ASD and non-ASD groups
        asd_scores = self.df[self.df['Class/ASD'] == 'YES'][[f'A{i}_Score' for i in range(1, 11)]].mean().values
        non_asd_scores = self.df[self.df['Class/ASD'] == 'NO'][[f'A{i}_Score' for i in range(1, 11)]].mean().values

        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        values += values[:1]
        asd_scores = np.append(asd_scores, asd_scores[0])
        non_asd_scores = np.append(non_asd_scores, non_asd_scores[0])

        self.analysis_canvas1.axes = self.analysis_canvas1.fig.add_subplot(111, polar=True)
        self.analysis_canvas1.axes.plot(angles, values, 'o-', linewidth=2, label='Patient Scores')
        self.analysis_canvas1.axes.fill(angles, values, alpha=0.25)
        self.analysis_canvas1.axes.plot(angles, asd_scores, 'o-', linewidth=2, label='ASD Average')
        self.analysis_canvas1.axes.plot(angles, non_asd_scores, 'o-', linewidth=2, label='Non-ASD Average')
        self.analysis_canvas1.axes.set_thetagrids(np.degrees(angles[:-1]), categories)
        self.analysis_canvas1.axes.set_title('Score Comparison with Averages')
        self.analysis_canvas1.axes.legend(loc='upper right')

    def plot_probability(self, probability):
        # Plot probability distribution
        labels = ['No ASD', 'ASD']
        colors = ['lightblue', 'lightcoral']

        bars = self.analysis_canvas2.axes.bar(labels, probability, color=colors)
        self.analysis_canvas2.axes.set_ylabel('Probability')
        self.analysis_canvas2.axes.set_title('Prediction Probability Distribution')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.analysis_canvas2.axes.text(bar.get_x() + bar.get_width() / 2., height,
                                            f'{probability[i]:.3f}', ha='center', va='bottom')

    def go_back(self):
        self.close()


def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = AutismScreeningAdult_Final(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()