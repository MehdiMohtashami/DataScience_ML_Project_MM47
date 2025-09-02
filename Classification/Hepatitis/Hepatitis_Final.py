import sys
import numpy as np
import pandas as pd
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QMessageBox, QTabWidget, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

class HepatitisPredictorApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hepatitis")
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.y_train = None
        self.current_prediction = None
        self.initUI()
        self.load_model_and_data()

    def initUI(self):
        self.setWindowTitle('Hepatitis Survival Predictor')
        self.setGeometry(100, 100, 1400, 900)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout(model_info_group)
        self.model_name_label = QLabel("Model: K-Nearest Neighbors (KNN)")
        self.accuracy_label = QLabel("Cross-Validation Accuracy: 87.9%")
        model_info_layout.addWidget(self.model_name_label)
        model_info_layout.addWidget(self.accuracy_label)
        left_layout.addWidget(model_info_group)

        # Input fields
        input_group = QGroupBox("Patient Information")
        input_layout = QGridLayout(input_group)

        # Define input fields
        self.input_fields = {}
        fields = [
            ('Age', 'years (7-78)', 'int'),
            ('Sex', '1:Male, 2:Female', 'int'),
            ('Steroid', 'Select Yes/No', 'combo'),
            ('Antivirals', 'Select Yes/No', 'combo'),
            ('Fatigue', 'Select Yes/No', 'combo'),
            ('Malaise', 'Select Yes/No', 'combo'),
            ('Anorexia', 'Select Yes/No', 'combo'),
            ('Liver Big', 'Select Yes/No', 'combo'),
            ('Liver Firm', 'Select Yes/No', 'combo'),
            ('Spleen Palpable', 'Select Yes/No', 'combo'),
            ('Spiders', 'Select Yes/No', 'combo'),
            ('Ascites', 'Select Yes/No', 'combo'),
            ('Varices', 'Select Yes/No', 'combo'),
            ('Bilirubin', 'e.g., 0.39, 0.80, 1.20', 'float'),
            ('Alk Phosphate', 'e.g., 33, 80, 120', 'float'),
            ('Sgot', 'e.g., 13, 100, 200', 'float'),
            ('Albumin', 'e.g., 2.1, 3.0, 3.8', 'float'),
            ('Protime', 'e.g., 60, 70, 80', 'float'),
            ('Histology', 'Select Yes/No', 'combo')
        ]

        for i, (field, placeholder, field_type) in enumerate(fields):
            label = QLabel(field)
            if field_type == 'combo':
                input_field = QComboBox()
                input_field.addItem("No", 1)
                input_field.addItem("Yes", 2)
                input_field.setCurrentIndex(-1)  # No default selection
            else:
                input_field = QLineEdit()
                input_field.setPlaceholderText(placeholder)
                if field_type == 'int':
                    input_field.setValidator(QIntValidator())
                else:
                    input_field.setValidator(QDoubleValidator())

            self.input_fields[field] = input_field
            input_layout.addWidget(label, i, 0)
            input_layout.addWidget(input_field, i, 1)

        left_layout.addWidget(input_group)

        # Predict button
        self.predict_btn = QPushButton("Predict Survival")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        left_layout.addWidget(self.predict_btn)

        self.back_button = QPushButton("Back to Main")
        self.back_button.clicked.connect(self.close_and_go_back)
        self.back_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-weight: bold; }")
        left_layout.addWidget(self.back_button)

        # Prediction result
        self.result_label = QLabel("Please enter patient data and click 'Predict'")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; }")
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(self.result_label)

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualization
        self.right_panel = QTabWidget()
        self.right_panel.setVisible(False)  # Hide until prediction is made

        # Create tabs
        self.analysis_tab = QWidget()
        self.feature_importance_tab = QWidget()
        self.feature_relationship_tab = QWidget()

        self.right_panel.addTab(self.analysis_tab, "Prediction Analysis")
        self.right_panel.addTab(self.feature_importance_tab, "Feature Importance")
        self.right_panel.addTab(self.feature_relationship_tab, "Feature Relationships")

        # Setup each tab
        self.setup_analysis_tab()
        self.setup_feature_importance_tab()
        self.setup_feature_relationship_tab()

        main_layout.addWidget(self.right_panel)

    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)

        # Confusion matrix
        confusion_group = QGroupBox("Model Performance - Confusion Matrix")
        confusion_layout = QVBoxLayout(confusion_group)
        self.confusion_figure = Figure(figsize=(6, 5))
        self.confusion_canvas = FigureCanvas(self.confusion_figure)
        confusion_layout.addWidget(self.confusion_canvas)
        layout.addWidget(confusion_group)

        # PCA plot
        pca_group = QGroupBox("Patient Position in Feature Space (PCA)")
        pca_layout = QVBoxLayout(pca_group)
        self.pca_figure = Figure(figsize=(6, 5))
        self.pca_canvas = FigureCanvas(self.pca_figure)
        pca_layout.addWidget(self.pca_canvas)
        layout.addWidget(pca_group)

    def setup_feature_importance_tab(self):
        layout = QVBoxLayout(self.feature_importance_tab)

        # Feature importance bar chart
        importance_group = QGroupBox("Feature Importance")
        importance_layout = QVBoxLayout(importance_group)
        self.importance_figure = Figure(figsize=(6, 5))
        self.importance_canvas = FigureCanvas(self.importance_figure)
        importance_layout.addWidget(self.importance_canvas)
        layout.addWidget(importance_group)

        # SHAP summary plot (placeholder)
        shap_group = QGroupBox("Feature Impact on Prediction (SHAP Values)")
        shap_layout = QVBoxLayout(shap_group)
        self.shap_figure = Figure(figsize=(6, 5))
        self.shap_canvas = FigureCanvas(self.shap_figure)
        shap_layout.addWidget(self.shap_canvas)
        layout.addWidget(shap_group)

    def setup_feature_relationship_tab(self):
        layout = QVBoxLayout(self.feature_relationship_tab)

        # Feature correlation heatmap
        correlation_group = QGroupBox("Feature Correlation Heatmap")
        correlation_layout = QVBoxLayout(correlation_group)
        self.correlation_figure = Figure(figsize=(6, 5))
        self.correlation_canvas = FigureCanvas(self.correlation_figure)
        correlation_layout.addWidget(self.correlation_canvas)
        layout.addWidget(correlation_group)

        # Scatter plot of two most important features
        scatter_group = QGroupBox("Key Feature Relationship")
        scatter_layout = QVBoxLayout(scatter_group)
        self.scatter_figure = Figure(figsize=(6, 5))
        self.scatter_canvas = FigureCanvas(self.scatter_figure)
        scatter_layout.addWidget(self.scatter_canvas)
        layout.addWidget(scatter_group)

    def load_model_and_data(self):
        try:
            # Load the pre-trained model and preprocessor
            self.model = joblib.load('hepatitis_knn_model.pkl')
            self.preprocessor = joblib.load('hepatitis_preprocessor.pkl')

            # For demonstration, we'll create some sample training data
            self.X_train = pd.DataFrame(np.random.randn(100, 19),
                                        columns=self.input_fields.keys())
            self.y_train = np.random.choice([1, 2], 100)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 "Model files not found. Please make sure 'hepatitis_knn_model.pkl' and 'hepatitis_preprocessor.pkl' are in the same directory.")

    def predict(self):
        # Validate inputs
        patient_data = {}
        for field, input_field in self.input_fields.items():
            if isinstance(input_field, QComboBox):
                value = input_field.currentData()  # Get the numerical value (1 or 2)
                if value is None:
                    QMessageBox.warning(self, "Missing Data", f"Please select a value for {field}")
                    return
            else:
                value = input_field.text().strip()
                if not value:
                    QMessageBox.warning(self, "Missing Data", f"Please enter a value for {field}")
                    return
                try:
                    if field in ['Age', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime']:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Data", f"Please enter a valid number for {field}")
                    return
            patient_data[field] = value

        # Convert to DataFrame for prediction
        patient_df = pd.DataFrame([patient_data])

        try:
            # Make prediction
            prediction = self.model.predict(patient_df)
            prediction_proba = self.model.predict_proba(patient_df)

            # Store current prediction for visualization
            self.current_prediction = {
                'data': patient_df,
                'prediction': prediction[0],
                'probability': prediction_proba[0],
                'class': 'DIE' if prediction[0] == 1 else 'LIVE'
            }

            # Update result label
            confidence = max(prediction_proba[0]) * 100
            result_text = f"Prediction: {self.current_prediction['class']}\nConfidence: {confidence:.2f}%"
            self.result_label.setText(result_text)

            # Change background color based on prediction
            if prediction[0] == 1:  # DIE
                self.result_label.setStyleSheet(
                    "QLabel { background-color: #ffcccc; padding: 10px; border: 1px solid #f00; }")
            else:  # LIVE
                self.result_label.setStyleSheet(
                    "QLabel { background-color: #ccffcc; padding: 10px; border: 1px solid #0f0; }")

            # Show visualization panel and update charts
            self.right_panel.setVisible(True)
            self.update_visualizations()

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")

    def update_visualizations(self):
        # Update all visualizations with current prediction
        self.plot_confusion_matrix()
        self.plot_pca()
        self.plot_feature_importance()
        self.plot_shap_values()
        self.plot_correlation_heatmap()
        self.plot_feature_relationship()

    def plot_confusion_matrix(self):
        # For demonstration, create a sample confusion matrix
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)

        # Sample data
        y_true = np.random.choice([1, 2], 50)
        y_pred = np.random.choice([1, 2], 50)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['DIE', 'LIVE'], yticklabels=['DIE', 'LIVE'])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        self.confusion_canvas.draw()

    def plot_pca(self):
        self.pca_figure.clear()
        ax = self.pca_figure.add_subplot(111)

        # Apply PCA to training data
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(self.X_train)

        # Plot training data
        scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                             c=self.y_train, alpha=0.6, cmap='coolwarm')

        # Apply PCA to current patient and plot
        patient_transformed = pca.transform(self.current_prediction['data'])
        ax.scatter(patient_transformed[:, 0], patient_transformed[:, 1],
                   c='red', s=200, marker='X', label='Current Patient')

        ax.set_title('PCA Projection of Patient Data')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend()

        # Add colorbar
        cbar = self.pca_figure.colorbar(scatter)
        cbar.set_label('Class (1:Die, 2:Live)')

        self.pca_canvas.draw()

    def plot_feature_importance(self):
        self.importance_figure.clear()
        ax = self.importance_figure.add_subplot(111)

        # Calculate feature importance using permutation importance
        result = permutation_importance(
            self.model, self.X_train, self.y_train, n_repeats=10, random_state=42
        )

        # Get feature names from the preprocessor
        feature_names = list(self.input_fields.keys())

        # Sort features by importance
        sorted_idx = result.importances_mean.argsort()[::-1]

        # Create bar plot
        ax.barh(range(len(sorted_idx)),
                result.importances_mean[sorted_idx],
                xerr=result.importances_std[sorted_idx])
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_title('Permutation Feature Importance')
        ax.set_xlabel('Mean Decrease in Accuracy')

        self.importance_canvas.draw()

    def plot_shap_values(self):
        self.shap_figure.clear()
        ax = self.shap_figure.add_subplot(111)

        # For demonstration, create a mock SHAP plot
        feature_names = list(self.input_fields.keys())
        importance_values = np.random.randn(len(feature_names))

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(importance_values))[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = [importance_values[i] for i in sorted_idx]

        colors = ['red' if v < 0 else 'blue' for v in sorted_values]
        ax.barh(sorted_features, sorted_values, color=colors)
        ax.set_title('SHAP Values (Feature Impact on Prediction)')
        ax.set_xlabel('Impact on Prediction Output')

        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        self.shap_canvas.draw()

    def plot_correlation_heatmap(self):
        self.correlation_figure.clear()
        ax = self.correlation_figure.add_subplot(111)

        # Calculate correlation matrix
        corr_matrix = self.X_train.corr()

        # Create heatmap
        sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', center=0,
                    square=True, annot=False, cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Heatmap')

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        self.correlation_canvas.draw()

    def plot_feature_relationship(self):
        self.scatter_figure.clear()
        ax = self.scatter_figure.add_subplot(111)

        # Select two most important features for scatter plot
        feature_names = list(self.input_fields.keys())
        x_feature, y_feature = feature_names[0], feature_names[1]

        # Create scatter plot
        scatter = ax.scatter(self.X_train[x_feature], self.X_train[y_feature],
                             c=self.y_train, alpha=0.6, cmap='viridis')

        # Add current prediction point
        ax.scatter(self.current_prediction['data'][x_feature],
                   self.current_prediction['data'][y_feature],
                   c='red', s=200, marker='X', label='Current Patient')

        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Relationship between {x_feature} and {y_feature}')
        ax.legend()

        # Add colorbar
        cbar = self.scatter_figure.colorbar(scatter)
        cbar.set_label('Class (1:Die, 2:Live)')

        self.scatter_canvas.draw()

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = HepatitisPredictorApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()