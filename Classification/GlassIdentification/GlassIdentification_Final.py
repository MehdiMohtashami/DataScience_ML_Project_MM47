import sys
import numpy as np
import pandas as pd
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGroupBox, QFormLayout,
                             QMessageBox, QScrollArea, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.decomposition import PCA


class GlassClassifierApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Heart Disease Prediction")
        self.model = None
        self.scaler = None
        self.feature_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        self.feature_ranges = {
            'RI': (1.51, 1.53),
            'Na': (10.0, 18.0),
            'Mg': (0.0, 4.5),
            'Al': (0.3, 3.5),
            'Si': (69.0, 76.0),
            'K': (0.0, 6.5),
            'Ca': (5.0, 17.0),
            'Ba': (0.0, 3.5),
            'Fe': (0.0, 0.6)
        }
        self.class_names = {
            1: 'building_windows_float_processed',
            2: 'building_windows_non_float_processed',
            3: 'vehicle_windows_float_processed',
            4: 'vehicle_windows_non_float_processed',
            5: 'containers',
            6: 'tableware',
            7: 'headlamps'
        }
        self.current_prediction = None
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle('Glass Classification App')
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_info)
        self.model_label = QLabel("Model: Random Forest (Accuracy: 86.05%)")
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        model_layout.addWidget(self.model_label)
        left_layout.addWidget(model_info)

        input_group = QGroupBox("Input Features")
        form_layout = QFormLayout(input_group)

        self.input_fields = {}
        for feature in self.feature_names:
            line_edit = QLineEdit()
            line_edit.setValidator(QDoubleValidator(0.0, 100.0, 6))
            min_val, max_val = self.feature_ranges[feature]
            line_edit.setPlaceholderText(f"Range: {min_val:.2f} - {max_val:.2f}")
            form_layout.addRow(QLabel(f"{feature}:"), line_edit)
            self.input_fields[feature] = line_edit

        left_layout.addWidget(input_group)

        self.predict_btn = QPushButton("Predict Glass Type")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.predict_btn)

        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)
        self.back_button.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(self.back_button)



        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("Please enter values and click 'Predict'")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; padding: 20px; }")
        result_layout.addWidget(self.result_label)
        left_layout.addWidget(result_group)

        main_layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.viz_tabs = QTabWidget()
        right_layout.addWidget(self.viz_tabs)

        self.feature_importance_tab = QWidget()
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_tab)
        self.viz_tabs.addTab(self.feature_importance_tab, "Feature Importance")

        self.pca_tab = QWidget()
        self.pca_layout = QVBoxLayout(self.pca_tab)
        self.viz_tabs.addTab(self.pca_tab, "PCA Visualization")

        self.relationships_tab = QWidget()
        self.relationships_layout = QVBoxLayout(self.relationships_tab)
        self.viz_tabs.addTab(self.relationships_tab, "Feature Relationships")

        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        self.viz_tabs.addTab(self.analysis_tab, "Prediction Analysis")

        main_layout.addWidget(right_panel)

        self.viz_tabs.setVisible(False)

    def load_model(self):
        try:
            self.model = joblib.load('best_glass_classification_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("Model loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            print(f"Error loading model: {e}")

    def predict(self):
        try:
            input_values = {}
            for feature, field in self.input_fields.items():
                text = field.text().strip()
                if not text:
                    QMessageBox.warning(self, "Input Error", f"Please enter a value for {feature}")
                    return
                try:
                    value = float(text)
                    min_val, max_val = self.feature_ranges[feature]
                    if value < min_val or value > max_val:
                        QMessageBox.warning(self, "Input Error",
                                            f"Value for {feature} should be between {min_val:.2f} and {max_val:.2f}")
                        return
                    input_values[feature] = value
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Invalid value for {feature}")
                    return

            input_array = np.array([input_values[feature] for feature in self.feature_names]).reshape(1, -1)
            scaled_input = self.scaler.transform(input_array)

            prediction = self.model.predict(scaled_input)[0]
            probability = np.max(self.model.predict_proba(scaled_input))

            self.current_prediction = {
                'values': input_values,
                'scaled': scaled_input,
                'prediction': prediction,
                'probability': probability
            }

            class_name = self.class_names.get(prediction, f"Type {prediction}")
            self.result_label.setText(f"Predicted Glass Type: {class_name}\nConfidence: {probability:.2%}")

            self.viz_tabs.setVisible(True)

            self.update_feature_importance()
            self.update_pca_visualization()
            self.update_feature_relationships()
            self.update_prediction_analysis()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during prediction: {str(e)}")
            print(f"Prediction error: {e}")

    def update_feature_importance(self):
        self.clear_layout(self.feature_importance_layout)

        try:
            importance_fig = Figure(figsize=(10, 6))
            importance_canvas = FigureCanvas(importance_fig)
            self.feature_importance_layout.addWidget(importance_canvas)

            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            ax = importance_fig.add_subplot(111)
            bars = ax.bar(range(len(importances)), importances[indices], align="center")

            for i, v in enumerate(importances[indices]):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importance in Random Forest Model")

            importance_canvas.draw()

        except Exception as e:
            error_label = QLabel(f"Could not generate feature importance: {str(e)}")
            self.feature_importance_layout.addWidget(error_label)
            print(f"Feature importance error: {e}")

    def update_pca_visualization(self):
        self.clear_layout(self.pca_layout)

        try:
            df = pd.read_csv('glass.csv')
            X = df.drop(['Id number', 'Type of Glass'], axis=1)
            y = df['Type of Glass']

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.scaler.transform(X))

            pca_fig = Figure(figsize=(10, 6))
            pca_canvas = FigureCanvas(pca_fig)
            self.pca_layout.addWidget(pca_canvas)

            ax = pca_fig.add_subplot(111)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)

            if self.current_prediction:
                current_pca = pca.transform(self.current_prediction['scaled'])
                ax.scatter(current_pca[0, 0], current_pca[0, 1],
                           c='red', s=200, marker='X',
                           label=f'Prediction: {self.current_prediction["prediction"]}')
                ax.legend()

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            ax.set_title("PCA Visualization of Glass Data")
            pca_fig.colorbar(scatter, label='Glass Type')

            pca_canvas.draw()

        except Exception as e:
            error_label = QLabel(f"Could not generate PCA visualization: {str(e)}")
            self.pca_layout.addWidget(error_label)
            print(f"PCA visualization error: {e}")

    def update_feature_relationships(self):
        self.clear_layout(self.relationships_layout)

        try:
            scroll = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            scroll.setWidget(scroll_widget)
            scroll.setWidgetResizable(True)
            self.relationships_layout.addWidget(scroll)

            df = pd.read_csv('glass.csv')

            importances = self.model.feature_importances_
            top_features = np.argsort(importances)[::-1][:2]
            feature1, feature2 = self.feature_names[top_features[0]], self.feature_names[top_features[1]]

            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            scroll_layout.addWidget(canvas)

            ax = fig.add_subplot(111)
            scatter = ax.scatter(df[feature1], df[feature2], c=df['Type of Glass'], cmap='viridis', alpha=0.7)

            if self.current_prediction:
                ax.scatter(self.current_prediction['values'][feature1],
                           self.current_prediction['values'][feature2],
                           c='red', s=200, marker='X',
                           label=f'Prediction: {self.current_prediction["prediction"]}')
                ax.legend()

            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_title(f"Relationship between {feature1} and {feature2}")
            fig.colorbar(scatter, label='Glass Type')

            canvas.draw()

            corr_fig = Figure(figsize=(10, 8))
            corr_canvas = FigureCanvas(corr_fig)
            scroll_layout.addWidget(corr_canvas)

            ax2 = corr_fig.add_subplot(111)
            correlation_matrix = df.drop('Id number', axis=1).corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
            ax2.set_title("Feature Correlation Heatmap")

            corr_canvas.draw()

        except Exception as e:
            error_label = QLabel(f"Could not generate feature relationships: {str(e)}")
            self.relationships_layout.addWidget(error_label)
            print(f"Feature relationships error: {e}")

    def update_prediction_analysis(self):
        self.clear_layout(self.analysis_layout)

        if not self.current_prediction:
            return

        try:
            # Get prediction probabilities
            probabilities = self.model.predict_proba(self.current_prediction['scaled'])[0]

            # Get all possible classes from the model
            all_classes = self.model.classes_

            # Create probability distribution chart
            prob_fig = Figure(figsize=(10, 6))
            prob_canvas = FigureCanvas(prob_fig)
            self.analysis_layout.addWidget(prob_canvas)

            ax = prob_fig.add_subplot(111)

            # Use the actual class labels from the model
            class_labels = [self.class_names.get(cls, f"Type {cls}") for cls in all_classes]

            bars = ax.bar(class_labels, probabilities, color='skyblue')

            # Highlight the predicted class
            pred_idx = np.where(all_classes == self.current_prediction['prediction'])[0][0]
            bars[pred_idx].set_color('red')

            ax.set_xlabel("Glass Types")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probability Distribution")
            ax.tick_params(axis='x', rotation=45)

            # Add probability values on top of bars
            for i, v in enumerate(probabilities):
                ax.text(i, v + 0.01, f'{v:.2%}', ha='center')

            prob_canvas.draw()

            # Add decision path explanation for random forest
            info_label = QLabel(
                f"The model is {self.current_prediction['probability']:.2%} confident that "
                f"the sample belongs to {self.class_names.get(self.current_prediction['prediction'])}. "
                "The feature importance and relationships shown in other tabs influenced this prediction."
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
            self.analysis_layout.addWidget(info_label)

        except Exception as e:
            error_label = QLabel(f"Could not generate prediction analysis: {str(e)}")
            self.analysis_layout.addWidget(error_label)
            print(f"Prediction analysis error: {e}")

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 10, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = GlassClassifierApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()