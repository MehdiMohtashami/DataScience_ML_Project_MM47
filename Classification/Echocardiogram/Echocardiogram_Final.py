import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.inspection import permutation_importance
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QScrollArea, QMessageBox,
                             QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Echocardiogram")
        self.setWindowTitle("Heart Attack Survival Predictor")
        self.setGeometry(100, 100, 1400, 900)

        # Load model and scaler
        try:
            self.model = joblib.load('svm_heart_attack_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.accuracy = 0.885  # From our previous evaluation

            # Get feature names from the scaler
            self.feature_names = self.scaler.feature_names_in_
            print(f"Loaded scaler feature names: {self.feature_names}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load model files: {str(e)}")
            sys.exit(1)

        # Load data for visualization
        try:
            self.df = pd.read_csv('echocardiogram.csv')
            self.preprocess_data()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load data file: {str(e)}")
            sys.exit(1)

        self.initUI()
        self.current_prediction = None

    def preprocess_data(self):
        # Clean and preprocess data for visualization
        # Create target variable
        def create_target(row):
            if pd.isna(row['Still-alive']) or pd.isna(row['Survival']):
                return np.nan
            if row['Still-alive'] == 1:
                return 1
            else:
                return 1 if row['Survival'] >= 12 else 0

        self.df['target'] = self.df.apply(create_target, axis=1)
        self.df_clean = self.df.drop(['Name', 'Group', 'Alive-at-1', 'Survival', 'Still-alive'], axis=1)
        self.df_clean = self.df_clean.dropna(subset=['target'])

        # Handle missing values
        numerical_features = ['Age-heart-attack', 'Pericardial-effusion', 'Fractional-shortening',
                              'Epss', 'Lvdd', 'Wall-motion-score', 'Wall-motion-index', 'Mult']
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        self.df_clean[numerical_features] = imputer.fit_transform(self.df_clean[numerical_features])

        # Prepare feature matrix for permutation importance
        self.X = self.df_clean.drop('target', axis=1)
        self.y = self.df_clean['target']

        # Ensure the column order matches the scaler
        self.X = self.X[self.feature_names]
        self.X_scaled = self.scaler.transform(self.X)

        # Calculate feature importance
        self.calculate_feature_importance()

    def calculate_feature_importance(self):
        # Calculate permutation importance
        try:
            result = permutation_importance(
                self.model, self.X_scaled, self.y, n_repeats=10, random_state=42
            )
            self.importances = result.importances_mean
            self.feature_names_imp = self.X.columns

            # Sort features by importance
            sorted_idx = np.argsort(self.importances)[::-1]
            self.sorted_importances = self.importances[sorted_idx]
            self.sorted_features = self.feature_names_imp[sorted_idx]
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            # Use default importance values if calculation fails
            self.sorted_features = self.X.columns
            self.sorted_importances = np.ones(len(self.X.columns))

    def initUI(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)

        # Model info group
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        model_layout.addWidget(QLabel(f"Model: SVM (RBF Kernel)"))
        model_layout.addWidget(QLabel(f"Accuracy: {self.accuracy:.3f}"))
        model_layout.addWidget(QLabel("Class weight: Balanced"))
        left_layout.addWidget(model_group)

        # Input group
        input_group = QGroupBox("Patient Information")
        input_layout = QGridLayout(input_group)

        self.input_fields = {}
        features = self.feature_names.tolist()

        ranges = {
            'Age-heart-attack': (35, 86),
            'Pericardial-effusion': (0, 1),
            'Fractional-shortening': (0.01, 0.61),
            'Epss': (0, 40),
            'Lvdd': (2.32, 6.78),
            'Wall-motion-score': (2, 39),
            'Wall-motion-index': (1, 3),
            'Mult': (0.14, 2.0)
        }

        for i, feature in enumerate(features):
            input_layout.addWidget(QLabel(feature), i, 0)
            field = QLineEdit()
            field.setValidator(QDoubleValidator())
            if feature in ranges:
                field.setPlaceholderText(f"Range: {ranges[feature][0]} - {ranges[feature][1]}")
            self.input_fields[feature] = field
            input_layout.addWidget(field, i, 1)

        left_layout.addWidget(input_group)

        # Predict button
        predict_btn = QPushButton("Predict Survival")
        predict_btn.clicked.connect(self.predict)
        left_layout.addWidget(predict_btn)

        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back) # متد close رو صدا میزنه
        left_layout.addWidget(back_button)

        # Add left panel to the main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualization
        self.tabs = QTabWidget()

        # Tab 1: Prediction Analysis
        self.prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(self.prediction_tab)

        self.prediction_label = QLabel("Please make a prediction to see results")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Bold))
        prediction_layout.addWidget(self.prediction_label)

        # Canvas for plots
        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)

        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.canvas1)
        plot_layout.addWidget(self.canvas2)

        prediction_layout.addLayout(plot_layout)
        self.tabs.addTab(self.prediction_tab, "Prediction Analysis")

        # Tab 2: Feature Importance
        self.feature_tab = QWidget()
        feature_layout = QVBoxLayout(self.feature_tab)

        self.canvas3 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas4 = MplCanvas(self, width=5, height=4, dpi=100)

        feature_plot_layout = QHBoxLayout()
        feature_plot_layout.addWidget(self.canvas3)
        feature_plot_layout.addWidget(self.canvas4)

        feature_layout.addLayout(feature_plot_layout)
        self.tabs.addTab(self.feature_tab, "Feature Importance")

        # Tab 3: Feature Relationships
        self.relationship_tab = QWidget()
        relationship_layout = QVBoxLayout(self.relationship_tab)

        self.canvas5 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas6 = MplCanvas(self, width=5, height=4, dpi=100)

        relationship_plot_layout = QHBoxLayout()
        relationship_plot_layout.addWidget(self.canvas5)
        relationship_plot_layout.addWidget(self.canvas6)

        relationship_layout.addLayout(relationship_plot_layout)
        self.tabs.addTab(self.relationship_tab, "Feature Relationships")

        main_layout.addWidget(self.tabs)

        # Initially hide all canvases
        self.hide_all_canvases()

    def hide_all_canvases(self):
        self.canvas1.setVisible(False)
        self.canvas2.setVisible(False)
        self.canvas3.setVisible(False)
        self.canvas4.setVisible(False)
        self.canvas5.setVisible(False)
        self.canvas6.setVisible(False)

    def show_all_canvases(self):
        self.canvas1.setVisible(True)
        self.canvas2.setVisible(True)
        self.canvas3.setVisible(True)
        self.canvas4.setVisible(True)
        self.canvas5.setVisible(True)
        self.canvas6.setVisible(True)

    def predict(self):
        try:
            # Get input values
            input_data = []
            valid_input = True

            for feature in self.feature_names:
                value = self.input_fields[feature].text()
                if value == '':
                    QMessageBox.warning(self, "Input Error", f"Please enter a value for {feature}")
                    valid_input = False
                    break
                input_data.append(float(value))

            if not valid_input:
                return

            # Convert to numpy array and reshape
            input_array = np.array(input_data).reshape(1, -1)

            # Create a DataFrame with the correct feature names and order
            input_df = pd.DataFrame(input_array, columns=self.feature_names)

            # Scale the input
            input_scaled = self.scaler.transform(input_df)

            # Make prediction
            prediction = self.model.predict(input_scaled)

            # Get probabilities if available, otherwise use decision function
            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(input_scaled)
            else:
                # Fallback for models without predict_proba
                decision = self.model.decision_function(input_scaled)
                # Convert decision function to pseudo-probabilities
                probability = 1 / (1 + np.exp(-decision))
                # Reshape to match predict_proba format
                probability = np.column_stack((1 - probability, probability))

            # Store current prediction
            self.current_prediction = {
                'input': input_array[0],
                'scaled_input': input_scaled[0],
                'prediction': prediction[0],
                'probability': probability[0],
                'features': self.feature_names.tolist()
            }

            # Update UI
            result_text = "Prediction: "
            if prediction[0] == 1:
                result_text += "Patient will likely survive at least one year"
                result_text += f"\nConfidence: {probability[0][1]:.2%}"
            else:
                result_text += "Patient is unlikely to survive one year"
                result_text += f"\nConfidence: {probability[0][0]:.2%}"

            self.prediction_label.setText(result_text)

            # Show all canvases and update plots
            self.show_all_canvases()
            self.update_plots()

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")
            print(f"Prediction error: {e}")

    def update_plots(self):
        if self.current_prediction is None:
            return

        try:
            # Clear all canvases
            for canvas in [self.canvas1, self.canvas2, self.canvas3,
                           self.canvas4, self.canvas5, self.canvas6]:
                canvas.axes.clear()

            # Plot 1: Feature Importance Bar Chart
            self.canvas1.axes.barh(self.sorted_features, self.sorted_importances)
            self.canvas1.axes.set_xlabel('Importance')
            self.canvas1.axes.set_title('Feature Importance (Permutation)')
            self.canvas1.fig.tight_layout()
            self.canvas1.draw()

            # Plot 2: Prediction Probability Pie Chart
            labels = ['Will Not Survive', 'Will Survive']
            sizes = self.current_prediction['probability']
            colors = ['#ff9999', '#66b3ff']
            explode = (0.1, 0) if self.current_prediction['prediction'] == 1 else (0, 0.1)

            self.canvas2.axes.pie(sizes, explode=explode, labels=labels, colors=colors,
                                  autopct='%1.1f%%', shadow=True, startangle=90)
            self.canvas2.axes.axis('equal')
            self.canvas2.axes.set_title('Prediction Probability')
            self.canvas2.draw()

            # Plot 3: Top Features Comparison
            top_n = min(4, len(self.sorted_features))
            top_features = self.sorted_features[:top_n]
            top_importances = self.sorted_importances[:top_n]

            self.canvas3.axes.bar(range(top_n), top_importances, tick_label=top_features)
            self.canvas3.axes.set_ylabel('Importance')
            self.canvas3.axes.set_title('Top Important Features')
            self.canvas3.axes.tick_params(axis='x', rotation=45)
            self.canvas3.fig.tight_layout()
            self.canvas3.draw()

            # Plot 4: Radar Chart for Top Features
            if len(self.sorted_features) >= 2:
                top_n = min(4, len(self.sorted_features))
                top_features = self.sorted_features[:top_n]

                angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False).tolist()
                angles += angles[:1]  # Close the circle

                # Get current patient's values for top features
                patient_values = []
                for feature in top_features:
                    idx = self.current_prediction['features'].index(feature)
                    patient_values.append(self.current_prediction['input'][idx])

                # Normalize to 0-1 scale for radar chart
                patient_values = np.array(patient_values)
                max_vals = self.X[top_features].max().values
                min_vals = self.X[top_features].min().values
                normalized_values = (patient_values - min_vals) / (max_vals - min_vals)
                normalized_values = normalized_values.tolist()
                normalized_values += normalized_values[:1]  # Close the circle

                self.canvas4.axes.plot(angles, normalized_values, 'o-', linewidth=2, label='Current Patient')
                self.canvas4.axes.fill(angles, normalized_values, alpha=0.25)
                self.canvas4.axes.set_xticks(angles[:-1])
                self.canvas4.axes.set_xticklabels(top_features)
                self.canvas4.axes.set_ylim(0, 1)
                self.canvas4.axes.set_title('Patient vs Normal Range (Top Features)')
                self.canvas4.axes.legend(loc='upper right')
                self.canvas4.draw()

            # Plot 5: Feature Relationships - Scatter plot of two most important features
            if len(self.sorted_features) >= 2:
                feat1, feat2 = self.sorted_features[0], self.sorted_features[1]

                # Get indices of these features
                idx1 = self.current_prediction['features'].index(feat1)
                idx2 = self.current_prediction['features'].index(feat2)

                # Create scatter plot
                survived = self.df_clean[self.df_clean['target'] == 1]
                not_survived = self.df_clean[self.df_clean['target'] == 0]

                self.canvas5.axes.scatter(survived[feat1], survived[feat2],
                                          alpha=0.7, label='Survived', color='green')
                self.canvas5.axes.scatter(not_survived[feat1], not_survived[feat2],
                                          alpha=0.7, label='Not Survived', color='red')

                # Plot current patient
                self.canvas5.axes.scatter(self.current_prediction['input'][idx1],
                                          self.current_prediction['input'][idx2],
                                          s=200, marker='*',
                                          color='blue' if self.current_prediction['prediction'] == 1 else 'orange',
                                          label='Current Patient')

                self.canvas5.axes.set_xlabel(feat1)
                self.canvas5.axes.set_ylabel(feat2)
                self.canvas5.axes.set_title('Feature Relationship: ' + feat1 + ' vs ' + feat2)
                self.canvas5.axes.legend()
                self.canvas5.fig.tight_layout()
                self.canvas5.draw()

            # Plot 6: Distribution of most important feature
            if len(self.sorted_features) >= 1:
                most_important = self.sorted_features[0]
                idx = self.current_prediction['features'].index(most_important)
                patient_value = self.current_prediction['input'][idx]

                survived = self.df_clean[self.df_clean['target'] == 1][most_important]
                not_survived = self.df_clean[self.df_clean['target'] == 0][most_important]

                self.canvas6.axes.hist([survived, not_survived], bins=10,
                                       alpha=0.7, label=['Survived', 'Not Survived'],
                                       color=['green', 'red'], density=True)

                # Add vertical line for current patient
                self.canvas6.axes.axvline(patient_value, color='blue', linestyle='--',
                                          linewidth=2, label='Current Patient')

                self.canvas6.axes.set_xlabel(most_important)
                self.canvas6.axes.set_ylabel('Density')
                self.canvas6.axes.set_title('Distribution of ' + most_important)
                self.canvas6.axes.legend()
                self.canvas6.fig.tight_layout()
                self.canvas6.draw()

        except Exception as e:
            print(f"Error updating plots: {e}")

    def close_and_go_back(self):
        self.close() #


if __name__ == '__main__':
    app = QApplication(sys.argv)
    parent = None
    window = MainWindow(parent)
    window.show()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())