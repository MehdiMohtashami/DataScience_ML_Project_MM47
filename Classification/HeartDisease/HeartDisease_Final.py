import sys
import numpy as np
import pandas as pd
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('future.no_silent_downcasting', True)


class HeartDiseaseApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HeartDisease")
        self.model = None
        self.feature_names = None
        self.current_prediction = None
        self.X_train = None
        self.y_train = None
        self.X_train_clean = None  # داده‌های تمیز شده برای PCA
        self.initUI()
        self.load_model()
        self.load_training_data()

    def initUI(self):
        self.setWindowTitle('HeartDisease Prediction System - Advanced Analysis')
        self.setGeometry(100, 100, 1600, 1000)

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
        self.model_label = QLabel("Model: Logistic Regression")
        self.accuracy_label = QLabel("Accuracy: 88.52%")
        self.model_label.setStyleSheet("QLabel { font-weight: bold; color: darkblue; }")
        self.accuracy_label.setStyleSheet("QLabel { font-weight: bold; color: darkgreen; }")
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.accuracy_label)

        # Input fields
        input_group = QGroupBox("Patient Information")
        input_layout = QGridLayout(input_group)

        # Define input fields with validation
        self.input_fields = {}
        fields = [
            ('age', 'Age (29-77)', 'int', (29, 77)),
            ('sex', 'Sex (0: Female, 1: Male)', 'combo', ['0', '1']),
            ('cp', 'Chest Pain Type (1-4)', 'combo', ['1', '2', '3', '4']),
            ('trestbps', 'Resting BP (94-200)', 'int', (94, 200)),
            ('chol', 'Cholesterol (126-564)', 'int', (126, 564)),
            ('fbs', 'Fasting Blood Sugar (0-1)', 'combo', ['0', '1']),
            ('restecg', 'Resting ECG (0-2)', 'combo', ['0', '1', '2']),
            ('thalach', 'Max Heart Rate (71-202)', 'int', (71, 202)),
            ('exang', 'Exercise Angina (0-1)', 'combo', ['0', '1']),
            ('oldpeak', 'ST Depression (0-6.2)', 'float', (0, 6.2)),
            ('slope', 'Slope (1-3)', 'combo', ['1', '2', '3']),
            ('ca', 'Major Vessels (0-3)', 'combo', ['0', '1', '2', '3']),
            ('thal', 'Thalassemia (3,6,7)', 'combo', ['3', '6', '7'])
        ]

        for i, (field_name, label, field_type, constraints) in enumerate(fields):
            input_layout.addWidget(QLabel(label), i, 0)

            if field_type == 'int':
                field = QLineEdit()
                field.setValidator(QIntValidator(constraints[0], constraints[1]))
                self.input_fields[field_name] = field
            elif field_type == 'float':
                field = QLineEdit()
                field.setValidator(QDoubleValidator(constraints[0], constraints[1], 2))
                self.input_fields[field_name] = field
            elif field_type == 'combo':
                field = QComboBox()
                field.addItems(constraints)
                self.input_fields[field_name] = field

            input_layout.addWidget(field, i, 1)

        # Prediction button
        predict_btn = QPushButton('Predict Heart Disease Risk')
        predict_btn.clicked.connect(self.predict)
        predict_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        #Closing
        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)
        back_button.setStyleSheet("QLabel { background-color: gray; font-size: 20px; font-weight: bold; padding: 15px;border-radius: 10px; }")
        # left_layout.addWidget(self.back_button)

        # Add widgets to left layout
        left_layout.addWidget(model_info)
        left_layout.addWidget(input_group)
        left_layout.addWidget(predict_btn)
        left_layout.addWidget(back_button)
        left_layout.addStretch()

        # Right panel for charts - using tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Prediction result
        self.result_group = QGroupBox("Prediction Result")
        self.result_group.setVisible(False)
        result_layout = QVBoxLayout(self.result_group)
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel { 
                font-size: 18px; 
                font-weight: bold; 
                padding: 15px;
                border-radius: 10px;
            }
        """)
        result_layout.addWidget(self.result_label)


        # Tab widget for charts
        self.tab_widget = QTabWidget()
        self.tab_widget.setVisible(False)

        # Create tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        self.tab_widget.addTab(self.tab1, "Feature Importance")
        self.tab_widget.addTab(self.tab2, "PCA Analysis")
        self.tab_widget.addTab(self.tab3, "Risk Radar")
        self.tab_widget.addTab(self.tab4, "Comparison")

        # Set layouts for tabs
        self.tab1.layout = QVBoxLayout(self.tab1)
        self.tab2.layout = QVBoxLayout(self.tab2)
        self.tab3.layout = QVBoxLayout(self.tab3)
        self.tab4.layout = QVBoxLayout(self.tab4)

        # Create figures and canvases for each tab
        self.figure1 = Figure(figsize=(10, 8))
        self.canvas1 = FigureCanvas(self.figure1)
        self.tab1.layout.addWidget(self.canvas1)

        self.figure2 = Figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.figure2)
        self.tab2.layout.addWidget(self.canvas2)

        self.figure3 = Figure(figsize=(10, 8))
        self.canvas3 = FigureCanvas(self.figure3)
        self.tab3.layout.addWidget(self.canvas3)

        self.figure4 = Figure(figsize=(10, 8))
        self.canvas4 = FigureCanvas(self.figure4)
        self.tab4.layout.addWidget(self.canvas4)

        # Add to right layout
        right_layout.addWidget(self.result_group)
        right_layout.addWidget(self.tab_widget)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def load_model(self):
        try:
            self.model = joblib.load('logistic_regression_model.joblib')
            self.feature_names = self.get_feature_names()
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Model file not found. Please train the model first.")

    def load_training_data(self):
        """Load and clean training data for visualization"""
        try:
            df = pd.read_csv('processed.cleveland.csv')
            df['target'] = (df['num'] > 0).astype(int)
            df = df.drop('num', axis=1)

            # Separate features and target
            self.X_train = df.drop('target', axis=1)
            self.y_train = df['target']

            # Clean data for PCA - handle missing values
            imputer = SimpleImputer(strategy='median')
            self.X_train_clean = imputer.fit_transform(self.X_train)

        except Exception as e:
            print(f"Error loading training data: {e}")
            self.X_train = None
            self.y_train = None
            self.X_train_clean = None

    def get_feature_names(self):
        """Extract feature names from the preprocessor"""
        try:
            # Extract numeric features
            numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

            # Extract categorical features (after one-hot encoding)
            categorical_features = []
            cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            for feature in cat_features:
                if feature == 'sex':
                    categorical_features.extend(['sex_0', 'sex_1'])
                elif feature == 'cp':
                    categorical_features.extend(['cp_1', 'cp_2', 'cp_3', 'cp_4'])
                elif feature == 'fbs':
                    categorical_features.extend(['fbs_0', 'fbs_1'])
                elif feature == 'restecg':
                    categorical_features.extend(['restecg_0', 'restecg_1', 'restecg_2'])
                elif feature == 'exang':
                    categorical_features.extend(['exang_0', 'exang_1'])
                elif feature == 'slope':
                    categorical_features.extend(['slope_1', 'slope_2', 'slope_3'])
                elif feature == 'ca':
                    categorical_features.extend(['ca_0', 'ca_1', 'ca_2', 'ca_3'])
                elif feature == 'thal':
                    categorical_features.extend(['thal_3', 'thal_6', 'thal_7'])

            return numeric_features + categorical_features
        except Exception as e:
            print(f"Error extracting feature names: {e}")
            # Fallback to default feature names
            return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                    'sex_0', 'sex_1', 'cp_1', 'cp_2', 'cp_3', 'cp_4',
                    'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2',
                    'exang_0', 'exang_1', 'slope_1', 'slope_2', 'slope_3',
                    'ca_0', 'ca_1', 'ca_2', 'ca_3', 'thal_3', 'thal_6', 'thal_7']

    def predict(self):
        # Collect input values
        input_data = {}
        for field_name, field in self.input_fields.items():
            if isinstance(field, QComboBox):
                input_data[field_name] = float(field.currentText())
            else:
                try:
                    input_data[field_name] = float(field.text())
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Please enter a valid value for {field_name}")
                    return

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        try:
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0]

            self.current_prediction = {
                'data': input_df,
                'prediction': prediction,
                'probability': probability
            }

            # Update result display
            if prediction == 1:
                result_text = "HeartDisease Detected!"
                color = "#d32f2f"
                risk_level = f"Risk Level: {probability[1] * 100:.2f}%"
            else:
                result_text = "No HeartDisease Detected"
                color = "#388e3c"
                risk_level = f"Risk Level: {probability[0] * 100:.2f}%"

            result_text += f"\n{risk_level}"
            self.result_label.setText(result_text)
            self.result_label.setStyleSheet(f"""
                QLabel {{ 
                    color: {color}; 
                    font-size: 18pt; 
                    font-weight: bold; 
                    padding: 15px;
                    border: 2px solid {color};
                    border-radius: 10px;
                    background-color: #f5f5f5;
                }}
            """)

            # Show results and charts
            self.result_group.setVisible(True)
            self.tab_widget.setVisible(True)

            # Update charts
            self.update_charts()

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")

    def update_charts(self):
        # Clear previous charts
        self.figure1.clear()
        self.figure2.clear()
        self.figure3.clear()
        self.figure4.clear()

        # Chart 1: Feature Importance
        ax1 = self.figure1.add_subplot(111)
        self.plot_feature_importance(ax1)
        self.figure1.tight_layout()

        # Chart 2: PCA Visualization
        ax2 = self.figure2.add_subplot(111)
        if self.X_train_clean is not None and self.y_train is not None:
            self.plot_pca_visualization(ax2)
        else:
            ax2.text(0.5, 0.5, 'Training data not available for PCA visualization',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        self.figure2.tight_layout()

        # Chart 3: Risk Factors Radar Chart
        ax3 = self.figure3.add_subplot(111, polar=True)
        self.plot_risk_radar(ax3)
        self.figure3.tight_layout()

        # Chart 4: Probability Distribution
        ax4 = self.figure4.add_subplot(111)
        self.plot_probability_distribution(ax4)
        self.figure4.tight_layout()

        # Update canvases
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

    def plot_feature_importance(self, ax):
        """Plot feature importance from the model"""
        try:
            if hasattr(self.model.named_steps['classifier'], 'coef_'):
                # For Logistic Regression
                importance = np.abs(self.model.named_steps['classifier'].coef_[0])

                # Ensure we have the right number of feature names
                if len(importance) > len(self.feature_names):
                    importance = importance[:len(self.feature_names)]
                elif len(importance) < len(self.feature_names):
                    self.feature_names = self.feature_names[:len(importance)]

                # Create a DataFrame for easier plotting
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True).tail(8)  # Top 8 features

                # Create a beautiful horizontal bar chart
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
                bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)

                # Customize the chart
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df['Feature'], fontsize=10)
                ax.set_xlabel('Importance (Absolute Coefficient Value)', fontsize=12)
                ax.set_title('Top 8 Important Features for HeartDisease Prediction\n', fontsize=14, fontweight='bold')

                # Add value labels
                for i, (bar, importance_val) in enumerate(zip(bars, importance_df['Importance'])):
                    width = bar.get_width()
                    ax.text(width + 0.001, i, f'{importance_val:.3f}',
                            va='center', fontsize=9, fontweight='bold')

                # Add grid
                ax.grid(axis='x', linestyle='--', alpha=0.7)

                # Remove spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            else:
                ax.text(0.5, 0.5, 'Feature importance not available for this model',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting feature importance: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def plot_pca_visualization(self, ax):
        """Plot PCA visualization with the current prediction point"""
        try:
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X_train_clean)

            # Plot training data with beautiful styling
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train,
                                 cmap='RdYlBu_r', alpha=0.7, s=60, edgecolors='w', linewidth=0.5)

            # Transform the current input using the same PCA
            # First, we need to handle missing values in the input
            imputer = SimpleImputer(strategy='median')
            input_processed = imputer.fit_transform(self.current_prediction['data'])
            input_pca = pca.transform(input_processed)

            # Plot current prediction with a star marker
            marker_color = 'green' if self.current_prediction['prediction'] == 0 else 'red'
            ax.scatter(input_pca[:, 0], input_pca[:, 1],
                       c=marker_color, s=400, marker='*', edgecolors='black', linewidth=2,
                       label='Current Prediction')

            # Customize the plot
            ax.set_xlabel('First Principal Component', fontsize=12)
            ax.set_ylabel('Second Principal Component', fontsize=12)
            ax.set_title('PCA Visualization of HeartDisease Data\n(★ marks the current prediction)',
                         fontsize=14, fontweight='bold')

            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                           markersize=10, label='HeartDisease', markeredgecolor='w'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                           markersize=10, label='No HeartDisease', markeredgecolor='w'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                           markersize=15, label='Current Prediction (No Disease)', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                           markersize=15, label='Current Prediction (Disease)', markeredgecolor='black')
            ]
            ax.legend(handles=legend_elements, loc='best')

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating PCA visualization: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def plot_risk_radar(self, ax):
        """Plot a radar chart of risk factors"""
        try:
            # Select key features for radar chart
            key_features = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak']
            feature_labels = ['Age', 'Cholesterol', 'Blood Pressure', 'Max Heart Rate', 'ST Depression']

            # Get values for current patient
            patient_values = [self.current_prediction['data'][feat].values[0] for feat in key_features]

            # Normalize values (0-1 scale)
            max_values = [77, 564, 200, 202, 6.2]  # Max values from data description
            normalized_values = [val / max_val for val, max_val in zip(patient_values, max_values)]

            # Complete the circle
            normalized_values += normalized_values[:1]
            angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
            angles += angles[:1]

            # Plot radar chart with beautiful styling
            ax.plot(angles, normalized_values, 'o-', linewidth=2,
                    label='Current Patient', color='#FF6B6B', markersize=8)
            ax.fill(angles, normalized_values, alpha=0.25, color='#FF6B6B')

            # Add feature labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=11)

            # Set yticks
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add title
            ax.set_title('Risk Factors Radar Chart\n(Normalized to 0-100% scale)',
                         fontsize=14, fontweight='bold', pad=20)

            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating radar chart: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def plot_probability_distribution(self, ax):
        """Plot probability distribution for the current prediction"""
        try:
            # Get probabilities
            probs = self.current_prediction['probability']
            classes = ['No HeartDisease', 'HeartDisease']
            colors = ['#4CAF50', '#F44336']

            # Create a beautiful bar chart
            bars = ax.bar(classes, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Customize the chart
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title('Prediction Probability Distribution\n', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add a horizontal line at 0.5 for reference
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            ax.text(1.5, 0.52, 'Decision Boundary (0.5)', ha='center', va='bottom', fontsize=10)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting probability: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def close_and_go_back(self):
        self.close()


def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = HeartDiseaseApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()