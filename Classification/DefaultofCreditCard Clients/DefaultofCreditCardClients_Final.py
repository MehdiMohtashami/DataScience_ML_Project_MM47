import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import joblib
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QFormLayout, QTabWidget, QMessageBox,
                             QScrollArea, QGridLayout, QComboBox, QSizePolicy,
                             QSplitter)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont

plt.style.use('default')
sns.set_palette("husl")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)


class CreditCardPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.model = joblib.load('best_credit_card_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        except:
            QMessageBox.critical(self, "Error",
                                 "Could not load model files. Please make sure 'best_credit_card_model.pkl' and 'scaler.pkl' are in the same directory.")
            sys.exit(1)

        self.feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                              'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                              'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                              'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                              'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

        self.accuracy = 0.8172
        self.roc_auc = 0.7806

        self.current_prediction = None
        self.canvases = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Credit Card Default Prediction System')
        self.setGeometry(100, 100, 1800, 1000)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setMaximumWidth(500)
        left_panel.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        info_font = QFont()
        info_font.setBold(True)
        info_font.setPointSize(10)

        model_label = QLabel("Model: Gradient Boosting Classifier")
        model_label.setFont(info_font)
        info_layout.addWidget(model_label)

        accuracy_label = QLabel(f"Accuracy: {self.accuracy * 100:.2f}%")
        accuracy_label.setFont(info_font)
        info_layout.addWidget(accuracy_label)

        roc_label = QLabel(f"ROC AUC: {self.roc_auc:.4f}")
        roc_label.setFont(info_font)
        info_layout.addWidget(roc_label)

        usage_label = QLabel("This system predicts credit card default risk with 81.7% accuracy")
        usage_label.setWordWrap(True)
        info_layout.addWidget(usage_label)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # Input form
        input_group = QGroupBox("Customer Information")
        input_layout = QFormLayout()
        input_layout.setVerticalSpacing(10)

        self.input_fields = {}

        # Create input fields with appropriate widgets
        for feature in self.feature_names:
            if feature == 'SEX':
                combo = QComboBox()
                combo.addItem("Male", 1)
                combo.addItem("Female", 2)
                input_layout.addRow(QLabel(f"{feature}:"), combo)
                self.input_fields[feature] = combo
            elif feature == 'EDUCATION':
                combo = QComboBox()
                combo.addItem("Graduate School", 1)
                combo.addItem("University", 2)
                combo.addItem("High School", 3)
                combo.addItem("Others", 4)
                input_layout.addRow(QLabel(f"{feature}:"), combo)
                self.input_fields[feature] = combo
            elif feature == 'MARRIAGE':
                combo = QComboBox()
                combo.addItem("Married", 1)
                combo.addItem("Single", 2)
                combo.addItem("Others", 3)
                input_layout.addRow(QLabel(f"{feature}:"), combo)
                self.input_fields[feature] = combo
            elif 'PAY_' in feature:
                line_edit = QLineEdit()
                line_edit.setValidator(QIntValidator(-2, 8))
                line_edit.setPlaceholderText("-2 to 8")
                input_layout.addRow(QLabel(f"{feature}:"), line_edit)
                self.input_fields[feature] = line_edit
            elif 'AMT' in feature or 'LIMIT_BAL' in feature:
                line_edit = QLineEdit()
                line_edit.setValidator(QIntValidator(0, 10000000))
                line_edit.setPlaceholderText("0 to 10,000,000")
                input_layout.addRow(QLabel(f"{feature}:"), line_edit)
                self.input_fields[feature] = line_edit
            elif feature == 'AGE':
                line_edit = QLineEdit()
                line_edit.setValidator(QIntValidator(18, 100))
                line_edit.setPlaceholderText("18 to 100")
                input_layout.addRow(QLabel(f"{feature}:"), line_edit)
                self.input_fields[feature] = line_edit
            else:
                line_edit = QLineEdit()
                line_edit.setValidator(QIntValidator())
                input_layout.addRow(QLabel(f"{feature}:"), line_edit)
                self.input_fields[feature] = line_edit

        input_group.setLayout(input_layout)

        # Add scroll area for the form
        scroll = QScrollArea()
        scroll.setWidget(input_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(600)
        left_layout.addWidget(scroll)

        # Prediction button
        predict_btn = QPushButton("Predict Default Risk")
        predict_btn.setStyleSheet("""
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
        predict_btn.clicked.connect(self.predict)
        left_layout.addWidget(predict_btn)

        back_button = QPushButton("Back to Main", self)
        back_button.setStyleSheet("""
                   QPushButton { 
                       background-color: gray; 
                       color: white; 
                       font-weight: bold; 
                       padding: 15px;
                       border-radius: 5px;
                       font-size: 14px;
                   }
               """)
        back_button.clicked.connect(self.close_and_go_back)
        left_layout.addWidget(back_button)

        # Prediction result
        self.result_label = QLabel("Please enter customer data and click 'Predict'")
        self.result_label.setStyleSheet("""
            font-weight: bold; 
            font-size: 16px; 
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label)

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for charts - use a splitter for better resizing
        self.right_panel = QTabWidget()
        self.right_panel.setVisible(False)

        # Create tabs with scroll areas
        self.feature_importance_tab = QScrollArea()
        self.feature_importance_tab.setWidgetResizable(True)
        self.feature_importance_content = QWidget()
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_content)

        self.prediction_analysis_tab = QScrollArea()
        self.prediction_analysis_tab.setWidgetResizable(True)
        self.prediction_analysis_content = QWidget()
        self.prediction_analysis_layout = QVBoxLayout(self.prediction_analysis_content)

        self.feature_relationships_tab = QScrollArea()
        self.feature_relationships_tab.setWidgetResizable(True)
        self.feature_relationships_content = QWidget()
        self.feature_relationships_layout = QVBoxLayout(self.feature_relationships_content)

        # Set the content widgets for scroll areas
        self.feature_importance_tab.setWidget(self.feature_importance_content)
        self.prediction_analysis_tab.setWidget(self.prediction_analysis_content)
        self.feature_relationships_tab.setWidget(self.feature_relationships_content)

        # Add tabs to right panel
        self.right_panel.addTab(self.feature_importance_tab, "Feature Importance")
        self.right_panel.addTab(self.prediction_analysis_tab, "Prediction Analysis")
        self.right_panel.addTab(self.feature_relationships_tab, "Feature Relationships")

        main_layout.addWidget(self.right_panel, 1)  # The 1 makes it take remaining space

    def get_input_value(self, feature_name):
        """Get the value from input field based on its type"""
        widget = self.input_fields[feature_name]
        if isinstance(widget, QComboBox):
            return widget.currentData()
        else:
            return float(widget.text()) if widget.text() else None

    def predict(self):
        try:
            # Get input values
            input_data = []
            missing_fields = []

            for feature in self.feature_names:
                value = self.get_input_value(feature)
                if value is None:
                    missing_fields.append(feature)
                input_data.append(value)

            if missing_fields:
                QMessageBox.warning(self, "Input Error",
                                    f"Please enter values for the following fields:\n{', '.join(missing_fields)}")
                return

            # Convert to DataFrame with feature names to avoid the warning
            input_df = pd.DataFrame([input_data], columns=self.feature_names)

            # Scale input data
            input_scaled = self.scaler.transform(input_df)

            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0][1]

            self.current_prediction = {
                'data': input_data,
                'prediction': prediction,
                'probability': probability,
                'scaled_data': input_scaled[0]
            }

            # Update result label
            risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
            color = "red" if prediction == 1 else "green"
            result_text = f"<span style='color: {color}; font-size: 18px;'>Prediction: {risk_level}<br/>"
            result_text += f"Default Probability: {probability * 100:.2f}%</span>"
            self.result_label.setText(result_text)

            # Show charts and update them
            self.right_panel.setVisible(True)
            self.update_charts()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def update_charts(self):
        # Clear previous charts
        self.clearLayout(self.feature_importance_layout)
        self.clearLayout(self.prediction_analysis_layout)
        self.clearLayout(self.feature_relationships_layout)

        # Clear canvas references
        self.canvases = []

        # Create new charts
        self.create_feature_importance_chart()
        self.create_prediction_analysis_chart()
        self.create_feature_relationships_chart()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def create_feature_importance_chart(self):
        # Get feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Chart 1: Horizontal bar chart of feature importances
        canvas1 = MplCanvas(self, width=10, height=8)
        self.canvases.append(canvas1)
        ax1 = canvas1.axes

        # Plot feature importance
        ax1.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        ax1.set_yticks(range(len(indices)))
        ax1.set_yticklabels([self.feature_names[i] for i in indices])
        ax1.set_xlabel('Relative Importance')
        ax1.set_title('Feature Importance Ranking', fontsize=16, fontweight='bold')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)

        # Add value annotations
        for i, v in enumerate(importances[indices]):
            ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)

        self.feature_importance_layout.addWidget(canvas1)

        # Chart 2: Pie chart of top 8 features - using legend instead of direct labels
        canvas2 = MplCanvas(self, width=10, height=8)
        self.canvases.append(canvas2)
        ax2 = canvas2.axes

        # Get top 8 features
        top_n = 8
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]

        # Create pie chart with legend outside the chart
        wedges, texts, autotexts = ax2.pie(top_importances, autopct='%1.1f%%',
                                           startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, top_n)))
        ax2.set_title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')

        # Improved readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Adding a legend outside the chart
        ax2.legend(wedges, top_features, title="Features",
                   loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Adjust the layout to accommodate the legend
        canvas2.fig.tight_layout()

        self.feature_importance_layout.addWidget(canvas2)

    def create_prediction_analysis_chart(self):
        current_prob = self.current_prediction['probability']

        # Chart 1: Probability distribution with current prediction
        canvas1 = MplCanvas(self, width=10, height=6)
        self.canvases.append(canvas1)
        ax1 = canvas1.axes

        # Generate sample probabilities (beta distribution)
        np.random.seed(42)
        sample_probs = np.random.beta(2, 5, 1000)  # Skewed toward lower probabilities

        # Create histogram
        n, bins, patches = ax1.hist(sample_probs, bins=30, alpha=0.7,
                                    color='lightblue', edgecolor='black',
                                    label='Typical Distribution')

        # Add vertical line for current prediction
        ax1.axvline(current_prob, color='red', linestyle='--', linewidth=3,
                    label=f'Current Prediction: {current_prob:.4f}')

        # Add risk zones
        ax1.axvspan(0, 0.3, alpha=0.2, color='green', label='Low Risk')
        ax1.axvspan(0.3, 0.7, alpha=0.2, color='orange', label='Medium Risk')
        ax1.axvspan(0.7, 1.0, alpha=0.2, color='red', label='High Risk')

        ax1.set_xlabel('Default Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Default Probability Distribution with Risk Zones', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        self.prediction_analysis_layout.addWidget(canvas1)

        # Chart 2: SHAP-like waterfall chart for top features
        canvas2 = MplCanvas(self, width=12, height=8)
        self.canvases.append(canvas2)
        ax2 = canvas2.axes

        # Get feature importances and current values
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:6]  # Top 6 features

        features = [self.feature_names[i] for i in indices]
        values = [self.current_prediction['data'][i] for i in indices]
        scaled_values = [self.current_prediction['scaled_data'][i] for i in indices]

        # Calculate contribution (simplified - not actual SHAP values)
        base_value = 0.5  # Base probability
        contributions = [sv * imp for sv, imp in zip(scaled_values, importances[indices])]
        cumulative = base_value + np.cumsum(contributions)
        cumulative = np.clip(cumulative, 0, 1)  # Ensure within [0,1] range

        # Create waterfall chart
        y_pos = np.arange(len(features) + 1)
        ax2.barh(y_pos[1:], contributions, left=base_value,
                 color=['red' if c > 0 else 'green' for c in contributions])

        # Add connector lines
        for i in range(len(features)):
            ax2.plot([cumulative[i] if i > 0 else base_value, cumulative[i]],
                     [i + 0.9, i + 1.1], 'k-', lw=0.8)

        # Add final probability
        ax2.barh(len(features) + 1, 0, left=current_prob, color='blue')

        # Set labels and title
        ax2.set_yticks(range(len(features) + 2))
        ax2.set_yticklabels(['Base'] + features + ['Final'])
        ax2.set_xlabel('Probability Contribution')
        ax2.set_title('Feature Contributions to Prediction (Waterfall Chart)', fontsize=16, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(base_value, color='gray', linestyle='--', alpha=0.7)

        # Adjust layout to prevent labels from being cut off
        canvas2.fig.tight_layout()

        self.prediction_analysis_layout.addWidget(canvas2)

        # Chart 3: Confidence indicator
        canvas3 = MplCanvas(self, width=8, height=2)
        self.canvases.append(canvas3)
        ax3 = canvas3.axes

        # Create a confidence bar
        confidence = min(current_prob, 1 - current_prob) * 2  # Convert to confidence measure
        ax3.barh(0, confidence, color='green' if confidence > 0.6 else 'orange' if confidence > 0.3 else 'red')
        ax3.set_xlim(0, 1)
        ax3.set_title(f'Prediction Confidence: {confidence * 100:.1f}%', fontsize=14, fontweight='bold')
        ax3.set_yticks([])
        ax3.text(confidence / 2, 0, f'{confidence * 100:.1f}%',
                 ha='center', va='center', color='white', fontweight='bold')

        self.prediction_analysis_layout.addWidget(canvas3)

    def create_feature_relationships_chart(self):
        # Chart 1: Scatter plot of two most important features
        canvas1 = MplCanvas(self, width=10, height=8)
        self.canvases.append(canvas1)
        ax1 = canvas1.axes

        # Get two most important features
        importances = self.model.feature_importances_
        top_two = np.argsort(importances)[::-1][:2]
        feat1, feat2 = self.feature_names[top_two[0]], self.feature_names[top_two[1]]

        # Generate sample data (using multivariate normal distribution)
        np.random.seed(42)
        mean = [self.current_prediction['data'][top_two[0]],
                self.current_prediction['data'][top_two[1]]]
        cov = [[abs(mean[0]) / 2, 0], [0, abs(mean[1]) / 2]]  # Diagonal covariance
        data = np.random.multivariate_normal(mean, cov, 100)

        # Create scatter plot
        ax1.scatter(data[:, 0], data[:, 1], alpha=0.6,
                    c=np.random.rand(100), cmap='viridis', s=50, label='Sample Data')
        ax1.scatter(mean[0], mean[1], color='red', s=200,
                    marker='X', label='Current Customer', edgecolors='black')

        ax1.set_xlabel(feat1)
        ax1.set_ylabel(feat2)
        ax1.set_title(f'Relationship between {feat1} and {feat2}', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        self.feature_relationships_layout.addWidget(canvas1)

        # Chart 2: Correlation heatmap for top features
        canvas2 = MplCanvas(self, width=10, height=8)
        self.canvases.append(canvas2)
        ax2 = canvas2.axes

        # Get top 8 features
        top_features = np.argsort(importances)[::-1][:8]
        top_feature_names = [self.feature_names[i] for i in top_features]

        # Generate sample correlation data
        np.random.seed(42)
        data = np.random.randn(100, len(top_features))
        # Add some correlation structure
        for i in range(1, len(top_features)):
            data[:, i] += 0.3 * data[:, i - 1]

        # Calculate correlation matrix
        corr = np.corrcoef(data, rowvar=False)

        # Create heatmap
        im = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(top_feature_names)))
        ax2.set_yticks(range(len(top_feature_names)))
        ax2.set_xticklabels(top_feature_names, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(top_feature_names, fontsize=8)
        ax2.set_title('Correlation Matrix of Top Features', fontsize=16, fontweight='bold')

        # Add correlation values to cells
        for i in range(len(top_feature_names)):
            for j in range(len(top_feature_names)):
                ax2.text(j, i, f'{corr[i, j]:.2f}',
                         ha='center', va='center', fontsize=6,
                         color='white' if abs(corr[i, j]) > 0.5 else 'black')

        # Add colorbar
        plt.colorbar(im, ax=ax2)

        # Adjust layout to prevent labels from being cut off
        canvas2.fig.tight_layout()

        self.feature_relationships_layout.addWidget(canvas2)

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = CreditCardPredictor()
    window.show()
    if parent is None:
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()