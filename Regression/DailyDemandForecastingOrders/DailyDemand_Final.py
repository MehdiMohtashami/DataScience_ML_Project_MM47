import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QMessageBox, QGroupBox, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator, QDoubleValidator, QFont


class RegressionApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DailyDemandForecastingOrders")

        # Load model and data
        self.model = joblib.load('best_regression_model.joblib')
        self.scaler = joblib.load('scaler.joblib')
        self.data = pd.read_csv('Daily_Demand_Forecasting_Orders.csv')

        # Prepare data
        self.X = self.data.drop('Target (Total orders)', axis=1)
        self.y = self.data['Target (Total orders)']
        self.X_scaled = self.scaler.transform(self.X)
        self.predictions = self.model.predict(self.X_scaled)

        # Calculate feature importance
        self.feature_importance = self.calculate_feature_importance()

        # Initialize UI
        self.init_ui()

        # Flags
        self.plots_visible = False

    def init_ui(self):
        self.setWindowTitle('Advanced Demand Forecasting')
        self.setGeometry(300, 300, 1400, 900)

        # Create main widgets
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QVBoxLayout(self.central_widget)

        # Model info header
        self.model_info = QLabel()
        self.calculate_accuracy()
        self.model_info.setAlignment(Qt.AlignCenter)
        self.model_info.setStyleSheet(
            "background-color: #2c3e50; color: white; font-size: 16px; "
            "font-weight: bold; padding: 10px; border-radius: 5px;"
        )
        main_layout.addWidget(self.model_info)

        # Create splitter for input and visualization
        self.splitter = QSplitter(Qt.Horizontal)

        # Input Panel
        self.input_panel = QWidget()
        self.create_input_panel()

        # Visualization Panel
        self.viz_panel = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_panel)

        self.splitter.addWidget(self.input_panel)
        self.splitter.addWidget(self.viz_panel)
        self.splitter.setSizes([400, 1000])

        main_layout.addWidget(self.splitter)

        # Initially hide visualization
        self.viz_panel.setVisible(False)

    def create_input_panel(self):
        layout = QVBoxLayout(self.input_panel)

        # Title
        title = QLabel("Enter Feature Values")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2980b9; "
            "margin-bottom: 20px;"
        )
        layout.addWidget(title)

        # Create input fields with examples
        self.input_fields = {}
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)

        # Feature examples from dataset
        examples = {
            'Week of the month': "1-5 (e.g., 3)",
            'Day of the week': "2-6 (e.g., 4)",
            'Non-urgent order': "e.g., 150.0",
            'Urgent order': "e.g., 130.0",
            'Order type A': "e.g., 45.0",
            'Order type B': "e.g., 115.0",
            'Order type C': "e.g., 150.0",
            'Fiscal sector orders': "e.g., 10.0",
            'Orders from the traffic controller sector': "e.g., 40000",
            'Banking orders (1)': "e.g., 30000",
            'Banking orders (2)': "e.g., 90000",
            'Banking orders (3)': "e.g., 20000"
        }

        for feature, example in examples.items():
            container = QWidget()
            hbox = QHBoxLayout(container)
            hbox.setContentsMargins(0, 0, 0, 0)

            label = QLabel(feature)
            label.setStyleSheet("font-weight: bold;")
            label.setFixedWidth(200)

            line_edit = QLineEdit()
            line_edit.setValidator(QDoubleValidator())
            line_edit.setPlaceholderText(example)
            line_edit.setStyleSheet("padding: 5px;")
            line_edit.setMinimumHeight(30)

            self.input_fields[feature] = line_edit

            hbox.addWidget(label)
            hbox.addWidget(line_edit)
            form_layout.addRow(container)

        # Wrap form in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form_widget.setLayout(form_layout)
        scroll.setWidget(form_widget)

        # Prediction button
        self.predict_btn = QPushButton("PREDICT DEMAND")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #27ae60; 
                color: white; 
                font-weight: bold; 
                font-size: 16px;
                height: 50px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            """
        )


        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)
        self.back_button.setStyleSheet(
            """
            QPushButton {
                background-color: gray; 
                color: white; 
                font-weight: bold; 
                font-size: 16px;
                height: 50px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: gray;
            }
            """
        )
        # Result display
        self.result_label = QLabel("Prediction will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e74c3c; "
            "margin: 20px; padding: 15px; border: 2px dashed #3498db; "
            "border-radius: 10px; background-color: #f8f9fa;"
        )

        layout.addWidget(scroll)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.back_button)
        layout.addWidget(self.result_label)
        layout.addStretch()

    def calculate_accuracy(self):
        """Calculate R² score for the model"""
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y, self.predictions)
        self.model_info.setText(
            f"Model: Linear Regression | Accuracy (R² Score): {r2:.4f} | "
            f"Dataset Size: {len(self.data)} records | Features: {len(self.X.columns)}"
        )
        return r2

    def calculate_feature_importance(self):
        """Calculate and sort feature importance"""
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            features = self.X.columns
            importance = pd.Series(coefficients, index=features)
            return importance.sort_values(key=abs, ascending=False)
        return None

    def create_visualizations(self):
        """Create all visualizations"""
        # Clear previous content
        for i in reversed(range(self.viz_layout.count())):
            widget = self.viz_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Create tab widget for analysis sections
        analysis_tabs = QTabWidget()

        # Tab 1: Prediction Analysis
        pred_tab = QWidget()
        pred_layout = QVBoxLayout(pred_tab)

        # Actual vs Predicted plot
        self.figure1 = plt.figure(figsize=(10, 6))
        self.canvas1 = FigureCanvas(self.figure1)
        ax1 = self.figure1.add_subplot(111)

        ax1.scatter(self.y, self.predictions, alpha=0.7,
                    edgecolor='k', s=80, label='Historical Data')
        ax1.plot([self.y.min(), self.y.max()],
                 [self.y.min(), self.y.max()], 'r--', lw=2, label='Ideal Prediction')

        # Add user prediction
        if hasattr(self, 'user_prediction'):
            ax1.scatter(self.user_prediction, self.user_prediction,
                        s=250, c='gold', edgecolors='k', marker='*',
                        label='Your Prediction')

        ax1.set_xlabel('Actual Demand', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Demand', fontsize=12, fontweight='bold')
        ax1.set_title('Actual vs Predicted Demand', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        self.canvas1.draw()

        pred_layout.addWidget(self.canvas1)

        # Tab 2: Feature Analysis
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)

        # Feature Importance plot
        self.figure2 = plt.figure(figsize=(10, 5))
        self.canvas2 = FigureCanvas(self.figure2)
        ax2 = self.figure2.add_subplot(111)

        if self.feature_importance is not None:
            # Get top 5 features
            top_features = self.feature_importance.head(5)

            colors = ['#3498db' if coef > 0 else '#e74c3c' for coef in top_features.values]

            ax2.barh(top_features.index, top_features.values, color=colors)
            ax2.set_title('Top 5 Influential Features', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Coefficient Value', fontsize=10, fontweight='bold')
            ax2.grid(axis='x', linestyle='--', alpha=0.7)
            ax2.axvline(0, color='k', linestyle='-', alpha=0.5)

        self.canvas2.draw()
        feature_layout.addWidget(self.canvas2)

        # Feature Relationship plot
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_feature = self.feature_importance.index[0]

            self.figure3 = plt.figure(figsize=(10, 5))
            self.canvas3 = FigureCanvas(self.figure3)
            ax3 = self.figure3.add_subplot(111)

            ax3.scatter(self.X[top_feature], self.y, alpha=0.7,
                        edgecolor='k', s=80, label='Historical Data')

            # Add user input if available
            if hasattr(self, 'user_input') and top_feature in self.user_input:
                user_x = self.user_input[top_feature]
                ax3.scatter(user_x, self.user_prediction,
                            s=250, c='gold', edgecolors='k', marker='*',
                            label='Your Input')

                # Add annotation
                ax3.annotate(f'Predicted: {self.user_prediction:.1f}',
                             (user_x, self.user_prediction),
                             textcoords="offset points",
                             xytext=(10, -15),
                             ha='center',
                             fontsize=10,
                             arrowprops=dict(arrowstyle="->"))

            ax3.set_xlabel(top_feature, fontsize=10, fontweight='bold')
            ax3.set_ylabel('Total Orders', fontsize=10, fontweight='bold')
            ax3.set_title(f'Relationship: {top_feature} vs Total Orders',
                          fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            self.canvas3.draw()
            feature_layout.addWidget(self.canvas3)

        # Add tabs
        analysis_tabs.addTab(pred_tab, "Prediction Analysis")
        analysis_tabs.addTab(feature_tab, "Feature Analysis")

        # Add to visualization layout
        self.viz_layout.addWidget(analysis_tabs)

        # Add interpretation notes
        notes = QLabel(
            "Interpretation Guide:\n"
            "- Blue bars indicate features that increase demand\n"
            "- Red bars indicate features that decrease demand\n"
            "- The golden star shows your prediction compared to historical data"
        )
        notes.setStyleSheet(
            "font-style: italic; color: #7f8c8d; "
            "background-color: #f8f9fa; padding: 10px; "
            "border-radius: 5px; margin-top: 10px;"
        )
        self.viz_layout.addWidget(notes)

    def predict(self):
        """Predict demand based on user input"""
        try:
            # Get input values
            input_values = []
            user_input = {}
            for feature, field in self.input_fields.items():
                value = field.text().strip()
                if not value:
                    QMessageBox.warning(self, "Input Error", f"Please enter a value for {feature}")
                    return
                input_values.append(float(value))
                user_input[feature] = float(value)

            # Store user input
            self.user_input = user_input

            # Scale input and predict
            input_array = np.array(input_values).reshape(1, -1)
            scaled_input = self.scaler.transform(input_array)
            prediction = self.model.predict(scaled_input)[0]

            # Display result
            self.result_label.setText(f"PREDICTED TOTAL DEMAND: {prediction:.2f} orders")

            # Store for visualization
            self.user_prediction = prediction

            # Show and update visualizations
            self.viz_panel.setVisible(True)
            self.create_visualizations()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def closeEvent(self, event):
        """Clean up on application close"""
        plt.close('all')
        event.accept()

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 10, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = RegressionApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == "__main__":
    main()