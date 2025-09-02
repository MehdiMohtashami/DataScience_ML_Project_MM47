import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QPushButton, QTabWidget, QGroupBox,
    QDoubleSpinBox, QSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class RealEstateApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RealEstateValuation")
        self.setWindowTitle("Real Estate Price Prediction System")
        self.setGeometry(100, 100, 1400, 900)
        self.first_prediction = True

        # Load model and scaler
        try:
            self.model = joblib.load('real_estate_rf_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 "Model files not found. Please make sure 'real_estate_rf_model.pkl' and 'scaler.pkl' are in the same directory.")
            sys.exit(1)

        # Load dataset for visualization
        self.dataset = self.load_dataset()
        self.X_test = None
        self.y_test = None
        self.test_predictions = None

        # Create UI
        self.init_ui()

    def load_dataset(self):
        """Load sample dataset for visualization purposes"""
        try:
            # Use CSV reader instead of Excel
            return pd.read_csv("Real estate valuation data set.csv")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
            return pd.DataFrame()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Input and model info
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)

        # Model info section
        model_info = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_info)

        model_name = QLabel("Algorithm: Random Forest Regressor")
        model_name.setFont(QFont("Arial", 12, QFont.Bold))

        model_accuracy = QLabel("Model Accuracy (R²): 80.59%")
        model_rmse = QLabel("Prediction Error (RMSE): 5.71")

        model_layout.addWidget(model_name)
        model_layout.addWidget(model_accuracy)
        model_layout.addWidget(model_rmse)

        # Input section
        input_group = QGroupBox("Property Features")
        form_layout = QFormLayout(input_group)

        # Create input fields
        self.transaction_date = QDoubleSpinBox()
        self.transaction_date.setRange(2012, 2014)
        self.transaction_date.setSingleStep(0.001)
        self.transaction_date.setValue(2013.333)
        self.transaction_date.setToolTip("Transaction date (e.g., 2013.500 = June 2013)")

        self.house_age = QDoubleSpinBox()
        self.house_age.setRange(0, 100)
        self.house_age.setValue(10.5)
        self.house_age.setToolTip("Age of the property in years")

        self.distance_to_mrt = QDoubleSpinBox()
        self.distance_to_mrt.setRange(0, 5000)
        self.distance_to_mrt.setValue(200.0)
        self.distance_to_mrt.setToolTip("Distance to nearest metro station in meters")

        self.convenience_stores = QSpinBox()
        self.convenience_stores.setRange(0, 20)
        self.convenience_stores.setValue(5)
        self.convenience_stores.setToolTip("Number of convenience stores in the area")

        self.latitude = QDoubleSpinBox()
        self.latitude.setRange(24.0, 25.5)
        self.latitude.setDecimals(5)
        self.latitude.setValue(24.98000)
        self.latitude.setToolTip("Geographic coordinate - latitude")

        self.longitude = QDoubleSpinBox()
        self.longitude.setRange(121.0, 122.0)
        self.longitude.setDecimals(5)
        self.longitude.setValue(121.54000)
        self.longitude.setToolTip("Geographic coordinate - longitude")

        # Add fields to form
        form_layout.addRow("Transaction Date (X1):", self.transaction_date)
        form_layout.addRow("House Age (years) (X2):", self.house_age)
        form_layout.addRow("Distance to MRT (m) (X3):", self.distance_to_mrt)
        form_layout.addRow("Convenience Stores (X4):", self.convenience_stores)
        form_layout.addRow("Latitude (X5):", self.latitude)
        form_layout.addRow("Longitude (X6):", self.longitude)

        # Prediction button
        self.predict_btn = QPushButton("Predict Price")
        self.predict_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.predict_btn.clicked.connect(self.predict_price)

        # ایجاد یک دکمه برای بازگشت
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        self.back_button.setStyleSheet("background-color: gray; color: white; padding: 10px;")
        self.back_button.clicked.connect(self.predict_price)

        # Result display
        self.result_label = QLabel("Click 'Predict Price' to see results")
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #1976D2; background-color: #E3F2FD; padding: 15px;")


        # Add widgets to input panel
        input_layout.addWidget(model_info)
        input_layout.addWidget(input_group)
        input_layout.addWidget(self.predict_btn)
        input_layout.addWidget(self.back_button)
        input_layout.addWidget(self.result_label)
        input_layout.addStretch()

        # Visualization panel (hidden initially)
        self.viz_panel = QTabWidget()
        self.viz_panel.setVisible(False)

        # Create tabs
        self.tab1 = QWidget()  # Actual vs Predicted
        self.tab2 = QWidget()  # Feature Importance
        self.tab3 = QWidget()  # Model Comparison
        self.tab4 = QWidget()  # Price Distribution
        self.tab5 = QWidget()  # Residual Analysis

        self.viz_panel.addTab(self.tab1, "Actual vs Predicted")
        self.viz_panel.addTab(self.tab2, "Feature Importance")
        self.viz_panel.addTab(self.tab3, "Model Comparison")
        self.viz_panel.addTab(self.tab4, "Price Distribution")
        self.viz_panel.addTab(self.tab5, "Residual Analysis")

        # Add panels to main layout
        main_layout.addWidget(input_panel, 1)
        main_layout.addWidget(self.viz_panel, 2)

    def predict_price(self):
        """Predict price based on user input"""
        # Collect input data
        input_data = [
            self.transaction_date.value(),
            self.house_age.value(),
            self.distance_to_mrt.value(),
            self.convenience_stores.value(),
            self.latitude.value(),
            self.longitude.value()
        ]

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale the input
        input_scaled = self.scaler.transform(input_array)

        # Predict price
        predicted_price = self.model.predict(input_scaled)[0]

        # Display result
        self.result_label.setText(
            f"Predicted Price: {predicted_price:.2f} (10,000 TWD/Ping)\n"
            f"Approx. {predicted_price * 33000:.0f} New Taiwan Dollars"
        )

        # Generate visualizations on first prediction
        if self.first_prediction:
            self.first_prediction = False
            self.generate_visualizations()
            self.viz_panel.setVisible(True)

        # Highlight user's prediction on charts
        self.highlight_prediction(predicted_price)

    def generate_visualizations(self):
        """Create all visualizations"""
        # Prepare test data for visualizations
        if self.dataset.empty:
            QMessageBox.warning(self, "Warning", "Dataset is empty. Using sample data for visualizations.")
            return

        X = self.dataset.drop('Y house price of unit area', axis=1)
        y = self.dataset['Y house price of unit area']

        # Split data
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        # Make predictions
        self.test_predictions = self.model.predict(self.X_test_scaled)

        # Create plots
        self.create_actual_vs_predicted_plot()
        self.create_feature_importance_plot()
        self.create_model_comparison_plot()
        self.create_price_distribution_plot()
        self.create_residual_plot()

    def highlight_prediction(self, predicted_price):
        """Highlight user's prediction on relevant charts"""
        # Highlight on Actual vs Predicted plot
        if hasattr(self, 'actual_vs_predicted_canvas'):
            self.highlight_on_actual_vs_predicted(predicted_price)

        # Highlight on Price Distribution plot
        if hasattr(self, 'price_distribution_canvas'):
            self.highlight_on_price_distribution(predicted_price)

    def create_figure_canvas(self, figure, tab):
        """Create a figure canvas for the plot"""
        canvas = FigureCanvas(figure)
        layout = QVBoxLayout(tab)
        layout.addWidget(canvas)
        canvas.draw()
        return canvas

    def create_actual_vs_predicted_plot(self):
        """Create Actual vs Predicted plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot test set predictions
        ax.scatter(self.y_test, self.test_predictions, alpha=0.6, color='#2196F3', label='Test Data')
        ax.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', lw=2, label='Perfect Prediction')

        ax.set_title('Actual vs Predicted Prices', fontsize=14)
        ax.set_xlabel('Actual Prices (10,000 TWD/Ping)', fontsize=12)
        ax.set_ylabel('Predicted Prices', fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend()

        self.actual_vs_predicted_canvas = self.create_figure_canvas(fig, self.tab1)

    def highlight_on_actual_vs_predicted(self, predicted_price):
        """Highlight user's prediction on Actual vs Predicted plot"""
        # We don't have the actual price for user input, so we'll show it as a horizontal line
        fig = self.actual_vs_predicted_canvas.figure
        ax = fig.axes[0]

        # Clear any previous highlights
        for artist in ax.lines[1:]:
            artist.remove()
        for artist in ax.collections[1:]:
            artist.remove()

        # Add horizontal line for prediction
        ax.axhline(y=predicted_price, color='g', linestyle='-', lw=2,
                   label='Your Prediction', alpha=0.7)

        # Add label
        ax.text(ax.get_xlim()[1] * 0.95, predicted_price,
                f'Your Prediction: {predicted_price:.2f}',
                verticalalignment='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))

        ax.legend()
        fig.canvas.draw()

    def create_feature_importance_plot(self):
        """Create Feature Importance plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        features = [
            'Distance to MRT (X3)',
            'House Age (X2)',
            'Latitude (X5)',
            'Longitude (X6)',
            'Transaction Date (X1)',
            'Convenience Stores (X4)'
        ]

        importance = [0.5653, 0.1694, 0.1251, 0.0705, 0.0429, 0.0266]

        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, align='center', color='teal')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # top to bottom
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance in Price Prediction', fontsize=14)
        ax.grid(axis='x', alpha=0.3)

        # Add percentage labels
        total = sum(importance)
        for i, v in enumerate(importance):
            percentage = f'{(v / total) * 100:.1f}%'
            ax.text(v + 0.01, i, percentage, va='center')

        self.create_figure_canvas(fig, self.tab2)

    def create_model_comparison_plot(self):
        """Create Model Comparison plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Lasso', 'Ridge', 'Linear', 'Decision Tree']
        r2_scores = [0.8059, 0.7964, 0.7688, 0.6838, 0.6814, 0.6811, 0.6038]

        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.barh(models, r2_scores, color=colors)
        ax.set_title('Regression Model Performance Comparison', fontsize=14)
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_xlim(0, 0.9)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.4f}',
                    ha='left', va='center')

        self.create_figure_canvas(fig, self.tab3)

    def create_price_distribution_plot(self):
        """Create Price Distribution plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram of all prices in dataset
        prices = self.dataset['Y house price of unit area']
        ax.hist(prices, bins=20, color='#9C27B0', alpha=0.7, edgecolor='black',
                label='Market Prices')

        # Add mean line
        mean_price = np.mean(prices)
        ax.axvline(mean_price, color='r', linestyle='--', lw=2,
                   label=f'Market Mean: {mean_price:.2f}')

        ax.set_title('Property Price Distribution', fontsize=14)
        ax.set_xlabel('Price (10,000 TWD/Ping)', fontsize=12)
        ax.set_ylabel('Number of Properties', fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend()

        self.price_distribution_canvas = self.create_figure_canvas(fig, self.tab4)

    def highlight_on_price_distribution(self, predicted_price):
        """Highlight user's prediction on Price Distribution plot"""
        fig = self.price_distribution_canvas.figure
        ax = fig.axes[0]

        # Clear any previous prediction markers
        for artist in ax.lines[1:]:
            if artist.get_label() != 'Market Mean':
                artist.remove()

        # Add vertical line for prediction
        ax.axvline(predicted_price, color='g', linestyle='-', lw=3,
                   label=f'Your Prediction: {predicted_price:.2f}')

        # Add text annotation
        ax.text(predicted_price, ax.get_ylim()[1] * 0.9,
                f'Your Prediction: {predicted_price:.2f}',
                rotation=90, verticalalignment='top', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))

        ax.legend()
        fig.canvas.draw()

    def create_residual_plot(self):
        """Create Residual Analysis plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate residuals
        residuals = self.y_test - self.test_predictions

        # Plot residuals
        ax.scatter(self.test_predictions, residuals, alpha=0.6, color='#FF9800')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)

        # Add labels
        ax.set_title('Prediction Residual Analysis', fontsize=14)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        ax.grid(alpha=0.2)

        # Add explanation
        ax.text(0.05, 0.95, "Good predictions have residuals near zero",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

        self.create_figure_canvas(fig, self.tab5)

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = RealEstateApp(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == "__main__":
    main()