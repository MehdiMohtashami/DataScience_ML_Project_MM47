import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLineEdit, QComboBox, QPushButton, QLabel, QTabWidget, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont


class AutoMPGPredictor(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoMPG")
        self.setWindowTitle("AutoMPG Predictor")
        self.setGeometry(100, 100, 1200, 800)

        # Load model and data
        self.model = joblib.load('best_auto_mpg_model.joblib')
        self.df = pd.read_csv('auto-mpg.csv')
        self.current_prediction = None

        def preprocess_data(df):
            print("\nMissing values status before preprocessing:")
            print(df.isna().sum())

            numeric_cols = ['displacement', 'horsepower', 'weight', 'acceleration']

            for col in numeric_cols:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

            origin_map = {1: 'American', 2: 'European', 3: 'Japanese'}
            df['origin'] = df['origin'].map(origin_map)

            if 'car name' in df.columns:
                df.drop('car name', axis=1, inplace=True)
            if 'car_name' in df.columns:
                df.drop('car_name', axis=1, inplace=True)

            return df

        df = preprocess_data(self.df)

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel - Input form
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Model info group
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        model_info = QLabel(
            "<b>Ridge Regression Model</b><br>"
            "<b>R²:</b> 0.8641 | <b>MAE:</b> 2.16 | <b>RMSE:</b> 2.70<br><br>"
            "This model predicts vehicle fuel efficiency (MPG) based on technical specifications."
        )
        model_info.setAlignment(Qt.AlignCenter)
        model_info.setStyleSheet("font-size: 14px;")
        model_layout.addWidget(model_info)

        # Input form group
        input_group = QGroupBox("Car Specifications")
        form_layout = QFormLayout()
        input_group.setLayout(form_layout)

        # Create input fields with validators
        double_validator = QDoubleValidator()

        self.cylinders = QComboBox()
        self.cylinders.addItems(["3", "4", "5", "6", "8"])

        self.displacement = QLineEdit()
        self.displacement.setValidator(double_validator)
        self.displacement.setPlaceholderText("e.g., 250.0")

        self.horsepower = QLineEdit()
        self.horsepower.setValidator(double_validator)
        self.horsepower.setPlaceholderText("e.g., 100.0")

        self.weight = QLineEdit()
        self.weight.setValidator(double_validator)
        self.weight.setPlaceholderText("e.g., 3500")

        self.acceleration = QLineEdit()
        self.acceleration.setValidator(double_validator)
        self.acceleration.setPlaceholderText("e.g., 15.0")

        self.model_year = QComboBox()
        self.model_year.addItems([str(year) for year in range(70, 83)])

        self.origin = QComboBox()
        self.origin.addItems(["American", "European", "Japanese"])

        # Add fields to form
        form_layout.addRow("Cylinders:", self.cylinders)
        form_layout.addRow("Displacement (cu.in):", self.displacement)
        form_layout.addRow("Horsepower (hp):", self.horsepower)
        form_layout.addRow("Weight (lbs):", self.weight)
        form_layout.addRow("Acceleration (0-60 sec):", self.acceleration)
        form_layout.addRow("Model Year (70-82):", self.model_year)
        form_layout.addRow("Origin:", self.origin)

        # Prediction button
        self.predict_btn = QPushButton("Predict MPG")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton {"
            "   background-color: #4CAF50;"
            "   color: white;"
            "   font-weight: bold;"
            "   padding: 12px;"
            "   border-radius: 6px;"
            "   font-size: 16px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #45a049;"
            "}"
        )
        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        self.back_button.setStyleSheet(
            "QPushButton {"
            "   background-color: gray;"
            "   color: white;"
            "   font-weight: bold;"
            "   padding: 12px;"
            "   border-radius: 6px;"
            "   font-size: 16px;"
            "}"
            "QPushButton:hover {"
            "   background-color: gray;"
            "}"
        )


        # Prediction result
        self.result_label = QLabel("Predicted MPG: --")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #2196F3;"
            "margin: 25px; background-color: #f0f8ff;"
            "border-radius: 8px; padding: 15px;"
        )

        # Add widgets to left layout
        left_layout.addWidget(model_group)
        left_layout.addWidget(input_group)
        left_layout.addWidget(self.predict_btn)
        left_layout.addWidget(self.back_button)
        left_layout.addWidget(self.result_label)
        left_layout.addStretch()

        # Right panel - Visualization tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setVisible(False)  # Hide until first prediction

        # Create tabs with simplified charts
        self.tab1 = QWidget()  # Prediction Analysis
        self.tab2 = QWidget()  # Feature Importance
        self.tab3 = QWidget()  # MPG Trends
        self.tab4 = QWidget()  # Feature Relationships

        self.tab_widget.addTab(self.tab1, "Prediction Analysis")
        self.tab_widget.addTab(self.tab2, "Feature Importance")
        self.tab_widget.addTab(self.tab3, "MPG Trends")
        self.tab_widget.addTab(self.tab4, "Feature Relationships")

        # Set up canvas for each tab
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()

        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.tab_widget, 1)

    def setup_tab1(self):
        layout = QVBoxLayout()
        self.tab1.setLayout(layout)

        self.canvas1 = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.canvas1)

        # Initial empty plot
        ax = self.canvas1.figure.subplots()
        ax.text(0.5, 0.5, "Perform a prediction to see analysis",
                ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')
        self.canvas1.draw()

    def setup_tab2(self):
        layout = QVBoxLayout()
        self.tab2.setLayout(layout)

        self.canvas2 = FigureCanvas(Figure(figsize=(10, 6)))
        layout.addWidget(self.canvas2)

        # Initial empty plot
        ax = self.canvas2.figure.subplots()
        ax.text(0.5, 0.5, "Predict MPG to see feature importance",
                ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')
        self.canvas2.draw()

    def setup_tab3(self):
        layout = QVBoxLayout()
        self.tab3.setLayout(layout)

        self.canvas3 = FigureCanvas(Figure(figsize=(10, 6)))
        layout.addWidget(self.canvas3)

        # Initial empty plot
        ax = self.canvas3.figure.subplots()
        ax.text(0.5, 0.5, "Predict MPG to see MPG trends",
                ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')
        self.canvas3.draw()

    def setup_tab4(self):
        layout = QVBoxLayout()
        self.tab4.setLayout(layout)

        self.canvas4 = FigureCanvas(Figure(figsize=(10, 6)))
        layout.addWidget(self.canvas4)

        # Initial empty plot
        ax = self.canvas4.figure.subplots()
        ax.text(0.5, 0.5, "Predict MPG to see feature relationships",
                ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')
        self.canvas4.draw()

    def predict(self):
        # Validate inputs
        if not self.validate_inputs():
            return

        # Get input values
        car_data = {
            'cylinders': int(self.cylinders.currentText()),
            'displacement': float(self.displacement.text()),
            'horsepower': float(self.horsepower.text()),
            'weight': float(self.weight.text()),
            'acceleration': float(self.acceleration.text()),
            'model year': int(self.model_year.currentText()),
            'origin': self.origin.currentText()
        }

        # Create DataFrame
        sample_car = pd.DataFrame([car_data])

        # Predict
        try:
            prediction = self.model.predict(sample_car)[0]
            self.current_prediction = {
                'data': car_data,
                'prediction': prediction
            }

            # Update result label
            self.result_label.setText(f"Predicted MPG: {prediction:.2f}")

            # Show visualizations
            self.update_visualizations()
            self.tab_widget.setVisible(True)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error",
                                 f"Error during prediction: {str(e)}")

    def validate_inputs(self):
        # Check all numerical fields
        fields = {
            "Displacement": self.displacement,
            "Horsepower": self.horsepower,
            "Weight": self.weight,
            "Acceleration": self.acceleration
        }

        for name, field in fields.items():
            if not field.text().strip():
                QMessageBox.warning(self, "Input Error", f"{name} is required!")
                field.setFocus()
                return False

            try:
                float(field.text())
            except ValueError:
                QMessageBox.warning(self, "Input Error",
                                    f"Invalid number format for {name}!")
                field.setFocus()
                return False

        return True

    def update_visualizations(self):
        if not self.current_prediction:
            return

        # Update all visualizations
        self.update_prediction_analysis()
        self.update_feature_importance()
        self.update_mpg_trends()
        self.update_feature_relationships()

    def update_prediction_analysis(self):
        # Clear previous figure
        self.canvas1.figure.clear()
        fig = self.canvas1.figure
        fig.set_facecolor('#f8f9fa')

        # Create subplots
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # 1. Actual vs Predicted
        sample = self.df.sample(100)
        y_true = sample['mpg']
        y_pred = self.model.predict(sample.drop('mpg', axis=1))

        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, color='#1f77b4', label='Cars in dataset')

        # Add current prediction
        ax1.scatter(
            [self.current_prediction['prediction']],
            [self.current_prediction['prediction']],
            s=150, c='#d62728', marker='*', label='Your Prediction'
        )

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min()) - 3
        max_val = max(y_true.max(), y_pred.max()) + 3
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

        ax1.set_xlabel('Actual MPG', fontsize=10)
        ax1.set_ylabel('Predicted MPG', fontsize=10)
        ax1.set_title('Actual vs Predicted MPG', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 2. MPG Distribution with Prediction
        sns.histplot(self.df['mpg'], kde=True, ax=ax2, color='#2ca02c')
        ax2.axvline(self.current_prediction['prediction'],
                    color='#d62728', linestyle='--', linewidth=2)

        # Add annotation
        ax2.annotate(
            f"Your Car: {self.current_prediction['prediction']:.1f} MPG",
            xy=(self.current_prediction['prediction'], 0),
            xytext=(5, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='#d62728'),
            color='#d62728',
            fontweight='bold'
        )

        ax2.set_xlabel('MPG', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('MPG Distribution with Your Prediction', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Adjust layout
        fig.tight_layout()
        self.canvas1.draw()

    def update_feature_importance(self):
        # Clear previous figure
        self.canvas2.figure.clear()
        fig = self.canvas2.figure
        fig.set_facecolor('#f8f9fa')
        ax = fig.add_subplot(1, 1, 1)

        try:
            # Extract the Ridge model and preprocessor
            preprocessor = self.model.named_steps['preprocessor']
            ridge_model = self.model.named_steps['regressor']

            # Get feature names and coefficients
            feature_names = preprocessor.get_feature_names_out()
            coefficients = ridge_model.coef_

            # Get top 10 features by absolute value
            top_idx = np.argsort(np.abs(coefficients))[-10:]
            top_features = np.array(feature_names)[top_idx]
            top_coeffs = coefficients[top_idx]

            # Simplify feature names
            simplified_names = []
            for name in top_features:
                if 'num__' in name:
                    simplified_names.append(name.split('__')[1])
                elif 'cat__' in name:
                    parts = name.split('__')
                    if len(parts) > 2:
                        simplified_names.append(f"{parts[1]}: {parts[2]}")
                    else:
                        simplified_names.append(name)
                else:
                    simplified_names.append(name)

            # Create horizontal bar chart
            colors = ['#1f77b4' if c > 0 else '#d62728' for c in top_coeffs]
            y_pos = np.arange(len(simplified_names))

            ax.barh(y_pos, top_coeffs, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(simplified_names)
            ax.set_xlabel('Coefficient Value', fontsize=10)
            ax.set_title('Top 10 Feature Importance (Ridge Regression)', fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)

            # Add value labels
            for i, v in enumerate(top_coeffs):
                ax.text(v, i, f" {v:.3f}",
                        color=('black' if abs(v) > 0.5 else 'black'),
                        va='center', fontsize=9)

        except Exception as e:
            ax.text(0.5, 0.5, "Feature importance not available",
                    ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')

        fig.tight_layout()
        self.canvas2.draw()

    def update_mpg_trends(self):
        # Clear previous figure
        self.canvas3.figure.clear()
        fig = self.canvas3.figure
        fig.set_facecolor('#f8f9fa')

        # Create subplots
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # 1. MPG by Model Year (Boxplot)
        sns.boxplot(x='model year', y='mpg', data=self.df, ax=ax1,
                    palette='pastel', width=0.7)

        # Add current prediction
        prediction_year = int(self.current_prediction['data']['model year'])
        ax1.scatter(
            [prediction_year - 70],
            [self.current_prediction['prediction']],
            s=100, c='red', marker='*',
            label=f'Your Prediction: {self.current_prediction["prediction"]:.1f} MPG'
        )

        ax1.set_xlabel('Model Year', fontsize=10)
        ax1.set_ylabel('MPG', fontsize=10)
        ax1.set_title('MPG Distribution by Model Year', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

        # 2. MPG by Origin (Bar plot)
        origin_mpg = self.df.groupby('origin')['mpg'].mean().sort_values()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        origin_mpg.plot(kind='bar', ax=ax2, color=colors, alpha=0.8)

        # Add current prediction
        user_origin = self.current_prediction['data']['origin']
        user_mpg = self.current_prediction['prediction']
        origin_idx = list(origin_mpg.index).index(user_origin)

        ax2.scatter(
            [origin_idx],
            [user_mpg],
            s=100, c='red', marker='*',
            label=f'Your Prediction: {user_mpg:.1f} MPG'
        )

        # Add value labels
        for i, v in enumerate(origin_mpg):
            ax2.text(i, v + 0.5, f"{v:.1f}",
                     ha='center', fontsize=9)

        ax2.set_xlabel('Origin', fontsize=10)
        ax2.set_ylabel('Average MPG', fontsize=10)
        ax2.set_title('Average MPG by Origin', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

        fig.tight_layout()
        self.canvas3.draw()

    def update_feature_relationships(self):
        # Clear previous figure
        self.canvas4.figure.clear()
        fig = self.canvas4.figure
        fig.set_facecolor('#f8f9fa')

        # Create subplots
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # 1. Weight vs MPG
        sns.regplot(
            x='weight',
            y='mpg',
            data=self.df,
            ax=ax1,
            scatter_kws={'alpha': 0.4, 'color': '#1f77b4'},
            line_kws={'color': '#d62728', 'alpha': 0.7}
        )

        # Add current prediction
        ax1.scatter(
            [self.current_prediction['data']['weight']],
            [self.current_prediction['prediction']],
            s=150, c='red', marker='*',
            label='Your Prediction'
        )

        # Add annotation
        ax1.annotate(
            f"{self.current_prediction['prediction']:.1f} MPG",
            xy=(self.current_prediction['data']['weight'], self.current_prediction['prediction']),
            xytext=(10, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='red'),
            color='red',
            fontweight='bold'
        )

        ax1.set_xlabel('Weight (lbs)', fontsize=10)
        ax1.set_ylabel('MPG', fontsize=10)
        ax1.set_title('Vehicle Weight vs Fuel Efficiency', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 2. Horsepower vs MPG
        sns.regplot(
            x='horsepower',
            y='mpg',
            data=self.df,
            ax=ax2,
            scatter_kws={'alpha': 0.4, 'color': '#ff7f0e'},
            line_kws={'color': '#d62728', 'alpha': 0.7}
        )

        # Add current prediction
        ax2.scatter(
            [self.current_prediction['data']['horsepower']],
            [self.current_prediction['prediction']],
            s=150, c='red', marker='*',
            label='Your Prediction'
        )

        # Add annotation
        ax2.annotate(
            f"{self.current_prediction['prediction']:.1f} MPG",
            xy=(self.current_prediction['data']['horsepower'], self.current_prediction['prediction']),
            xytext=(10, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='red'),
            color='red',
            fontweight='bold'
        )

        ax2.set_xlabel('Horsepower (hp)', fontsize=10)
        ax2.set_ylabel('MPG', fontsize=10)
        ax2.set_title('Engine Power vs Fuel Efficiency', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)

        fig.tight_layout()
        self.canvas4.draw()

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 9, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = AutoMPGPredictor(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == "__main__":
    main()
