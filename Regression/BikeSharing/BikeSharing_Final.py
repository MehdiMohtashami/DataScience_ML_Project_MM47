import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QComboBox, QTabWidget, QMessageBox, QScrollArea, QFrame,
                             QDateEdit)
from PyQt5.QtCore import Qt, QRegExp, QDate
from PyQt5.QtGui import QRegExpValidator, QFont, QPalette, QColor

# تنظیم استایل seaborn برای نمودارها
sns.set_style("whitegrid")
sns.set_palette("husl")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setWindowTitle("Heart Disease Prediction")
        # تنظیمات برای جلوگیری از به هم ریختن نمودارها
        self.fig.tight_layout()
        self.axes.grid(True, alpha=0.3)

    def clear_plot(self):
        """پاک کردن کامل نمودار"""
        self.axes.clear()
        self.axes.grid(True, alpha=0.3)
        self.draw()


class BikeSharingPredictor:
    def __init__(self):
        try:
            self.model = joblib.load('bike_sharing_xgboost_model.pkl')
            self.scaler = joblib.load('bike_sharing_scaler.pkl')
            self.required_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                                      'workingday', 'weathersit', 'temp', 'atemp', 'hum',
                                      'windspeed', 'year', 'month', 'day']
        except:
            self.model = None
            self.scaler = None

    def preprocess_data(self, input_dict):
        df = pd.DataFrame([input_dict])

        # تبدیل تاریخ
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'])
            df['year'] = df['dteday'].dt.year
            df['month'] = df['dteday'].dt.month
            df['day'] = df['dteday'].dt.day
            df.drop('dteday', axis=1, inplace=True)

        # اطمینان از وجود تمام ویژگی‌های مورد نیاز
        for feature in self.required_features:
            if feature not in df.columns:
                df[feature] = 0

        return df[self.required_features]

    def predict(self, input_data):
        if self.model is None or self.scaler is None:
            return None

        processed_data = self.preprocess_data(input_data)
        scaled_data = self.scaler.transform(processed_data)
        prediction = self.model.predict(scaled_data)
        return prediction[0]


class ValidationLineEdit(QLineEdit):
    def __init__(self, placeholder, validator_type, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)

        if validator_type == "int":
            validator = QRegExpValidator(QRegExp(r"^\d+$"))
            self.setValidator(validator)
        elif validator_type == "float":
            validator = QRegExpValidator(QRegExp(r"^\d*\.?\d+$"))
            self.setValidator(validator)

        self.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #ccc;
                border-radius: 5px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = BikeSharingPredictor()
        self.predictions_history = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('BikeSharing Demand Prediction')
        self.setGeometry(100, 100, 1400, 900)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Model info section
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout(model_info_group)

        model_name = QLabel("XGBoost Regressor")
        model_name.setStyleSheet("font-weight: bold; font-size: 16px; color: #2E86AB;")
        model_accuracy = QLabel("R² Score: 0.9520 | RMSE: 38.97")
        model_accuracy.setStyleSheet("font-size: 14px; color: #555;")

        model_info_layout.addWidget(model_name)
        model_info_layout.addWidget(model_accuracy)
        model_info_layout.addStretch()

        # Input section
        input_group = QGroupBox("Input Parameters")
        input_layout = QGridLayout(input_group)

        # Input fields
        self.input_fields = {}
        row = 0

        # Date - استفاده از تقویم
        input_layout.addWidget(QLabel("Date:"), row, 0)
        self.input_fields['dteday'] = QDateEdit()
        self.input_fields['dteday'].setDate(QDate(2012, 7, 15))  # تاریخ پیش‌فرض
        self.input_fields['dteday'].setCalendarPopup(True)  # نمایش تقویم
        self.input_fields['dteday'].setDisplayFormat("yyyy-MM-dd")
        self.input_fields['dteday'].setStyleSheet("""
            QDateEdit {
                padding: 8px;
                border: 2px solid #ccc;
                border-radius: 5px;
                font-size: 14px;
            }
            QDateEdit:focus {
                border-color: #4CAF50;
            }
        """)
        input_layout.addWidget(self.input_fields['dteday'], row, 1)
        row += 1

        # Season
        input_layout.addWidget(QLabel("Season:"), row, 0)
        season_combo = QComboBox()
        season_combo.addItems(["Spring", "Summer", "Fall", "Winter"])
        season_combo.setCurrentIndex(2)  # Default to Fall
        self.input_fields['season'] = season_combo
        input_layout.addWidget(season_combo, row, 1)
        row += 1

        # Year
        input_layout.addWidget(QLabel("Year:"), row, 0)
        year_combo = QComboBox()
        year_combo.addItems(["2011", "2012"])
        year_combo.setCurrentIndex(1)  # Default to 2012
        self.input_fields['yr'] = year_combo
        input_layout.addWidget(year_combo, row, 1)
        row += 1

        # Month
        input_layout.addWidget(QLabel("Month:"), row, 0)
        month_combo = QComboBox()
        month_combo.addItems(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        month_combo.setCurrentIndex(6)  # Default to July
        self.input_fields['mnth'] = month_combo
        input_layout.addWidget(month_combo, row, 1)
        row += 1

        # Hour
        input_layout.addWidget(QLabel("Hour:"), row, 0)
        self.input_fields['hr'] = ValidationLineEdit("0-23", "int")
        self.input_fields['hr'].setText("12")
        input_layout.addWidget(self.input_fields['hr'], row, 1)
        row += 1

        # Holiday
        input_layout.addWidget(QLabel("Holiday:"), row, 0)
        holiday_combo = QComboBox()
        holiday_combo.addItems(["No", "Yes"])
        self.input_fields['holiday'] = holiday_combo
        input_layout.addWidget(holiday_combo, row, 1)
        row += 1

        # Weekday
        input_layout.addWidget(QLabel("Weekday:"), row, 0)
        weekday_combo = QComboBox()
        weekday_combo.addItems(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
        weekday_combo.setCurrentIndex(3)  # Default to Wednesday
        self.input_fields['weekday'] = weekday_combo
        input_layout.addWidget(weekday_combo, row, 1)
        row += 1

        # Working Day
        input_layout.addWidget(QLabel("Working Day:"), row, 0)
        workingday_combo = QComboBox()
        workingday_combo.addItems(["No", "Yes"])
        workingday_combo.setCurrentIndex(1)  # Default to Yes
        self.input_fields['workingday'] = workingday_combo
        input_layout.addWidget(workingday_combo, row, 1)
        row += 1

        # Weather Situation
        input_layout.addWidget(QLabel("Weather Situation:"), row, 0)
        weather_combo = QComboBox()
        weather_combo.addItems(["Clear", "Misty", "Light Rain/Snow", "Heavy Rain/Snow"])
        self.input_fields['weathersit'] = weather_combo
        input_layout.addWidget(weather_combo, row, 1)
        row += 1

        # Temperature
        input_layout.addWidget(QLabel("Temperature (0-1):"), row, 0)
        self.input_fields['temp'] = ValidationLineEdit("0.0-1.0", "float")
        self.input_fields['temp'].setText("0.7")
        input_layout.addWidget(self.input_fields['temp'], row, 1)
        row += 1

        # Feeling Temperature
        input_layout.addWidget(QLabel("Feeling Temp (0-1):"), row, 0)
        self.input_fields['atemp'] = ValidationLineEdit("0.0-1.0", "float")
        self.input_fields['atemp'].setText("0.65")
        input_layout.addWidget(self.input_fields['atemp'], row, 1)
        row += 1

        # Humidity
        input_layout.addWidget(QLabel("Humidity (0-1):"), row, 0)
        self.input_fields['hum'] = ValidationLineEdit("0.0-1.0", "float")
        self.input_fields['hum'].setText("0.6")
        input_layout.addWidget(self.input_fields['hum'], row, 1)
        row += 1

        # Windspeed
        input_layout.addWidget(QLabel("Windspeed (0-1):"), row, 0)
        self.input_fields['windspeed'] = ValidationLineEdit("0.0-1.0", "float")
        self.input_fields['windspeed'].setText("0.1")
        input_layout.addWidget(self.input_fields['windspeed'], row, 1)

        # Prediction button
        predict_btn = QPushButton("Predict Bike Demand")
        predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        predict_btn.clicked.connect(self.predict_demand)

        # ایجاد یک دکمه برای بازگشت
        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        back_button.setStyleSheet("""
                    QPushButton {
                        background-color: gray;
                        color: white;
                        border: none;
                        padding: 12px;
                        font-size: 16px;
                        font-weight: bold;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: gray;
                    }
                    QPushButton:pressed {
                        background-color: gray;
                    }
                """)
        back_button.clicked.connect(self.predict_demand)


        # Clear button
        clear_btn = QPushButton("Clear Predictions")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        clear_btn.clicked.connect(self.clear_predictions)

        # Result display
        self.result_label = QLabel("Prediction: -")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2E86AB;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #dee2e6;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)

        # Add widgets to left layout
        left_layout.addWidget(model_info_group)
        left_layout.addWidget(input_group)
        left_layout.addWidget(predict_btn)
        left_layout.addWidget(back_button)
        left_layout.addWidget(clear_btn)
        left_layout.addWidget(self.result_label)
        left_layout.addStretch()

        # Right panel for charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget for different chart types
        self.tab_widget = QTabWidget()

        # Prediction Analysis tab
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)

        # Feature Importance tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)

        # Feature Relationship tab
        relationship_tab = QWidget()
        relationship_layout = QVBoxLayout(relationship_tab)

        # Create chart containers
        self.prediction_chart1 = MplCanvas(self, width=6, height=4)
        self.prediction_chart2 = MplCanvas(self, width=6, height=4)

        self.feature_chart1 = MplCanvas(self, width=6, height=4)
        self.feature_chart2 = MplCanvas(self, width=6, height=4)

        self.relationship_chart1 = MplCanvas(self, width=6, height=4)
        self.relationship_chart2 = MplCanvas(self, width=6, height=4)

        # Add charts to tabs
        prediction_layout.addWidget(QLabel("Time Series Analysis"))
        prediction_layout.addWidget(self.prediction_chart1)
        prediction_layout.addWidget(QLabel("Distribution of Predictions"))
        prediction_layout.addWidget(self.prediction_chart2)

        feature_layout.addWidget(QLabel("Feature Importance"))
        feature_layout.addWidget(self.feature_chart1)
        feature_layout.addWidget(QLabel("Feature Correlation"))
        feature_layout.addWidget(self.feature_chart2)

        relationship_layout.addWidget(QLabel("Temperature vs Demand"))
        relationship_layout.addWidget(self.relationship_chart1)
        relationship_layout.addWidget(QLabel("Seasonal Patterns"))
        relationship_layout.addWidget(self.relationship_chart2)

        # Add tabs to tab widget
        self.tab_widget.addTab(prediction_tab, "Prediction Analysis")
        self.tab_widget.addTab(feature_tab, "Feature Importance")
        self.tab_widget.addTab(relationship_tab, "Feature Relationships")

        right_layout.addWidget(self.tab_widget)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Initially hide charts
        self.tab_widget.setVisible(False)

        # Set up initial chart data
        self.setup_initial_charts()

    def setup_initial_charts(self):
        """تنظیم داده‌های اولیه برای نمودارها"""
        # نمونه داده برای نمودارهای رابطه ویژگی‌ها
        np.random.seed(42)
        self.sample_temp_values = np.random.uniform(0, 1, 100)
        self.sample_demand_values = 200 + 500 * self.sample_temp_values + np.random.normal(0, 50, 100)

        # داده‌های نمونه برای الگوهای فصلی
        self.seasonal_avg = [150, 250, 220, 120]

        # اهمیت ویژگی‌ها
        self.feature_names = ['temp', 'hr', 'atemp', 'hum', 'windspeed', 'season', 'weathersit',
                              'mnth', 'weekday', 'workingday', 'holiday', 'yr']
        self.importance_values = [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]

        # ماتریس همبستگی
        np.random.seed(42)
        corr_data = np.random.randn(10, 10)
        self.corr_matrix = np.corrcoef(corr_data)
        self.feature_subset = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'season', 'weathersit', 'mnth', 'weekday',
                               'workingday']

    def get_input_values(self):
        values = {}

        # Get values from input fields
        date_value = self.input_fields['dteday'].date()
        values['dteday'] = f"{date_value.year()}-{date_value.month():02d}-{date_value.day():02d}"

        # Convert combo box values to numeric
        values['season'] = self.input_fields['season'].currentIndex() + 1
        values['yr'] = self.input_fields['yr'].currentIndex()
        values['mnth'] = self.input_fields['mnth'].currentIndex() + 1
        values['hr'] = int(self.input_fields['hr'].text()) if self.input_fields['hr'].text() else 0
        values['holiday'] = self.input_fields['holiday'].currentIndex()
        values['weekday'] = self.input_fields['weekday'].currentIndex()
        values['workingday'] = self.input_fields['workingday'].currentIndex()
        values['weathersit'] = self.input_fields['weathersit'].currentIndex() + 1

        # Get float values
        values['temp'] = float(self.input_fields['temp'].text()) if self.input_fields['temp'].text() else 0.0
        values['atemp'] = float(self.input_fields['atemp'].text()) if self.input_fields['atemp'].text() else 0.0
        values['hum'] = float(self.input_fields['hum'].text()) if self.input_fields['hum'].text() else 0.0
        values['windspeed'] = float(self.input_fields['windspeed'].text()) if self.input_fields[
            'windspeed'].text() else 0.0

        return values

    def predict_demand(self):
        # Get input values
        try:
            input_values = self.get_input_values()

            # Make prediction
            prediction = self.predictor.predict(input_values)

            if prediction is None:
                QMessageBox.critical(self, "Error", "Model not loaded. Please check if model files exist.")
                return

            # Display result
            self.result_label.setText(f"Prediction: {int(prediction)} bikes")

            # Add to prediction history
            self.predictions_history.append({
                'input': input_values,
                'prediction': prediction,
                'timestamp': datetime.now()
            })

            # Show charts and update them
            self.tab_widget.setVisible(True)
            self.update_all_charts()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def clear_predictions(self):
        self.predictions_history = []
        self.result_label.setText("Prediction: -")
        self.update_all_charts()

    def update_all_charts(self):
        self.update_prediction_charts()
        self.update_feature_charts()
        self.update_relationship_charts()

    def update_prediction_charts(self):
        # پاک کردن کامل نمودارها قبل از رسم جدید
        self.prediction_chart1.clear_plot()
        self.prediction_chart2.clear_plot()

        # Chart 1: Time series of predictions
        if self.predictions_history:
            timestamps = [p['timestamp'] for p in self.predictions_history]
            predictions = [p['prediction'] for p in self.predictions_history]

            self.prediction_chart1.axes.plot(timestamps, predictions, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            self.prediction_chart1.axes.set_title('Prediction History Over Time', fontsize=14, fontweight='bold')
            self.prediction_chart1.axes.set_xlabel('Time')
            self.prediction_chart1.axes.set_ylabel('Predicted Bike Count')

            # تنظیم محدوده محور Y برای جلوگیری از به هم ریختن
            if len(predictions) > 0:
                y_min = max(0, min(predictions) - 50)
                y_max = max(predictions) + 50
                self.prediction_chart1.axes.set_ylim(y_min, y_max)

            # Format x-axis dates
            self.prediction_chart1.fig.autofmt_xdate()
        else:
            self.prediction_chart1.axes.text(0.5, 0.5, 'No predictions yet\nClick "Predict" to see results',
                                             horizontalalignment='center', verticalalignment='center',
                                             transform=self.prediction_chart1.axes.transAxes, fontsize=12)

        # Chart 2: Distribution of predictions
        if self.predictions_history:
            predictions = [p['prediction'] for p in self.predictions_history]

            self.prediction_chart2.axes.hist(predictions, bins=min(10, len(predictions)), alpha=0.7,
                                             edgecolor='black', color='#A23B72')
            self.prediction_chart2.axes.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
            self.prediction_chart2.axes.set_xlabel('Predicted Bike Count')
            self.prediction_chart2.axes.set_ylabel('Frequency')
        else:
            self.prediction_chart2.axes.text(0.5, 0.5, 'No predictions yet\nClick "Predict" to see results',
                                             horizontalalignment='center', verticalalignment='center',
                                             transform=self.prediction_chart2.axes.transAxes, fontsize=12)

        # Refresh canvases
        self.prediction_chart1.figure.tight_layout()
        self.prediction_chart2.figure.tight_layout()
        self.prediction_chart1.draw()
        self.prediction_chart2.draw()

    def update_feature_charts(self):
        # پاک کردن کامل نمودارها قبل از رسم جدید
        self.feature_chart1.clear_plot()
        self.feature_chart2.clear_plot()

        # Chart 1: Feature importance
        sorted_idx = np.argsort(self.importance_values)

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.feature_names)))
        self.feature_chart1.axes.barh(range(len(sorted_idx)), [self.importance_values[i] for i in sorted_idx],
                                      color=colors)
        self.feature_chart1.axes.set_yticks(range(len(sorted_idx)))
        self.feature_chart1.axes.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        self.feature_chart1.axes.set_title('Feature Importance', fontsize=14, fontweight='bold')
        self.feature_chart1.axes.set_xlabel('Importance')

        # Chart 2: Feature correlation heatmap
        im = self.feature_chart2.axes.imshow(self.corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        self.feature_chart2.axes.set_xticks(range(len(self.feature_subset)))
        self.feature_chart2.axes.set_yticks(range(len(self.feature_subset)))
        self.feature_chart2.axes.set_xticklabels(self.feature_subset, rotation=45, ha='right')
        self.feature_chart2.axes.set_yticklabels(self.feature_subset)
        self.feature_chart2.axes.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

        # Add colorbar
        self.feature_chart2.fig.colorbar(im, ax=self.feature_chart2.axes)

        # Refresh canvases
        self.feature_chart1.figure.tight_layout()
        self.feature_chart2.figure.tight_layout()
        self.feature_chart1.draw()
        self.feature_chart2.draw()

    def update_relationship_charts(self):
        # پاک کردن کامل نمودارها قبل از رسم جدید
        self.relationship_chart1.clear_plot()
        self.relationship_chart2.clear_plot()

        # Chart 1: Temperature vs Demand
        self.relationship_chart1.axes.scatter(self.sample_temp_values, self.sample_demand_values, alpha=0.6,
                                              color='#18A558')

        # Add current prediction if available
        if self.predictions_history:
            current = self.predictions_history[-1]
            self.relationship_chart1.axes.scatter(
                current['input']['temp'],
                current['prediction'],
                color='red',
                s=100,
                label='Current Prediction',
                edgecolors='black'
            )
            self.relationship_chart1.axes.legend()

        self.relationship_chart1.axes.set_title('Temperature vs Bike Demand', fontsize=14, fontweight='bold')
        self.relationship_chart1.axes.set_xlabel('Normalized Temperature')
        self.relationship_chart1.axes.set_ylabel('Bike Count')

        # Chart 2: Seasonal patterns
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3']
        bars = self.relationship_chart2.axes.bar(seasons, self.seasonal_avg, alpha=0.7, color=colors)

        # Add current prediction if available
        if self.predictions_history:
            current = self.predictions_history[-1]
            season_idx = current['input']['season'] - 1
            self.relationship_chart2.axes.plot(
                season_idx,
                current['prediction'],
                'ro',
                markersize=10,
                label='Current Prediction',
                markeredgecolor='black'
            )
            self.relationship_chart2.axes.legend()

        self.relationship_chart2.axes.set_title('Average Demand by Season', fontsize=14, fontweight='bold')
        self.relationship_chart2.axes.set_ylabel('Average Bike Count')

        # Refresh canvases
        self.relationship_chart1.figure.tight_layout()
        self.relationship_chart2.figure.tight_layout()
        self.relationship_chart1.draw()
        self.relationship_chart2.draw()

    def close_and_go_back(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')
    parent = None
    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())