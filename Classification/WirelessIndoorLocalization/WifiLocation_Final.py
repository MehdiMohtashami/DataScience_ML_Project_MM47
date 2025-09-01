import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QScrollArea, QMessageBox, QSizePolicy, QFormLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QFont
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class WiFiApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WirelessIndoorLocalization")
        self.setWindowTitle("WiFi Indoor Localization Predictor")
        self.setGeometry(100, 100, 1200, 900)

        # Load model and scaler
        try:
            self.model = joblib.load('knn_wifi_model.pkl')
            self.scaler = joblib.load('standard_scaler.pkl')
            self.accuracy = 0.9850  # From our evaluation
        except:
            QMessageBox.critical(self, "Error",
                                 "Could not load model files. Please make sure knn_wifi_model.pkl and standard_scaler.pkl are in the same directory.")
            sys.exit(1)

        # Load sample data for visualizations
        self.df = pd.read_csv('wifi_localization.csv')
        self.feature_names = list(self.df.columns[:-1])

        # Create a central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add title
        title = QLabel("WiFi Indoor Localization Predictor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin: 10px;")
        scroll_layout.addWidget(title)

        # Model info section
        model_info = QLabel(f"Using K-Nearest Neighbors model with accuracy: {self.accuracy * 100:.2f}%")
        model_info.setAlignment(Qt.AlignCenter)
        model_info.setStyleSheet("margin-bottom: 10px;")
        scroll_layout.addWidget(model_info)

        # Input section - استفاده از QFormLayout برای ترازبندی بهتر
        input_group = QGroupBox("Input WiFi Signal Strengths (values between -100 and 0)")
        input_layout = QFormLayout()

        self.input_fields = []
        for i in range(7):
            input_field = QLineEdit()
            input_field.setPlaceholderText("e.g., -64")
            input_field.setValidator(QIntValidator(-100, 0))
            input_field.setMaximumWidth(100)
            input_field.setAlignment(Qt.AlignLeft)
            self.input_fields.append(input_field)
            input_layout.addRow(f"Wifi {i + 1}:", input_field)

        input_group.setLayout(input_layout)
        scroll_layout.addWidget(input_group)

        # Predict button
        self.predict_btn = QPushButton("Predict Location")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        scroll_layout.addWidget(self.predict_btn)

        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        self.back_button.setStyleSheet(
            "QPushButton { background-color: gray; color: white; font-weight: bold; padding: 10px; }")
        scroll_layout.addWidget(self.back_button)


        # Result display
        self.result_label = QLabel("Predicted Room: ")
        result_font = QFont()
        result_font.setPointSize(14)
        result_font.setBold(True)
        self.result_label.setFont(result_font)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: #e8f5e9; padding: 15px; border: 1px solid #c8e6c9;")
        scroll_layout.addWidget(self.result_label)

        # Visualization section
        vis_group = QGroupBox("Prediction Analysis")
        vis_layout = QVBoxLayout()

        # Create tabs or sections for different visualizations
        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas3 = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas4 = MplCanvas(self, width=5, height=4, dpi=100)

        # Initially hide canvases
        self.canvas1.setVisible(False)
        self.canvas2.setVisible(False)
        self.canvas3.setVisible(False)
        self.canvas4.setVisible(False)

        # Add visualization titles
        self.vis_title1 = QLabel("PCA Projection of WiFi Signals")
        self.vis_title1.setAlignment(Qt.AlignCenter)
        self.vis_title1.setVisible(False)

        self.vis_title2 = QLabel("Prediction Confidence")
        self.vis_title2.setAlignment(Qt.AlignCenter)
        self.vis_title2.setVisible(False)

        self.vis_title3 = QLabel("Feature Importance")
        self.vis_title3.setAlignment(Qt.AlignCenter)
        self.vis_title3.setVisible(False)

        self.vis_title4 = QLabel("Signal Strength Comparison")
        self.vis_title4.setAlignment(Qt.AlignCenter)
        self.vis_title4.setVisible(False)

        # Arrange visualizations in a grid
        vis_grid = QGridLayout()
        vis_grid.addWidget(self.vis_title1, 0, 0)
        vis_grid.addWidget(self.vis_title2, 0, 1)
        vis_grid.addWidget(self.canvas1, 1, 0)
        vis_grid.addWidget(self.canvas2, 1, 1)
        vis_grid.addWidget(self.vis_title3, 2, 0)
        vis_grid.addWidget(self.vis_title4, 2, 1)
        vis_grid.addWidget(self.canvas3, 3, 0)
        vis_grid.addWidget(self.canvas4, 3, 1)

        vis_group.setLayout(vis_grid)
        scroll_layout.addWidget(vis_group)

        # Set the scroll content
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def predict(self):
        # Get input values
        try:
            input_values = [int(field.text()) if field.text() else 0 for field in self.input_fields]
            if any(not field.text() for field in self.input_fields):
                QMessageBox.warning(self, "Input Error", "Please fill all fields with values between -100 and 0")
                return
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid integer values between -100 and 0")
            return

        # Create DataFrame with correct feature names to avoid warnings
        input_df = pd.DataFrame([input_values], columns=self.feature_names)

        # Scale the input
        input_scaled = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(input_scaled)[0]

        # Update result label
        self.result_label.setText(f"Predicted Room: {prediction}")

        # Show visualizations
        self.canvas1.setVisible(True)
        self.canvas2.setVisible(True)
        self.canvas3.setVisible(True)
        self.canvas4.setVisible(True)
        self.vis_title1.setVisible(True)
        self.vis_title2.setVisible(True)
        self.vis_title3.setVisible(True)
        self.vis_title4.setVisible(True)

        # Generate visualizations
        self.generate_pca_plot(input_scaled, prediction)
        self.generate_confidence_plot(input_scaled)
        self.generate_feature_importance(input_scaled, prediction)
        self.generate_signal_comparison(input_values, prediction)

    def generate_pca_plot(self, input_scaled, prediction):
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.scaler.transform(self.df[self.feature_names]))
        input_pca = pca.transform(input_scaled)

        # Clear previous plot
        self.canvas1.axes.clear()

        # Create scatter plot
        rooms = self.df['Room'].unique()
        colors = ['red', 'blue', 'green', 'orange']

        for i, room in enumerate(rooms):
            mask = self.df['Room'] == room
            self.canvas1.axes.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                      alpha=0.5, c=colors[i], label=f'Room {room}', s=30)

        # Plot the new point
        self.canvas1.axes.scatter(input_pca[0, 0], input_pca[0, 1],
                                  c='black', marker='*', s=200,
                                  label=f'Your input (Room {prediction})', edgecolors='yellow', linewidth=2)

        self.canvas1.axes.set_xlabel('Principal Component 1')
        self.canvas1.axes.set_ylabel('Principal Component 2')
        self.canvas1.axes.set_title('PCA Projection of WiFi Signals')
        self.canvas1.axes.legend()
        self.canvas1.fig.tight_layout()
        self.canvas1.draw()

    def generate_confidence_plot(self, input_scaled):
        # Get probabilities for each class
        try:
            probabilities = self.model.predict_proba(input_scaled)[0]
        except:
            # For models without predict_proba, use uniform distribution
            probabilities = [0.25, 0.25, 0.25, 0.25]

        # Clear previous plot
        self.canvas2.axes.clear()

        # Create bar plot
        rooms = [1, 2, 3, 4]
        colors = ['red', 'blue', 'green', 'orange']

        bars = self.canvas2.axes.bar(rooms, probabilities, color=colors, alpha=0.7)
        self.canvas2.axes.set_xlabel('Room')
        self.canvas2.axes.set_ylabel('Confidence')
        self.canvas2.axes.set_title('Prediction Confidence')
        self.canvas2.axes.set_ylim(0, 1)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            self.canvas2.axes.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                   f'{prob:.3f}', ha='center', va='bottom')

        self.canvas2.fig.tight_layout()
        self.canvas2.draw()

    def generate_feature_importance(self, input_scaled, prediction):
        # Clear previous plot
        self.canvas3.axes.clear()

        # Calculate feature importance based on how much each feature contributes to distance
        # Get the nearest neighbors
        distances, indices = self.model.kneighbors(input_scaled)

        # Get the training data
        X_train = self.scaler.transform(self.df[self.feature_names])

        # Calculate the average absolute difference for each feature between the input and its neighbors
        feature_differences = []
        for i in range(7):
            input_val = input_scaled[0, i]
            neighbor_vals = X_train[indices[0], i]
            avg_diff = np.mean(np.abs(input_val - neighbor_vals))
            feature_differences.append(avg_diff)

        # Normalize the differences to get importance scores
        feature_importance = np.array(feature_differences) / np.sum(feature_differences)

        # Create bar plot
        features = [f'Wifi {i + 1}' for i in range(7)]

        bars = self.canvas3.axes.barh(features, feature_importance, alpha=0.7)
        self.canvas3.axes.set_xlabel('Importance Score')
        self.canvas3.axes.set_title('Feature Importance (Based on Neighbor Differences)')
        self.canvas3.axes.set_xlim(0, max(feature_importance) * 1.1)

        # Add value labels on bars
        for bar, imp in zip(bars, feature_importance):
            width = bar.get_width()
            self.canvas3.axes.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                                   f'{width:.3f}', ha='left', va='center')

        self.canvas3.fig.tight_layout()
        self.canvas3.draw()

    def generate_signal_comparison(self, input_values, prediction):
        # Compare input signals with average signals for the predicted room
        # Clear previous plot
        self.canvas4.axes.clear()

        # Get average signals for the predicted room
        room_data = self.df[self.df['Room'] == prediction]
        avg_signals = room_data.mean().values[:-1]  # Exclude Room column

        # Create comparison plot
        features = [f'Wifi {i + 1}' for i in range(7)]
        x = np.arange(len(features))

        width = 0.35
        bars1 = self.canvas4.axes.bar(x - width / 2, input_values, width, label='Your Input', alpha=0.7,
                                      color='skyblue')
        bars2 = self.canvas4.axes.bar(x + width / 2, avg_signals, width, label=f'Room {prediction} Average', alpha=0.7,
                                      color='lightcoral')

        self.canvas4.axes.set_xlabel('WiFi Features')
        self.canvas4.axes.set_ylabel('Signal Strength')
        self.canvas4.axes.set_title('Your Input vs Room Average')
        self.canvas4.axes.set_xticks(x)
        self.canvas4.axes.set_xticklabels(features, rotation=45)
        self.canvas4.axes.legend()

        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                self.canvas4.axes.text(bar.get_x() + bar.get_width() / 2., height,
                                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        add_labels(bars1)
        add_labels(bars2)

        self.canvas4.fig.tight_layout()
        self.canvas4.draw()

    def close_and_go_back(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    parent = None
    window = WiFiApp(parent)
    window.show()
    sys.exit(app.exec_())