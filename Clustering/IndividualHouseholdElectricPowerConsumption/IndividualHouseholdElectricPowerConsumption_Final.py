import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTabWidget, QFormLayout)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class PowerConsumptionUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IndividualHouseholdElectricPowerConsumption")
        self.setWindowTitle("Household Power Consumption Clustering")
        self.setGeometry(100, 100, 1200, 800)

        try:
            self.kmeans = joblib.load('kmeans_model.joblib')
        except FileNotFoundError:
            print("The model file kmeans_model.joblib was not found!")
            sys.exit()

        try:
            df = pd.read_csv('household_power_consumption.txt', sep=',', na_values='?')
            df.fillna(df.mean(numeric_only=True), inplace=True)
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df.set_index('Datetime', inplace=True)
            df.drop(['Date', 'Time'], axis=1, inplace=True)
            df['Unmetered'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
            self.df_hourly = df.resample('h').mean()
            self.df_hourly = self.df_hourly.ffill()
            # اضافه کردن ستون Cluster به df_hourly
            self.features = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
            self.scaler = StandardScaler()
            self.scaler.fit(self.df_hourly[self.features])
            self.df_hourly['Cluster'] = self.kmeans.predict(self.scaler.transform(self.df_hourly[self.features]))
        except FileNotFoundError:
            print("The data file household_power_consumption.txt was not found!")
            sys.exit()

        # UI components
        self.init_ui()

        self.prediction_made = False
        self.last_prediction = None

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        model_info = QLabel("Clustering Model: KMeans (K=4, Silhouette Score=0.7243)")
        model_info.setAlignment(Qt.AlignCenter)
        model_info.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(model_info)

        self.prediction_label = QLabel("Please enter values and click Predict.")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 14px; color: green;")
        main_layout.addWidget(self.prediction_label)

        input_widget = QWidget()
        input_layout = QFormLayout(input_widget)
        self.inputs = {}
        sample_values = {
            'Global_active_power': '0.5-5 (e.g., 2.0 kW)',
            'Sub_metering_1': '0-50 (e.g., 10 Wh)',
            'Sub_metering_2': '0-50 (e.g., 5 Wh)',
            'Sub_metering_3': '0-50 (e.g., 15 Wh)'
        }

        for feature in self.features:
            self.inputs[feature] = QLineEdit()
            self.inputs[feature].setPlaceholderText(sample_values[feature])
            input_layout.addRow(f"{feature}:", self.inputs[feature])

        predict_button = QPushButton("Predict Cluster")
        predict_button.clicked.connect(self.predict_cluster)
        input_layout.addRow(predict_button)
        main_layout.addWidget(input_widget)

        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)
        input_layout.addRow(back_button)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.prediction_tab = QWidget()
        self.prediction_layout = QHBoxLayout(self.prediction_tab)
        self.tabs.addTab(self.prediction_tab, "Prediction Analysis")

        self.importance_tab = QWidget()
        self.importance_layout = QHBoxLayout(self.importance_tab)
        self.tabs.addTab(self.importance_tab, "Feature Importance")

        self.relationship_tab = QWidget()
        self.relationship_layout = QHBoxLayout(self.relationship_tab)
        self.tabs.addTab(self.relationship_tab, "Feature Relationship")

    def predict_cluster(self):
        try:
            input_data = []
            for feature in self.features:
                value = self.inputs[feature].text()
                if not value.replace('.', '', 1).isdigit():
                    self.prediction_label.setText(f"Error: Please enter a valid number for {feature}.")
                    self.prediction_label.setStyleSheet("font-size: 14px; color: red;")
                    return
                input_data.append(float(value))

            input_data = np.array(input_data).reshape(1, -1)
            input_scaled = self.scaler.transform(input_data)
            cluster = self.kmeans.predict(input_scaled)[0]
            self.last_prediction = (input_data[0], cluster)

            self.prediction_label.setText(f"Your input belongs to Cluster {cluster}.")
            self.prediction_label.setStyleSheet("font-size: 14px; color: green;")
            self.prediction_made = True
            self.update_charts()

        except Exception as e:
            self.prediction_label.setText(f"Error: {str(e)}")
            self.prediction_label.setStyleSheet("font-size: 14px; color: red;")

    def update_charts(self):
        for i in reversed(range(self.prediction_layout.count())):
            self.prediction_layout.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.importance_layout.count())):
            self.importance_layout.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.relationship_layout.count())):
            self.relationship_layout.itemAt(i).widget().setParent(None)

        if self.prediction_made:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.scatterplot(x=self.df_hourly.index, y='Global_active_power', hue='Cluster', 
                           data=self.df_hourly, palette='viridis', ax=ax1)
            if self.last_prediction:
                ax1.scatter(self.df_hourly.index[-1], self.last_prediction[0][0], 
                           c='red', s=100, marker='*', label='Your Prediction')
            ax1.set_title('Cluster Distribution Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Global Active Power (kW)')
            ax1.legend()
            canvas1 = FigureCanvas(fig1)
            self.prediction_layout.addWidget(canvas1)

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.violinplot(x='Cluster', y='Global_active_power', data=self.df_hourly, ax=ax2, palette='muted')
            if self.last_prediction:
                ax2.axhline(self.last_prediction[0][0], c='red', linestyle='--', label='Your Prediction')
            ax2.set_title('Power Distribution by Cluster')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Global Active Power (kW)')
            ax2.legend()
            canvas2 = FigureCanvas(fig2)
            self.prediction_layout.addWidget(canvas2)

            variances = pd.DataFrame({col: self.df_hourly.groupby('Cluster')[col].var() for col in self.features})
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            variances.mean().plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Average Feature Variance Across Clusters')
            ax3.set_xlabel('Features')
            ax3.set_ylabel('Variance')
            plt.xticks(rotation=45)
            canvas3 = FigureCanvas(fig3)
            self.importance_layout.addWidget(canvas3)

            fig4, ax4 = plt.subplots(figsize=(5, 4))
            cluster_counts = self.df_hourly['Cluster'].value_counts()
            ax4.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
            ax4.set_title('Cluster Distribution')
            canvas4 = FigureCanvas(fig4)
            self.importance_layout.addWidget(canvas4)

            fig5, ax5 = plt.subplots(figsize=(5, 4))
            sns.heatmap(self.df_hourly[self.features].corr(), annot=True, cmap='coolwarm', ax=ax5)
            ax5.set_title('Feature Correlation')
            canvas5 = FigureCanvas(fig5)
            self.relationship_layout.addWidget(canvas5)

            fig6, ax6 = plt.subplots(figsize=(5, 4))
            sns.scatterplot(x='Sub_metering_1', y='Global_active_power', hue='Cluster', 
                           data=self.df_hourly, palette='deep', ax=ax6)
            if self.last_prediction:
                ax6.scatter(self.last_prediction[0][1], self.last_prediction[0][0], 
                           c='red', s=100, marker='*', label='Your Prediction')
            ax6.set_title('Sub_metering_1 vs Global_active_power')
            ax6.set_xlabel('Sub_metering_1 (Wh)')
            ax6.set_ylabel('Global Active Power (kW)')
            ax6.legend()
            canvas6 = FigureCanvas(fig6)
            self.relationship_layout.addWidget(canvas6)

            plt.tight_layout()

    def close_and_go_back(self):
        self.close()
def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 10, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = PowerConsumptionUI(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()
