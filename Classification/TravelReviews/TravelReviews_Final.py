import sys
import pandas as pd
import numpy as np
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QDoubleSpinBox, QPushButton, QTabWidget, QFormLayout)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns


class TravelReviewsUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Heart Disease Prediction")
        self.setWindowTitle("TravelReviews Classifier")
        self.setGeometry(100, 100, 1200, 800)

        # Load model and scaler
        self.model = joblib.load('travel_reviews_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

        # Load dataset for feature relationship analysis
        self.df = pd.read_csv('tripadvisor_review.csv')
        self.df_numeric = self.df.drop('User ID', axis=1)
        self.labels = pd.qcut(self.df_numeric.mean(axis=1), q=3, labels=['Low', 'Medium', 'High'])

        # Initialize UI
        self.init_ui()
        self.prediction = None
        self.inputs = []

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Model info
        model_info = QLabel("Model: Logistic Regression | Accuracy: 0.98")
        model_info.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(model_info, alignment=Qt.AlignCenter)

        # Input section
        input_widget = QWidget()
        input_layout = QFormLayout(input_widget)
        self.spin_boxes = []
        for i in range(1, 11):
            spin_box = QDoubleSpinBox()
            spin_box.setRange(0.0, 4.0)
            spin_box.setSingleStep(0.1)
            spin_box.setValue(2.0)
            label = QLabel(f"Category {i} (Enter rating 0 to 4, e.g., Art Galleries, Dance Clubs, etc.):")
            input_layout.addRow(label, spin_box)
            self.spin_boxes.append(spin_box)
        main_layout.addWidget(input_widget)

        # Predict button
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict)
        main_layout.addWidget(predict_button, alignment=Qt.AlignCenter)

        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        main_layout.addWidget(back_button, alignment=Qt.AlignCenter)


        # Tabs for charts
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Prediction Analysis Tab
        self.prediction_tab = QWidget()
        self.prediction_layout = QHBoxLayout(self.prediction_tab)
        self.tabs.addTab(self.prediction_tab, "Prediction Analysis")

        # Feature Importance Tab
        self.importance_tab = QWidget()
        self.importance_layout = QHBoxLayout(self.importance_tab)
        self.tabs.addTab(self.importance_tab, "Feature Importance")

        # Feature Relationship Tab
        self.relationship_tab = QWidget()
        self.relationship_layout = QHBoxLayout(self.relationship_tab)
        self.tabs.addTab(self.relationship_tab, "Feature Relationship")

    def predict(self):
        # Get inputs
        self.inputs = [spin_box.value() for spin_box in self.spin_boxes]
        input_data = self.scaler.transform([self.inputs])
        self.prediction = self.model.predict(input_data)[0]

        # Clear previous charts
        self.clear_layout(self.prediction_layout)
        self.clear_layout(self.importance_layout)
        self.clear_layout(self.relationship_layout)

        # Update charts
        self.update_prediction_charts()
        self.update_importance_charts()
        self.update_relationship_charts()

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def update_prediction_charts(self):
        # Radar Chart for user input
        fig1 = Figure(figsize=(5, 4))
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111, polar=True)
        categories = [f'Cat {i}' for i in range(1, 11)]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        values = self.inputs + self.inputs[:1]
        ax1.fill(angles, values, color='blue', alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_title(f'User Input Ratings (Prediction: {self.prediction})')
        self.prediction_layout.addWidget(canvas1)

        # Bar Plot for class distribution
        fig2 = Figure(figsize=(5, 4))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        class_counts = self.labels.value_counts()
        classes = ['Low', 'Medium', 'High']
        counts = [class_counts.get(c, 0) for c in classes]
        user_class = [1 if c == self.prediction else 0 for c in classes]
        bar_width = 0.35
        x = np.arange(len(classes))
        ax2.bar(x - bar_width / 2, counts, bar_width, label='Dataset Distribution', color='skyblue')
        ax2.bar(x + bar_width / 2, user_class, bar_width, label='User Prediction', color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes)
        ax2.set_title('Class Distribution vs User Prediction')
        ax2.legend()
        self.prediction_layout.addWidget(canvas2)

    def update_importance_charts(self):
        # Bar Plot for feature importance
        fig1 = Figure(figsize=(5, 4))
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        importance = pd.DataFrame({
            'Feature': [f'Category {i}' for i in range(1, 11)],
            'Coefficient': abs(self.model.coef_[0])
        }).sort_values(by='Coefficient', ascending=False)
        sns.barplot(x='Coefficient', y='Feature', data=importance, ax=ax1, palette='viridis')
        ax1.set_title('Feature Importance (Logistic Regression)')
        self.importance_layout.addWidget(canvas1)

        # Pie Chart for top features
        fig2 = Figure(figsize=(5, 4))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        top_features = importance.head(4)
        ax2.pie(top_features['Coefficient'], labels=top_features['Feature'], autopct='%1.1f%%',
                colors=sns.color_palette('pastel'))
        ax2.set_title('Top Feature Contributions')
        self.importance_layout.addWidget(canvas2)

    def update_relationship_charts(self):
        # Scatter Plot for Category 3 vs Category 6
        fig1 = Figure(figsize=(5, 4))
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        sns.scatterplot(x=self.df_numeric['Category 3'], y=self.df_numeric['Category 6'], hue=self.labels, ax=ax1)
        if self.prediction:
            ax1.scatter(self.inputs[2], self.inputs[5], c='red', s=100, marker='*', label='User Input')
        ax1.set_title('Category 3 vs Category 6')
        ax1.legend()
        self.relationship_layout.addWidget(canvas1)

        # Correlation Heatmap
        fig2 = Figure(figsize=(5, 4))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        corr = self.df_numeric.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
        ax2.set_title('Feature Correlation Heatmap')
        self.relationship_layout.addWidget(canvas2)

    def close_and_go_back(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    parent = None
    window = TravelReviewsUI(parent)
    window.show()
    sys.exit(app.exec_())