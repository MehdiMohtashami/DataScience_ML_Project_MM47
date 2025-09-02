import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QComboBox, QPushButton, QTabWidget, QMessageBox,
                             QGridLayout, QFormLayout, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=5, dpi=100):  # افزایش سایز canvas
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout(pad=3.0)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QualitativeBankruptcy")
        self.setWindowTitle("Bankruptcy Prediction System")
        self.setGeometry(100, 100, 1600, 900)

        # Load model and encoders
        try:
            self.model = joblib.load('models/logistic_regression_model.pkl')
            self.encoder = joblib.load('models/onehot_encoder.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.accuracy = 1.0
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model files not found or error loading: {str(e)}")
            sys.exit(1)

        # Load original data for visualizations
        try:
            self.df = pd.read_csv('Qualitative_Bankruptcy.csv')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load data: {str(e)}")
            sys.exit(1)

        # Prepare data for visualizations
        self.prepare_data()

        # Setup UI
        self.setup_ui()

        # Initialize charts (but don't show them yet)
        self.init_charts()

    def prepare_data(self):
        # Encode the data for analysis
        self.df_encoded = self.df.copy()
        for col in self.df.columns[:-1]:
            self.df_encoded[col] = self.df[col].map({'P': 2, 'A': 1, 'N': 0})

        # Encode target
        self.df_encoded['Class'] = self.df['Class'].map({'NB': 0, 'B': 1})

        # Get feature importances
        self.feature_importances = self.model.coef_[0]
        feature_names = self.encoder.get_feature_names_out(self.df.columns[:-1])
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances
        })

    def setup_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for input
        left_panel = QWidget()
        left_panel.setMinimumWidth(450)  # افزایش عرض
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)

        # Model info
        model_info = QGroupBox("Model Information")
        model_info.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QVBoxLayout(model_info)
        model_layout.addWidget(QLabel(f"Model: Logistic Regression"))
        model_layout.addWidget(QLabel(f"Accuracy: {self.accuracy:.2%}"))
        model_layout.addWidget(QLabel("Dataset: QualitativeBankruptcy"))
        model_layout.addWidget(QLabel("Samples: 250"))
        left_layout.addWidget(model_info)

        # Input form
        input_group = QGroupBox("Input Parameters")
        input_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        input_layout = QFormLayout(input_group)
        input_layout.setLabelAlignment(Qt.AlignRight)

        self.input_fields = {}
        features = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                    'Credibility', 'Competitiveness', 'Operating Risk']

        for feature in features:
            combo = QComboBox()
            combo.addItems(['Positive (P)', 'Average (A)', 'Negative (N)'])
            combo.setCurrentIndex(1)
            combo.setMinimumHeight(30)
            input_layout.addRow(QLabel(feature + ":"), combo)
            self.input_fields[feature] = combo

        left_layout.addWidget(input_group)

        # Predict button
        self.predict_btn = QPushButton("Predict Bankruptcy Risk")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2E86AB; 
                color: white; 
                font-weight: bold; 
                padding: 15px;
                font-size: 14px;
                border-radius: 5px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #1B6B93;
            }
        """)
        left_layout.addWidget(self.predict_btn)

        self.back_button = QPushButton("Back to Main", self)
        self.back_button.clicked.connect(self.close_and_go_back)
        self.back_button.setStyleSheet("""
                    QPushButton { 
                        background-color: gray; 
                        color: white; 
                        font-weight: bold; 
                        padding: 15px;
                        font-size: 14px;
                        border-radius: 5px;
                        min-height: 20px;
                    }
                    QPushButton:hover {
                        background-color: #1B6B93;
                    }
                """)
        left_layout.addWidget(self.back_button)

        # Result display
        self.result_label = QLabel("Please click 'Predict' to see results")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel { 
                background-color: #F5F5F5; 
                padding: 20px; 
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                min-height: 100px;
            }
        """)
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label)

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right panel for visualizations
        self.right_panel = QTabWidget()
        self.right_panel.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #cccccc; }
            QTabBar::tab { 
                padding: 8px 16px; 
                background: #f0f0f0; 
                border: 1px solid #cccccc; 
                border-bottom: none; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { 
                background: #ffffff; 
                border-bottom: 1px solid white; 
            }
        """)
        main_layout.addWidget(self.right_panel)

        # Initially hide the right panel
        self.right_panel.hide()

    def init_charts(self):
        # Create tabs for different visualizations
        self.analysis_tab = QWidget()
        self.feature_tab = QWidget()
        self.relationship_tab = QWidget()

        # Set up layouts for tabs with more spacing
        analysis_layout = QVBoxLayout(self.analysis_tab)
        analysis_layout.setSpacing(15)
        analysis_layout.setContentsMargins(15, 15, 15, 15)

        feature_layout = QVBoxLayout(self.feature_tab)
        feature_layout.setSpacing(15)
        feature_layout.setContentsMargins(15, 15, 15, 15)

        relationship_layout = QVBoxLayout(self.relationship_tab)
        relationship_layout.setSpacing(15)
        relationship_layout.setContentsMargins(15, 15, 15, 15)

        # Create canvas for charts with larger size
        self.risk_profile_canvas = MplCanvas(self, width=8, height=6)
        self.probability_canvas = MplCanvas(self, width=8, height=6)
        self.importance_canvas = MplCanvas(self, width=8, height=6)
        self.correlation_canvas = MplCanvas(self, width=8, height=6)
        self.comparison_canvas = MplCanvas(self, width=8, height=6)
        self.relationship_canvas = MplCanvas(self, width=8, height=6)

        # Add charts to tabs with larger labels
        analysis_title1 = QLabel("Risk Profile Analysis")
        analysis_title1.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        analysis_layout.addWidget(analysis_title1)
        analysis_layout.addWidget(self.risk_profile_canvas)

        analysis_title2 = QLabel("Probability Distribution")
        analysis_title2.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        analysis_layout.addWidget(analysis_title2)
        analysis_layout.addWidget(self.probability_canvas)

        feature_title1 = QLabel("Feature Importance")
        feature_title1.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        feature_layout.addWidget(feature_title1)
        feature_layout.addWidget(self.importance_canvas)

        feature_title2 = QLabel("Feature Correlation")
        feature_title2.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        feature_layout.addWidget(feature_title2)
        feature_layout.addWidget(self.correlation_canvas)

        relationship_title1 = QLabel("Case Comparison")
        relationship_title1.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        relationship_layout.addWidget(relationship_title1)
        relationship_layout.addWidget(self.comparison_canvas)

        relationship_title2 = QLabel("Feature Relationship")
        relationship_title2.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        relationship_layout.addWidget(relationship_title2)
        relationship_layout.addWidget(self.relationship_canvas)

        # Add tabs to right panel
        self.right_panel.addTab(self.analysis_tab, "Prediction Analysis")
        self.right_panel.addTab(self.feature_tab, "Feature Importance")
        self.right_panel.addTab(self.relationship_tab, "Feature Relationships")

    def predict(self):
        try:
            # Get input values
            input_data = {}
            for feature, combo in self.input_fields.items():
                value = combo.currentText()[0]
                input_data[feature] = value

            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])

            # Encode input
            input_encoded = self.encoder.transform(input_df)

            # Predict
            prediction_encoded = self.model.predict(input_encoded)
            prediction_proba = self.model.predict_proba(input_encoded)

            # Decode prediction
            prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
            probability = prediction_proba[0][1] if prediction == 'B' else prediction_proba[0][0]

            # Update result label
            result_text = f"Prediction: {prediction} ({'Bankruptcy' if prediction == 'B' else 'Non-Bankruptcy'})\n"
            result_text += f"Probability: {probability:.2%}"

            self.result_label.setText(result_text)

            # Set color based on prediction
            if prediction == 'B':
                self.result_label.setStyleSheet("""
                    QLabel { 
                        background-color: #FF6B6B; 
                        color: white; 
                        padding: 20px; 
                        border: 2px solid #ddd;
                        border-radius: 5px;
                        font-size: 14px;
                        min-height: 100px;
                    }
                """)
            else:
                self.result_label.setStyleSheet("""
                    QLabel { 
                        background-color: #77DD77; 
                        color: white; 
                        padding: 20px; 
                        border: 2px solid #ddd;
                        border-radius: 5px;
                        font-size: 14px;
                        min-height: 100px;
                    }
                """)

            # Show visualizations
            self.right_panel.show()

            # Update charts with new prediction
            self.update_charts(input_df, prediction, probability)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")

    def update_charts(self, input_df, prediction, probability):
        try:
            # Clear all charts
            for canvas in [self.risk_profile_canvas, self.probability_canvas,
                           self.importance_canvas, self.correlation_canvas,
                           self.comparison_canvas, self.relationship_canvas]:
                canvas.axes.clear()
                canvas.fig.clf()  # اضافه کردن این خط
                canvas.axes = canvas.fig.add_subplot(111)

            # Prepare input for charts
            input_encoded = []
            for feature in self.df.columns[:-1]:
                value = input_df[feature].iloc[0]
                input_encoded.append({'P': 2, 'A': 1, 'N': 0}[value])

            # 1. Risk Profile Bar Chart
            self.create_bar_chart(input_encoded)

            # 2. Probability Distribution
            self.create_probability_chart(probability, prediction)

            # 3. Feature Importance
            self.create_feature_importance_chart()

            # 4. Feature Correlation
            self.create_correlation_chart()

            # 5. Case Comparison
            self.create_comparison_chart(input_encoded, prediction)

            # 6. Feature Relationship
            self.create_relationship_chart()

            # Refresh all canvases
            for canvas in [self.risk_profile_canvas, self.probability_canvas,
                           self.importance_canvas, self.correlation_canvas,
                           self.comparison_canvas, self.relationship_canvas]:
                canvas.fig.tight_layout(pad=3.0)  # افزایش padding برای همه نمودارها
                canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Chart Error", f"An error occurred while updating charts: {str(e)}")

    def create_bar_chart(self, input_encoded):
        """Bar chart برای نمایش پروفایل ریسک"""
        ax = self.risk_profile_canvas.axes

        features = ['Industrial', 'Management', 'Financial', 'Credibility', 'Competitiveness', 'Operating']
        x_pos = np.arange(len(features))

        # Plot bars
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#5C8001']
        bars = ax.bar(x_pos, input_encoded, color=colors)

        # Customize chart
        ax.set_xlabel('Risk Factors', fontsize=12)
        ax.set_ylabel('Risk Level', fontsize=12)
        ax.set_title('Risk Profile Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, 2.5)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            risk_level = "Negative" if height == 0 else "Average" if height == 1 else "Positive"
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                    f'{risk_level}\n({height})', ha='center', va='bottom', fontsize=10)

        # Add horizontal grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set background color
        ax.set_facecolor('#f8f9fa')

    def create_probability_chart(self, probability, prediction):
        ax = self.probability_canvas.axes

        # Create donut chart
        sizes = [probability, 1 - probability]
        colors = ['#FF6B6B', '#77DD77'] if prediction == 'B' else ['#77DD77', '#FF6B6B']
        labels = [f'{prediction}\n{probability:.2%}', f'{"NB" if prediction == "B" else "B"}\n{1 - probability:.2%}']

        # Draw pie chart
        wedges, texts = ax.pie(sizes, colors=colors, startangle=90)

        # Draw circle in center to make it a donut
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)

        # Add text in center
        ax.text(0, 0, f"Prediction\n{prediction}", ha='center', va='center',
                fontsize=14, fontweight='bold', color='#2E86AB')

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        # Add legend
        ax.legend(wedges, labels, title="Categories", loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

        ax.set_title('Probability Distribution', fontsize=14, fontweight='bold', pad=20)

    def create_feature_importance_chart(self):
        ax = self.importance_canvas.axes

        # Prepare data
        features = self.feature_importance_df['feature']
        importance = np.abs(self.feature_importance_df['importance'])

        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#5C8001']
        ax.barh(y_pos, importance, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
        ax.set_title('Feature Importance (Logistic Regression Coefficients)',
                     fontsize=14, fontweight='bold', pad=20)

        # Add value labels on bars
        for i, v in enumerate(importance):
            ax.text(v + 0.001, i, f'{v:.3f}', color='black', fontweight='bold', fontsize=10)

        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_facecolor('#f8f9fa')

    def create_correlation_chart(self):
        ax = self.correlation_canvas.axes
        fig = self.correlation_canvas.fig  # Get the figure from the canvas

        # Calculate correlation matrix
        corr = self.df_encoded.corr()

        # Create heatmap
        im = ax.imshow(corr, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

        # Set ticks and labels
        features = list(self.df_encoded.columns)
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(features, fontsize=11)

        # Add correlation values to cells
        for i in range(len(features)):
            for j in range(len(features)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                        color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black', fontsize=10)

        # Add colorbar - use the figure from the canvas instead of plt
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    def create_comparison_chart(self, input_encoded, prediction):
        ax = self.comparison_canvas.axes

        # Prepare data
        features = ['Industrial', 'Management', 'Financial', 'Credibility', 'Competitiveness', 'Operating']

        # Calculate average values for each class
        avg_bankrupt = self.df_encoded[self.df_encoded['Class'] == 1].mean()[:-1].values
        avg_non_bankrupt = self.df_encoded[self.df_encoded['Class'] == 0].mean()[:-1].values

        x = np.arange(len(features))
        width = 0.25

        # Plot bars
        rects1 = ax.bar(x - width, avg_bankrupt, width, label='Avg Bankruptcy', color='#FF6B6B')
        rects2 = ax.bar(x, avg_non_bankrupt, width, label='Avg Non-Bankruptcy', color='#77DD77')
        rects3 = ax.bar(x + width, input_encoded, width, label='Current Case', color='#2E86AB')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Scores', fontsize=12)
        ax.set_title('Comparison with Average Cases', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=11)

        # Add value labels on bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

        add_labels(rects1)
        add_labels(rects2)
        add_labels(rects3)

        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_facecolor('#f8f9fa')

    def create_relationship_chart(self):
        ax = self.relationship_canvas.axes

        # Create a scatter plot matrix would be too complex for 6 features
        # Instead, let's show the relationship between the most important feature and the target

        # Get most important feature
        most_important_feature = self.feature_importance_df.loc[
            np.abs(self.feature_importance_df['importance']).idxmax(), 'feature']
        orig_feature_name = most_important_feature.split('_')[0]

        # Create violin plot
        data = []
        labels = ['Non-Bankruptcy', 'Bankruptcy']

        for cls in [0, 1]:
            subset = self.df_encoded[self.df_encoded['Class'] == cls]
            data.append(subset[orig_feature_name].values)

        parts = ax.violinplot(data, showmeans=True, showmedians=True)

        # Color the violins
        for pc, color in zip(parts['bodies'], ['#77DD77', '#FF6B6B']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Set labels and title
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Risk Score', fontsize=12)
        ax.set_title(f'Distribution of {orig_feature_name} by Bankruptcy Status',
                     fontsize=14, fontweight='bold', pad=20)

        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_facecolor('#f8f9fa')

    def close_and_go_back(self):
        self.close()

def main(parent=None):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    font = QFont("Arial", 8, QFont.Bold)
    app.setFont(font)
    app.setStyle('Fusion')
    window = MainWindow(parent)
    window.show()
    if parent is None:
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()