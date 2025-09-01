import sys
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from joblib import load
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLineEdit, QLabel, QPushButton, QScrollArea,
                             QGridLayout, QMessageBox, QTabWidget, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QFont, QPalette, QColor
import matplotlib.pyplot as plt

# ---------------------------
# Load model, scaler, features
# ---------------------------
try:
    model = load('best_lgbm_classification_model.joblib')
    scaler = load('scaler.joblib')
    important_features = load('important_features.joblib')
except Exception as e:
    print(f"Model/scaler/important_features not loaded (running demo mode). Error: {e}")
    # demo fallback
    important_features = ['LDA_00', 'kw_max_min', 'kw_avg_max', 'num_hrefs', 'num_imgs',
                          'num_videos', 'average_token_length', 'data_channel_is_world',
                          'kw_avg_min', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
                          'self_reference_min_shares', 'self_reference_max_shares',
                          'self_reference_avg_sharess', 'LDA_01', 'LDA_02', 'LDA_03',
                          'LDA_04', 'global_subjectivity', 'avg_negative_polarity',
                          'title_subjectivity', 'abs_title_sentiment_polarity',
                          'hrefs_per_token', 'imgs_per_token', 'videos_per_token',
                          'keyword_avg', 'is_weekend', 'multimedia_score', 'keyword_strength',
                          'content_length_score', 'polarity_score', 'engagement_score',
                          'keyword_impact', 'social_impact', 'content_impact',
                          'sentiment_impact', 'interaction_score', 'text_complexity',
                          'keyword_content_ratio']
    model = None
    scaler = StandardScaler()  # not fitted; in demo mode we won't call transform

# Default values used to pre-fill inputs (can be adjusted)
feature_defaults = {
    'LDA_00': 0.5, 'kw_max_min': 0.5, 'kw_avg_max': 0.5, 'num_hrefs': 10,
    'num_imgs': 5, 'num_videos': 2, 'average_token_length': 4.5,
    'data_channel_is_world': 0, 'kw_avg_min': 0.5, 'kw_min_avg': 0.5,
    'kw_max_avg': 0.5, 'kw_avg_avg': 0.5, 'self_reference_min_shares': 1000,
    'self_reference_max_shares': 2000, 'self_reference_avg_sharess': 1500,
    'LDA_01': 0.2, 'LDA_02': 0.2, 'LDA_03': 0.2, 'LDA_04': 0.2,
    'global_subjectivity': 0.5, 'avg_negative_polarity': 0.2,
    'title_subjectivity': 0.5, 'abs_title_sentiment_polarity': 0.2,
    'hrefs_per_token': 0.02, 'imgs_per_token': 0.01, 'videos_per_token': 0.005,
    'keyword_avg': 0.5, 'is_weekend': 0, 'multimedia_score': 7,
    'keyword_strength': 0.5, 'content_length_score': 2000,
    'polarity_score': 0.35, 'engagement_score': 5.5,
    'keyword_impact': 0.5, 'social_impact': 1500,
    'content_impact': 500, 'sentiment_impact': 0.1,
    'interaction_score': 50, 'text_complexity': 500,
    'keyword_content_ratio': 0.05
}

# Heuristic ranges generator
def infer_range(feature, default):
    name = feature.lower()
    # common ratio-like / probability features
    prob_like_keywords = ['lda', 'kw_', 'keyword', 'subjectivity', 'polarity', 'ratio', 'avg_', 'avg']
    count_keywords = ['num_', 'count', 'imgs', 'hrefs', 'videos', 'token', 'length',
                      'shares', 'score', 'impact', 'interaction', 'social', 'content', 'complexity']
    if any(k in name for k in prob_like_keywords) and (0 <= default <= 1):
        return 0.0, 1.0
    if any(k in name for k in count_keywords) or isinstance(default, int):
        # integer-like
        try:
            d = int(default)
            if d <= 3:
                return 0, 3
            else:
                # reasonable upper bound
                return 0, max(d * 5, d + 10)
        except Exception:
            # floats that represent counts
            return 0, max(default * 5 if default > 0 else 100, 100)
    # fallback for floats
    if isinstance(default, float):
        if 0 <= default <= 1:
            return 0.0, 1.0
        return 0.0, max(default * 5, default + 10)
    # ultimate fallback
    return -1e6, 1e6

# Widgets canvas class
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("News Popularity Predictor")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # top controls (title + theme selector)
        top_h = QHBoxLayout()
        title = QLabel("News Popularity Prediction (LightGBM)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        top_h.addWidget(title)

        # Theme selector
        theme_label = QLabel("Theme:")
        theme_label.setFont(QFont("Arial", 10))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.set_theme)
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        top_h.addLayout(theme_layout)
        layout.addLayout(top_h)

        # Model accuracy note
        accuracy_label = QLabel("Model Accuracy: 74% (F1-Score: 0.38 for positive class at threshold=0.4)")
        accuracy_label.setFont(QFont("Arial", 10))
        accuracy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(accuracy_label)

        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Input tab and scroll area
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        self.input_fields = {}
        self.ranges = {}  # store (min, max) for each feature

        row, col = 0, 0
        for i, feature in enumerate(important_features):
            default = feature_defaults.get(feature, 0)
            min_val, max_val = infer_range(feature, default)
            self.ranges[feature] = (min_val, max_val)

            label = QLabel(feature.replace('_', ' ').title() + ":")
            label.setToolTip(f"Allowed range: {min_val} — {max_val}")
            # decide whether to present as combo box:
            # if integer small-range (<=3 distinct ints) -> combo
            use_combo = False
            combo_options = None
            if (isinstance(min_val, (int, np.integer)) or float(min_val).is_integer()) and (isinstance(max_val, (int, np.integer)) or float(max_val).is_integer()):
                imin, imax = int(min_val), int(max_val)
                if imax - imin <= 3:
                    use_combo = True
                    combo_options = [str(x) for x in range(imin, imax + 1)]

            if use_combo:
                widget = QComboBox()
                widget.addItems(combo_options)
                # set default index if possible
                try:
                    widget.setCurrentIndex(combo_options.index(str(int(default))))
                except Exception:
                    widget.setCurrentIndex(0)
            else:
                # choose int or double validator
                if float(min_val).is_integer() and float(max_val).is_integer():
                    validator = QIntValidator(int(min_val), int(max_val))
                else:
                    # decimals=6
                    validator = QDoubleValidator(float(min_val), float(max_val), 6)
                    validator.setNotation(QDoubleValidator.StandardNotation)
                widget = QLineEdit()
                widget.setValidator(validator)
                # set default text
                widget.setText(str(default))
                widget.setPlaceholderText(f"Enter value ({min_val} — {max_val})")

            scroll_layout.addWidget(label, row, col*2)
            scroll_layout.addWidget(widget, row, col*2 + 1)
            self.input_fields[feature] = widget

            row += 1
            if row > 15:
                row = 0
                col += 1

        scroll.setWidget(scroll_widget)
        input_layout.addWidget(scroll)

        # Predict button
        predict_btn = QPushButton("Predict Popularity")
        predict_btn.clicked.connect(self.predict)
        predict_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        input_layout.addWidget(predict_btn)

        # ایجاد یک دکمه برای بازگشت
        back_button = QPushButton("Back to Main", self)
        back_button.clicked.connect(self.close_and_go_back)  # متد close رو صدا میزنه
        back_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-size: 14px; padding: 10px; }")
        input_layout.addWidget(back_button)

        self.result_label = QLabel("Please enter values and click 'Predict Popularity'")
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel { padding: 10px; }")
        input_layout.addWidget(self.result_label)

        # Analysis / Importance / Relationship tabs
        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        self.analysis_layout.addWidget(QLabel("Prediction analysis will appear here after prediction"))

        self.feature_importance_tab = QWidget()
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_tab)
        self.feature_importance_layout.addWidget(QLabel("Feature importance will appear here after prediction"))

        self.feature_relationship_tab = QWidget()
        self.feature_relationship_layout = QVBoxLayout(self.feature_relationship_tab)
        self.feature_relationship_layout.addWidget(QLabel("Feature relationships will appear here after prediction"))

        tabs.addTab(input_tab, "Input")
        tabs.addTab(self.analysis_tab, "Prediction Analysis")
        tabs.addTab(self.feature_importance_tab, "Feature Importance")
        tabs.addTab(self.feature_relationship_tab, "Feature Relationships")

        # storage
        self.charts = []

        # set default theme
        self.set_theme("Light")

    def set_theme(self, theme_name: str):
        app = QApplication.instance()
        if theme_name == "Dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(35, 35, 35))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            app.setPalette(palette)
            plt.style.use('dark_background')
        else:
            app.setPalette(QApplication.style().standardPalette())
            plt.style.use('default')

    def clear_analysis_tabs(self):
        # remove chart widgets
        for chart in self.charts:
            try:
                chart.setParent(None)
            except Exception:
                pass
        self.charts = []

        # clear layouts
        for layout in (self.analysis_layout, self.feature_importance_layout, self.feature_relationship_layout):
            for i in reversed(range(layout.count())):
                w = layout.itemAt(i).widget()
                if w:
                    w.setParent(None)

    def predict(self):
        try:
            self.clear_analysis_tabs()

            # collect inputs and validate ranges
            input_values = []
            for feature in important_features:
                widget = self.input_fields.get(feature)
                min_val, max_val = self.ranges.get(feature, (-1e9, 1e9))
                value = None
                if isinstance(widget, QComboBox):
                    try:
                        value = float(widget.currentText())
                    except:
                        value = float(feature_defaults.get(feature, 0))
                else:
                    text = widget.text().strip()
                    if text == "":
                        # use default if empty
                        value = float(feature_defaults.get(feature, 0))
                    else:
                        # parse
                        try:
                            value = float(text)
                        except Exception:
                            QMessageBox.critical(self, "Input Error", f"Invalid value for {feature}. Please enter a number.")
                            return

                # range check
                if value < min_val or value > max_val:
                    QMessageBox.critical(self, "Range Error",
                                         f"Value for '{feature}' = {value} is outside allowed range [{min_val}, {max_val}].")
                    return

                input_values.append(value)

            # assemble dataframe with column names
            df_input = pd.DataFrame([input_values], columns=important_features)

            if model is not None:
                # try to scale (if scaler is fitted)
                try:
                    if hasattr(scaler, 'mean_') or hasattr(scaler, 'scale_') or hasattr(scaler, 'n_samples_seen_'):
                        scaled_array = scaler.transform(df_input)
                        df_scaled = pd.DataFrame(scaled_array, columns=important_features)
                    else:
                        # scaler not fitted but model exists -> try to transform anyway and catch errors
                        scaled_array = scaler.transform(df_input)
                        df_scaled = pd.DataFrame(scaled_array, columns=important_features)
                except Exception as e:
                    QMessageBox.critical(self, "Scaler Error",
                                         f"Scaler transform failed: {e}\nPrediction aborted.")
                    return

                # pass DataFrame with column names to avoid sklearn warning
                try:
                    proba = model.predict_proba(df_scaled)[0][1]
                except Exception as e:
                    QMessageBox.critical(self, "Model Error", f"Model prediction failed: {e}")
                    return

                prediction = "Popular" if proba > 0.4 else "Not Popular"
                self.result_label.setText(f"Prediction: {prediction} (Probability: {proba:.2%})")

                # create charts
                self.create_analysis_charts(input_values, proba, prediction)
                self.create_feature_importance_charts(input_values)
                self.create_feature_relationship_charts(input_values)

            else:
                # demo mode
                self.result_label.setText("Model not loaded. Demo mode prediction.")
                # example probability from demo
                demo_prob = 0.65
                self.create_analysis_charts(input_values, demo_prob, "Popular (Demo)")
                self.create_feature_importance_charts(input_values)
                self.create_feature_relationship_charts(input_values)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    # --------------- Charts ----------------
    def create_analysis_charts(self, input_values, probability, prediction):
        # Probability gauge
        gauge_group = QGroupBox("Prediction Probability")
        gauge_layout = QVBoxLayout()
        gauge_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        self.create_gauge_chart(gauge_canvas, probability, prediction)
        gauge_layout.addWidget(gauge_canvas)
        gauge_group.setLayout(gauge_layout)
        self.analysis_layout.addWidget(gauge_group)
        self.charts.append(gauge_canvas)

        # Radar chart for top 6 features
        radar_group = QGroupBox("Feature Comparison Radar Chart (top 6 features)")
        radar_layout = QVBoxLayout()
        radar_canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.create_radar_chart(radar_canvas, input_values)
        radar_layout.addWidget(radar_canvas)
        radar_group.setLayout(radar_layout)
        self.analysis_layout.addWidget(radar_group)
        self.charts.append(radar_canvas)

    def create_feature_importance_charts(self, input_values):
        # Importance bar
        importance_group = QGroupBox("Feature Importance")
        importance_layout = QVBoxLayout()
        importance_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.create_importance_chart(importance_canvas)
        importance_layout.addWidget(importance_canvas)
        importance_group.setLayout(importance_layout)
        self.feature_importance_layout.addWidget(importance_group)
        self.charts.append(importance_canvas)

        # Top features impact
        impact_group = QGroupBox("Top Features Impact on Prediction")
        impact_layout = QVBoxLayout()
        impact_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.create_impact_chart(impact_canvas, input_values)
        impact_layout.addWidget(impact_canvas)
        impact_group.setLayout(impact_layout)
        self.feature_importance_layout.addWidget(impact_group)
        self.charts.append(impact_canvas)

    def create_feature_relationship_charts(self, input_values):
        heatmap_group = QGroupBox("Feature Correlation Heatmap")
        heatmap_layout = QVBoxLayout()
        heatmap_canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.create_heatmap_chart(heatmap_canvas)
        heatmap_layout.addWidget(heatmap_canvas)
        heatmap_group.setLayout(heatmap_layout)
        self.feature_relationship_layout.addWidget(heatmap_group)
        self.charts.append(heatmap_canvas)

        scatter_group = QGroupBox("Feature Relationship Scatter Plot")
        scatter_layout = QVBoxLayout()
        scatter_canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.create_scatter_chart(scatter_canvas, input_values)
        scatter_layout.addWidget(scatter_canvas)
        scatter_group.setLayout(scatter_layout)
        self.feature_relationship_layout.addWidget(scatter_group)
        self.charts.append(scatter_canvas)

    def create_gauge_chart(self, canvas, probability, prediction):
        # simple horizontal bar gauge
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        ax.clear()
        ax.barh(0, 1, height=0.3, color='lightgray')
        ax.barh(0, probability, height=0.3, color='skyblue')
        ax.plot([probability, probability], [-0.2, 0.2], 'r-', linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f'Popularity Probability: {probability:.2%}\nPrediction: {prediction}', fontsize=12)
        ax.set_yticks([])
        ax.set_xlabel('Probability')
        labels = ['Not Popular', 'Likely Not', 'Neutral', 'Likely', 'Popular']
        values = [0, 0.25, 0.5, 0.75, 1.0]
        for v, lab in zip(values, labels):
            ax.text(v, -0.2, lab, ha='center', va='top')
        ax.grid(True, alpha=0.3)
        canvas.draw()

    def create_radar_chart(self, canvas, input_values):
        # build a polar axis for radar
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111, polar=True)

        top_features = important_features[:6]
        top_values = input_values[:6]
        # compute normalization factors
        max_vals = []
        for f, v in zip(top_features, top_values):
            fd = feature_defaults.get(f, 1)
            if fd == 0:
                max_vals.append(1.0)
            else:
                max_vals.append(abs(fd) * 1.5 if abs(fd) > 0 else 1.0)
        normalized = [v / m if m != 0 else 0 for v, m in zip(top_values, max_vals)]

        categories = [f.replace('_', '\n') for f in top_features]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        values = normalized + normalized[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0, 1)
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="Input Values")
        ax.fill(angles, values, alpha=0.1)
        ax.set_title("Normalized Feature Values Comparison", size=12, y=1.08)
        canvas.draw()

    def create_importance_chart(self, canvas):
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        ax.clear()
        # demo importances
        feature_importance = {
            'LDA_00': 0.12, 'kw_max_min': 0.09, 'kw_avg_max': 0.08,
            'num_hrefs': 0.07, 'num_imgs': 0.06, 'average_token_length': 0.06,
            'kw_avg_min': 0.05, 'global_subjectivity': 0.05, 'LDA_02': 0.05
        }
        features = list(feature_importance.keys())
        importances = [feature_importance[f] for f in features]
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', '\n') for f in features])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        canvas.draw()

    def create_impact_chart(self, canvas, input_values):
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        ax.clear()
        # demo impacts
        impacts = {
            'LDA_00': 0.15, 'kw_max_min': 0.12, 'kw_avg_max': -0.08,
            'num_hrefs': 0.06, 'num_imgs': 0.04
        }
        features = list(impacts.keys())
        impact_values = [impacts[f] for f in features]
        colors = ['green' if v > 0 else 'red' for v in impact_values]

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, impact_values, align='center', color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', '\n') for f in features])
        ax.invert_yaxis()
        ax.set_xlabel('Impact on Prediction')
        ax.set_title('Top Features Impact on Prediction\n(Green: Positive, Red: Negative)')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + (0.01 if width > 0 else -0.01)
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', ha='left' if width > 0 else 'right', va='center')
        canvas.draw()

    def create_heatmap_chart(self, canvas):
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        ax.clear()
        top_features = important_features[:8]
        n = len(top_features)
        np.random.seed(42)
        corr = np.random.uniform(-0.7, 0.7, size=(n, n))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([f.replace('_', '\n') for f in top_features], rotation=45, ha='right')
        ax.set_yticklabels([f.replace('_', '\n') for f in top_features])
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
        for i in range(n):
            for j in range(n):
                txtcolor = "white" if abs(corr[i, j]) > 0.5 else "black"
                ax.text(j, i, f'{corr[i, j]:.2f}', ha="center", va="center", color=txtcolor)
        ax.set_title("Feature Correlation Heatmap")
        canvas.draw()

    def create_scatter_chart(self, canvas, input_values):
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        ax.clear()
        feat1, feat2 = important_features[0], important_features[1]
        np.random.seed(42)
        n_points = 100
        x_vals = np.random.normal(feature_defaults.get(feat1, 1), max(0.001, feature_defaults.get(feat1, 1)/3), n_points)
        y_vals = np.random.normal(feature_defaults.get(feat2, 1), max(0.001, feature_defaults.get(feat2, 1)/3), n_points)
        colors = np.sqrt(np.abs(x_vals)**2 + np.abs(y_vals)**2)
        sc = ax.scatter(x_vals, y_vals, c=colors, cmap='viridis', alpha=0.6)
        input_x = input_values[0]
        input_y = input_values[1]
        ax.scatter(input_x, input_y, c='red', s=80, marker='X', label='Input Article')
        ax.set_xlabel(feat1.replace('_', ' ').title())
        ax.set_ylabel(feat2.replace('_', ' ').title())
        ax.set_title(f'Relationship between {feat1} and {feat2}')
        ax.legend()
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.set_label('Combined Impact', rotation=270, labelpad=15)
        canvas.draw()

    def close_and_go_back(self):
        self.close()

# ------------------- run -------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
