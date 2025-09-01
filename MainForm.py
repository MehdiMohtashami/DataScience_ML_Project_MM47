import sys
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QListWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QFileDialog, QListWidgetItem,
    QSplitter, QMessageBox, QSizePolicy
)

ROOT_DIR = Path(__file__).resolve().parent

CATEGORIES = ["Classification", "Clustering", "Regression"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}


class Dataset:
    def __init__(self, name: str, category: str, path: Path):
        self.name = name
        self.category = category
        self.path = path
        self.final_py: Optional[Path] = None
        self.hover_img: Optional[Path] = None
        self.summary_file: Optional[Path] = None
        self.gallery_imgs: List[Path] = []
        self._discover()

    def _discover(self):
        finals = list(self.path.glob("*_Final.py"))
        if finals:
            self.final_py = finals[0]
        for p in self.path.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem.lower().endswith("_hover"):
                self.hover_img = p
                break
        for p in self.path.iterdir():
            if p.is_file() and p.name.lower().startswith("summery"):
                self.summary_file = p
                break
        if not self.summary_file:
            for p in self.path.iterdir():
                if p.is_file() and p.name.lower().startswith("summary"):
                    self.summary_file = p
                    break
        for p in self.path.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                if self.hover_img and p.resolve() == self.hover_img.resolve():
                    continue
                self.gallery_imgs.append(p)


def discover_all(root: Path) -> List[Dataset]:
    datasets = []
    for cat in CATEGORIES:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        for sub in sorted(cat_dir.iterdir()):
            if sub.is_dir():
                datasets.append(Dataset(sub.name, cat, sub))
    return datasets


class HubWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle("ML Datasets Hub")
        self.resize(1400, 860)

        self.root = root
        self.datasets: List[Dataset] = discover_all(self.root)
        self.dataset_map = {f"{d.category}/{d.name}": d for d in self.datasets}
        self.is_dark_mode = False

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        self.bg_path = None
        pic_dir = self.root / "Picture"
        if pic_dir.exists():
            pics = [p for p in pic_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if pics:
                self.bg_path = str(pics[0].as_posix())
                central.setStyleSheet(f"QWidget{{background-image: url('{self.bg_path}'); background-repeat: no-repeat; background-position: center;}}")

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)

        btn_box = QWidget()
        btn_layout = QHBoxLayout(btn_box)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_class = QPushButton("Classification")
        self.btn_cluster = QPushButton("Clustering")
        self.btn_regress = QPushButton("Regression")
        for b in (self.btn_class, self.btn_cluster, self.btn_regress):
            b.setMinimumHeight(36)
            btn_layout.addWidget(b)
        left_layout.addWidget(btn_box)

        self.search_line = QLineEdit()
        self.search_line.setPlaceholderText("Search dataset name...")
        self.search_line.setStyleSheet('QLineEdit{font-size: 16px}')
        left_layout.addWidget(self.search_line)

        self.dataset_list = QListWidget()
        self.dataset_list.setAlternatingRowColors(True)
        left_layout.addWidget(self.dataset_list, stretch=1)

        main_layout.addWidget(left_widget, 2)

        middle_split = QSplitter(Qt.Vertical)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setIconSize(QSize(320, 200))
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setSpacing(8)
        middle_split.addWidget(self.gallery)

        self.summ_text = QTextEdit()
        self.summ_text.setReadOnly(True)
        middle_split.addWidget(self.summ_text)
        middle_split.setSizes([520, 320])

        main_layout.addWidget(middle_split, 5)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.welcome = QLabel("<h2>Welcome to the Datasets Hub.</h2><div>Select a category or type and search.</div>")
        self.welcome.setWordWrap(True)
        self.welcome.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_layout.addWidget(self.welcome)

        self.hover_label = QLabel("Hover preview")
        self.hover_label.setAlignment(Qt.AlignCenter)
        self.hover_label.setMinimumHeight(260)
        self.hover_label.setStyleSheet("QLabel{border: 1px dashed rgba(0,0,0,0.2); padding: 6px;}")
        right_layout.addWidget(self.hover_label)

        self.btn_goto = QPushButton("GO TO")
        self.btn_save_charts = QPushButton("Save Charts")
        # self.btn_save_charts.setStyleSheet('QPushButton{background: #E0FFFF}')##B2FFFF
        self.btn_save_summary = QPushButton("Save Summery")
        # self.btn_save_summary.setStyleSheet('QPushButton{background: #B2FFFF}')##A5E3E0
        self.btn_toggle_dark = QPushButton("Toggle Dark Mode")
        # self.btn_toggle_dark.setStyleSheet('QPushButton{background: #A5E3E0}')##A5E3E0
        for b in (self.btn_goto, self.btn_save_charts, self.btn_save_summary, self.btn_toggle_dark):
            b.setMinimumHeight(38)
            right_layout.addWidget(b)

        right_layout.addStretch(1)
        main_layout.addWidget(right_widget, 2)

        self.btn_class.clicked.connect(lambda: self.load_category("Classification"))
        # self.btn_class.setStyleSheet('QPushButton{background: #A5E3E0; font-size: 20px;}')
        self.btn_cluster.clicked.connect(lambda: self.load_category("Clustering"))#7FFFD4
        # self.btn_cluster.setStyleSheet('QPushButton{background: #7FFFD4; font-size: 20px;}')
        self.btn_regress.clicked.connect(lambda: self.load_category("Regression"))#A5E3E0
        # self.btn_regress.setStyleSheet('QPushButton{background: #96DED1; font-size: 20px;}')

        self.search_line.textChanged.connect(self.search_datasets)

        self.dataset_list.itemClicked.connect(self.on_dataset_selected)
        self.dataset_list.itemDoubleClicked.connect(self.open_selected_dataset)

        self.btn_goto.clicked.connect(self.open_selected_dataset)
        self.btn_save_charts.clicked.connect(self.save_charts)
        self.btn_save_summary.clicked.connect(self.save_summary)
        self.btn_toggle_dark.clicked.connect(self.toggle_dark_mode)

        self.gallery.itemDoubleClicked.connect(self.open_gallery_image)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)

        self.load_all_datasets()
        self.update_clock()

    def update_clock(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.setWindowTitle(f"ML Datasets Hub - {current_time}")

    def toggle_dark_mode(self):
        self.is_dark_mode = not self.is_dark_mode

        dark_color = "#2b2b2b"
        text_color = "#f0f0f0"
        mid_color = "#3c3c3c"
        button_color = "#444444"
        button_hover = "#555555"
        button_pressed = "#333333"
        border_color = "#555555"

        from PyQt5.QtGui import QPalette, QColor

        if self.is_dark_mode:
            # ---------- QPalette ----------
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(dark_color))
            palette.setColor(QPalette.WindowText, QColor(text_color))
            palette.setColor(QPalette.Base, QColor(mid_color))
            palette.setColor(QPalette.AlternateBase, QColor("#454545"))
            palette.setColor(QPalette.Text, QColor(text_color))
            palette.setColor(QPalette.Button, QColor(button_color))
            palette.setColor(QPalette.ButtonText, QColor(text_color))
            palette.setColor(QPalette.Highlight, QColor(button_hover))
            palette.setColor(QPalette.HighlightedText, QColor(text_color))
            self.setPalette(palette)

            # ---------- Widgets ----------
            for w in [self.centralWidget(), self.gallery, self.summ_text, self.search_line,
                      self.hover_label]:
                w.setStyleSheet(f"background-color: {mid_color}; color: {text_color}; border:1px solid {border_color};")
                w.setAutoFillBackground(True)

            # ---------- Buttons ----------
            for b in [self.btn_class, self.btn_cluster, self.btn_regress,
                      self.btn_goto, self.btn_save_charts, self.btn_save_summary, self.btn_toggle_dark]:
                b.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {button_color};
                        color: {text_color};
                        border: 1px solid {border_color};
                        padding: 6px;
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        background-color: {button_hover};
                    }}
                    QPushButton:pressed {{
                        background-color: {button_pressed};
                    }}
                """)

            # ---------- QListWidget ----------
            self.dataset_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: {mid_color};
                    alternate-background-color: #454545;
                    color: {text_color};
                    border: 1px solid {border_color};
                }}
                QListWidget::item:selected {{
                    background-color: {button_hover};
                    color: {text_color};
                }}
            """)
            # Re-apply items color
            for i in range(self.dataset_list.count()):
                item = self.dataset_list.item(i)
                item.setForeground(QColor(text_color))

            # ---------- Splitter ----------
            self.setStyleSheet(f"""
                QSplitter::handle {{
                    background-color: #444444;
                }}
                QSplitter::handle:hover {{
                    background-color: #666666;
                }}
            """)
            # hover_label border
            self.hover_label.setStyleSheet("QLabel{border: 1px dashed rgba(255,255,255,0.3); padding: 6px;}")
        else:
            # ---------- روشن ----------
            self.setPalette(QApplication.style().standardPalette())
            # Reset all widgets
            for w in [self.centralWidget(), self.gallery, self.summ_text, self.search_line,
                      self.hover_label, self.dataset_list]:
                w.setStyleSheet("")
                w.setAutoFillBackground(False)

            for b in [self.btn_class, self.btn_cluster, self.btn_regress,
                      self.btn_goto, self.btn_save_charts, self.btn_save_summary, self.btn_toggle_dark]:
                b.setStyleSheet("")

            # Reset QListWidget items
            for i in range(self.dataset_list.count()):
                item = self.dataset_list.item(i)
                item.setForeground(QApplication.palette().color(QPalette.Text))

            # Reset hover_label border
            self.hover_label.setStyleSheet("QLabel{border: 1px dashed rgba(0,0,0,0.3); padding: 6px;}")

            # Restore background image if exists
            if self.bg_path:
                self.centralWidget().setStyleSheet(
                    f"QWidget{{background-image: url('{self.bg_path}'); background-repeat: no-repeat; background-position: center;}}"
                )

    def load_all_datasets(self):
        self.dataset_list.clear()
        keys = sorted(self.dataset_map.keys())
        for key in keys:
            ds = self.dataset_map[key]
            item = QListWidgetItem(f"{ds.category} / {ds.name}")
            if ds.hover_img and ds.hover_img.exists():
                item.setIcon(QIcon(str(ds.hover_img)))
            item.setData(Qt.UserRole, key)
            self.dataset_list.addItem(item)

    def load_category(self, category: str):
        self.dataset_list.clear()
        for key, ds in sorted(self.dataset_map.items()):
            if ds.category.lower() == category.lower():
                item = QListWidgetItem(ds.name)
                if ds.hover_img and ds.hover_img.exists():
                    item.setIcon(QIcon(str(ds.hover_img)))
                item.setData(Qt.UserRole, key)
                self.dataset_list.addItem(item)
        self.clear_preview()

    def search_datasets(self):
        query = self.search_line.text().strip().lower()
        self.dataset_list.clear()
        if not query:
            self.load_all_datasets()
            return
        for key, ds in sorted(self.dataset_map.items()):
            if query in ds.name.lower() or query in ds.category.lower():
                item = QListWidgetItem(f"{ds.category} / {ds.name}")
                if ds.hover_img and ds.hover_img.exists():
                    item.setIcon(QIcon(str(ds.hover_img)))
                item.setData(Qt.UserRole, key)
                self.dataset_list.addItem(item)
        self.clear_preview()

    def clear_preview(self):
        self.hover_label.setText("Hover preview")
        self.hover_label.setPixmap(QPixmap())
        self.gallery.clear()
        self.summ_text.clear()

    def on_dataset_selected(self, item: QListWidgetItem):
        key = item.data(Qt.UserRole)
        if not key:
            return
        ds = self.dataset_map.get(key)
        if not ds:
            return
        if ds.hover_img and ds.hover_img.exists():
            try:
                pix = QPixmap(str(ds.hover_img))
                if not pix.isNull():
                    scaled = pix.scaled(self.hover_label.width(), self.hover_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.hover_label.setPixmap(scaled)
                else:
                    self.hover_label.setText("No hover image")
            except Exception:
                self.hover_label.setText("No hover image")
        else:
            self.hover_label.setText("No hover image")

        self.gallery.clear()
        for p in ds.gallery_imgs:
            if p.exists():
                icon = QIcon(str(p))
                gw_item = QListWidgetItem(icon, p.name)
                gw_item.setData(Qt.UserRole, str(p))
                self.gallery.addItem(gw_item)

        self.summ_text.clear()
        if ds.summary_file and ds.summary_file.exists():
            try:
                txt = ds.summary_file.read_text(encoding="utf-8", errors="ignore")
                self.summ_text.setPlainText(txt)
            except Exception:
                self.summ_text.setPlainText("[Error reading summary file]")
        else:
            self.summ_text.setPlainText("[No Summery / Summary file found]")

    def open_selected_dataset(self, *args):
        item = self.dataset_list.currentItem()
        if not item:
            QMessageBox.information(self, "Open Dataset", "Select a dataset first.")
            return
        key = item.data(Qt.UserRole)
        ds = self.dataset_map.get(key)
        if not ds:
            QMessageBox.warning(self, "Open Dataset", "Dataset not found.")
            return
        if not ds.final_py or not ds.final_py.exists():
            QMessageBox.warning(self, "Open Dataset", "No *_Final.py found for this dataset.")
            return
        try:
            python_exec = sys.executable or "python"
            subprocess.Popen([python_exec, str(ds.final_py)], cwd=str(ds.path))
        except Exception as e:
            QMessageBox.critical(self, "Open Dataset", f"Failed to open: {e}")

    def save_charts(self):
        item = self.dataset_list.currentItem()
        if not item:
            QMessageBox.information(self, "Save Charts", "Select a dataset first.")
            return
        key = item.data(Qt.UserRole)
        ds = self.dataset_map.get(key)
        if not ds:
            return
        if not ds.gallery_imgs:
            QMessageBox.information(self, "Save Charts", "No chart images found for this dataset.")
            return
        dest = QFileDialog.getExistingDirectory(self, "Choose folder to save charts")
        if not dest:
            return
        copied = 0
        for p in ds.gallery_imgs:
            try:
                shutil.copy2(str(p), str(Path(dest) / p.name))
                copied += 1
            except Exception:
                pass
        QMessageBox.information(self, "Save Charts", f"Copied {copied} image(s) to:\n{dest}")

    def save_summary(self):
        item = self.dataset_list.currentItem()
        if not item:
            QMessageBox.information(self, "Save Summery", "Select a dataset first.")
            return
        key = item.data(Qt.UserRole)
        ds = self.dataset_map.get(key)
        if not ds:
            return
        if ds.summary_file and ds.summary_file.exists():
            suggested = f"{ds.name}_Summery.txt"
            fn, _ = QFileDialog.getSaveFileName(self, "Save Summery as", suggested, "Text files (*.txt);;Markdown (*.md);;All files (*)")
            if not fn:
                return
            try:
                shutil.copy2(str(ds.summary_file), fn)
                QMessageBox.information(self, "Save Summery", "Saved successfully.")
                return
            except Exception:
                pass
        fn, _ = QFileDialog.getSaveFileName(self, "Save Summery as", f"{ds.name}_Summery.txt", "Text files (*.txt);;All files (*)")
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(self.summ_text.toPlainText())
            QMessageBox.information(self, "Save Summery", "Saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Summery", f"Failed to save: {e}")

    def open_gallery_image(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "Open Image", "File not found.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QMessageBox.warning(self, "Open Image", f"Could not open image: {e}")


def main():
    app = QApplication(sys.argv)
    w = HubWindow(ROOT_DIR)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()