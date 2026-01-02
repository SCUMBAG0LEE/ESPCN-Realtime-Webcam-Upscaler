import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QPushButton, QFileDialog, QMessageBox, 
                             QGridLayout, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QRect, pyqtSignal, QPoint
from PIL import Image

class ImageLabel(QLabel):
    """ Custom Label that handles mouse selection """
    selection_changed = pyqtSignal(QRect) 

    def __init__(self):
        super().__init__()
        self.selection_rect = QRect()
        self.is_selecting = False
        self.start_point = None
        self.setMouseTracking(False) 

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.is_selecting = True
            self.selection_rect = QRect()
            self.update() 

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.selection_rect = QRect(self.start_point, event.pos()).normalized()
            self.update()
            self.selection_changed.emit(self.selection_rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            self.selection_changed.emit(self.selection_rect)

    def paintEvent(self, event):
        super().paintEvent(event) 
        if not self.selection_rect.isNull():
            painter = QPainter(self)
            pen = QPen(QColor("red"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)

    def set_selection(self, rect):
        self.selection_rect = rect
        self.update()

class ArrowOverlay(QWidget):
    """ Transparent layer for drawing arrows on top of images """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.active_rect = QRect()
        self.top_labels = []
        self.zoom_labels = []
        self.show_arrows = False

    def update_geometry_data(self, rect, top_labels, zoom_labels):
        self.active_rect = rect
        self.top_labels = top_labels
        self.zoom_labels = zoom_labels
        self.show_arrows = True
        self.update() 

    def paintEvent(self, event):
        if not self.show_arrows or self.active_rect.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(QColor(0, 120, 255, 200), 3, Qt.PenStyle.SolidLine) 
        painter.setPen(pen)
        painter.setBrush(QColor(0, 120, 255, 200))

        for i in range(3):
            if i >= len(self.top_labels) or i >= len(self.zoom_labels): continue
            
            top_lbl = self.top_labels[i]
            bot_lbl = self.zoom_labels[i]

            # Coordinate mapping
            top_origin = top_lbl.mapTo(self, QPoint(0,0))
            start_x = top_origin.x() + self.active_rect.center().x()
            start_y = top_origin.y() + self.active_rect.bottom()

            bot_origin = bot_lbl.mapTo(self, QPoint(0,0))
            end_x = bot_origin.x() + (bot_lbl.width() // 2)
            end_y = bot_origin.y()

            sx, sy = float(start_x), float(start_y)
            ex, ey = float(end_x), float(end_y)

            path = QPainterPath()
            path.moveTo(sx, sy)
            c1x, c1y = sx, sy + (ey - sy) * 0.5
            c2x, c2y = ex, sy + (ey - sy) * 0.5
            path.cubicTo(c1x, c1y, c2x, c2y, ex, ey)
            painter.drawPath(path)
            
            arrow_head = QPainterPath()
            arrow_head.moveTo(ex, ey)
            arrow_head.lineTo(ex - 6, ey - 12)
            arrow_head.lineTo(ex + 6, ey - 12)
            arrow_head.closeSubpath()
            painter.drawPath(arrow_head)

class VisualArea(QWidget):
    """ Container for the Grid that will be saved """
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")
        self.arrow_layer = ArrowOverlay(self)

    def resizeEvent(self, event):
        # Resize the overlay when this widget resizes
        self.arrow_layer.resize(self.size())
        super().resizeEvent(event)

class MagnifyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comparison Tool: GT vs Bicubic vs ESPCN")
        self.resize(1280, 900) # Window size

        # Configuration
        self.display_size = (380, 380)
        self.zoom_size = (380, 380)
        self.suffix_bicubic = "_bicubic2x"
        self.suffix_espcn = "_espcn2x"

        self.images_pil = [] 
        self.scale_factor = 1.0
        self.top_labels = []  
        self.zoom_labels = [] 

        # --- Main Layout Structure ---
        # 1. Main Wrapper
        wrapper = QWidget()
        self.setCentralWidget(wrapper)
        wrapper_layout = QVBoxLayout(wrapper)

        # 2. Scroll Area (Solves cropping/squashing)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True) # Allow inner widget to expand
        wrapper_layout.addWidget(self.scroll)

        # 3. Visual Area (Inside Scroll Area)
        self.visual_area = VisualArea() 
        self.scroll.setWidget(self.visual_area)

        # 4. Controls (Outside Scroll Area)
        self.btn_save = QPushButton("Save Comparison Image")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; font-size: 14px; 
                padding: 12px; border-radius: 6px; margin-top: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.btn_save.clicked.connect(self.save_snapshot)
        wrapper_layout.addWidget(self.btn_save)
        
        if self.load_images_auto():
            self.setup_ui()
        else:
            sys.exit() 

    def load_images_auto(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth Image (e.g. Nature02.png)")
        if not fname: return False

        directory = os.path.dirname(fname)
        base_name = os.path.basename(fname)
        name_no_ext, ext = os.path.splitext(base_name)

        path_gt = fname
        path_bic = os.path.join(directory, f"{name_no_ext}{self.suffix_bicubic}{ext}")
        path_esp = os.path.join(directory, f"{name_no_ext}{self.suffix_espcn}{ext}")

        missing = []
        if not os.path.exists(path_bic): missing.append(path_bic)
        if not os.path.exists(path_esp): missing.append(path_esp)
        
        if missing:
            QMessageBox.critical(self, "Error", f"Missing companion files:\n{missing}")
            return False

        try:
            self.images_pil = [
                Image.open(path_gt).convert("RGB"),
                Image.open(path_bic).convert("RGB"),
                Image.open(path_esp).convert("RGB")
            ]
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return False

    def setup_ui(self):
        orig_w, orig_h = self.images_pil[0].size
        self.scale_factor = min(self.display_size[0]/orig_w, self.display_size[1]/orig_h)
        new_w = int(orig_w * self.scale_factor)
        new_h = int(orig_h * self.scale_factor)

        # Set up grid on Visual Area
        grid = QGridLayout(self.visual_area)
        grid.setSpacing(15) 
        # Generous bottom margin to prevent crop
        grid.setContentsMargins(20, 20, 20, 50) 
        
        titles = ["Ground Truth", "Bicubic", "ESPCN"]

        for i in range(3):
            # Title
            lbl_title = QLabel(titles[i])
            lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: black;")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl_title, 0, i)

            # Top Image
            img_label = ImageLabel()
            img_label.setFixedSize(new_w, new_h)
            pil_resized = self.images_pil[i].resize((new_w, new_h), Image.LANCZOS)
            img_label.setPixmap(self.pil2pixmap(pil_resized))
            img_label.selection_changed.connect(self.sync_selection)
            
            grid.addWidget(img_label, 1, i)
            self.top_labels.append(img_label)

            # Spacer
            grid.setRowMinimumHeight(2, 60)

            # Zoom Image
            zoom_panel = QLabel()
            zoom_panel.setFixedSize(self.zoom_size[0], self.zoom_size[1])
            zoom_panel.setStyleSheet("border: 2px solid #ccc; background: #ddd;")
            zoom_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            grid.addWidget(zoom_panel, 3, i)
            self.zoom_labels.append(zoom_panel)
        
        # Force a minimum height so it can't be squished by window size
        # Height = Title + Image + Space + Title + Zoom + Padding
        estimated_height = 30 + new_h + 60 + 30 + self.zoom_size[1] + 60
        self.visual_area.setMinimumHeight(estimated_height)

    def pil2pixmap(self, pil_img):
        # Convert to RGBA to ensure 4-byte memory alignment
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        
        # Dump data as RGBA
        data = pil_img.tobytes("raw", "RGBA")
        
        # Create QImage using Format_RGBA8888
        qim = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
        
        return QPixmap.fromImage(qim)

    def sync_selection(self, rect):
        for lbl in self.top_labels:
            lbl.set_selection(rect)

        x1 = int(rect.x() / self.scale_factor)
        y1 = int(rect.y() / self.scale_factor)
        w = int(rect.width() / self.scale_factor)
        h = int(rect.height() / self.scale_factor)

        if w > 0 and h > 0:
            for i in range(3):
                crop = self.images_pil[i].crop((x1, y1, x1+w, y1+h))
                display_crop = crop.resize(self.zoom_size, Image.NEAREST)
                self.zoom_labels[i].setPixmap(self.pil2pixmap(display_crop))

        self.visual_area.arrow_layer.update_geometry_data(rect, self.top_labels, self.zoom_labels)
        self.visual_area.arrow_layer.raise_() 

    def save_snapshot(self):
        # Grab the full visual area (even if scrolled out of view)
        pixmap = self.visual_area.grab()
        path, _ = QFileDialog.getSaveFileName(self, "Save Comparison", "comparison_output.png", "PNG Images (*.png)")
        if path:
            pixmap.save(path)
            QMessageBox.information(self, "Success", f"Saved to:\n{path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagnifyApp()
    if window.images_pil: 
        window.show()
        sys.exit(app.exec())