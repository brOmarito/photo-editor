"""
Image Preview Widget.
Provides a before/after comparison view for original and edited images.

Features:
  - Side-by-side view
  - Slider overlay comparison
  - Zoom and pan
  - Toggle between original and edited
"""

import cv2
import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont


class ImagePreviewWidget(QWidget):
    """
    Widget that displays original and edited images with comparison tools.

    Supports three comparison modes:
      - Side by side
      - Slider overlay (drag a divider across the image)
      - Toggle (click to switch between before and after)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_image: Optional[np.ndarray] = None
        self._edited_image: Optional[np.ndarray] = None
        self._original_pixmap: Optional[QPixmap] = None
        self._edited_pixmap: Optional[QPixmap] = None
        self._mode = "slider"  # "side_by_side", "slider", "toggle"
        self._showing_original = False  # For toggle mode
        self._slider_pos = 0.5  # 0.0 to 1.0, position of divider
        self._dragging = False
        self._zoom_level = 1.0

        self._setup_ui()

    def _setup_ui(self):
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._btn_side_by_side = QPushButton("Side by Side")
        self._btn_side_by_side.setCheckable(True)
        self._btn_side_by_side.clicked.connect(lambda: self._set_mode("side_by_side"))

        self._btn_slider = QPushButton("Slider")
        self._btn_slider.setCheckable(True)
        self._btn_slider.setChecked(True)
        self._btn_slider.clicked.connect(lambda: self._set_mode("slider"))

        self._btn_toggle = QPushButton("Toggle (Click)")
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.clicked.connect(lambda: self._set_mode("toggle"))

        for btn in (self._btn_side_by_side, self._btn_slider, self._btn_toggle):
            btn.setFixedHeight(28)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    border: 1px solid #555; border-radius: 4px;
                    padding: 2px 10px; font-size: 11px;
                }
                QPushButton:checked {
                    background-color: #0078d4; color: white; border: 1px solid #0078d4;
                }
                QPushButton:hover { background-color: #4a4a4a; }
            """)
            toolbar.addWidget(btn)

        toolbar.addStretch()

        self._label_info = QLabel("")
        self._label_info.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._label_info)

        layout.addLayout(toolbar)

        # Canvas area for rendering
        self._canvas = PreviewCanvas(self)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMinimumHeight(200)
        layout.addWidget(self._canvas)

    def _set_mode(self, mode: str):
        """Switch comparison mode."""
        self._mode = mode
        self._btn_side_by_side.setChecked(mode == "side_by_side")
        self._btn_slider.setChecked(mode == "slider")
        self._btn_toggle.setChecked(mode == "toggle")
        self._canvas.set_mode(mode)
        self._canvas.update()

    def set_images(self, original: Optional[np.ndarray],
                   edited: Optional[np.ndarray]):
        """
        Set the original and edited images for comparison.

        Args:
            original: BGR numpy array (uint8) or None.
            edited: BGR numpy array (uint8) or None.
        """
        self._original_image = original
        self._edited_image = edited

        self._original_pixmap = self._array_to_pixmap(original) if original is not None else None
        self._edited_pixmap = self._array_to_pixmap(edited) if edited is not None else None

        self._canvas.set_pixmaps(self._original_pixmap, self._edited_pixmap)

        # Update info label
        if original is not None:
            h, w = original.shape[:2]
            self._label_info.setText(f"{w} × {h}")
        else:
            self._label_info.setText("")

        self._canvas.update()

    def clear(self):
        """Clear both images."""
        self.set_images(None, None)

    @staticmethod
    def _array_to_pixmap(image: np.ndarray) -> QPixmap:
        """
        Convert a BGR numpy array to a QPixmap.

        Args:
            image: BGR uint8 numpy array.

        Returns:
            QPixmap for display.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image.copy())


class PreviewCanvas(QWidget):
    """Custom paint widget for rendering the before/after comparison."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original: Optional[QPixmap] = None
        self._edited: Optional[QPixmap] = None
        self._mode = "slider"
        self._slider_pos = 0.5
        self._dragging = False
        self._showing_original = False
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_pixmaps(self, original: Optional[QPixmap], edited: Optional[QPixmap]):
        self._original = original
        self._edited = edited

    def set_mode(self, mode: str):
        self._mode = mode

    def paintEvent(self, event):
        """Render the comparison view."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect()

        # Draw background
        painter.fillRect(rect, QColor("#1e1e1e"))

        if self._original is None and self._edited is None:
            # No images loaded — show placeholder
            painter.setPen(QColor("#555"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(rect, Qt.AlignCenter,
                             "Drag & drop images here\nor use the file picker")
            painter.end()
            return

        if self._mode == "side_by_side":
            self._paint_side_by_side(painter, rect)
        elif self._mode == "slider":
            self._paint_slider(painter, rect)
        elif self._mode == "toggle":
            self._paint_toggle(painter, rect)

        painter.end()

    def _fit_pixmap(self, pixmap: QPixmap, target_rect: QRect) -> tuple:
        """Scale a pixmap to fit within target_rect while preserving aspect ratio."""
        if pixmap is None:
            return None, QRect()

        scaled = pixmap.scaled(target_rect.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        # Center within the target rect
        x = target_rect.x() + (target_rect.width() - scaled.width()) // 2
        y = target_rect.y() + (target_rect.height() - scaled.height()) // 2
        return scaled, QRect(x, y, scaled.width(), scaled.height())

    def _paint_side_by_side(self, painter: QPainter, rect: QRect):
        """Draw original on the left, edited on the right."""
        mid = rect.width() // 2

        # Left half = original
        left_rect = QRect(0, 0, mid - 2, rect.height())
        if self._original:
            scaled, dest = self._fit_pixmap(self._original, left_rect)
            if scaled:
                painter.drawPixmap(dest, scaled)
            painter.setPen(QColor("#888"))
            painter.drawText(left_rect.adjusted(8, 8, 0, 0), Qt.AlignLeft | Qt.AlignTop, "Original")

        # Divider
        painter.setPen(QPen(QColor("#555"), 2))
        painter.drawLine(mid, 0, mid, rect.height())

        # Right half = edited
        right_rect = QRect(mid + 2, 0, mid - 2, rect.height())
        if self._edited:
            scaled, dest = self._fit_pixmap(self._edited, right_rect)
            if scaled:
                painter.drawPixmap(dest, scaled)
            painter.setPen(QColor("#888"))
            painter.drawText(right_rect.adjusted(8, 8, 0, 0), Qt.AlignLeft | Qt.AlignTop, "Edited")

    def _paint_slider(self, painter: QPainter, rect: QRect):
        """Draw with a draggable slider dividing original and edited."""
        # Scale the edited image to fill the widget (aspect-ratio preserved)
        display_pixmap = self._edited if self._edited else self._original
        if display_pixmap is None:
            return

        scaled, dest = self._fit_pixmap(display_pixmap, rect)
        if scaled is None:
            return

        # Also scale original to match
        orig_scaled = None
        if self._original:
            orig_scaled = self._original.scaled(
                scaled.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        divider_x = int(dest.x() + dest.width() * self._slider_pos)

        # Draw edited (full)
        painter.drawPixmap(dest, scaled)

        # Draw original (left of divider) by clipping
        if orig_scaled:
            painter.save()
            clip_rect = QRect(dest.x(), dest.y(),
                              divider_x - dest.x(), dest.height())
            painter.setClipRect(clip_rect)
            painter.drawPixmap(dest, orig_scaled)
            painter.restore()

        # Draw divider line
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(divider_x, dest.y(), divider_x, dest.y() + dest.height())

        # Divider handle
        handle_y = dest.y() + dest.height() // 2
        painter.setBrush(QColor("white"))
        painter.drawEllipse(QPoint(divider_x, handle_y), 8, 8)

        # Labels
        painter.setPen(QColor("#ddd"))
        painter.setFont(QFont("Segoe UI", 10))
        if self._slider_pos > 0.1:
            painter.drawText(dest.x() + 8, dest.y() + 20, "Original")
        if self._slider_pos < 0.9:
            painter.drawText(dest.x() + dest.width() - 60, dest.y() + 20, "Edited")

    def _paint_toggle(self, painter: QPainter, rect: QRect):
        """Show either original or edited, toggled on click."""
        pixmap = self._original if self._showing_original else self._edited
        if pixmap is None:
            pixmap = self._edited if self._showing_original else self._original
        if pixmap is None:
            return

        scaled, dest = self._fit_pixmap(pixmap, rect)
        if scaled:
            painter.drawPixmap(dest, scaled)

        label = "Original" if self._showing_original else "Edited"
        painter.setPen(QColor("#ddd"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(dest.x() + 8, dest.y() + 20, label)
        painter.setPen(QColor("#888"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(dest.x() + 8, dest.y() + 36, "Click to toggle")

    # ---- Mouse events for slider interaction ----

    def mousePressEvent(self, event):
        if self._mode == "slider":
            self._dragging = True
            self._update_slider_pos(event.position().x())
        elif self._mode == "toggle":
            self._showing_original = not self._showing_original
            self.update()

    def mouseMoveEvent(self, event):
        if self._mode == "slider" and self._dragging:
            self._update_slider_pos(event.position().x())

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def _update_slider_pos(self, x: float):
        """Update slider position from mouse x coordinate."""
        display_pixmap = self._edited if self._edited else self._original
        if display_pixmap is None:
            return

        _, dest = self._fit_pixmap(display_pixmap, self.rect())
        if dest.width() > 0:
            pos = (x - dest.x()) / dest.width()
            self._slider_pos = max(0.0, min(1.0, pos))
            self.update()
