"""
Main Window Module.
The primary application window for the Photo Editor application.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  Menu Bar                                               │
  ├───────────┬─────────────────────────────────────────────┤
  │           │                                             │
  │ File List │          Image Preview                      │
  │ (left)    │        (center, largest)                    │
  │           │                                             │
  │           │                                             │
  ├───────────┼────────────────────────────┬────────────────┤
  │           │   Progress Bar             │  Settings      │
  │           │                            │  Panel (right) │
  └───────────┴────────────────────────────┴────────────────┘

Features:
  - Drag-and-drop image/folder support
  - File and folder picker dialogs
  - Preview with before/after comparison
  - Real-time progress tracking
  - Threaded batch processing (non-blocking UI)
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,  QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QMenuBar, QMenu, QStatusBar, QApplication, QSizePolicy,
    QFrame, QAbstractItemView,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QMimeData, QUrl, QSize, QTimer,
)
from PySide6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent, QFont

from src.ui.image_preview import ImagePreviewWidget
from src.ui.settings_panel import SettingsPanel, ImageSettings
from src.ui.progress_widget import ProgressWidget
from src.core.image_loader import discover_images, load_image, is_supported_format
from src.core.image_classifier import classify_image, get_category_display_name
from src.core.editing_pipeline import process_single_image, EditResult
from src.core.editing_profiles import EditStyle, EditStrength
from src.core.image_classifier import ImageCategory
from src.utils.logger import get_logger

logger = get_logger("photo_editor.ui")


# ======================================================================
# Preview Worker Thread — renders a single preview off the UI thread
# ======================================================================

class PreviewWorker(QThread):
    """
    Worker thread that renders a single-image preview in the background.
    This keeps the UI responsive while an edit preview is being computed.
    Emits the result (original + edited arrays) when done.
    """

    # Signal: (original_image, edited_image, file_path)
    preview_ready = Signal(object, object, str)

    def __init__(self, file_path: str, style: EditStyle,
                 strength: EditStrength,
                 category_override: Optional[ImageCategory],
                 disabled_edits: set, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.style = style
        self.strength = strength
        self.category_override = category_override
        self.disabled_edits = disabled_edits
        self._cancelled = False

    def cancel(self):
        """Mark this preview as cancelled (result will be discarded)."""
        self._cancelled = True

    def run(self):
        """Load, classify, and edit the image for preview."""
        try:
            if self._cancelled:
                return

            original = load_image(self.file_path)
            if original is None or self._cancelled:
                return

            result = process_single_image(
                self.file_path,
                style=self.style,
                edit_strength=self.strength,
                category_override=self.category_override,
                disabled_edits=self.disabled_edits,
                keep_images=True,
                preview_only=True,
            )

            if not self._cancelled and result.edited_image is not None:
                self.preview_ready.emit(original, result.edited_image, self.file_path)
        except Exception as e:
            logger.error(f"Preview render failed: {e}")


# ======================================================================
# Worker Thread — runs batch processing off the UI thread
# ======================================================================

class BatchWorker(QThread):
    """
    Worker thread that processes images in the background.
    Each image uses its own *ImageSettings* so edits are fully
    independent per file.
    """

    # Signals
    progress = Signal(int, int, str, bool, str, float)  # current, total, filename, success, error, duration
    finished = Signal(list)   # List[EditResult]
    step_update = Signal(str)  # Current edit step name

    def __init__(self, file_paths: List[str],
                 per_image_settings: dict,  # {path: ImageSettings}
                 output_dir: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.per_image_settings = per_image_settings
        self.output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the batch processing."""
        self._cancelled = True

    def run(self):
        """Run the batch processing pipeline."""
        results: List[EditResult] = []
        total = len(self.file_paths)

        for i, file_path in enumerate(self.file_paths):
            if self._cancelled:
                logger.info("Batch processing cancelled.")
                break

            def step_callback(step_name, step_idx, total_steps):
                self.step_update.emit(step_name)

            # Look up this image's independent settings
            s = self.per_image_settings.get(file_path, ImageSettings())

            result = process_single_image(
                file_path,
                style=s.style,
                edit_strength=s.strength,
                output_dir=self.output_dir,
                output_format=s.get_output_format_value(),
                quality=s.quality,
                category_override=s.get_category_override_enum(),
                disabled_edits=s.disabled_edits,
                step_callback=step_callback,
                keep_images=False,  # Don't hold all images in memory
            )
            results.append(result)

            self.progress.emit(
                i + 1, total,
                os.path.basename(file_path),
                result.success,
                result.error or "",
                result.duration,
            )

        self.finished.emit(results)


# ======================================================================
# Main Window
# ======================================================================

class MainWindow(QMainWindow):
    """Primary application window."""

    APP_STYLE = """
        QMainWindow { background-color: #252526; }
        QMenuBar {
            background-color: #2d2d2d; color: #d4d4d4;
            border-bottom: 1px solid #3c3c3c;
        }
        QMenuBar::item:selected { background-color: #3c3c3c; }
        QMenu {
            background-color: #2d2d2d; color: #d4d4d4;
            border: 1px solid #3c3c3c;
        }
        QMenu::item:selected { background-color: #0078d4; }
        QStatusBar {
            background-color: #007acc; color: white;
            font-size: 11px;
        }
        QSplitter::handle { background-color: #3c3c3c; }
        QSplitter::handle:horizontal { width: 3px; }
        QSplitter::handle:vertical { height: 3px; }
        QListWidget {
            background-color: #1e1e1e; color: #d4d4d4;
            border: 1px solid #3c3c3c; border-radius: 4px;
            font-size: 11px; outline: none;
        }
        QListWidget::item {
            padding: 4px 8px; border-bottom: 1px solid #2d2d2d;
        }
        QListWidget::item:selected {
            background-color: #094771; color: white;
        }
        QListWidget::item:hover { background-color: #2a2d2e; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Editor — Batch Photo Editing")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)
        self.setStyleSheet(self.APP_STYLE)
        self.setAcceptDrops(True)

        self._file_paths: List[str] = []
        self._per_image_settings: dict = {}  # {file_path: ImageSettings}
        self._worker: Optional[BatchWorker] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._results: List[EditResult] = []
        self._current_original: Optional[np.ndarray] = None  # cached for re-render
        self._current_preview_path: Optional[str] = None     # which file is previewed

        # Debounce timer: when settings change, wait 400ms of inactivity
        # before re-rendering the preview. This avoids queuing dozens of
        # renders while the user drags a slider or clicks quickly.
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(400)  # ms
        self._preview_debounce.timeout.connect(self._refresh_preview)

        self._build_menu_bar()
        self._build_ui()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Menu Bar
    # ------------------------------------------------------------------
    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_files = QAction("Open Image(s)...", self)
        open_files.setShortcut("Ctrl+O")
        open_files.triggered.connect(self._on_open_files)
        file_menu.addAction(open_files)

        open_folder = QAction("Open Folder...", self)
        open_folder.setShortcut("Ctrl+Shift+O")
        open_folder.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder)

        file_menu.addSeparator()

        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self._on_clear)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Central UI Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # --- Left panel: file list + action buttons ---
        left_panel = QWidget()
        left_panel.setFixedWidth(260)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Action buttons
        btn_style = """
            QPushButton {
                background-color: #0078d4; color: white;
                border: none; border-radius: 4px;
                padding: 8px 16px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1a8ae8; }
            QPushButton:pressed { background-color: #005fa3; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """

        btn_row1 = QHBoxLayout()
        self._btn_open_files = QPushButton("📁 Files")
        self._btn_open_files.setToolTip("Select individual image files")
        self._btn_open_files.clicked.connect(self._on_open_files)
        self._btn_open_files.setStyleSheet(btn_style)
        btn_row1.addWidget(self._btn_open_files)

        self._btn_open_folder = QPushButton("📂 Folder")
        self._btn_open_folder.setToolTip("Select a folder of images")
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        self._btn_open_folder.setStyleSheet(btn_style)
        btn_row1.addWidget(self._btn_open_folder)

        left_layout.addLayout(btn_row1)

        # File list label
        list_header = QHBoxLayout()
        self._file_count_label = QLabel("No images loaded")
        self._file_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        list_header.addWidget(self._file_count_label)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedSize(50, 20)
        self._btn_clear.setStyleSheet("""
            QPushButton {
                background: transparent; color: #888;
                border: 1px solid #555; border-radius: 3px; font-size: 10px;
            }
            QPushButton:hover { color: #d4d4d4; border-color: #888; }
        """)
        self._btn_clear.clicked.connect(self._on_clear)
        list_header.addWidget(self._btn_clear)

        left_layout.addLayout(list_header)

        # File list widget
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        left_layout.addWidget(self._file_list, 1)

        # Process button
        self._btn_process = QPushButton("▶  Process All")
        self._btn_process.setFixedHeight(40)
        self._btn_process.setEnabled(False)
        self._btn_process.setStyleSheet("""
            QPushButton {
                background-color: #16825d; color: white;
                border: none; border-radius: 6px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1a9e6f; }
            QPushButton:pressed { background-color: #107048; }
            QPushButton:disabled { background-color: #3c3c3c; color: #666; }
        """)
        self._btn_process.clicked.connect(self._on_process)
        left_layout.addWidget(self._btn_process)

        main_layout.addWidget(left_panel)

        # --- Center: preview + progress ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self._preview = ImagePreviewWidget()
        center_layout.addWidget(self._preview, 1)

        # Classification info bar
        self._class_info = QLabel("")
        self._class_info.setStyleSheet(
            "color: #4ec9b0; font-size: 11px; padding: 2px 8px;"
        )
        self._class_info.setVisible(False)
        center_layout.addWidget(self._class_info)

        self._progress = ProgressWidget()
        self._progress.cancel_requested.connect(self._on_cancel)
        center_layout.addWidget(self._progress)

        main_layout.addWidget(center_panel, 1)

        # --- Right panel: settings ---
        self._settings = SettingsPanel()
        # Re-render preview whenever any editing option changes
        self._settings.settings_changed.connect(self._on_settings_changed)
        main_layout.addWidget(self._settings)

    # ------------------------------------------------------------------
    # Status Bar
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — drag & drop images or use File menu")

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle dropped files and folders."""
        urls = event.mimeData().urls()
        paths = []
        for url in urls:
            path = url.toLocalFile()
            if path:
                paths.append(path)

        if paths:
            self._add_paths(paths)

    # ------------------------------------------------------------------
    # File / Folder Picker Actions
    # ------------------------------------------------------------------
    def _on_open_files(self):
        """Open file picker dialog for selecting individual images."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images",
            "", 
            "Images (*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp "
            "*.cr2 *.cr3 *.nef *.nrw *.arw *.dng *.orf *.rw2 *.raf *.raw *.pef);;"
            "All Files (*)"
        )
        if files:
            self._add_paths(files)

    def _on_open_folder(self):
        """Open folder picker dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder"
        )
        if folder:
            self._add_paths([folder])

    # ------------------------------------------------------------------
    # File Management
    # ------------------------------------------------------------------
    def _add_paths(self, paths: List[str]):
        """Discover images from the given paths and add them to the file list."""
        new_files = []
        for path in paths:
            new_files.extend(discover_images(path))

        # Deduplicate
        existing = set(self._file_paths)
        for f in new_files:
            if f not in existing:
                self._file_paths.append(f)
                self._per_image_settings[f] = ImageSettings()  # default settings
                existing.add(f)

        self._refresh_file_list()
        self._status_bar.showMessage(
            f"Loaded {len(self._file_paths)} image(s)"
        )

    def _refresh_file_list(self):
        """Update the file list widget from self._file_paths."""
        self._file_list.clear()
        for fp in self._file_paths:
            item = QListWidgetItem(os.path.basename(fp))
            item.setToolTip(fp)
            self._file_list.addItem(item)

        count = len(self._file_paths)
        self._file_count_label.setText(
            f"{count} image{'s' if count != 1 else ''}"
        )
        self._btn_process.setEnabled(count > 0)

        # Select first item
        if count > 0:
            self._file_list.setCurrentRow(0)

    def _on_clear(self):
        """Clear all loaded files."""
        self._file_paths.clear()
        self._file_list.clear()
        self._preview.clear()
        self._class_info.setVisible(False)
        self._file_count_label.setText("No images loaded")
        self._btn_process.setEnabled(False)
        self._progress.reset()
        self._results.clear()
        self._per_image_settings.clear()
        self._current_original = None
        self._current_preview_path = None
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.cancel()
        self._preview_debounce.stop()
        self._status_bar.showMessage("Ready")

    def _on_file_selected(self, row: int):
        """Handle file selection from the list — save current settings, load new image's settings, kick off preview."""
        if row < 0 or row >= len(self._file_paths):
            return

        # --- Save settings for the previously selected image ---
        if self._current_preview_path and self._current_preview_path in self._per_image_settings:
            self._per_image_settings[self._current_preview_path] = (
                self._settings.get_all_settings()
            )

        file_path = self._file_paths[row]
        self._status_bar.showMessage(f"Loading preview: {os.path.basename(file_path)}")
        QApplication.processEvents()

        # --- Load this image's independent settings into the panel ---
        if file_path in self._per_image_settings:
            self._settings.set_all_settings(
                self._per_image_settings[file_path], emit=False
            )

        original = load_image(file_path)
        if original is None:
            self._preview.clear()
            self._class_info.setVisible(False)
            self._current_original = None
            self._current_preview_path = None
            self._status_bar.showMessage("Failed to load image")
            return

        # Cache the loaded original so settings changes can re-render
        # without reloading the file from disk.
        self._current_original = original
        self._current_preview_path = file_path

        # Classify for informational display
        classification = classify_image(original)
        category = classification["category"]
        confidence = classification["confidence"]
        features = classification["features"]

        self._class_info.setText(
            f"Detected: {get_category_display_name(category)} "
            f"({confidence:.0%} confidence)  |  "
            f"{features['width']}×{features['height']}  |  "
            f"Faces: {features['num_faces']}  |  "
            f"Brightness: {features['brightness']:.0f}"
        )
        self._class_info.setVisible(True)

        # Show the original immediately while the edit renders in the background
        self._preview.set_images(original, None)

        # Launch background preview edit
        self._start_preview_render(file_path)

    # ------------------------------------------------------------------
    # Live Preview Rendering
    # ------------------------------------------------------------------
    def _on_settings_changed(self):
        """
        Called whenever any editing setting changes (style, strength,
        individual edit toggles, category override, etc.).

        Persists the change to the currently selected image's settings
        and starts a debounce timer so that rapid successive changes
        only trigger a single preview re-render.
        """
        # Persist to this image's entry
        if self._current_preview_path and self._current_preview_path in self._per_image_settings:
            self._per_image_settings[self._current_preview_path] = (
                self._settings.get_all_settings()
            )
        self._preview_debounce.start()  # (re)start the 400 ms timer

    def _refresh_preview(self):
        """
        Re-render the preview for the currently selected image using
        the latest settings. Called by the debounce timer.
        """
        if self._current_preview_path is None:
            return
        self._status_bar.showMessage("Updating preview...")
        self._start_preview_render(self._current_preview_path)

    def _start_preview_render(self, file_path: str):
        """
        Launch (or replace) a background thread to render the preview
        edit for the given file using the current settings.

        If a previous preview render is still running, it is cancelled
        so only the latest request completes.
        """
        # Cancel any in-flight preview render
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
            self._preview_worker.quit()
            # Don't wait — the old thread will exit on its own

        self._preview_worker = PreviewWorker(
            file_path,
            style=self._settings.get_style(),
            strength=self._settings.get_strength(),
            category_override=self._settings.get_category_override(),
            disabled_edits=self._settings.get_disabled_edits(),
            parent=self,
        )
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_worker.start()

    def _on_preview_ready(self, original, edited, file_path: str):
        """
        Slot called when the PreviewWorker finishes rendering.
        Only updates the display if the file_path still matches
        the currently selected image (guards against stale results
        from a cancelled/slow render).
        """
        if file_path != self._current_preview_path:
            # The user selected a different image while we were rendering.
            return
        self._preview.set_images(original, edited)
        self._status_bar.showMessage(f"Viewing: {os.path.basename(file_path)}")

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------
    def _on_process(self):
        """Start batch processing all loaded images."""
        if not self._file_paths:
            return

        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "Processing is already in progress.")
            return

        # Persist the panel to the currently viewed image before we start
        if self._current_preview_path and self._current_preview_path in self._per_image_settings:
            self._per_image_settings[self._current_preview_path] = (
                self._settings.get_all_settings()
            )

        # Setup progress
        self._progress.reset()
        self._progress.start(len(self._file_paths))
        self._btn_process.setEnabled(False)
        self._start_time = time.time()

        # Create and start worker thread with per-image settings
        self._worker = BatchWorker(
            self._file_paths,
            per_image_settings=self._per_image_settings,
            parent=self,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.step_update.connect(self._on_step_update)
        self._worker.start()

        self._status_bar.showMessage("Processing...")

    def _on_worker_progress(self, current: int, total: int, filename: str,
                            success: bool, error: str, duration: float):
        """Handle progress updates from the worker thread."""
        self._progress.update_progress(
            current, total, filename, success, error, duration
        )

    def _on_step_update(self, step_name: str):
        """Handle individual edit step updates."""
        self._status_bar.showMessage(f"Applying: {step_name}")

    def _on_worker_finished(self, results: List[EditResult]):
        """Handle batch processing completion."""
        self._results = results
        total_time = time.time() - self._start_time
        success_count = sum(1 for r in results if r.success)

        self._progress.finish(len(results), success_count, total_time)
        self._btn_process.setEnabled(True)
        self._worker = None

        # Show output folder location
        if results and any(r.output_path for r in results):
            first_output = next(
                (r.output_path for r in results if r.output_path), None
            )
            if first_output:
                output_dir = os.path.dirname(first_output)
                self._status_bar.showMessage(
                    f"Complete! Output: {output_dir}"
                )
                self._progress.log_info(f"Output folder: {output_dir}")

    def _on_cancel(self):
        """Cancel the running batch process."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._status_bar.showMessage("Cancelling...")

    # ------------------------------------------------------------------
    # Help / About
    # ------------------------------------------------------------------
    def _on_about(self):
        QMessageBox.about(
            self,
            "About Photo Editor",
            "<h2>Photo Editor</h2>"
            "<p>Batch photo editing application with automatic image "
            "classification and professional editing profiles.</p>"
            "<p>Version 1.0.0</p>"
            "<p>Features:<br>"
            "• Multi-format support (JPG, PNG, TIFF, RAW)<br>"
            "• Automatic image classification<br>"
            "• Professional editing profiles<br>"
            "• Before/after comparison<br>"
            "• Non-destructive editing<br>"
            "• Metadata preservation</p>"
        )
