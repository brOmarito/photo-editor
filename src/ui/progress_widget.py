"""
Progress Widget Module.
Displays batch processing progress with per-file status, timing, and error log.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCursor


class ProgressWidget(QWidget):
    """
    Widget that shows batch processing progress.

    Features:
      - Overall progress bar
      - Current file label
      - Elapsed time
      - Error log view
      - Cancel button
    """

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.reset()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        # Top bar: progress + cancel
        top = QHBoxLayout()

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m  (%p%)")
        self._progress_bar.setFixedHeight(22)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3c3c3c; border-radius: 4px;
                background-color: #2d2d2d; text-align: center;
                color: #d4d4d4; font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #0078d4; border-radius: 3px;
            }
        """)
        top.addWidget(self._progress_bar, 1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedSize(70, 22)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #c42b1c; color: white;
                border: none; border-radius: 4px; font-size: 11px;
            }
            QPushButton:hover { background-color: #d43a2b; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._cancel_btn.setEnabled(False)
        top.addWidget(self._cancel_btn)

        layout.addLayout(top)

        # Status line
        status_layout = QHBoxLayout()

        self._current_file_label = QLabel("")
        self._current_file_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._current_file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_layout.addWidget(self._current_file_label)

        self._time_label = QLabel("")
        self._time_label.setStyleSheet("color: #888; font-size: 11px;")
        status_layout.addWidget(self._time_label)

        layout.addLayout(status_layout)

        # Error / log area (collapsed by default)
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setFixedHeight(100)
        self._log_area.setVisible(False)
        self._log_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #3c3c3c; border-radius: 4px;
                font-family: Consolas, monospace; font-size: 10px;
                padding: 4px;
            }
        """)
        layout.addWidget(self._log_area)

    def reset(self):
        """Reset progress to initial state."""
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(100)
        self._current_file_label.setText("")
        self._time_label.setText("")
        self._log_area.clear()
        self._log_area.setVisible(False)
        self._cancel_btn.setEnabled(False)
        self._error_count = 0

    def start(self, total: int):
        """Begin tracking progress for a batch of total images."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(0)
        self._cancel_btn.setEnabled(True)
        self._error_count = 0

    def update_progress(self, current: int, total: int, filename: str,
                        success: bool, error_msg: str = "",
                        duration: float = 0.0):
        """
        Update progress for the current file.

        Args:
            current: Current file index (1-based).
            total: Total number of files.
            filename: Name of the current file.
            success: Whether the current file was processed successfully.
            error_msg: Error message if processing failed.
            duration: Processing time for this file in seconds.
        """
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

        if current < total:
            self._current_file_label.setText(f"Processing: {filename}")
        else:
            self._current_file_label.setText("Complete!")

        self._time_label.setText(f"{duration:.1f}s")

        if not success:
            self._error_count += 1
            self._log_area.setVisible(True)
            self._log_error(filename, error_msg)

    def finish(self, total: int, success_count: int, total_time: float):
        """
        Mark batch processing as complete.

        Args:
            total: Total files processed.
            success_count: Number of successfully processed files.
            total_time: Total elapsed time in seconds.
        """
        fail_count = total - success_count
        self._current_file_label.setText(
            f"Done! {success_count}/{total} succeeded"
            + (f", {fail_count} failed" if fail_count > 0 else "")
        )
        self._time_label.setText(f"Total: {total_time:.1f}s")
        self._cancel_btn.setEnabled(False)

    def _log_error(self, filename: str, error_msg: str):
        """Append an error to the log area."""
        self._log_area.append(
            f'<span style="color:#c42b1c;">✗ {filename}:</span> '
            f'<span style="color:#aaa;">{error_msg}</span>'
        )
        # Auto-scroll to bottom
        cursor = self._log_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_area.setTextCursor(cursor)

    def log_info(self, message: str):
        """Append an informational message to the log."""
        self._log_area.setVisible(True)
        self._log_area.append(
            f'<span style="color:#4ec9b0;">{message}</span>'
        )
