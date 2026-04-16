"""
Photo Editor — Batch Photo Editing Application
================================================
Entry point for the desktop application.

Usage:
    python main.py

Requirements:
    pip install -r requirements.txt
"""

import sys
import os

# Ensure the project root is in the Python path so 'src' can be imported
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from src.ui.main_window import MainWindow
from src.utils.logger import setup_logger


def main():
    """Launch the Photo Editor application."""
    # Initialise logging
    logger = setup_logger("photo_editor")
    logger.info("Starting Photo Editor application...")

    # Enable High-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Photo Editor")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("PhotoEditor")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Apply global dark theme style
    app.setStyleSheet("""
        QToolTip {
            background-color: #2d2d2d; color: #d4d4d4;
            border: 1px solid #555; padding: 4px;
            font-size: 11px;
        }
        QScrollBar:vertical {
            background: #1e1e1e; width: 10px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #555; min-height: 30px; border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover { background: #777; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0; background: none;
        }
        QScrollBar:horizontal {
            background: #1e1e1e; height: 10px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #555; min-width: 30px; border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover { background: #777; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0; background: none;
        }
    """)

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("Application window shown. Ready.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
