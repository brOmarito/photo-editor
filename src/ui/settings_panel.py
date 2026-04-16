"""
Settings Panel Widget.
Provides UI controls for all user-configurable editing options.

Includes:
  - Editing style selector (Auto, Natural, Vibrant, High Contrast, Portrait)
  - Strength level (Light, Medium, Strong)
  - Individual edit toggles
  - Output format selection
  - Quality slider
  - Resize option
  - Image category confirmation/override
"""

from dataclasses import dataclass, field
from typing import Optional, Set

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QGroupBox, QCheckBox, QSpinBox,
    QScrollArea, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal

from src.core.editing_profiles import EditStyle, EditStrength, get_available_styles
from src.core.image_classifier import ImageCategory, get_category_display_name


@dataclass
class ImageSettings:
    """Complete set of editing settings for a single image."""

    style: EditStyle = EditStyle.AUTO
    strength: EditStrength = EditStrength.MEDIUM
    category_override: Optional[str] = "auto"  # "auto" or an ImageCategory value
    disabled_edits: Set[str] = field(default_factory=set)
    output_format: Optional[str] = None  # None ⇒ same as original
    quality: int = 95
    resize_enabled: bool = False
    resize_max_dimension: int = 2048

    # ----- convenience helpers used by the pipeline ----- #

    def get_category_override_enum(self) -> Optional[ImageCategory]:
        """Return *None* for auto-detect, or the ImageCategory enum."""
        if self.category_override == "auto" or self.category_override is None:
            return None
        return ImageCategory(self.category_override)

    def get_output_format_value(self) -> Optional[str]:
        """Return the format string or *None* for same-as-original."""
        return self.output_format if self.output_format else None


class SettingsPanel(QWidget):
    """
    Panel containing all editing settings and options.
    Emits signals when settings change so the pipeline can respond.
    """

    # Signal emitted whenever any setting changes
    settings_changed = Signal()

    # Stylesheet shared across the panel
    PANEL_STYLE = """
        QGroupBox {
            font-weight: bold; font-size: 12px; color: #d4d4d4;
            border: 1px solid #3c3c3c; border-radius: 6px;
            margin-top: 10px; padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px; padding: 0 4px;
        }
        QLabel { color: #cccccc; font-size: 11px; }
        QComboBox {
            background-color: #3c3c3c; color: #d4d4d4;
            border: 1px solid #555; border-radius: 4px;
            padding: 4px 8px; min-height: 24px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #2d2d2d; color: #d4d4d4;
            selection-background-color: #0078d4;
        }
        QCheckBox { color: #cccccc; font-size: 11px; spacing: 6px; }
        QCheckBox::indicator {
            width: 16px; height: 16px; border-radius: 3px;
            border: 1px solid #555; background-color: #3c3c3c;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d4; border: 1px solid #0078d4;
        }
        QSlider::groove:horizontal {
            height: 6px; background: #3c3c3c; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px; height: 16px; margin: -5px 0;
            background: #0078d4; border-radius: 8px;
        }
        QSlider::sub-page:horizontal { background: #0078d4; border-radius: 3px; }
        QSpinBox {
            background-color: #3c3c3c; color: #d4d4d4;
            border: 1px solid #555; border-radius: 4px;
            padding: 2px 6px; min-height: 22px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet(self.PANEL_STYLE)
        self._setup_ui()

    def _setup_ui(self):
        """Build the settings panel UI."""
        # Use a scroll area so everything is accessible even at small window sizes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(8, 8, 8, 8)

        self._build_style_section()
        self._build_strength_section()
        self._build_category_section()
        self._build_edits_section()
        self._build_output_section()

        self._layout.addStretch()

        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Style Section
    # ------------------------------------------------------------------
    def _build_style_section(self):
        group = QGroupBox("Editing Style")
        layout = QVBoxLayout(group)

        self._style_combo = QComboBox()
        styles = get_available_styles()
        for s in styles:
            self._style_combo.addItem(s["name"], s["value"])
        self._style_combo.setToolTip("Choose the overall editing style")
        self._style_combo.currentIndexChanged.connect(self._on_setting_changed)

        layout.addWidget(self._style_combo)

        # Description label
        self._style_desc = QLabel(styles[0]["description"])
        self._style_desc.setWordWrap(True)
        self._style_desc.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        layout.addWidget(self._style_desc)

        self._style_combo.currentIndexChanged.connect(self._update_style_desc)

        self._layout.addWidget(group)

    def _update_style_desc(self, index):
        styles = get_available_styles()
        if 0 <= index < len(styles):
            self._style_desc.setText(styles[index]["description"])

    # ------------------------------------------------------------------
    # Strength Section
    # ------------------------------------------------------------------
    def _build_strength_section(self):
        group = QGroupBox("Edit Strength")
        layout = QVBoxLayout(group)

        self._strength_combo = QComboBox()
        self._strength_combo.addItem("Light", EditStrength.LIGHT.value)
        self._strength_combo.addItem("Medium", EditStrength.MEDIUM.value)
        self._strength_combo.addItem("Strong", EditStrength.STRONG.value)
        self._strength_combo.setCurrentIndex(1)  # Default: Medium
        self._strength_combo.setToolTip("Controls the intensity of all edits")
        self._strength_combo.currentIndexChanged.connect(self._on_setting_changed)

        layout.addWidget(self._strength_combo)
        self._layout.addWidget(group)

    # ------------------------------------------------------------------
    # Category Override Section
    # ------------------------------------------------------------------
    def _build_category_section(self):
        group = QGroupBox("Image Category")
        layout = QVBoxLayout(group)

        self._category_combo = QComboBox()
        self._category_combo.addItem("Auto-detect", "auto")
        categories = [
            ImageCategory.PORTRAIT, ImageCategory.LANDSCAPE,
            ImageCategory.INDOOR, ImageCategory.OUTDOOR,
            ImageCategory.NIGHT, ImageCategory.PRODUCT,
            ImageCategory.EVENT, ImageCategory.PET,
            ImageCategory.DOCUMENT,
        ]
        for cat in categories:
            self._category_combo.addItem(get_category_display_name(cat), cat.value)

        self._category_combo.setToolTip(
            "Override the auto-detected image type, or leave as Auto-detect"
        )
        self._category_combo.currentIndexChanged.connect(self._on_setting_changed)

        layout.addWidget(self._category_combo)
        self._layout.addWidget(group)

    # ------------------------------------------------------------------
    # Individual Edit Toggles
    # ------------------------------------------------------------------
    def _build_edits_section(self):
        group = QGroupBox("Edit Steps (toggle on/off)")
        layout = QVBoxLayout(group)

        self._edit_checkboxes = {}
        edit_names = [
            ("auto_levels", "Auto Levels"),
            ("adjust_exposure", "Exposure Correction"),
            ("adjust_white_balance", "White Balance"),
            ("recover_highlights_shadows", "Highlight/Shadow Recovery"),
            ("adjust_contrast", "Contrast Enhancement"),
            ("apply_tone_curve", "Tone Curve"),
            ("adjust_vibrance", "Vibrance"),
            ("adjust_saturation", "Saturation"),
            ("adjust_temperature", "Colour Temperature"),
            ("dehaze", "Dehaze"),
            ("reduce_noise", "Noise Reduction"),
            ("sharpen", "Sharpening"),
            ("auto_straighten", "Auto Straighten"),
            ("smooth_skin", "Skin Smoothing"),
            ("apply_vignette", "Vignette"),
        ]

        for func_name, display_name in edit_names:
            cb = QCheckBox(display_name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_setting_changed)
            layout.addWidget(cb)
            self._edit_checkboxes[func_name] = cb

        self._layout.addWidget(group)

    # ------------------------------------------------------------------
    # Output Settings
    # ------------------------------------------------------------------
    def _build_output_section(self):
        group = QGroupBox("Output Settings")
        layout = QVBoxLayout(group)

        # Format
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItem("Same as original", "")
        self._format_combo.addItem("JPEG (.jpg)", ".jpg")
        self._format_combo.addItem("PNG (.png)", ".png")
        self._format_combo.addItem("TIFF (.tiff)", ".tiff")
        self._format_combo.addItem("WebP (.webp)", ".webp")
        self._format_combo.currentIndexChanged.connect(self._on_setting_changed)
        fmt_layout.addWidget(self._format_combo)
        layout.addLayout(fmt_layout)

        # Quality
        qual_layout = QHBoxLayout()
        qual_layout.addWidget(QLabel("Quality:"))
        self._quality_slider = QSlider(Qt.Horizontal)
        self._quality_slider.setRange(50, 100)
        self._quality_slider.setValue(95)
        self._quality_slider.setTickInterval(5)
        self._quality_slider.valueChanged.connect(self._on_setting_changed)
        qual_layout.addWidget(self._quality_slider)
        self._quality_label = QLabel("95")
        self._quality_label.setFixedWidth(24)
        self._quality_slider.valueChanged.connect(
            lambda v: self._quality_label.setText(str(v))
        )
        qual_layout.addWidget(self._quality_label)
        layout.addLayout(qual_layout)

        # Resize option
        self._resize_check = QCheckBox("Resize output")
        self._resize_check.setChecked(False)
        self._resize_check.stateChanged.connect(self._on_resize_toggled)
        layout.addWidget(self._resize_check)

        resize_layout = QHBoxLayout()
        resize_layout.addWidget(QLabel("Max dimension:"))
        self._resize_spin = QSpinBox()
        self._resize_spin.setRange(100, 10000)
        self._resize_spin.setValue(2048)
        self._resize_spin.setSuffix(" px")
        self._resize_spin.setEnabled(False)
        self._resize_spin.valueChanged.connect(self._on_setting_changed)
        resize_layout.addWidget(self._resize_spin)
        layout.addLayout(resize_layout)

        self._layout.addWidget(group)

    def _on_resize_toggled(self, state):
        self._resize_spin.setEnabled(bool(state))
        self._on_setting_changed()

    def _on_setting_changed(self, *_args):
        """Relay any widget change as a single settings_changed signal."""
        self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Public API — read current settings
    # ------------------------------------------------------------------

    def get_style(self) -> EditStyle:
        """Return the currently selected editing style."""
        value = self._style_combo.currentData()
        return EditStyle(value)

    def get_strength(self) -> EditStrength:
        """Return the currently selected edit strength."""
        value = self._strength_combo.currentData()
        return EditStrength(value)

    def get_category_override(self):
        """Return the category override, or None for auto-detect."""
        value = self._category_combo.currentData()
        if value == "auto":
            return None
        return ImageCategory(value)

    def get_disabled_edits(self) -> set:
        """Return a set of function names that the user has disabled."""
        disabled = set()
        for func_name, cb in self._edit_checkboxes.items():
            if not cb.isChecked():
                disabled.add(func_name)
        return disabled

    def get_output_format(self):
        """Return the selected output format, or None for same as original."""
        value = self._format_combo.currentData()
        return value if value else None

    def get_quality(self) -> int:
        """Return the quality slider value (50–100)."""
        return self._quality_slider.value()

    def get_resize_settings(self) -> dict:
        """Return resize settings."""
        return {
            "enabled": self._resize_check.isChecked(),
            "max_dimension": self._resize_spin.value(),
        }

    # ------------------------------------------------------------------
    # Snapshot / Restore — used for per-image settings
    # ------------------------------------------------------------------

    def get_all_settings(self) -> ImageSettings:
        """Capture a snapshot of every widget into an *ImageSettings* object."""
        return ImageSettings(
            style=self.get_style(),
            strength=self.get_strength(),
            category_override=self._category_combo.currentData(),
            disabled_edits=self.get_disabled_edits(),
            output_format=self._format_combo.currentData() or None,
            quality=self._quality_slider.value(),
            resize_enabled=self._resize_check.isChecked(),
            resize_max_dimension=self._resize_spin.value(),
        )

    def set_all_settings(self, s: ImageSettings, *, emit: bool = False):
        """
        Load an *ImageSettings* snapshot into the widgets.

        By default signals are **blocked** while values are written so
        that the caller can avoid unwanted preview re-renders during the
        load.  Pass *emit=True* to allow the normal signal flow.
        """
        # Block child-widget signals so _on_setting_changed is not fired
        # for every single widget update.
        if not emit:
            self.blockSignals(True)
            for w in (
                self._style_combo, self._strength_combo,
                self._category_combo, self._format_combo,
                self._quality_slider, self._resize_check,
                self._resize_spin,
            ):
                w.blockSignals(True)
            for cb in self._edit_checkboxes.values():
                cb.blockSignals(True)

        # ---- Style ----
        idx = self._style_combo.findData(s.style.value)
        if idx >= 0:
            self._style_combo.setCurrentIndex(idx)

        # ---- Strength ----
        idx = self._strength_combo.findData(s.strength.value)
        if idx >= 0:
            self._strength_combo.setCurrentIndex(idx)

        # ---- Category ----
        cat_data = s.category_override if s.category_override else "auto"
        idx = self._category_combo.findData(cat_data)
        if idx >= 0:
            self._category_combo.setCurrentIndex(idx)

        # ---- Edit toggles ----
        for func_name, cb in self._edit_checkboxes.items():
            cb.setChecked(func_name not in s.disabled_edits)

        # ---- Output format ----
        fmt_data = s.output_format if s.output_format else ""
        idx = self._format_combo.findData(fmt_data)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)

        # ---- Quality ----
        self._quality_slider.setValue(s.quality)
        self._quality_label.setText(str(s.quality))

        # ---- Resize ----
        self._resize_check.setChecked(s.resize_enabled)
        self._resize_spin.setValue(s.resize_max_dimension)
        self._resize_spin.setEnabled(s.resize_enabled)

        # Restore signals
        if not emit:
            for w in (
                self._style_combo, self._strength_combo,
                self._category_combo, self._format_combo,
                self._quality_slider, self._resize_check,
                self._resize_spin,
            ):
                w.blockSignals(False)
            for cb in self._edit_checkboxes.values():
                cb.blockSignals(False)
            self.blockSignals(False)
