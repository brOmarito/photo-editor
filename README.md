# Photo Editor — Batch Photo Editing Application

A professional-grade desktop application for batch photo editing with automatic
image classification and multiple editing profiles. Mimics an automated Lightroom
workflow: non-destructive, metadata-preserving, and safe.

## Features

- **Multi-format support**: JPG, JPEG, PNG, TIFF, BMP, WebP, and RAW formats
  (CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF, PEF, etc.)
- **Automatic image classification**: Detects portraits, landscapes, indoor,
  outdoor, night, product, event, document, and more
- **5 editing styles**: Auto, Natural, Vibrant, High Contrast, Portrait
- **3 strength levels**: Light, Medium, Strong
- **Professional editing pipeline**: Exposure, white balance, contrast, tone
  curves, vibrance, sharpening, noise reduction, skin smoothing, and more
- **Before/after comparison**: Side-by-side, slider overlay, and toggle views
- **Drag-and-drop**: Drop files or folders directly into the app
- **Non-destructive**: Original files are never modified
- **Metadata preservation**: EXIF, GPS, camera info preserved in output
- **GPU acceleration**: Uses CUDA via OpenCV when available
- **Batch processing**: Process entire folders with progress tracking and cancellation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

### 3. Load Images

- **Drag & drop** files or folders onto the window
- Use **File → Open Image(s)** to pick files (Ctrl+O)
- Use **File → Open Folder** to pick a folder (Ctrl+Shift+O)

### 4. Configure Settings (optional)

- Choose an **Editing Style** (Auto recommended)
- Set **Edit Strength** (Light / Medium / Strong)
- Toggle individual edit steps on/off
- Override the detected image category if needed

### 5. Process

Click **▶ Process All** to edit all loaded images. Output is saved to a
folder alongside the originals (e.g., `Edited_Natural/`).

## Project Structure

```
photo-editor/
├── main.py                        # Application entry point
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── src/
│   ├── __init__.py
│   ├── core/                      # Core editing engine
│   │   ├── image_loader.py        # Image loading (standard + RAW)
│   │   ├── image_classifier.py    # Automatic image classification
│   │   ├── image_edits.py         # Individual editing operations
│   │   ├── editing_profiles.py    # Preset editing profiles
│   │   ├── editing_pipeline.py    # Pipeline orchestration
│   │   ├── metadata.py            # EXIF/metadata handling
│   │   └── file_output.py         # File saving with quality settings
│   ├── ui/                        # User interface (PySide6)
│   │   ├── main_window.py         # Main application window
│   │   ├── image_preview.py       # Before/after preview widget
│   │   ├── settings_panel.py      # Settings controls
│   │   └── progress_widget.py     # Progress bar and error log
│   └── utils/                     # Utilities
│       ├── logger.py              # Logging configuration
│       └── gpu_utils.py           # GPU acceleration helpers
└── logs/                          # Log files (auto-created)
```

## Editing Pipeline

Each image passes through these steps (in order):

| Step | Description |
|------|-------------|
| Auto Levels | Stretches histogram to use full tonal range |
| Exposure Correction | Gamma correction toward target brightness |
| White Balance | Gray World algorithm to remove colour casts |
| Highlight/Shadow Recovery | Compresses dynamic range in extremes |
| Contrast | CLAHE adaptive contrast on L channel |
| Tone Curve | S-curve for tonal enhancement |
| Vibrance | Selective saturation boost for muted colours |
| Saturation | Flat saturation adjust (profile-dependent) |
| Colour Temperature | Warm/cool shift (profile-dependent) |
| Dehaze | Dark channel prior haze removal |
| Noise Reduction | Non-Local Means denoising |
| Sharpening | Unsharp mask |
| Auto Straighten | Hough line-based tilt correction |
| Skin Smoothing | Bilateral filter on skin regions |
| Vignette | Radial edge darkening |

Each step can be toggled on/off and its strength is controlled by the
selected profile and global strength setting.

## Image Categories

The classifier detects:

| Category | How it's detected |
|----------|-------------------|
| Portrait | Face detection (Haar cascade), face-area ratio |
| Landscape | Wide aspect ratio, high colourfulness, good brightness |
| Indoor | Moderate brightness, lower saturation |
| Outdoor | Good brightness, moderate-high saturation |
| Night | Very low overall brightness |
| Product | Low background variance, moderate saturation |
| Event | Multiple faces detected |
| Document | High edge density, low colourfulness |

## Output

- Output folders are created alongside the source (never modifying originals)
- Folder naming: `Edited`, `Edited_Natural`, `Edited_Vibrant`, etc.
- File naming: `IMG_1024_edited.jpg`, `IMG_1024_natural.jpg`, etc.
- RAW files are saved as TIFF by default
- Quality defaults to 95 for JPEG
- EXIF metadata is preserved when possible

## Optional Dependencies

- **rawpy**: Required for RAW format support (`pip install rawpy`)
- **piexif**: Required for EXIF metadata preservation (`pip install piexif`)
- **OpenCV with CUDA**: For GPU acceleration (requires custom OpenCV build)

## Requirements

- Python 3.9+
- Windows / macOS / Linux
- ~500 MB RAM minimum (more for large batches)
