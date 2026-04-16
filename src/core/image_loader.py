"""
Image Loader Module.
Handles loading images from files, supporting multiple formats including RAW.
Validates files and provides consistent numpy array output for the pipeline.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger("photo_editor.loader")

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
STANDARD_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
RAW_FORMATS = {
    ".cr2", ".cr3",     # Canon
    ".nef", ".nrw",     # Nikon
    ".arw", ".srf",     # Sony
    ".orf",             # Olympus
    ".rw2",             # Panasonic
    ".raf",             # Fujifilm
    ".dng",             # Adobe DNG (universal)
    ".pef",             # Pentax
    ".raw",             # Generic RAW
    ".srw",             # Samsung
    ".x3f",             # Sigma
}
ALL_SUPPORTED = STANDARD_FORMATS | RAW_FORMATS

# Try to import rawpy for RAW support; it's optional
try:
    import rawpy
    RAW_SUPPORT = True
    logger.info("RAW image support enabled (rawpy available).")
except ImportError:
    RAW_SUPPORT = False
    logger.info("rawpy not installed — RAW image support disabled.")


def is_supported_format(file_path: str) -> bool:
    """
    Check whether the file extension is a supported image format.

    Args:
        file_path: Path to the image file.

    Returns:
        True if the format is supported.
    """
    ext = Path(file_path).suffix.lower()
    if ext in RAW_FORMATS and not RAW_SUPPORT:
        return False
    return ext in ALL_SUPPORTED


def discover_images(path: str) -> List[str]:
    """
    Discover all supported image files at the given path.
    If *path* is a file, returns a one-element list (if supported).
    If *path* is a directory, returns all supported images found recursively.

    Args:
        path: File path or directory path to scan.

    Returns:
        Sorted list of absolute paths to supported image files.
    """
    path = os.path.abspath(path)
    results: List[str] = []

    if os.path.isfile(path):
        if is_supported_format(path):
            results.append(path)
        else:
            logger.warning(f"Unsupported file format: {path}")
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in files:
                fpath = os.path.join(root, fname)
                if is_supported_format(fpath):
                    results.append(fpath)
        logger.info(f"Discovered {len(results)} image(s) in '{path}'")
    else:
        logger.error(f"Path does not exist: {path}")

    return sorted(results)


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image file and return it as a BGR numpy array (OpenCV convention).

    For standard formats, uses OpenCV / Pillow.
    For RAW formats, uses rawpy to demosaic and then converts to BGR.

    Args:
        file_path: Absolute path to the image file.

    Returns:
        Image as numpy array (uint8, BGR) or None on failure.
    """
    file_path = os.path.abspath(file_path)
    ext = Path(file_path).suffix.lower()

    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    # ------------------------------------------------------------------
    # RAW format loading
    # ------------------------------------------------------------------
    if ext in RAW_FORMATS:
        return _load_raw(file_path)

    # ------------------------------------------------------------------
    # Standard format loading
    # ------------------------------------------------------------------
    return _load_standard(file_path)


def _load_standard(file_path: str) -> Optional[np.ndarray]:
    """
    Load a standard image format (JPG, PNG, TIFF, etc.).
    Tries OpenCV first; falls back to Pillow for formats OpenCV may
    struggle with (e.g. 16-bit TIFF, certain PNGs).

    Args:
        file_path: Path to the image file.

    Returns:
        BGR numpy array (uint8) or None.
    """
    try:
        # IMREAD_UNCHANGED preserves bit depth and alpha channels
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            # Fallback to Pillow
            return _load_with_pillow(file_path)

        # Convert 16-bit images to 8-bit for the editing pipeline
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)

        # If grayscale, convert to BGR so the pipeline is consistent
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # BGRA → BGR (drop alpha for editing, alpha handled at save)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        logger.debug(f"Loaded (OpenCV): {file_path}  shape={image.shape}")
        return image

    except Exception as e:
        logger.warning(f"OpenCV load failed for {file_path}: {e}. Trying Pillow.")
        return _load_with_pillow(file_path)


def _load_with_pillow(file_path: str) -> Optional[np.ndarray]:
    """
    Fallback loader using Pillow, converting to BGR numpy array.

    Args:
        file_path: Path to the image file.

    Returns:
        BGR numpy array (uint8) or None.
    """
    try:
        pil_image = Image.open(file_path)
        pil_image = pil_image.convert("RGB")
        rgb_array = np.array(pil_image, dtype=np.uint8)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        logger.debug(f"Loaded (Pillow): {file_path}  shape={bgr_array.shape}")
        return bgr_array
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None


def _load_raw(file_path: str) -> Optional[np.ndarray]:
    """
    Load a RAW image file using rawpy, demosaic, and convert to BGR uint8.

    Uses reasonable default parameters that produce a good starting point
    for the editing pipeline (auto white balance, no auto brightness so
    we can control exposure ourselves).

    Args:
        file_path: Path to the RAW image file.

    Returns:
        BGR numpy array (uint8) or None.
    """
    if not RAW_SUPPORT:
        logger.error(f"Cannot load RAW file (rawpy not installed): {file_path}")
        return None

    try:
        with rawpy.imread(file_path) as raw:
            # Demosaic with sensible defaults:
            #  - use_camera_wb: use the camera's white balance if available
            #  - no_auto_bright: let our pipeline handle exposure
            #  - output_bps: 8 bits per sample for uint8 pipeline
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                half_size=False,  # full resolution
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        logger.debug(f"Loaded (RAW): {file_path}  shape={bgr.shape}")
        return bgr

    except Exception as e:
        logger.error(f"Failed to load RAW image {file_path}: {e}")
        return None


def get_image_info(file_path: str) -> dict:
    """
    Get basic information about an image file without fully loading it.

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary with keys: path, filename, extension, size_bytes,
        width, height, channels (some may be None if reading fails).
    """
    file_path = os.path.abspath(file_path)
    p = Path(file_path)
    info = {
        "path": file_path,
        "filename": p.name,
        "extension": p.suffix.lower(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "width": None,
        "height": None,
        "channels": None,
    }

    try:
        if info["extension"] in RAW_FORMATS:
            if RAW_SUPPORT:
                with rawpy.imread(file_path) as raw:
                    sizes = raw.sizes
                    info["width"] = sizes.width
                    info["height"] = sizes.height
                    info["channels"] = 3
        else:
            pil_image = Image.open(file_path)
            info["width"], info["height"] = pil_image.size
            mode_channels = {"L": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
            info["channels"] = mode_channels.get(pil_image.mode, 3)
    except Exception as e:
        logger.debug(f"Could not read image dimensions for {file_path}: {e}")

    return info
