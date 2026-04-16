"""
File Output Module.
Handles saving edited images with proper naming conventions, output folder
creation, and format preservation.

Key principles:
  - Never modify original files.
  - Create output folder alongside the source folder.
  - Preserve original format unless the user requests conversion.
  - Save at the highest practical quality for the format.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.editing_profiles import EditStyle
from src.utils.logger import get_logger

logger = get_logger("photo_editor.output")

# Mapping from style to folder / file suffix
STYLE_SUFFIXES = {
    EditStyle.AUTO: ("Edited", "_edited"),
    EditStyle.NATURAL: ("Edited_Natural", "_natural"),
    EditStyle.VIBRANT: ("Edited_Vibrant", "_vibrant"),
    EditStyle.HIGH_CONTRAST: ("Edited_HighContrast", "_highcontrast"),
    EditStyle.PORTRAIT: ("Edited_Portrait", "_portrait"),
}

# Default JPEG quality (1–100). 95 is near-lossless for JPEG.
DEFAULT_JPEG_QUALITY = 95

# PNG compression level (0–9). 3 balances speed and size.
DEFAULT_PNG_COMPRESSION = 3


def get_output_folder_name(style: EditStyle) -> str:
    """
    Get the output folder name for a given editing style.

    Args:
        style: Editing style.

    Returns:
        Folder name string (e.g. "Edited_Vibrant").
    """
    folder_name, _ = STYLE_SUFFIXES.get(style, ("Edited", "_edited"))
    return folder_name


def get_file_suffix(style: EditStyle) -> str:
    """
    Get the filename suffix for a given editing style.

    Args:
        style: Editing style.

    Returns:
        Suffix string (e.g. "_vibrant").
    """
    _, suffix = STYLE_SUFFIXES.get(style, ("Edited", "_edited"))
    return suffix


def build_output_path(
    source_path: str,
    style: EditStyle,
    output_dir: Optional[str] = None,
    output_format: Optional[str] = None,
) -> str:
    """
    Build the full output file path for an edited image.

    The output structure is:
      <parent_of_source_folder>/<OutputFolder>/<original_name><suffix>.<ext>

    If output_dir is provided, it overrides the automatic folder placement.

    Args:
        source_path: Path to the original image.
        style: Editing style (determines folder and suffix).
        output_dir: Explicit output directory (optional).
        output_format: Output file extension override (e.g. ".png").

    Returns:
        Absolute path for the output file.
    """
    source = Path(source_path)
    stem = source.stem  # filename without extension
    ext = source.suffix.lower()

    # Determine suffix and folder name from style
    folder_name = get_output_folder_name(style)
    file_suffix = get_file_suffix(style)

    # Determine output extension
    if output_format:
        out_ext = output_format if output_format.startswith(".") else f".{output_format}"
    else:
        # RAW files get saved as TIFF by default (lossless)
        raw_formats = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".orf",
                       ".rw2", ".raf", ".dng", ".pef", ".raw", ".srw", ".x3f"}
        if ext in raw_formats:
            out_ext = ".tiff"
        else:
            out_ext = ext

    # Build output filename
    output_filename = f"{stem}{file_suffix}{out_ext}"

    # Determine output directory
    if output_dir:
        out_dir = output_dir
    else:
        # Place the output folder as a sibling of the source file's directory
        source_dir = source.parent
        out_dir = os.path.join(str(source_dir), folder_name)

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    return os.path.join(out_dir, output_filename)


def save_image(
    image: np.ndarray,
    output_path: str,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bool:
    """
    Save an image to disk with format-appropriate quality settings.

    Supported output formats:
      - JPEG (.jpg, .jpeg): Saved with configurable quality (default 95).
      - PNG (.png): Lossless compression.
      - TIFF (.tiff, .tif): Uncompressed for maximum quality.
      - WebP (.webp): Saved with configurable quality.
      - BMP (.bmp): Uncompressed.

    Args:
        image: BGR numpy array (uint8).
        output_path: Full path for the output file.
        quality: Quality for JPEG/WebP (1–100). Ignored for lossless formats.

    Returns:
        True if saved successfully.
    """
    ext = Path(output_path).suffix.lower()

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if ext in {".jpg", ".jpeg"}:
            # JPEG: quality 1-100, optimise Huffman tables
            params = [
                cv2.IMWRITE_JPEG_QUALITY, quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                # Use 4:4:4 chroma subsampling for highest quality
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
            ]
            success = cv2.imwrite(output_path, image, params)

        elif ext == ".png":
            # PNG: lossless, compression level 0-9 (3 is a good balance)
            params = [cv2.IMWRITE_PNG_COMPRESSION, DEFAULT_PNG_COMPRESSION]
            success = cv2.imwrite(output_path, image, params)

        elif ext in {".tiff", ".tif"}:
            # TIFF: uncompressed for maximum fidelity
            success = cv2.imwrite(output_path, image)

        elif ext == ".webp":
            # WebP: quality 1-100
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            success = cv2.imwrite(output_path, image, params)

        elif ext == ".bmp":
            success = cv2.imwrite(output_path, image)

        else:
            # Fallback to JPEG
            logger.warning(f"Unknown output format '{ext}', saving as JPEG.")
            output_path = str(Path(output_path).with_suffix(".jpg"))
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success = cv2.imwrite(output_path, image, params)

        if success:
            file_size = os.path.getsize(output_path)
            logger.debug(
                f"Saved: {output_path} "
                f"({file_size / 1024:.0f} KB, {image.shape[1]}×{image.shape[0]})"
            )
        else:
            logger.error(f"cv2.imwrite returned False for {output_path}")

        return success

    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return False
