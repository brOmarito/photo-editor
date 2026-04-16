"""
Metadata Handler Module.
Preserves EXIF and other metadata when saving edited images.

Copies metadata from the original image to the edited output, including:
  - EXIF data (camera, settings, date taken)
  - GPS data
  - Thumbnail (regenerated)
  - ICC colour profile

Uses piexif for EXIF manipulation. Falls back gracefully if piexif is
not available or the image format doesn't support EXIF.
"""

import os
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("photo_editor.metadata")

# Try to import piexif for EXIF handling
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False
    logger.info("piexif not installed — EXIF metadata preservation disabled.")

# Pillow for ICC profile handling
try:
    from PIL import Image as PILImage
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


# Formats that support EXIF data
EXIF_FORMATS = {".jpg", ".jpeg", ".tiff", ".tif", ".webp"}


def copy_metadata(source_path: str, dest_path: str) -> bool:
    """
    Copy EXIF metadata from the source image to the destination image.

    This preserves:
      - Camera information (make, model, lens)
      - Shooting settings (ISO, aperture, shutter speed)
      - Date and time taken
      - GPS coordinates
      - Orientation

    The thumbnail is removed since it would show the original, not the
    edited version (and would waste space).

    Args:
        source_path: Path to the original image.
        dest_path: Path to the edited image.

    Returns:
        True if metadata was copied successfully.
    """
    if not PIEXIF_AVAILABLE:
        logger.debug("piexif not available, skipping metadata copy")
        return False

    src_ext = Path(source_path).suffix.lower()
    dst_ext = Path(dest_path).suffix.lower()

    # Both files must be in EXIF-compatible formats
    if src_ext not in EXIF_FORMATS or dst_ext not in EXIF_FORMATS:
        logger.debug(
            f"Metadata copy skipped: source={src_ext}, dest={dst_ext} "
            f"(not EXIF-compatible)"
        )
        return False

    try:
        # Read EXIF from source
        exif_dict = piexif.load(source_path)

        # Remove thumbnail — it shows the original, not the edit
        exif_dict.pop("thumbnail", None)
        if "1st" in exif_dict:
            exif_dict["1st"] = {}

        # Add a note that the image was edited
        user_comment = b"Edited by Photo Editor"
        if "Exif" in exif_dict:
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
                b"ASCII\x00\x00\x00" + user_comment
            )

        # Serialize and write to destination
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, dest_path)

        logger.debug(f"Metadata copied: {source_path} → {dest_path}")
        return True

    except Exception as e:
        logger.warning(f"Metadata copy failed: {e}")
        return False


def copy_icc_profile(source_path: str, dest_path: str) -> bool:
    """
    Copy the ICC colour profile from source to destination.

    ICC profiles ensure consistent colour reproduction across devices.
    This preserves colour accuracy from the original camera/scanner.

    Args:
        source_path: Path to the original image.
        dest_path: Path to the edited image.

    Returns:
        True if ICC profile was copied successfully.
    """
    if not PILLOW_AVAILABLE:
        return False

    try:
        src_img = PILImage.open(source_path)
        icc_profile = src_img.info.get("icc_profile")

        if icc_profile is None:
            return False

        dst_img = PILImage.open(dest_path)
        dst_ext = Path(dest_path).suffix.lower()

        # Save with the ICC profile embedded
        save_kwargs = {"icc_profile": icc_profile}

        if dst_ext in {".jpg", ".jpeg"}:
            # Preserve existing EXIF when re-saving JPEG
            exif_data = dst_img.info.get("exif", b"")
            if exif_data:
                save_kwargs["exif"] = exif_data
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0  # 4:4:4 chroma

        dst_img.save(dest_path, **save_kwargs)
        logger.debug(f"ICC profile copied: {source_path} → {dest_path}")
        return True

    except Exception as e:
        logger.debug(f"ICC profile copy failed: {e}")
        return False


def read_exif_summary(file_path: str) -> Optional[dict]:
    """
    Read a summary of EXIF data for display in the UI.

    Returns a dictionary with human-readable keys like:
      camera, lens, iso, aperture, shutter_speed, date_taken, gps, etc.

    Returns None if EXIF cannot be read.
    """
    if not PIEXIF_AVAILABLE:
        return None

    ext = Path(file_path).suffix.lower()
    if ext not in EXIF_FORMATS:
        return None

    try:
        exif_dict = piexif.load(file_path)
    except Exception:
        return None

    summary = {}

    # Camera make and model
    _0th = exif_dict.get("0th", {})
    make = _0th.get(piexif.ImageIFD.Make, b"")
    model = _0th.get(piexif.ImageIFD.Model, b"")
    if make:
        summary["camera_make"] = make.decode("utf-8", errors="ignore").strip()
    if model:
        summary["camera_model"] = model.decode("utf-8", errors="ignore").strip()

    # EXIF data
    _exif = exif_dict.get("Exif", {})

    # ISO
    iso = _exif.get(piexif.ExifIFD.ISOSpeedRatings)
    if iso:
        summary["iso"] = iso

    # Aperture (FNumber)
    fnumber = _exif.get(piexif.ExifIFD.FNumber)
    if fnumber:
        summary["aperture"] = f"f/{fnumber[0] / fnumber[1]:.1f}"

    # Shutter speed (ExposureTime)
    exposure = _exif.get(piexif.ExifIFD.ExposureTime)
    if exposure:
        if exposure[0] < exposure[1]:
            summary["shutter_speed"] = f"{exposure[0]}/{exposure[1]}s"
        else:
            summary["shutter_speed"] = f"{exposure[0] / exposure[1]:.1f}s"

    # Focal length
    focal = _exif.get(piexif.ExifIFD.FocalLength)
    if focal:
        summary["focal_length"] = f"{focal[0] / focal[1]:.0f}mm"

    # Date taken
    date_taken = _exif.get(piexif.ExifIFD.DateTimeOriginal)
    if date_taken:
        summary["date_taken"] = date_taken.decode("utf-8", errors="ignore")

    # GPS
    _gps = exif_dict.get("GPS", {})
    if _gps:
        lat = _gps.get(piexif.GPSIFD.GPSLatitude)
        lon = _gps.get(piexif.GPSIFD.GPSLongitude)
        if lat and lon:
            summary["has_gps"] = True

    return summary if summary else None
