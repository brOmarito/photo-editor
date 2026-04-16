"""
Image Classifier Module.
Automatically classifies images into categories so the editing pipeline can
choose an appropriate editing profile.

Classification uses heuristic analysis (no deep learning model required):
  - Face detection  → Portrait
  - Color / saturation analysis → Landscape vs Indoor
  - Brightness analysis → Night photography
  - Edge density → Documents / screenshots
  - Dominant colour analysis → Product photography hints

Categories:
  portrait, landscape, indoor, outdoor, night, product, event, pet, document, unknown
"""

from enum import Enum
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("photo_editor.classifier")


class ImageCategory(str, Enum):
    """Possible image categories that drive editing profile selection."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NIGHT = "night"
    PRODUCT = "product"
    EVENT = "event"
    PET = "pet"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Lazy-loaded Haar cascade classifiers (OpenCV ships these by default)
# ---------------------------------------------------------------------------
_face_cascade: Optional[cv2.CascadeClassifier] = None
_eye_cascade: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Load the frontal-face Haar cascade (lazy singleton)."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _get_eye_cascade() -> cv2.CascadeClassifier:
    """Load the eye Haar cascade (lazy singleton)."""
    global _eye_cascade
    if _eye_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        _eye_cascade = cv2.CascadeClassifier(cascade_path)
    return _eye_cascade


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _compute_brightness(image_bgr: np.ndarray) -> float:
    """
    Compute the average perceived brightness of an image (0 – 255).
    Uses the V channel from HSV colour space.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _compute_saturation(image_bgr: np.ndarray) -> float:
    """
    Compute the average colour saturation (0 – 255).
    Uses the S channel from HSV colour space.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def _compute_colorfulness(image_bgr: np.ndarray) -> float:
    """
    Compute a colourfulness metric based on opponent colour space.
    Higher values indicate more colourful images.
    Reference: Hasler & Süsstrunk (2003).
    """
    B, G, R = image_bgr[:, :, 0].astype(float), image_bgr[:, :, 1].astype(float), image_bgr[:, :, 2].astype(float)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return std + 0.3 * mean


def _compute_edge_density(image_bgr: np.ndarray) -> float:
    """
    Compute the fraction of the image that contains strong edges.
    High edge density suggests documents, screenshots, or text-heavy images.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def _detect_faces(image_bgr: np.ndarray) -> list:
    """
    Detect faces in the image using Haar cascade.
    Returns list of (x, y, w, h) rectangles.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Use a smaller image for speed; faces detected at reduced resolution
    scale = 1.0
    if max(gray.shape) > 1000:
        scale = 1000 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return []

    # Scale face rects back to original resolution
    result = []
    for (x, y, w, h) in faces:
        result.append((
            int(x / scale), int(y / scale),
            int(w / scale), int(h / scale)
        ))
    return result


def _compute_aspect_ratio(image_bgr: np.ndarray) -> float:
    """Return width / height."""
    h, w = image_bgr.shape[:2]
    return w / h if h > 0 else 1.0


def _dominant_color_variance(image_bgr: np.ndarray) -> float:
    """
    Estimate how uniform the background is by looking at the variance
    of the most common colour cluster. Low variance suggests a product
    shot with a plain background.
    """
    # Downsample for speed
    small = cv2.resize(image_bgr, (64, 64))
    pixels = small.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Find the largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster_idx = unique[np.argmax(counts)]
    cluster_pixels = pixels[labels.flatten() == largest_cluster_idx]
    return float(np.mean(np.std(cluster_pixels, axis=0)))


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_image(image_bgr: np.ndarray) -> dict:
    """
    Classify an image into a category using heuristic feature analysis.

    The classifier computes multiple features and uses a rule-based system
    to determine the most likely category. It also returns a confidence
    score (0.0 – 1.0) and all computed features for debugging.

    Args:
        image_bgr: Image as BGR numpy array (uint8).

    Returns:
        Dictionary with keys:
            category (ImageCategory): Predicted category.
            confidence (float): Confidence score 0-1.
            features (dict): All computed feature values.
            faces (int): Number of faces detected.
    """
    h, w = image_bgr.shape[:2]

    # ------------------------------------------------------------------
    # Step 1: Compute features
    # ------------------------------------------------------------------
    brightness = _compute_brightness(image_bgr)
    saturation = _compute_saturation(image_bgr)
    colorfulness = _compute_colorfulness(image_bgr)
    edge_density = _compute_edge_density(image_bgr)
    aspect_ratio = _compute_aspect_ratio(image_bgr)
    faces = _detect_faces(image_bgr)
    num_faces = len(faces)
    bg_variance = _dominant_color_variance(image_bgr)

    features = {
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "colorfulness": round(colorfulness, 2),
        "edge_density": round(edge_density, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "num_faces": num_faces,
        "bg_variance": round(bg_variance, 2),
        "width": w,
        "height": h,
    }

    logger.debug(f"Image features: {features}")

    # ------------------------------------------------------------------
    # Step 2: Rule-based classification
    # ------------------------------------------------------------------

    # Documents / screenshots: very high edge density, low colour
    if edge_density > 0.15 and colorfulness < 40:
        category = ImageCategory.DOCUMENT
        confidence = min(0.5 + edge_density * 2, 0.95)

    # Night photography: very dark overall
    elif brightness < 50:
        # If faces are present in dark photo → event (e.g. concert, party)
        if num_faces >= 2:
            category = ImageCategory.EVENT
            confidence = 0.65
        else:
            category = ImageCategory.NIGHT
            confidence = min(0.5 + (50 - brightness) / 50, 0.90)

    # Portraits: faces detected taking up significant area
    elif num_faces >= 1:
        # Check how much of the image area is covered by faces
        face_area = sum(fw * fh for (_, _, fw, fh) in faces)
        image_area = w * h
        face_ratio = face_area / image_area

        if num_faces >= 3 or (num_faces >= 2 and face_ratio < 0.15):
            # Multiple faces, relatively small → event / group photo
            category = ImageCategory.EVENT
            confidence = 0.60 + min(num_faces * 0.05, 0.25)
        elif face_ratio > 0.03:
            category = ImageCategory.PORTRAIT
            confidence = min(0.60 + face_ratio * 3, 0.95)
        else:
            # Tiny face in a large scene → probably landscape / outdoor
            category = ImageCategory.OUTDOOR
            confidence = 0.50

    # Product photography: low background variance, moderate saturation
    elif bg_variance < 10 and saturation < 80:
        category = ImageCategory.PRODUCT
        confidence = min(0.55 + (10 - bg_variance) / 20, 0.85)

    # Landscape: wide aspect ratio, high colourfulness, good brightness
    elif aspect_ratio > 1.3 and colorfulness > 50 and brightness > 80:
        category = ImageCategory.LANDSCAPE
        confidence = min(0.50 + colorfulness / 200, 0.85)

    # Outdoor: good brightness and moderate-high saturation
    elif brightness > 100 and saturation > 50:
        category = ImageCategory.OUTDOOR
        confidence = 0.55

    # Indoor: moderate brightness, lower saturation
    elif 50 <= brightness <= 150 and saturation < 60:
        category = ImageCategory.INDOOR
        confidence = 0.50

    # Fallback
    else:
        category = ImageCategory.UNKNOWN
        confidence = 0.30

    result = {
        "category": category,
        "confidence": round(confidence, 2),
        "features": features,
        "faces": num_faces,
    }

    logger.info(
        f"Classification: {category.value} "
        f"(confidence={confidence:.0%}, faces={num_faces})"
    )
    return result


def get_category_display_name(category: ImageCategory) -> str:
    """
    Return a user-friendly display name for an image category.

    Args:
        category: ImageCategory enum value.

    Returns:
        Human-readable category name.
    """
    names = {
        ImageCategory.PORTRAIT: "Portrait",
        ImageCategory.LANDSCAPE: "Landscape",
        ImageCategory.INDOOR: "Indoor Photo",
        ImageCategory.OUTDOOR: "Outdoor Photo",
        ImageCategory.NIGHT: "Night Photography",
        ImageCategory.PRODUCT: "Product Photo",
        ImageCategory.EVENT: "Event / Group",
        ImageCategory.PET: "Pet / Animal",
        ImageCategory.DOCUMENT: "Document / Screenshot",
        ImageCategory.UNKNOWN: "General",
    }
    return names.get(category, "General")
