"""
Image Edits Module.
Contains individual, composable image editing operations that mimic a
professional photographer's Lightroom-style workflow.

Every function takes a BGR uint8 numpy array and returns a new BGR uint8
numpy array (non-destructive — the input is never mutated).

Parameters use a normalised "strength" value (0.0 = no change, 1.0 = full
effect) so editing profiles can dial each adjustment up or down.
"""

import cv2
import numpy as np
from skimage import exposure as ski_exposure
from skimage import restoration as ski_restoration

from src.utils.logger import get_logger
from src.utils.gpu_utils import get_gpu_accelerator

logger = get_logger("photo_editor.edits")


# ======================================================================
# Helper Utilities
# ======================================================================

def _clamp(image: np.ndarray) -> np.ndarray:
    """Clamp pixel values to valid uint8 range [0, 255]."""
    return np.clip(image, 0, 255).astype(np.uint8)


def _blend(original: np.ndarray, edited: np.ndarray, strength: float) -> np.ndarray:
    """
    Linearly blend between original and edited images based on strength.

    Args:
        original: Original image (BGR uint8).
        edited: Fully-edited image (BGR uint8).
        strength: 0.0 = original only, 1.0 = edited only.

    Returns:
        Blended image (BGR uint8).
    """
    strength = max(0.0, min(1.0, strength))
    if strength == 0.0:
        return original.copy()
    if strength == 1.0:
        return edited
    blended = cv2.addWeighted(original, 1.0 - strength, edited, strength, 0)
    return _clamp(blended)


# ======================================================================
# Exposure Correction
# ======================================================================

def adjust_exposure(image: np.ndarray, strength: float = 0.5,
                    target_brightness: float = 128.0) -> np.ndarray:
    """
    Correct image exposure toward a target brightness.

    How it works:
    1. Convert to HSV and measure current mean brightness (V channel).
    2. Compute the gamma needed to shift mean brightness toward the target.
    3. Apply gamma correction, blended by strength.

    Using gamma correction rather than linear scaling preserves highlight
    and shadow detail better than simple brightness shifts.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).
        target_brightness: Desired mean brightness (0 – 255).

    Returns:
        Exposure-corrected BGR image.
    """
    if strength <= 0:
        return image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    current_brightness = np.mean(hsv[:, :, 2])

    if current_brightness < 1:
        current_brightness = 1

    # Compute gamma: gamma < 1 brightens, gamma > 1 darkens
    ratio = target_brightness / current_brightness
    # Limit extreme corrections
    ratio = max(0.3, min(3.0, ratio))
    gamma = 1.0 / ratio

    # Build and apply lookup table for gamma correction
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(image, lut)

    return _blend(image, corrected, strength)


# ======================================================================
# White Balance Adjustment
# ======================================================================

def adjust_white_balance(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Correct colour casts using the Gray World assumption.

    How it works:
    The Gray World algorithm assumes the average colour of a scene should
    be neutral grey. We scale each channel so its mean equals the overall
    mean luminance, effectively removing tinted colour casts from
    artificial lighting or incorrect camera white balance.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).

    Returns:
        White-balance-corrected BGR image.
    """
    if strength <= 0:
        return image.copy()

    result = image.astype(np.float32)
    avg_b, avg_g, avg_r = [np.mean(result[:, :, i]) for i in range(3)]
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r

    corrected = _clamp(result)
    return _blend(image, corrected, strength)


# ======================================================================
# Contrast & Tone Balancing
# ======================================================================

def adjust_contrast(image: np.ndarray, strength: float = 0.5,
                    clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram
    Equalisation).

    How it works:
    CLAHE divides the image into small tiles and performs histogram
    equalisation on each tile independently, with a clip limit to prevent
    over-amplification of noise. This produces natural-looking contrast
    enhancement that adapts to local regions — similar to how a pro
    photographer adjusts tone curves in Lightroom.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0). Higher = more contrast.
        clip_limit: CLAHE clip limit (higher = more contrast per tile).

    Returns:
        Contrast-enhanced BGR image.
    """
    if strength <= 0:
        return image.copy()

    # Scale clip limit by strength
    effective_clip = 1.0 + (clip_limit * strength)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE only to the L (lightness) channel to avoid colour shifts
    clahe = cv2.createCLAHE(clipLimit=effective_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return _blend(image, result, strength)


# ======================================================================
# Highlight & Shadow Recovery
# ======================================================================

def recover_highlights_shadows(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Recover detail in highlights (bright areas) and shadows (dark areas).

    How it works:
    1. Convert to LAB colour space and work on the L channel.
    2. Create a "midtone mask" — pixels near middle brightness pass through,
       while very bright/dark pixels are identified for adjustment.
    3. Compress dynamic range by pulling highlights down and shadows up
       toward the midtones, similar to the Highlights/Shadows sliders in
       Lightroom.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).

    Returns:
        Image with recovered highlights and shadows.
    """
    if strength <= 0:
        return image.copy()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]  # L range in OpenCV LAB: 0-255

    # Define highlight and shadow thresholds
    shadow_thresh = 60
    highlight_thresh = 200

    # Shadow recovery: brighten pixels below threshold
    shadow_mask = np.clip((shadow_thresh - l_channel) / shadow_thresh, 0, 1)
    shadow_boost = shadow_mask * 30 * strength  # max +30 luminance
    l_channel += shadow_boost

    # Highlight recovery: darken pixels above threshold
    highlight_mask = np.clip((l_channel - highlight_thresh) / (255 - highlight_thresh), 0, 1)
    highlight_pull = highlight_mask * 25 * strength  # max -25 luminance
    l_channel -= highlight_pull

    lab[:, :, 0] = np.clip(l_channel, 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return _blend(image, result, strength)


# ======================================================================
# Colour Correction & Vibrance
# ======================================================================

def adjust_vibrance(image: np.ndarray, strength: float = 0.5,
                    amount: float = 30.0) -> np.ndarray:
    """
    Increase colour vibrance without over-saturating already-vivid colours.

    How it works:
    Unlike a flat saturation boost, vibrance selectively increases
    saturation more for muted colours and less for already-saturated ones.
    This produces vivid but natural-looking results — exactly how the
    Vibrance slider works in Lightroom vs the Saturation slider.

    We use a mask based on current saturation: low-saturation pixels
    receive a larger boost while high-saturation pixels are left alone.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).
        amount: Maximum saturation boost for desaturated pixels.

    Returns:
        Vibrance-enhanced BGR image.
    """
    if strength <= 0:
        return image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_channel = hsv[:, :, 1]

    # Create an inverse-saturation mask: less saturated pixels get more boost
    # Normalise saturation to 0–1 range, invert, and square for non-linearity
    sat_normalized = s_channel / 255.0
    vibrance_mask = (1.0 - sat_normalized) ** 1.5

    boost = vibrance_mask * amount * strength
    hsv[:, :, 1] = np.clip(s_channel + boost, 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend(image, result, strength)


def adjust_saturation(image: np.ndarray, strength: float = 0.5,
                      factor: float = 1.2) -> np.ndarray:
    """
    Adjust overall colour saturation.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).
        factor: Saturation multiplier (>1 = more saturated, <1 = less).

    Returns:
        Saturation-adjusted BGR image.
    """
    if strength <= 0:
        return image.copy()

    effective_factor = 1.0 + (factor - 1.0) * strength
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= effective_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


# ======================================================================
# Sharpening
# ======================================================================

def sharpen(image: np.ndarray, strength: float = 0.5,
            kernel_size: int = 3) -> np.ndarray:
    """
    Sharpen the image using unsharp masking.

    How it works:
    Unsharp masking creates a blurred version of the image, then
    subtracts it from the original to isolate high-frequency detail.
    Adding this detail layer back at a controlled strength enhances
    edges and fine texture — the standard "Sharpening" in Lightroom.

    We use Gaussian blur for the low-pass filter, which produces
    cleaner results than a simple kernel convolution.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).
        kernel_size: Blur kernel size (larger = coarser sharpening).

    Returns:
        Sharpened BGR image.
    """
    if strength <= 0:
        return image.copy()

    gpu = get_gpu_accelerator()
    ksize = (kernel_size * 2 + 1, kernel_size * 2 + 1)
    sigma = kernel_size * 0.5

    blurred = gpu.gaussian_blur(image, ksize, sigma)

    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(
        image, 1.0 + strength * 0.8,
        blurred, -strength * 0.8,
        0
    )
    return _clamp(sharpened)


# ======================================================================
# Noise Reduction
# ======================================================================

def reduce_noise(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Reduce image noise using Non-Local Means Denoising.

    How it works:
    Non-Local Means (NLM) works by finding similar patches across the
    entire image and averaging them, which preserves edges much better
    than simple blurring. OpenCV's fastNlMeansDenoisingColored is an
    optimised implementation. Filter strength is scaled by our strength
    parameter.

    This is the digital equivalent of noise reduction in Lightroom's
    Detail panel.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0). Higher = smoother.

    Returns:
        Denoised BGR image.
    """
    if strength <= 0:
        return image.copy()

    # Scale filter strength: NLM h parameter (3–15 typical range)
    h = 3 + strength * 12
    h_color = h

    denoised = cv2.fastNlMeansDenoisingColored(
        image, None,
        h=h,
        hForColorComponents=h_color,
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised


# ======================================================================
# Lens Correction / Straightening (perspective)
# ======================================================================

def auto_straighten(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Attempt to auto-straighten a slightly tilted image.

    How it works:
    1. Detect edges using Canny.
    2. Find dominant lines using Hough Line Transform.
    3. Compute the median angle of near-horizontal lines.
    4. Rotate the image to correct the tilt.

    Only applies correction if the detected tilt is small (< 5°).

    Args:
        image: Input BGR image.
        strength: Edit strength (0–1). At 0, no rotation is applied.

    Returns:
        Straightened BGR image (same dimensions, padded with black if needed).
    """
    if strength <= 0:
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=min(image.shape[1] // 4, 100),
                            maxLineGap=10)

    if lines is None or len(lines) < 3:
        return image.copy()

    # Collect angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines (within ±15° of horizontal)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return image.copy()

    median_angle = np.median(angles)

    # Only correct small tilts (< 5 degrees)
    if abs(median_angle) > 5:
        return image.copy()

    correction_angle = -median_angle * strength
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT_101)

    logger.debug(f"Auto-straighten: corrected {correction_angle:.2f}° tilt")
    return rotated


# ======================================================================
# Skin Smoothing (for Portraits)
# ======================================================================

def _detect_skin_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect skin regions across all human skin tones using a multi-colour-space
    approach.

    Why multiple colour spaces:
    No single colour space reliably captures the full range of human skin
    tones. HSV works well for lighter skin but misses darker complexions.
    YCrCb is more robust across the spectrum because its chrominance
    channels (Cr, Cb) stay relatively stable regardless of luminance —
    meaning it handles very light to very dark skin with the same
    thresholds. We combine both for maximum coverage.

    The detection pipeline:
    1. **YCrCb mask**: Cr (red chroma) and Cb (blue chroma) fall in a
       well-studied range for all human skin tones regardless of
       ethnicity. The thresholds (Cr: 135–180, Cb: 85–135) are based
       on Chai & Ngan (1999) and cover skin from Fitzpatrick Type I
       (very fair) through Type VI (deeply pigmented).
    2. **HSV mask — warm hues**: Captures the orange/red/yellow hue range
       (H: 0–50) that skin occupies. We use a low saturation floor (15)
       and a wide value range (30–255) to include both pale and dark skin.
    3. **HSV mask — reddish wrap-around**: Some deeper skin tones have
       hue values that wrap around past 170 in OpenCV's 0–179 range.
       This band catches those.
    4. **Combine with AND/OR logic**: A pixel is marked as skin if it
       passes the YCrCb test AND at least one of the HSV tests. This
       intersection eliminates false positives (wood, clothing, etc.)
       while retaining true skin across all tones.
    5. **Morphological cleanup**: Close small gaps and smooth the mask
       boundary so the smoothing filter blends seamlessly.

    Args:
        image: Input BGR image (uint8).

    Returns:
        Grayscale mask (uint8, 0–255) where 255 = skin.
    """
    # --- YCrCb colour space (best cross-tone skin detector) ---
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Cr and Cb ranges that span all major skin tones:
    #   Cr 135-180: red chrominance present in all skin
    #   Cb  85-135: blue chrominance inversely present in skin
    #   Y   30-255: allow full luminance range (very dark to very bright)
    lower_ycrcb = np.array([30, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # --- HSV colour space (complementary, hue-based) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Band 1: Warm hues (0–50). Low saturation floor (15) admits pale skin;
    # wide value range (30–255) admits dark skin.
    lower_hsv1 = np.array([0, 15, 30], dtype=np.uint8)
    upper_hsv1 = np.array([50, 255, 255], dtype=np.uint8)
    mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)

    # Band 2: Reddish wrap-around (170–179). Some deep / cool-undertone
    # skin tones register here in OpenCV's 0–179 hue scale.
    lower_hsv2 = np.array([170, 15, 30], dtype=np.uint8)
    upper_hsv2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)

    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # --- Combine: require YCrCb AND HSV agreement ---
    # This intersection dramatically reduces false positives while
    # catching skin across the full Fitzpatrick scale.
    skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # --- Morphological cleanup ---
    # Close small holes inside detected skin (e.g. nostrils, eyebrows)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close)

    # Dilate slightly so the smoothing filter extends just past visible
    # skin edges, avoiding a hard transition line
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel_dilate, iterations=1)

    # Gaussian blur the mask for a soft, feathered boundary
    skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

    return skin_mask


def smooth_skin(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply mild skin smoothing suitable for portrait photography.

    How it works:
    1. Detect skin regions using a multi-colour-space approach (YCrCb + HSV)
       that reliably covers all human skin tones — from very fair
       (Fitzpatrick Type I) through deeply pigmented (Fitzpatrick Type VI).
    2. Apply bilateral filtering to smooth those regions while
       preserving edges (pores, eyes, lips stay sharp).
    3. Blend the smoothed version back only in skin areas, weighted by
       both the mask confidence and the strength parameter.

    Bilateral filtering is the industry standard for skin retouching
    because it smooths noise/texture while preserving edge contrast,
    unlike Gaussian blur which makes everything uniformly soft.

    The strength of smoothing adapts automatically: the bilateral
    filter's sigma parameters scale with the strength setting, so
    "light" gives a barely-perceptible polish while "strong" gives
    a more magazine-style finish. In all cases the mask confines the
    effect exclusively to skin, leaving hair, eyes, lips, and
    background untouched.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0). Keep low for natural results.

    Returns:
        Image with smoothed skin regions.
    """
    if strength <= 0:
        return image.copy()

    # Detect skin across all tones (see _detect_skin_mask for details)
    skin_mask = _detect_skin_mask(image)

    # Bilateral filter for skin smoothing
    # d: filter diameter (5–15 px, larger = more smoothing)
    # sigma_colour: how much colour difference is tolerated (preserves edges)
    # sigma_space: spatial reach of the filter
    d = int(5 + strength * 10)
    sigma_colour = 50 + strength * 50
    sigma_space = 50 + strength * 50
    smoothed = cv2.bilateralFilter(image, d, sigma_colour, sigma_space)

    # Blend: apply smoothed only where skin is detected, scaled by strength
    mask_float = (skin_mask / 255.0 * strength)[:, :, np.newaxis]
    result = (image.astype(np.float32) * (1 - mask_float) +
              smoothed.astype(np.float32) * mask_float)

    return _clamp(result)


# ======================================================================
# Vignette
# ======================================================================

def apply_vignette(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply a subtle vignette effect (darken edges).

    How it works:
    Creates a radial gradient mask centered on the image, where the
    edges are darker. This draws the viewer's eye toward the center
    of the frame — a common technique in portrait and fine-art
    photography.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).

    Returns:
        Image with vignette effect.
    """
    if strength <= 0:
        return image.copy()

    h, w = image.shape[:2]

    # Create radial gradient
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    # Normalise distances to range 0–1
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist_norm = dist / max_dist

    # Vignette mask: 1 at center, darker at edges
    vignette_mask = 1.0 - (dist_norm ** 2) * strength * 0.6
    vignette_mask = np.clip(vignette_mask, 0.3, 1.0)

    result = image.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= vignette_mask

    return _clamp(result)


# ======================================================================
# Colour Temperature (Warm / Cool)
# ======================================================================

def adjust_temperature(image: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Shift colour temperature: positive = warmer (amber), negative = cooler (blue).

    How it works:
    Adjusts the Blue and Red channels inversely. Warming boosts reds/yellows
    and reduces blues; cooling does the opposite. This simulates the
    Kelvin / Temperature slider in Lightroom.

    Args:
        image: Input BGR image.
        strength: -1.0 (cold / blue) to +1.0 (warm / amber). 0 = no change.

    Returns:
        Temperature-adjusted BGR image.
    """
    if abs(strength) < 0.01:
        return image.copy()

    result = image.astype(np.float32)
    shift = strength * 20  # max ±20 per channel

    result[:, :, 0] -= shift  # Blue channel
    result[:, :, 2] += shift  # Red channel

    return _clamp(result)


# ======================================================================
# Tone Curve (S-curve for cinematic look)
# ======================================================================

def apply_tone_curve(image: np.ndarray, strength: float = 0.5,
                     shadows_lift: float = 0.0,
                     highlights_compress: float = 0.0) -> np.ndarray:
    """
    Apply a subtle S-curve to enhance tonal range.

    How it works:
    An S-curve darkens shadows slightly and brightens highlights,
    increasing contrast in the midtones. Optional shadows_lift raises
    the black point (for a "faded" look) and highlights_compress lowers
    the white point (for reduced glare).

    Args:
        image: Input BGR image.
        strength: Overall curve strength (0.0 – 1.0).
        shadows_lift: Raise the black point (0 – 30).
        highlights_compress: Lower the white point (0 – 30).

    Returns:
        Tone-curve-adjusted BGR image.
    """
    if strength <= 0:
        return image.copy()

    # Build look-up table for S-curve
    lut = np.arange(256, dtype=np.float32)

    # S-curve via sine function (smooth, natural)
    curve_strength = strength * 0.3  # keep subtle
    lut = lut + curve_strength * 128 * np.sin(np.pi * lut / 255.0)

    # Apply shadows lift (raise black point)
    if shadows_lift > 0:
        lut = lut * (1 - shadows_lift / 255.0) + shadows_lift

    # Apply highlights compression (lower white point)
    if highlights_compress > 0:
        lut = lut * (1 - highlights_compress / 255.0)

    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply LUT to the L channel of LAB to avoid colour shifts
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


# ======================================================================
# Dehaze (for hazy / foggy images)
# ======================================================================

def dehaze(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Remove haze or fog from an image using dark channel prior.

    How it works:
    The Dark Channel Prior (He et al., 2009) observes that in most
    outdoor images, at least one colour channel has very low intensity
    in non-sky patches. Haze raises these minimums. By estimating the
    haze transmission map and atmospheric light, we can invert the
    hazing process. We use a simplified version for speed.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).

    Returns:
        Dehazed BGR image.
    """
    if strength <= 0:
        return image.copy()

    img_float = image.astype(np.float64) / 255.0

    # Compute dark channel (minimum across colour channels in a local patch)
    patch_size = 15
    dark_channel = np.min(img_float, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(dark_channel, kernel)

    # Estimate atmospheric light (brightest pixels in dark channel)
    flat_dark = dark_channel.ravel()
    num_pixels = len(flat_dark)
    top_fraction = max(int(num_pixels * 0.001), 1)
    indices = np.argsort(flat_dark)[-top_fraction:]

    flat_img = img_float.reshape(-1, 3)
    atm_light = np.mean(flat_img[indices], axis=0)
    atm_light = np.clip(atm_light, 0.1, 1.0)

    # Estimate transmission map
    normalised = img_float / atm_light
    transmission = 1 - strength * 0.9 * np.min(normalised, axis=2)
    transmission = np.clip(transmission, 0.1, 1.0)

    # Refine transmission with guided filter (approximate with bilateral)
    transmission = cv2.bilateralFilter(
        transmission.astype(np.float32), 9, 0.1, 10
    )

    # Recover scene
    transmission_3d = transmission[:, :, np.newaxis]
    result = (img_float - atm_light) / np.maximum(transmission_3d, 0.1) + atm_light
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return _blend(image, result, strength)


# ======================================================================
# Auto Levels (stretch histogram to full range)
# ======================================================================

def auto_levels(image: np.ndarray, strength: float = 0.5,
                clip_percent: float = 1.0) -> np.ndarray:
    """
    Stretch the histogram of each channel so the full tonal range is used.

    How it works:
    Clips the darkest and brightest N% of pixels, then linearly maps
    the remaining range to 0–255. This is the digital equivalent of
    "Auto Levels" in Photoshop / the auto-tone feature in Lightroom.

    Args:
        image: Input BGR image.
        strength: Edit strength (0.0 – 1.0).
        clip_percent: Percentage of pixels to clip at each end.

    Returns:
        Levels-adjusted BGR image.
    """
    if strength <= 0:
        return image.copy()

    result = image.copy().astype(np.float32)

    for c in range(3):
        channel = result[:, :, c]
        low = np.percentile(channel, clip_percent)
        high = np.percentile(channel, 100 - clip_percent)

        if high - low < 10:
            continue

        channel = (channel - low) * 255.0 / (high - low)
        result[:, :, c] = channel

    corrected = _clamp(result)
    return _blend(image, corrected, strength)
