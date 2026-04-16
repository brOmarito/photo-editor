"""
Editing Pipeline Module.
Orchestrates the full image editing workflow: load → classify → edit → save.

Runs editing steps in sequence from an EditProfile, emitting progress
signals so the UI can display real-time updates. Supports cancellation
and error recovery.
"""

import os
import time
import traceback
from typing import Optional, List, Dict, Any, Callable

import numpy as np
import cv2

from src.core import image_edits
from src.core.image_loader import load_image, discover_images
from src.core.image_classifier import classify_image, ImageCategory
from src.core.editing_profiles import (
    EditProfile, EditStyle, EditStrength, EditStep, get_profile,
)
from src.core.metadata import copy_metadata
from src.core.file_output import save_image, build_output_path
from src.utils.logger import get_logger

logger = get_logger("photo_editor.pipeline")


# Map from EditStep.function_name → actual function in image_edits module
EDIT_FUNCTIONS = {
    "auto_levels": image_edits.auto_levels,
    "adjust_exposure": image_edits.adjust_exposure,
    "adjust_white_balance": image_edits.adjust_white_balance,
    "recover_highlights_shadows": image_edits.recover_highlights_shadows,
    "adjust_contrast": image_edits.adjust_contrast,
    "apply_tone_curve": image_edits.apply_tone_curve,
    "adjust_vibrance": image_edits.adjust_vibrance,
    "adjust_saturation": image_edits.adjust_saturation,
    "adjust_temperature": image_edits.adjust_temperature,
    "dehaze": image_edits.dehaze,
    "reduce_noise": image_edits.reduce_noise,
    "sharpen": image_edits.sharpen,
    "auto_straighten": image_edits.auto_straighten,
    "smooth_skin": image_edits.smooth_skin,
    "apply_vignette": image_edits.apply_vignette,
}


class EditResult:
    """
    Result container for a single image edit.

    Attributes:
        file_path: Original file path.
        output_path: Path where the edited image was saved (or None).
        success: Whether processing succeeded.
        error: Error message if failed.
        category: Detected image category.
        style: Editing style applied.
        duration: Processing time in seconds.
        original_image: Original image array (for preview).
        edited_image: Edited image array (for preview).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_path: Optional[str] = None
        self.success: bool = False
        self.error: Optional[str] = None
        self.category: Optional[ImageCategory] = None
        self.confidence: float = 0.0
        self.style: Optional[EditStyle] = None
        self.duration: float = 0.0
        self.original_image: Optional[np.ndarray] = None
        self.edited_image: Optional[np.ndarray] = None


def apply_profile(image: np.ndarray, profile: EditProfile,
                  step_callback: Optional[Callable] = None,
                  disabled_edits: Optional[set] = None,
                  edit_intensities: Optional[dict] = None) -> np.ndarray:
    """
    Apply all enabled edit steps from a profile to an image.

    Args:
        image: Input BGR image (uint8).
        profile: EditProfile containing ordered EditSteps.
        step_callback: Optional callback(step_name, step_index, total_steps)
                        called before each step for progress updates.
        disabled_edits: Optional set of function_name strings to skip.
        edit_intensities: Optional dict mapping function_name to an
                          intensity percentage (0–100).  100 = full
                          effect (default), 0 = no effect.  Values
                          between 0 and 100 alpha-blend the step’s
                          result with the pre-step image.

    Returns:
        Edited BGR image (uint8).
    """
    result = image.copy()
    total = len(profile.steps)
    disabled = disabled_edits or set()
    intensities = edit_intensities or {}

    for i, step in enumerate(profile.steps):
        if not step.enabled:
            continue
        if step.function_name in disabled:
            logger.debug(f"  Skipping disabled edit: {step.name}")
            continue

        func = EDIT_FUNCTIONS.get(step.function_name)
        if func is None:
            logger.warning(f"  Unknown edit function: {step.function_name}")
            continue

        if step_callback:
            step_callback(step.name, i, total)

        try:
            logger.debug(f"  Applying: {step.name} ({step.function_name}) params={step.params}")
            step_result = func(result, **step.params)

            # --- Intensity blending ---
            intensity = intensities.get(step.function_name, 100)
            if intensity >= 100:
                result = step_result
            elif intensity <= 0:
                pass  # keep result unchanged (effectively skip)
            else:
                alpha = intensity / 100.0
                result = cv2.addWeighted(
                    step_result, alpha, result, 1.0 - alpha, 0
                )
        except Exception as e:
            logger.warning(f"  Edit step '{step.name}' failed: {e}")
            # Continue with what we have rather than failing entirely

    return result


def process_single_image(
    file_path: str,
    style: EditStyle = EditStyle.AUTO,
    edit_strength: EditStrength = EditStrength.MEDIUM,
    output_dir: Optional[str] = None,
    output_format: Optional[str] = None,
    quality: int = 95,
    category_override: Optional[ImageCategory] = None,
    disabled_edits: Optional[set] = None,
    edit_intensities: Optional[dict] = None,
    step_callback: Optional[Callable] = None,
    keep_images: bool = False,
    preview_only: bool = False,
) -> EditResult:
    """
    Process a single image through the full pipeline.

    Steps:
    1. Load the image.
    2. Classify image type (or use override).
    3. Get the editing profile.
    4. Apply all edit steps.
    5. Save the result (skipped when *preview_only* is True).
    6. Copy metadata from original (skipped when *preview_only* is True).

    Args:
        file_path: Path to the source image.
        style: Editing style to apply.
        edit_strength: Global strength level.
        output_dir: Output directory (auto-determined if None).
        output_format: Output file format (preserves original if None).
        quality: JPEG/WebP quality (1-100).
        category_override: Force a specific image category.
        disabled_edits: Set of edit function names to skip.
        edit_intensities: Dict mapping edit function names to 0–100 intensity.
        step_callback: Progress callback for individual edit steps.
        keep_images: If True, store original/edited arrays in result (for preview).
        preview_only: If True, skip save and metadata steps (preview rendering).

    Returns:
        EditResult with processing outcome.
    """
    result = EditResult(file_path)
    start_time = time.time()

    try:
        # ----------------------------------------------------------
        # Step 1: Load
        # ----------------------------------------------------------
        logger.info(f"Processing: {os.path.basename(file_path)}")
        original = load_image(file_path)
        if original is None:
            result.error = "Failed to load image (unsupported or corrupted)"
            logger.error(f"  {result.error}")
            return result

        if keep_images:
            result.original_image = original.copy()

        # ----------------------------------------------------------
        # Step 2: Classify
        # ----------------------------------------------------------
        if category_override:
            category = category_override
            confidence = 1.0
        else:
            classification = classify_image(original)
            category = classification["category"]
            confidence = classification["confidence"]

        result.category = category
        result.confidence = confidence

        # ----------------------------------------------------------
        # Step 3: Get profile
        # ----------------------------------------------------------
        profile = get_profile(style, category, edit_strength)
        result.style = profile.style

        # ----------------------------------------------------------
        # Step 4: Apply edits
        # ----------------------------------------------------------
        edited = apply_profile(original, profile, step_callback,
                               disabled_edits, edit_intensities)

        if keep_images:
            result.edited_image = edited.copy()

        # ----------------------------------------------------------
        # Step 5: Save (skipped for preview-only renders)
        # ----------------------------------------------------------
        if not preview_only:
            out_path = build_output_path(
                file_path, style, output_dir=output_dir,
                output_format=output_format
            )
            save_image(edited, out_path, quality=quality)
            result.output_path = out_path

            # ----------------------------------------------------------
            # Step 6: Copy metadata
            # ----------------------------------------------------------
            try:
                copy_metadata(file_path, out_path)
            except Exception as e:
                logger.warning(f"  Metadata copy failed (non-critical): {e}")

        result.success = True
        result.duration = time.time() - start_time
        logger.info(
            f"  Done: {os.path.basename(out_path)} "
            f"({category.value}, {result.duration:.1f}s)"
        )

    except Exception as e:
        result.error = str(e)
        result.duration = time.time() - start_time
        logger.error(f"  Failed: {e}\n{traceback.format_exc()}")

    return result


def process_batch(
    paths: List[str],
    style: EditStyle = EditStyle.AUTO,
    edit_strength: EditStrength = EditStrength.MEDIUM,
    output_dir: Optional[str] = None,
    output_format: Optional[str] = None,
    quality: int = 95,
    category_override: Optional[ImageCategory] = None,
    disabled_edits: Optional[set] = None,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> List[EditResult]:
    """
    Process a batch of images.

    Args:
        paths: List of file/folder paths to process.
        style: Editing style.
        edit_strength: Global strength.
        output_dir: Output directory override.
        output_format: Output format override.
        quality: Save quality.
        category_override: Force category for all images.
        disabled_edits: Set of edits to skip.
        progress_callback: Called with (current_index, total, file_name, result).
        cancel_check: Called before each image; return True to cancel.

    Returns:
        List of EditResult objects (one per image).
    """
    # Discover all images from the given paths
    all_files: List[str] = []
    for path in paths:
        all_files.extend(discover_images(path))

    if not all_files:
        logger.warning("No supported images found in the provided paths.")
        return []

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    all_files = unique_files

    total = len(all_files)
    logger.info(f"Batch processing: {total} image(s)")

    results: List[EditResult] = []

    for i, file_path in enumerate(all_files):
        # Check for cancellation
        if cancel_check and cancel_check():
            logger.info("Batch processing cancelled by user.")
            break

        result = process_single_image(
            file_path,
            style=style,
            edit_strength=edit_strength,
            output_dir=output_dir,
            output_format=output_format,
            quality=quality,
            category_override=category_override,
            disabled_edits=disabled_edits,
        )
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, os.path.basename(file_path), result)

    # Summary
    success_count = sum(1 for r in results if r.success)
    fail_count = sum(1 for r in results if not r.success)
    total_time = sum(r.duration for r in results)
    logger.info(
        f"Batch complete: {success_count}/{len(results)} succeeded, "
        f"{fail_count} failed, total time {total_time:.1f}s"
    )

    return results
