"""
Editing Profiles Module.
Defines preset editing profiles that bundle individual edit operations with
appropriate parameters for different styles and image categories.

Each profile is a dictionary mapping edit function names to their parameters.
The pipeline iterates through these in order to produce the final result.

Profiles are modulated by:
  - Style: auto, natural, vibrant, high_contrast, portrait
  - Category: the classified image type
  - Strength: light (0.3), medium (0.6), strong (0.9)

The "auto" style selects parameters based on the image category.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from src.core.image_classifier import ImageCategory
from src.utils.logger import get_logger

logger = get_logger("photo_editor.profiles")


class EditStyle(str, Enum):
    """Available editing styles the user can choose from."""
    AUTO = "auto"
    NATURAL = "natural"
    VIBRANT = "vibrant"
    HIGH_CONTRAST = "high_contrast"
    PORTRAIT = "portrait"


class EditStrength(str, Enum):
    """Strength levels for the editing pipeline."""
    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"


# Multiplier applied to all strength values based on EditStrength
STRENGTH_MULTIPLIERS = {
    EditStrength.LIGHT: 0.4,
    EditStrength.MEDIUM: 0.7,
    EditStrength.STRONG: 1.0,
}


@dataclass
class EditStep:
    """
    A single step in the editing pipeline.

    Attributes:
        name: Human-readable name for UI display.
        function_name: Name of the function in image_edits module.
        enabled: Whether this step is active.
        params: Keyword arguments to pass to the function.
    """
    name: str
    function_name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EditProfile:
    """
    A complete editing profile consisting of ordered edit steps.

    Attributes:
        style: The style this profile represents.
        category: The image category this profile is tuned for.
        steps: Ordered list of editing steps to apply.
    """
    style: EditStyle
    category: ImageCategory
    steps: List[EditStep] = field(default_factory=list)


# ======================================================================
# Profile Definitions
# ======================================================================

def _base_steps() -> List[EditStep]:
    """
    Return the base set of editing steps common to all profiles.
    These represent a complete professional editing workflow in order.
    """
    return [
        EditStep("Auto Levels", "auto_levels", True, {"strength": 0.3, "clip_percent": 0.5}),
        EditStep("Exposure Correction", "adjust_exposure", True, {"strength": 0.5, "target_brightness": 128}),
        EditStep("White Balance", "adjust_white_balance", True, {"strength": 0.4}),
        EditStep("Highlight/Shadow Recovery", "recover_highlights_shadows", True, {"strength": 0.5}),
        EditStep("Contrast", "adjust_contrast", True, {"strength": 0.4, "clip_limit": 2.0}),
        EditStep("Tone Curve", "apply_tone_curve", True, {"strength": 0.3}),
        EditStep("Vibrance", "adjust_vibrance", True, {"strength": 0.4, "amount": 25}),
        EditStep("Saturation", "adjust_saturation", False, {"strength": 0.0, "factor": 1.0}),
        EditStep("Colour Temperature", "adjust_temperature", False, {"strength": 0.0}),
        EditStep("Dehaze", "dehaze", False, {"strength": 0.0}),
        EditStep("Noise Reduction", "reduce_noise", True, {"strength": 0.3}),
        EditStep("Sharpening", "sharpen", True, {"strength": 0.35, "kernel_size": 3}),
        EditStep("Straighten", "auto_straighten", True, {"strength": 0.5}),
        EditStep("Skin Smoothing", "smooth_skin", False, {"strength": 0.0}),
        EditStep("Vignette", "apply_vignette", False, {"strength": 0.0}),
    ]


def _find_step(steps: List[EditStep], function_name: str) -> Optional[EditStep]:
    """Find a step by function name."""
    for step in steps:
        if step.function_name == function_name:
            return step
    return None


# ------------------------------------------------------------------
# Natural Style
# ------------------------------------------------------------------

def get_natural_profile(category: ImageCategory) -> EditProfile:
    """
    Natural style: Subtle corrections that preserve the original look.
    Suitable for documentary, journalism, and when authenticity matters.
    """
    steps = _base_steps()

    # Light touch on everything
    _find_step(steps, "auto_levels").params["strength"] = 0.25
    _find_step(steps, "adjust_exposure").params["strength"] = 0.35
    _find_step(steps, "adjust_white_balance").params["strength"] = 0.3
    _find_step(steps, "recover_highlights_shadows").params["strength"] = 0.3
    _find_step(steps, "adjust_contrast").params["strength"] = 0.2
    _find_step(steps, "adjust_contrast").params["clip_limit"] = 1.5
    _find_step(steps, "apply_tone_curve").params["strength"] = 0.15
    _find_step(steps, "adjust_vibrance").params["strength"] = 0.2
    _find_step(steps, "adjust_vibrance").params["amount"] = 15
    _find_step(steps, "reduce_noise").params["strength"] = 0.2
    _find_step(steps, "sharpen").params["strength"] = 0.25

    # Category-specific tweaks
    if category == ImageCategory.PORTRAIT:
        skin = _find_step(steps, "smooth_skin")
        skin.enabled = True
        skin.params["strength"] = 0.15

    elif category == ImageCategory.LANDSCAPE:
        _find_step(steps, "adjust_vibrance").params["strength"] = 0.3
        dehaze = _find_step(steps, "dehaze")
        dehaze.enabled = True
        dehaze.params["strength"] = 0.2

    elif category == ImageCategory.NIGHT:
        _find_step(steps, "adjust_exposure").params["target_brightness"] = 100
        _find_step(steps, "reduce_noise").params["strength"] = 0.4
        _find_step(steps, "recover_highlights_shadows").params["strength"] = 0.4

    elif category == ImageCategory.DOCUMENT:
        _find_step(steps, "adjust_contrast").params["strength"] = 0.4
        _find_step(steps, "sharpen").params["strength"] = 0.4
        _find_step(steps, "adjust_vibrance").enabled = False

    return EditProfile(style=EditStyle.NATURAL, category=category, steps=steps)


# ------------------------------------------------------------------
# Vibrant Style
# ------------------------------------------------------------------

def get_vibrant_profile(category: ImageCategory) -> EditProfile:
    """
    Vibrant style: Punchy colours and enhanced visual impact.
    Popular for social media, travel, and lifestyle photography.
    """
    steps = _base_steps()

    _find_step(steps, "auto_levels").params["strength"] = 0.4
    _find_step(steps, "adjust_exposure").params["strength"] = 0.5
    _find_step(steps, "adjust_white_balance").params["strength"] = 0.4
    _find_step(steps, "recover_highlights_shadows").params["strength"] = 0.5
    _find_step(steps, "adjust_contrast").params["strength"] = 0.5
    _find_step(steps, "adjust_contrast").params["clip_limit"] = 2.5
    _find_step(steps, "apply_tone_curve").params["strength"] = 0.4
    _find_step(steps, "adjust_vibrance").params["strength"] = 0.7
    _find_step(steps, "adjust_vibrance").params["amount"] = 40

    # Enable saturation boost for vibrant style
    sat = _find_step(steps, "adjust_saturation")
    sat.enabled = True
    sat.params["strength"] = 0.5
    sat.params["factor"] = 1.25

    _find_step(steps, "reduce_noise").params["strength"] = 0.25
    _find_step(steps, "sharpen").params["strength"] = 0.45

    # Category-specific tweaks
    if category == ImageCategory.LANDSCAPE:
        _find_step(steps, "adjust_vibrance").params["strength"] = 0.8
        dehaze = _find_step(steps, "dehaze")
        dehaze.enabled = True
        dehaze.params["strength"] = 0.4
        vig = _find_step(steps, "apply_vignette")
        vig.enabled = True
        vig.params["strength"] = 0.2

    elif category == ImageCategory.PORTRAIT:
        skin = _find_step(steps, "smooth_skin")
        skin.enabled = True
        skin.params["strength"] = 0.2
        # Warm tone for portraits
        temp = _find_step(steps, "adjust_temperature")
        temp.enabled = True
        temp.params["strength"] = 0.15

    elif category == ImageCategory.NIGHT:
        _find_step(steps, "adjust_exposure").params["target_brightness"] = 110
        _find_step(steps, "reduce_noise").params["strength"] = 0.45
        _find_step(steps, "adjust_contrast").params["strength"] = 0.6

    elif category == ImageCategory.PRODUCT:
        _find_step(steps, "sharpen").params["strength"] = 0.5
        _find_step(steps, "adjust_contrast").params["strength"] = 0.5

    return EditProfile(style=EditStyle.VIBRANT, category=category, steps=steps)


# ------------------------------------------------------------------
# High Contrast / Dramatic Style
# ------------------------------------------------------------------

def get_high_contrast_profile(category: ImageCategory) -> EditProfile:
    """
    High Contrast / Dramatic style: Bold tones, deep shadows, bright highlights.
    Ideal for artistic, editorial, and fine-art photography.
    """
    steps = _base_steps()

    _find_step(steps, "auto_levels").params["strength"] = 0.5
    _find_step(steps, "auto_levels").params["clip_percent"] = 1.5
    _find_step(steps, "adjust_exposure").params["strength"] = 0.5
    _find_step(steps, "adjust_white_balance").params["strength"] = 0.35
    _find_step(steps, "recover_highlights_shadows").params["strength"] = 0.3  # Less recovery = more drama
    _find_step(steps, "adjust_contrast").params["strength"] = 0.7
    _find_step(steps, "adjust_contrast").params["clip_limit"] = 3.0
    _find_step(steps, "apply_tone_curve").params["strength"] = 0.6
    _find_step(steps, "adjust_vibrance").params["strength"] = 0.4
    _find_step(steps, "adjust_vibrance").params["amount"] = 25
    _find_step(steps, "reduce_noise").params["strength"] = 0.2
    _find_step(steps, "sharpen").params["strength"] = 0.5
    _find_step(steps, "sharpen").params["kernel_size"] = 3

    # Strong vignette for dramatic effect
    vig = _find_step(steps, "apply_vignette")
    vig.enabled = True
    vig.params["strength"] = 0.4

    # Category-specific tweaks
    if category == ImageCategory.PORTRAIT:
        skin = _find_step(steps, "smooth_skin")
        skin.enabled = True
        skin.params["strength"] = 0.15
        vig.params["strength"] = 0.5

    elif category == ImageCategory.LANDSCAPE:
        dehaze = _find_step(steps, "dehaze")
        dehaze.enabled = True
        dehaze.params["strength"] = 0.3
        _find_step(steps, "adjust_vibrance").params["strength"] = 0.5

    elif category == ImageCategory.NIGHT:
        _find_step(steps, "adjust_exposure").params["target_brightness"] = 90
        _find_step(steps, "adjust_contrast").params["strength"] = 0.8
        _find_step(steps, "reduce_noise").params["strength"] = 0.35

    return EditProfile(style=EditStyle.HIGH_CONTRAST, category=category, steps=steps)


# ------------------------------------------------------------------
# Portrait Style
# ------------------------------------------------------------------

def get_portrait_profile(category: ImageCategory) -> EditProfile:
    """
    Portrait style: Optimised for flattering skin tones and soft look.
    Suitable for headshots, family portraits, and fashion photography.
    """
    steps = _base_steps()

    _find_step(steps, "auto_levels").params["strength"] = 0.3
    _find_step(steps, "adjust_exposure").params["strength"] = 0.45
    _find_step(steps, "adjust_exposure").params["target_brightness"] = 135  # slightly bright
    _find_step(steps, "adjust_white_balance").params["strength"] = 0.4
    _find_step(steps, "recover_highlights_shadows").params["strength"] = 0.5
    _find_step(steps, "adjust_contrast").params["strength"] = 0.3
    _find_step(steps, "adjust_contrast").params["clip_limit"] = 1.8
    _find_step(steps, "apply_tone_curve").params["strength"] = 0.25
    _find_step(steps, "adjust_vibrance").params["strength"] = 0.3
    _find_step(steps, "adjust_vibrance").params["amount"] = 20
    _find_step(steps, "reduce_noise").params["strength"] = 0.35
    _find_step(steps, "sharpen").params["strength"] = 0.3

    # Enable skin smoothing
    skin = _find_step(steps, "smooth_skin")
    skin.enabled = True
    skin.params["strength"] = 0.3

    # Warm colour temperature for flattering skin
    temp = _find_step(steps, "adjust_temperature")
    temp.enabled = True
    temp.params["strength"] = 0.1

    # Subtle vignette to draw focus to the subject
    vig = _find_step(steps, "apply_vignette")
    vig.enabled = True
    vig.params["strength"] = 0.25

    return EditProfile(style=EditStyle.PORTRAIT, category=category, steps=steps)


# ------------------------------------------------------------------
# Auto Style (delegates to category-appropriate profile)
# ------------------------------------------------------------------

def get_auto_profile(category: ImageCategory) -> EditProfile:
    """
    Auto style: Automatically selects the best editing approach based
    on image classification. Maps each category to the most suitable
    style profile.

    Mapping:
      - portrait → portrait style
      - landscape / outdoor → vibrant style
      - night → high_contrast style
      - document → natural style (minimal changes)
      - everything else → natural style with category tweaks
    """
    if category == ImageCategory.PORTRAIT:
        profile = get_portrait_profile(category)
    elif category in (ImageCategory.LANDSCAPE, ImageCategory.OUTDOOR):
        profile = get_vibrant_profile(category)
    elif category == ImageCategory.NIGHT:
        profile = get_high_contrast_profile(category)
    elif category == ImageCategory.DOCUMENT:
        profile = get_natural_profile(category)
    elif category == ImageCategory.PRODUCT:
        profile = get_natural_profile(category)
        # Product-specific: brighter, sharper, clean
        for step in profile.steps:
            if step.function_name == "sharpen":
                step.params["strength"] = 0.45
            if step.function_name == "adjust_exposure":
                step.params["target_brightness"] = 140
    elif category == ImageCategory.EVENT:
        profile = get_natural_profile(category)
        # Event: moderate vibrance, handle mixed lighting
        for step in profile.steps:
            if step.function_name == "adjust_white_balance":
                step.params["strength"] = 0.5
            if step.function_name == "adjust_vibrance":
                step.params["strength"] = 0.35
    else:
        profile = get_natural_profile(category)

    profile.style = EditStyle.AUTO
    return profile


# ======================================================================
# Public API
# ======================================================================

def get_profile(style: EditStyle, category: ImageCategory,
                edit_strength: EditStrength = EditStrength.MEDIUM) -> EditProfile:
    """
    Get an editing profile for the given style, category, and strength.

    This is the main entry point for obtaining a profile. The returned
    profile has all step strengths scaled by the global strength multiplier.

    Args:
        style: Desired editing style.
        category: Image category from classifier.
        edit_strength: Global strength level (light / medium / strong).

    Returns:
        EditProfile with all steps configured.
    """
    profile_getters = {
        EditStyle.AUTO: get_auto_profile,
        EditStyle.NATURAL: get_natural_profile,
        EditStyle.VIBRANT: get_vibrant_profile,
        EditStyle.HIGH_CONTRAST: get_high_contrast_profile,
        EditStyle.PORTRAIT: get_portrait_profile,
    }

    getter = profile_getters.get(style, get_natural_profile)
    profile = getter(category)

    # Apply global strength multiplier to all step strengths
    multiplier = STRENGTH_MULTIPLIERS[edit_strength]
    for step in profile.steps:
        if "strength" in step.params:
            step.params["strength"] *= multiplier

    logger.info(
        f"Profile: style={style.value}, category={category.value}, "
        f"strength={edit_strength.value} (×{multiplier})"
    )
    return profile


def get_available_styles() -> List[Dict[str, str]]:
    """Return list of available styles with display names and descriptions."""
    return [
        {"value": EditStyle.AUTO.value, "name": "Auto", "description": "Automatically choose the best style based on image content"},
        {"value": EditStyle.NATURAL.value, "name": "Natural", "description": "Subtle corrections that preserve the original look"},
        {"value": EditStyle.VIBRANT.value, "name": "Vibrant", "description": "Punchy colours and enhanced visual impact"},
        {"value": EditStyle.HIGH_CONTRAST.value, "name": "High Contrast", "description": "Bold tones and dramatic mood"},
        {"value": EditStyle.PORTRAIT.value, "name": "Portrait", "description": "Flattering skin tones and soft, polished look"},
    ]
