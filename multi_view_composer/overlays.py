"""Overlay rendering for multi-view composer with configurable templates."""

import cv2
import numpy as np
import threading
from typing import Dict, Any, Tuple

from .config import (
    ViewerConfig, TextOverlayConfig, OverlayStyle,
    CentermarkConfig, BorderConfig
)
from .template_engine import (
    build_context, render_template, evaluate_condition, evaluate_color_rules
)
from .logging_config import get_logger


logger = get_logger("overlays")


# Thread-safe cache for rendered overlay text and colors
# Key: (overlay_id, data_cache_key) -> (text, color, visible)
_overlay_cache: Dict[Tuple[str, tuple], Tuple[str, Tuple[int, int, int], bool]] = {}
_cache_lock = threading.Lock()


# OpenCV font mapping
FONT_MAP = {
    "HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}


def make_cache_key(data: Dict[str, Any]) -> tuple:
    """Create a hashable cache key from a dictionary."""
    return tuple(sorted(data.items()))


def get_cv_font(font_name: str) -> int:
    """Get OpenCV font constant from string name."""
    return FONT_MAP.get(font_name, cv2.FONT_HERSHEY_SIMPLEX)


def draw_text_box(
    img: np.ndarray,
    text: str,
    y_offset: int,
    color: Tuple[int, int, int],
    style: OverlayStyle
) -> None:
    """
    Draw a text box with background on image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        text: Text to render
        y_offset: Y position offset
        color: Text color (BGR)
        style: OverlayStyle configuration
    """
    h, w = img.shape[:2]
    font = get_cv_font(style.font)

    # Draw background rectangle
    cv2.rectangle(
        img,
        (0, y_offset),
        (w, y_offset + style.box_height),
        style.background_color,
        -1
    )

    # Draw text
    text_y = y_offset + style.padding_top
    cv2.putText(
        img,
        text,
        (style.padding_left, text_y),
        font,
        style.font_scale,
        color,
        style.thickness,
        cv2.LINE_AA
    )


def _compute_overlay(
    overlay_config: TextOverlayConfig,
    sensor_data: Dict[str, Any],
    cache_key: tuple
) -> Tuple[str, Tuple[int, int, int], bool]:
    """
    Compute overlay text and color, with thread-safe caching.

    Returns:
        Tuple of (text, color, visible)
    """
    full_key = (overlay_config.id, cache_key)

    # Check cache first (with lock)
    with _cache_lock:
        if full_key in _overlay_cache:
            return _overlay_cache[full_key]

    # Build context from sensor data and overlay variables
    context = build_context(sensor_data, overlay_config.variables)

    # Check visibility condition
    visible = True
    if overlay_config.visible_when:
        visible = evaluate_condition(overlay_config.visible_when, context)

    if not visible:
        result = ("", (255, 255, 255), False)
        with _cache_lock:
            _overlay_cache[full_key] = result
        return result

    # Render template
    text = render_template(overlay_config.template, context)

    # Determine color
    if overlay_config.color:
        color = overlay_config.color
    elif overlay_config.color_rules:
        color = evaluate_color_rules(
            overlay_config.color_rules,
            context,
            default_color=(255, 255, 255)
        )
    else:
        color = (255, 255, 255)

    result = (text, color, True)
    with _cache_lock:
        _overlay_cache[full_key] = result
    return result


def draw_text_overlay(
    img: np.ndarray,
    overlay_config: TextOverlayConfig,
    sensor_data: Dict[str, Any],
    y_offset: int,
    default_style: OverlayStyle,
    cache_key: tuple = None
) -> int:
    """
    Draw a single text overlay based on config.

    Args:
        img: BGR image to draw on (modified in-place)
        overlay_config: Overlay configuration
        sensor_data: Dictionary of sensor values
        y_offset: Current Y position offset
        default_style: Default overlay style
        cache_key: Optional cache key for sensor data (enables caching)

    Returns:
        New y_offset after this overlay
    """
    # Compute text and color (with caching if cache_key provided)
    if cache_key is not None:
        text, color, visible = _compute_overlay(overlay_config, sensor_data, cache_key)
    else:
        # No caching - compute directly
        context = build_context(sensor_data, overlay_config.variables)

        visible = True
        if overlay_config.visible_when:
            visible = evaluate_condition(overlay_config.visible_when, context)

        if not visible:
            return y_offset

        text = render_template(overlay_config.template, context)

        if overlay_config.color:
            color = overlay_config.color
        elif overlay_config.color_rules:
            color = evaluate_color_rules(
                overlay_config.color_rules,
                context,
                default_color=(255, 255, 255)
            )
        else:
            color = (255, 255, 255)

    if not visible:
        return y_offset

    # Get style (overlay-specific or default)
    style = overlay_config.style if overlay_config.style else default_style

    # Draw the text box
    draw_text_box(img, text, y_offset, color, style)

    return y_offset + style.box_height


def draw_centermark(
    img: np.ndarray,
    config: CentermarkConfig
) -> None:
    """
    Draw crosshair centermark on image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        config: Centermark configuration
    """
    if not config.enabled:
        return

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(w * config.size_ratio)
    cv2.line(img, (cx - size, cy), (cx + size, cy), config.color, config.thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), config.color, config.thickness)


def draw_border(
    img: np.ndarray,
    config: BorderConfig
) -> None:
    """
    Draw border around image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        config: Border configuration
    """
    if not config.enabled:
        return

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), config.color, config.thickness)


def draw_camera_overlays(
    img: np.ndarray,
    camera_name: str,
    dynamic_data: Dict[str, Any],
    config: ViewerConfig,
    tree_index: int = 0,
    draw_centermark_flag: bool = False
) -> None:
    """
    Draw all configured overlays on a camera image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        camera_name: Name of the camera
        dynamic_data: Dictionary of dynamic values for templates
        config: ViewerConfig with overlay settings
        tree_index: Layout index (overlays only on index 0)
        draw_centermark_flag: Whether to draw centermark
    """
    # Draw centermark if requested
    if draw_centermark_flag:
        draw_centermark(img, config.centermark)

    # Only draw text overlays on tree_index 0
    if tree_index != 0:
        return

    # Get cache key for caching template rendering
    cache_key = make_cache_key(dynamic_data)

    y_offset = 0

    # Draw each configured overlay that targets this camera
    for overlay in config.text_overlays:
        if camera_name in overlay.cameras:
            y_offset = draw_text_overlay(
                img,
                overlay,
                dynamic_data,
                y_offset,
                config.default_overlay_style,
                cache_key
            )
