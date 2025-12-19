"""
Multi-View Composer Package

A high-performance multi-view image composer with configurable text overlays
and flexible YAML-based layouts.

Example usage:
    from multi_view_composer import MultiViewComposer

    # Load from YAML config file
    composer = MultiViewComposer("config.yaml")

    # Update camera images
    composer.update_camera_image("cam1", cv2_image, active=True)

    # Update dynamic data for overlays (any keyword arguments)
    composer.update_dynamic_data(temperature=25.5, speed=10.0)

    # Generate output frames
    frames = composer.generate_frame()
"""

from .generator import MultiViewComposer
from .camera import CameraConfig, create_camera_configs
from .overlays import draw_centermark, draw_camera_overlays, draw_border
from .layout import (
    LayoutManager,
    LayoutNode,
    Direction,
    hconcat_resize,
    vconcat_resize,
    create_placeholder,
    build_layout_from_config,
    compute_layout_from_config,
)
from .config import (
    ViewerConfig,
    CameraDefinition,
    OverlayStyle,
    TextOverlayConfig,
    ColorRule,
    VariableConfig,
    CentermarkConfig,
    BorderConfig,
    LayoutNodeConfig,
    ConfigError,
    load_config,
)
from .template_engine import (
    evaluate_condition,
    evaluate_formula,
    resolve_variable,
    render_template,
    evaluate_color_rules,
    build_context,
)
from .logging_config import setup_logging, get_logger

__version__ = "4.0.0"
__all__ = [
    # Main class
    "MultiViewComposer",
    # Camera
    "CameraConfig",
    "CameraDefinition",
    "create_camera_configs",
    # Overlays
    "draw_centermark",
    "draw_camera_overlays",
    "draw_border",
    # Layout
    "LayoutManager",
    "LayoutNode",
    "Direction",
    "hconcat_resize",
    "vconcat_resize",
    "create_placeholder",
    "build_layout_from_config",
    "compute_layout_from_config",
    # Config
    "ViewerConfig",
    "CameraDefinition",
    "OverlayStyle",
    "TextOverlayConfig",
    "ColorRule",
    "VariableConfig",
    "CentermarkConfig",
    "BorderConfig",
    "LayoutNodeConfig",
    "ConfigError",
    "load_config",
    # Template Engine
    "evaluate_condition",
    "evaluate_formula",
    "resolve_variable",
    "render_template",
    "evaluate_color_rules",
    "build_context",
    # Logging
    "setup_logging",
    "get_logger",
]
