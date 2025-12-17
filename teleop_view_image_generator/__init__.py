"""
Teleop View Image Generator Package

A ROS-independent image processing package for teleop viewer applications.
Supports fully configurable text overlays and layouts via YAML configuration.

Example usage with config file:
    from teleop_view_image_generator import TeleopImageGenerator

    # Load from YAML config file
    generator = TeleopImageGenerator("config.yaml")

    # Update camera images
    generator.update_camera_image("ee_cam", cv2_image, active=True)

    # Update sensor data for overlays
    generator.update_sensor_data(laser_distance=35.0, robot_status="SCANNING")

    # Generate output frames
    frames = generator.generate_frame()

Example usage with ViewerConfig object:
    from teleop_view_image_generator import TeleopImageGenerator, load_config

    # Load and modify config
    config = load_config("config.yaml")
    generator = TeleopImageGenerator(config)

Configuration features:
    - Unlimited text overlays with template variables ({laser_distance}, etc.)
    - Conditional text and colors based on sensor values
    - Flexible tree-based layout definitions
    - Automatic filtering of unused cameras for performance
"""

from .generator import TeleopImageGenerator
from .camera import CameraConfig, get_default_camera_configs
from .overlays import SensorData, draw_centermark, draw_camera_overlays, draw_border
from .layout import (
    LayoutManager,
    LayoutNode,
    Direction,
    hconcat_resize,
    vconcat_resize,
    concatenate_horizontal_layout,
    concatenate_vertical_layout,
    create_placeholder,
    compute_horizontal_layout_sizes,
    compute_vertical_layout_sizes,
    build_layout_from_config,
    compute_layout_from_config,
)
from .config import (
    ViewerConfig,
    OverlayStyle,
    TextOverlayConfig,
    ColorRule,
    VariableConfig,
    CentermarkConfig,
    BorderConfig,
    LayoutNodeConfig,
    load_config,
    get_default_text_overlays,
)
from .template_engine import (
    evaluate_condition,
    evaluate_formula,
    resolve_variable,
    render_template,
    evaluate_color_rules,
    build_context,
)
from .sample_images import (
    generate_sample_images,
    cleanup_sample_images,
    create_synthetic_image,
    SampleImageContext,
)

__version__ = "2.0.0"
__all__ = [
    # Main class
    "TeleopImageGenerator",
    # Camera
    "CameraConfig",
    "get_default_camera_configs",
    # Overlays
    "SensorData",
    "draw_centermark",
    "draw_camera_overlays",
    "draw_border",
    # Layout
    "LayoutManager",
    "LayoutNode",
    "Direction",
    "hconcat_resize",
    "vconcat_resize",
    "concatenate_horizontal_layout",
    "concatenate_vertical_layout",
    "create_placeholder",
    "compute_horizontal_layout_sizes",
    "compute_vertical_layout_sizes",
    "build_layout_from_config",
    "compute_layout_from_config",
    # Config
    "ViewerConfig",
    "OverlayStyle",
    "TextOverlayConfig",
    "ColorRule",
    "VariableConfig",
    "CentermarkConfig",
    "BorderConfig",
    "LayoutNodeConfig",
    "load_config",
    "get_default_text_overlays",
    # Template Engine
    "evaluate_condition",
    "evaluate_formula",
    "resolve_variable",
    "render_template",
    "evaluate_color_rules",
    "build_context",
    # Sample Images
    "generate_sample_images",
    "cleanup_sample_images",
    "create_synthetic_image",
    "SampleImageContext",
]
