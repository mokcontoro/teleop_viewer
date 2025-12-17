"""
Teleop View Image Generator Package

A ROS-independent image processing package for teleop viewer applications.

Example usage:
    from teleop_view_image_generator import TeleopImageGenerator

    config = {
        "resolutions": {
            "ee_cam": (480, 848, 3),
            "ifm": (800, 1280, 3),
            "monitor_cam": (720, 1280, 3),
            "recovery_cam": (530, 848, 3),
        },
        "hardware": {
            "old_elbow_cam": True,
            "camera_mount": "D",
        },
        "use_vertical": False,
    }

    generator = TeleopImageGenerator(config)
    generator.update_camera_image("ee_cam", cv2_image, active=True)
    generator.update_sensor_data(laser_distance=35.0, robot_status="SCANNING")
    frames = generator.generate_frame()
"""

from .generator import TeleopImageGenerator
from .camera import CameraConfig, get_default_camera_configs
from .overlays import SensorData, draw_centermark, draw_camera_overlays
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
)

__version__ = "1.0.0"
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
]
