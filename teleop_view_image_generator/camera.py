"""Camera configuration and data structures."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    name: str
    resolution: Tuple[int, int, int]  # (height, width, channels)
    rotate: Optional[int]  # cv2 rotation constant or None
    centermark: bool
    overlay_types: List[str] = field(default_factory=list)  # ["laser", "status", "pressure"]

    # Computed target sizes for each layout (set during initialization)
    target_sizes: List[Tuple[int, int]] = field(default_factory=list)

    # Runtime data
    active: bool = False
    raw_image: Optional[np.ndarray] = None  # Original BGR image from source
    processed_images: List[Optional[np.ndarray]] = field(default_factory=list)  # Processed for each layout

    def get_effective_resolution(self) -> Tuple[int, int, int]:
        """Get resolution after rotation."""
        if self.rotate is not None:
            return (self.resolution[1], self.resolution[0], self.resolution[2])
        return self.resolution


# Default camera configurations
def get_default_camera_configs(
    resolutions: dict,
    hardware: dict,
    num_layouts: int = 1
) -> dict:
    """
    Create default camera configurations.

    Args:
        resolutions: Dict with keys 'ee_cam', 'ifm', 'monitor_cam', 'recovery_cam'
        hardware: Dict with keys 'old_elbow_cam', 'camera_mount'
        num_layouts: Number of output layouts (1 or 2)

    Returns:
        Dict mapping camera name to CameraConfig
    """
    import cv2

    old_elbow_cam = hardware.get("old_elbow_cam", True)
    camera_mount = hardware.get("camera_mount", "D")

    ee_res = tuple(resolutions.get("ee_cam", [480, 848, 3]))
    ifm_res = tuple(resolutions.get("ifm", [800, 1280, 3]))
    monitor_res = tuple(resolutions.get("monitor_cam", [720, 1280, 3]))
    recovery_res = tuple(resolutions.get("recovery_cam", [530, 848, 3]))

    configs = {}

    # EE Camera
    configs["ee_cam"] = CameraConfig(
        name="ee_cam",
        resolution=ee_res,
        rotate=None,
        centermark=True,
        overlay_types=[],
        target_sizes=[ee_res[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    # IFM Camera 1 (or recovery cam depending on mount)
    ifm1_res = ifm_res if camera_mount == "D" else recovery_res
    configs["ifm_camera1"] = CameraConfig(
        name="ifm_camera1",
        resolution=ifm1_res,
        rotate=None,
        centermark=False,
        overlay_types=[],
        target_sizes=[ifm1_res[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    # IFM Camera 2 (elbow cam)
    ifm2_rotate = cv2.ROTATE_90_COUNTERCLOCKWISE if old_elbow_cam else None
    ifm2_effective = (ifm_res[1], ifm_res[0], 3) if old_elbow_cam else ifm_res
    configs["ifm_camera2"] = CameraConfig(
        name="ifm_camera2",
        resolution=ifm_res,
        rotate=ifm2_rotate,
        centermark=False,
        overlay_types=["laser"],
        target_sizes=[ifm2_effective[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    # Front Monitor Camera
    front_effective = (monitor_res[1], monitor_res[0], 3)
    configs["front_monitor_cam"] = CameraConfig(
        name="front_monitor_cam",
        resolution=monitor_res,
        rotate=cv2.ROTATE_90_COUNTERCLOCKWISE,
        centermark=False,
        overlay_types=["status", "pressure"],
        target_sizes=[front_effective[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    # Back Monitor Camera
    back_rotate = cv2.ROTATE_90_CLOCKWISE if old_elbow_cam else None
    back_effective = (monitor_res[1], monitor_res[0], 3) if old_elbow_cam else monitor_res
    configs["back_monitor_cam"] = CameraConfig(
        name="back_monitor_cam",
        resolution=monitor_res,
        rotate=back_rotate,
        centermark=False,
        overlay_types=["status", "pressure", "laser"],
        target_sizes=[back_effective[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    # A1 Cameras
    a1_effective = (ee_res[1], ee_res[0], 3)
    configs["A1_cam1"] = CameraConfig(
        name="A1_cam1",
        resolution=ee_res,
        rotate=cv2.ROTATE_90_CLOCKWISE,
        centermark=False,
        overlay_types=[],
        target_sizes=[a1_effective[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    configs["A1_cam2"] = CameraConfig(
        name="A1_cam2",
        resolution=ee_res,
        rotate=cv2.ROTATE_90_COUNTERCLOCKWISE,
        centermark=False,
        overlay_types=[],
        target_sizes=[a1_effective[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    # Boxwall Monitor Camera
    configs["boxwall_monitor_cam"] = CameraConfig(
        name="boxwall_monitor_cam",
        resolution=monitor_res,
        rotate=None,
        centermark=False,
        overlay_types=[],
        target_sizes=[monitor_res[:2]] * num_layouts,
        processed_images=[None] * num_layouts,
    )

    return configs
