"""Main TeleopImageGenerator class - core image processing engine."""

from __future__ import annotations
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple, Union, Set

from .camera import CameraConfig, get_default_camera_configs
from .overlays import SensorData, draw_camera_overlays, draw_border
from .layout import LayoutManager, create_placeholder
from .config import ViewerConfig, load_config, LayoutNodeConfig


logger = logging.getLogger(__name__)


def _get_cameras_from_layout(layout: LayoutNodeConfig) -> Set[str]:
    """Extract all camera names used in a layout tree."""
    cameras = set()
    if layout.camera:
        cameras.add(layout.camera)
    for child in layout.children:
        cameras.update(_get_cameras_from_layout(child))
    return cameras


class TeleopImageGenerator:
    """
    Core image processing engine for teleop viewer - no ROS dependencies.

    This class handles:
    - Camera image processing (resize, rotate)
    - Overlay rendering (configurable via YAML)
    - Layout concatenation (configurable via YAML)

    Usage:
        # From YAML file path
        generator = TeleopImageGenerator("config.yaml")

        # Or from ViewerConfig object
        config = load_config("config.yaml")
        generator = TeleopImageGenerator(config)

        # Update images and sensor data
        generator.update_camera_image("ee_cam", image, active=True)
        generator.update_dynamic_data(laser_distance=35.0, ...)
        frames = generator.generate_frame()
    """

    def __init__(self, config: Union[ViewerConfig, str]):
        """
        Initialize the image generator.

        Args:
            config: Configuration - either:
                - ViewerConfig: Config object with full overlay/layout settings
                - str: Path to YAML config file
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load config
        if isinstance(config, str):
            self.config = load_config(config)
        elif isinstance(config, ViewerConfig):
            self.config = config
        else:
            raise TypeError(
                f"config must be ViewerConfig or str path, got {type(config).__name__}. "
                "Use load_config('config.yaml') to load from file."
            )

        # Initialize cameras
        resolutions = self.config.resolutions
        hardware = self.config.hardware

        # Determine number of layouts and which cameras are used
        if not self.config.layouts:
            raise ValueError("No layouts defined in config. At least one layout is required.")

        self.num_layouts = len(self.config.layouts)

        # Get cameras actually used in layouts
        used_cameras: Set[str] = set()
        for layout in self.config.layouts.values():
            used_cameras.update(_get_cameras_from_layout(layout))

        # Get all camera configs, then filter to only used ones
        all_cameras = get_default_camera_configs(resolutions, hardware, self.num_layouts)
        self.cameras: Dict[str, CameraConfig] = {
            name: cam for name, cam in all_cameras.items()
            if name in used_cameras
        }

        # Compute layout and target sizes using tree-based algorithm
        camera_sizes = {
            name: cam.get_effective_resolution()[:2]
            for name, cam in self.cameras.items()
        }

        # Create layout manager from config
        self.layout_manager = LayoutManager(
            camera_sizes,
            layout_configs=self.config.layouts,
            active_layout=self.config.active_layout
        )

        # Update camera target_sizes from computed layout
        for tree_index in range(self.num_layouts):
            for name, cam in self.cameras.items():
                computed_size = self.layout_manager.get_target_size(name, tree_index)
                if computed_size != (480, 640):  # Not default
                    cam.target_sizes[tree_index] = computed_size

        # Sensor data
        self.sensor_data = SensorData()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.logger.info(f"Initialized with {len(self.cameras)} cameras, "
                        f"{self.num_layouts} layout(s)")

    def update_camera_image(
        self,
        camera_name: str,
        image: np.ndarray,
        active: bool = True
    ) -> bool:
        """
        Update a camera's raw image.

        Args:
            camera_name: Name of the camera (e.g., "ee_cam", "ifm_camera1")
            image: BGR image as numpy array
            active: Whether camera is active

        Returns:
            True if camera exists and was updated
        """
        if camera_name not in self.cameras:
            self.logger.warning(f"Unknown camera: {camera_name}")
            return False

        cam = self.cameras[camera_name]
        cam.raw_image = image
        cam.active = active
        return True

    def update_dynamic_data(
        self,
        laser_distance: Optional[float] = None,
        laser_active: Optional[bool] = None,
        pressure_manifold: Optional[float] = None,
        pressure_base: Optional[float] = None,
        robot_status: Optional[str] = None,
        is_manual_review: Optional[bool] = None
    ) -> None:
        """
        Update sensor values for overlays.

        Args:
            laser_distance: Laser distance in mm
            laser_active: Whether laser sensor is active
            pressure_manifold: Manifold pressure in bar
            pressure_base: Base pressure in bar
            robot_status: Robot status string
            is_manual_review: Whether in manual review mode
        """
        if laser_distance is not None:
            self.sensor_data.laser_distance = laser_distance
        if laser_active is not None:
            self.sensor_data.laser_active = laser_active
        if pressure_manifold is not None:
            self.sensor_data.pressure_manifold = pressure_manifold
        if pressure_base is not None:
            self.sensor_data.pressure_base = pressure_base
        if robot_status is not None:
            self.sensor_data.robot_status = robot_status
        if is_manual_review is not None:
            self.sensor_data.is_manual_review = is_manual_review

    def _process_camera(self, camera_name: str) -> bool:
        """
        Process a single camera's image for all layouts.

        Args:
            camera_name: Name of the camera to process

        Returns:
            True if processing was successful
        """
        cam = self.cameras.get(camera_name)
        if cam is None:
            return False

        img = cam.raw_image
        if img is None or not cam.active:
            cam.processed_images = [None] * self.num_layouts
            return False

        # Process for each layout
        for tree_index in range(self.num_layouts):
            target_h, target_w = cam.target_sizes[tree_index][:2]

            # Compute resize dimensions (before rotation)
            if cam.rotate is not None:
                resize_w, resize_h = target_h, target_w
            else:
                resize_w, resize_h = target_w, target_h

            # Resize with fast interpolation
            if img.shape[:2] != (resize_h, resize_w):
                processed = cv2.resize(
                    img, (resize_w, resize_h),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                processed = img.copy()

            # Rotate if needed
            if cam.rotate is not None:
                processed = cv2.rotate(processed, cam.rotate)

            # Draw overlays using config
            draw_camera_overlays(
                processed,
                cam.name,
                self.sensor_data,
                self.config,
                tree_index,
                cam.centermark
            )

            # Draw border
            draw_border(processed, self.config.border)

            cam.processed_images[tree_index] = processed

        return True

    def _get_processed_image(
        self,
        camera_name: str,
        tree_index: int
    ) -> np.ndarray:
        """
        Get processed image for a camera, or placeholder if not available.

        Args:
            camera_name: Name of the camera
            tree_index: Layout index

        Returns:
            Processed BGR image or black placeholder
        """
        cam = self.cameras.get(camera_name)
        if cam and cam.active and cam.processed_images[tree_index] is not None:
            return cam.processed_images[tree_index]

        # Return black placeholder
        if cam:
            h, w = cam.target_sizes[tree_index][:2]
        else:
            h, w = 480, 640
        return create_placeholder(h, w)

    def generate_frame(self) -> List[np.ndarray]:
        """
        Process all camera images and generate concatenated output frames.

        Returns:
            List of BGR images (one per layout configuration)
        """
        # Process all cameras in parallel
        camera_names = list(self.cameras.keys())
        futures = [
            self.executor.submit(self._process_camera, name)
            for name in camera_names
        ]
        for f in futures:
            f.result()  # Wait for completion

        # Generate output for each layout using the layout manager
        outputs = []
        for tree_index in range(self.num_layouts):
            def get_image(name: str, ti=tree_index) -> np.ndarray:
                return self._get_processed_image(name, ti)

            output = self.layout_manager.concatenate(get_image, tree_index)
            outputs.append(output)

        return outputs

    def get_camera_names(self) -> List[str]:
        """Get list of all camera names."""
        return list(self.cameras.keys())

    def get_camera_config(self, camera_name: str) -> Optional[CameraConfig]:
        """Get configuration for a specific camera."""
        return self.cameras.get(camera_name)

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=False)
        self.logger.info("Generator shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
