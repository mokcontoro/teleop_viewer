#!/usr/bin/python3
"""
Teleop Viewer - Optimized Version

Uses the teleop_view_image_generator package for image processing.
All settings (layouts, overlays, styling) are configured via YAML config file.

Benchmark Results:
  | Version  | FPS   | ms/frame | Speedup |
  |----------|-------|----------|---------|
  | Original | 20.5  | 48.69    | 1.0x    |
  | Improved | 56.7  | 17.63    | 2.76x   |

Key Features:
- Fully configurable text overlays via YAML (templates, colors, conditions)
- Flexible tree-based layout system
- Automatic camera filtering (only processes cameras in layout)
- Parallel image processing with ThreadPoolExecutor
- Image caching for file-based sources
- Pre-compiled regex patterns for template rendering

Usage:
    python teleop_viewer_improved.py                 # Use default config.yaml
    python teleop_viewer_improved.py -c custom.yaml  # Use custom config

See config.yaml for the full configuration format.
"""

from __future__ import annotations
import cv2
import numpy as np
import logging
import time
import os
import glob
from typing import Optional, Dict, List

from teleop_view_image_generator import (
    TeleopImageGenerator,
    load_config as load_viewer_config,
    generate_sample_images,
    cleanup_sample_images,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Teleop Viewer")


class ImageLoader:
    """Image loader with caching for file-based image sources."""

    __slots__ = ['input_directory', 'camera_images', 'camera_indices', 'image_cache']

    def __init__(self, input_directory: str):
        self.input_directory = input_directory
        self.camera_images: Dict[str, List[str]] = {}
        self.camera_indices: Dict[str, int] = {}
        self.image_cache: Dict[str, np.ndarray] = {}
        self._scan_directories()

    def _scan_directories(self):
        """Scan input directory for camera subdirectories."""
        if not os.path.exists(self.input_directory):
            logger.warning(f"Input directory does not exist: {self.input_directory}")
            return

        camera_names = [
            "ee_cam", "ifm_camera1", "ifm_camera2", "front_monitor_cam",
            "back_monitor_cam", "A1_cam1", "A1_cam2", "boxwall_monitor_cam", "recovery_cam"
        ]

        for camera_name in camera_names:
            camera_dir = os.path.join(self.input_directory, camera_name)
            if os.path.isdir(camera_dir):
                image_files = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]:
                    image_files.extend(glob.glob(os.path.join(camera_dir, ext)))
                if image_files:
                    self.camera_images[camera_name] = sorted(image_files)
                    self.camera_indices[camera_name] = 0
                    logger.info(f"Found {len(image_files)} images for {camera_name}")

    def load_image(self, camera_name: str) -> Optional[np.ndarray]:
        """Load image as BGR array with caching."""
        if camera_name not in self.camera_images:
            return None

        images = self.camera_images[camera_name]
        if not images:
            return None

        idx = self.camera_indices[camera_name]
        image_path = images[idx]

        # Check cache first
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        # Load and cache
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            self.image_cache[image_path] = img
        return img

    def advance_frame(self, camera_name: str) -> bool:
        """Advance to next frame. Returns True if looped back to start."""
        if camera_name not in self.camera_images:
            return False
        images = self.camera_images[camera_name]
        if not images:
            return False
        self.camera_indices[camera_name] = (self.camera_indices[camera_name] + 1) % len(images)
        return self.camera_indices[camera_name] == 0

    def has_images(self, camera_name: str) -> bool:
        return camera_name in self.camera_images and len(self.camera_images[camera_name]) > 0


class TeleopViewer:
    """Teleop viewer using the teleop_view_image_generator package."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the teleop viewer.

        Args:
            config_path: Path to YAML config file (relative to script or absolute)
        """
        self.logger = logging.getLogger("TeleopViewer")
        self._generated_sample_images = False
        self._script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load config from YAML file
        if not os.path.isabs(config_path):
            config_path = os.path.join(self._script_dir, config_path)
        self.viewer_config = load_viewer_config(config_path)

        # Initialize generator with ViewerConfig
        self.generator = TeleopImageGenerator(self.viewer_config)

        # Settings from config
        self.fps = self.viewer_config.fps
        self.window_name = self.viewer_config.window_name

        # Initialize image loader - generate sample images if needed
        input_dir = self.viewer_config.input_directory
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(self._script_dir, input_dir)

        if not os.path.exists(input_dir):
            self.logger.info("Sample images not found, generating synthetic images...")
            generate_sample_images(self._script_dir, num_frames=2)
            self._generated_sample_images = True

        self.image_loader = ImageLoader(input_dir)

        # Set initial sensor values from config
        sensors = self.viewer_config.sensors
        self.generator.update_sensor_data(
            laser_distance=sensors.get("laser_distance", 35.0),
            laser_active=sensors.get("laser_active", True),
            pressure_manifold=sensors.get("pressure_manifold", 0.5),
            pressure_base=sensors.get("pressure_base", 0.3),
            robot_status=sensors.get("robot_status", "Stopped"),
            is_manual_review=sensors.get("is_manual_review", True),
        )

        self.frame_counter = 0
        self.running = True

    def _load_all_camera_images(self):
        """Load images from disk and feed them to the generator."""
        for camera_name in self.generator.get_camera_names():
            img = self.image_loader.load_image(camera_name)
            if img is not None:
                self.generator.update_camera_image(camera_name, img, active=True)
            else:
                self.generator.update_camera_image(camera_name, np.zeros((1, 1, 3), dtype=np.uint8), active=False)

    def run(self):
        """Main loop with OpenCV display."""
        frame_time = 1.0 / self.fps
        self.logger.info(f"Starting viewer at {self.fps} FPS. Press 'q' or ESC to quit.")

        while self.running:
            start_time = time.perf_counter()

            try:
                # Load all camera images from disk
                self._load_all_camera_images()

                # Generate frames using the package
                frames = self.generator.generate_frame()

                # Display frames
                for idx, frame in enumerate(frames):
                    window = self.window_name if idx == 0 else f"{self.window_name} (Vertical)"
                    cv2.imshow(window, frame)

                self.frame_counter += 1

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord('n'):
                    for camera_name in self.generator.get_camera_names():
                        self.image_loader.advance_frame(camera_name)

                # Frame rate control
                elapsed = time.perf_counter() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.exception(f"Error: {e}")
                break

        self.generator.shutdown()
        cv2.destroyAllWindows()

        # Clean up generated sample images
        if self._generated_sample_images:
            self.logger.info("Cleaning up generated sample images...")
            cleanup_sample_images(self._script_dir)

        self.logger.info("Viewer stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Teleop Viewer - Optimized Version")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    viewer = TeleopViewer(config_path=args.config)
    viewer.run()


if __name__ == "__main__":
    main()
