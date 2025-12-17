"""Utility for generating and managing synthetic sample images for testing."""

import os
import shutil
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


# Default camera resolutions (height, width)
DEFAULT_CAMERA_SIZES: Dict[str, Tuple[int, int]] = {
    "ee_cam": (480, 848),
    "ifm_camera1": (800, 1280),
    "ifm_camera2": (800, 1280),
    "front_monitor_cam": (720, 1280),
    "back_monitor_cam": (720, 1280),
    "A1_cam1": (480, 848),
    "A1_cam2": (480, 848),
    "boxwall_monitor_cam": (480, 848),
    "recovery_cam": (530, 848),
}

# Colors for each camera (BGR)
CAMERA_COLORS: Dict[str, Tuple[int, int, int]] = {
    "ee_cam": (50, 50, 150),              # Dark red
    "ifm_camera1": (50, 150, 50),         # Dark green
    "ifm_camera2": (150, 50, 50),         # Dark blue
    "front_monitor_cam": (50, 150, 150),  # Dark yellow
    "back_monitor_cam": (150, 50, 150),   # Dark magenta
    "A1_cam1": (150, 150, 50),            # Dark cyan
    "A1_cam2": (100, 100, 100),           # Gray
    "boxwall_monitor_cam": (80, 120, 80), # Olive
    "recovery_cam": (120, 80, 120),       # Purple
}


def create_synthetic_image(
    height: int,
    width: int,
    color: Tuple[int, int, int],
    label: str,
    frame_number: int = 0
) -> np.ndarray:
    """
    Create a synthetic colored image with a label.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        color: BGR color tuple
        label: Text label to display
        frame_number: Optional frame number to display

    Returns:
        BGR image as numpy array
    """
    img = np.full((height, width, 3), color, dtype=np.uint8)

    # Add some visual interest with a gradient
    for i in range(height):
        factor = 0.7 + 0.3 * (i / height)
        img[i, :] = np.clip(img[i, :] * factor, 0, 255).astype(np.uint8)

    # Add label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label}"
    if frame_number > 0:
        text = f"{label} #{frame_number}"

    text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Draw text with shadow for visibility
    cv2.putText(img, text, (text_x + 2, text_y + 2), font, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def generate_sample_images(
    output_dir: str,
    camera_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    num_frames: int = 2
) -> str:
    """
    Generate synthetic sample images for all cameras.

    Args:
        output_dir: Directory to create sample images in
        camera_sizes: Optional dict of camera_name -> (height, width)
        num_frames: Number of frames to generate per camera

    Returns:
        Path to the created sample_images directory
    """
    if camera_sizes is None:
        camera_sizes = DEFAULT_CAMERA_SIZES

    sample_dir = os.path.join(output_dir, "sample_images")

    # Create directory structure
    os.makedirs(sample_dir, exist_ok=True)

    for camera_name, (height, width) in camera_sizes.items():
        camera_dir = os.path.join(sample_dir, camera_name)
        os.makedirs(camera_dir, exist_ok=True)

        color = CAMERA_COLORS.get(camera_name, (100, 100, 100))

        for frame_num in range(1, num_frames + 1):
            img = create_synthetic_image(height, width, color, camera_name, frame_num)
            filename = os.path.join(camera_dir, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return sample_dir


def cleanup_sample_images(output_dir: str) -> None:
    """
    Remove the generated sample_images directory.

    Args:
        output_dir: Directory containing sample_images folder
    """
    sample_dir = os.path.join(output_dir, "sample_images")
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)


class SampleImageContext:
    """
    Context manager for temporary sample images.

    Usage:
        with SampleImageContext() as sample_dir:
            # sample_dir contains generated images
            # ... run tests ...
        # Images are automatically cleaned up
    """

    def __init__(
        self,
        output_dir: str = ".",
        camera_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        num_frames: int = 2
    ):
        self.output_dir = output_dir
        self.camera_sizes = camera_sizes
        self.num_frames = num_frames
        self.sample_dir = None

    def __enter__(self) -> str:
        self.sample_dir = generate_sample_images(
            self.output_dir,
            self.camera_sizes,
            self.num_frames
        )
        return self.sample_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_sample_images(self.output_dir)
        return False
