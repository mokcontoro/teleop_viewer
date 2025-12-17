#!/usr/bin/env python3
"""
Example: Using the multi_view_composer package

This example demonstrates how to use the MultiViewComposer
with synthetic images (no external image files required).

Usage:
    python example.py
"""

import cv2
import numpy as np
import os
import random
import time
from multi_view_composer import MultiViewComposer, load_config, ViewerConfig


def create_synthetic_image(height: int, width: int, color: tuple, label: str) -> np.ndarray:
    """Create a synthetic colored image with a label."""
    img = np.full((height, width, 3), color, dtype=np.uint8)

    # Add label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 1.0, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, label, (text_x, text_y), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def main():
    # Try to load config from file, or use a minimal config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        composer = MultiViewComposer(config_path)
    else:
        print("Config file not found, using default config")
        # Create minimal config programmatically
        from multi_view_composer.config import (
            ViewerConfig, OverlayStyle, TextOverlayConfig,
            CentermarkConfig, BorderConfig, LayoutNodeConfig,
            ColorRule, VariableConfig, VariableCondition
        )

        config = ViewerConfig(
            resolutions={
                "ee_cam": [480, 848, 3],
                "ifm": [800, 1280, 3],
                "monitor_cam": [720, 1280, 3],
            },
            hardware={"old_elbow_cam": True, "camera_mount": "D"},
            text_overlays=[
                TextOverlayConfig(
                    id="status",
                    template="Status: {robot_status}",
                    cameras=["back_monitor_cam", "front_monitor_cam"],
                    color=(255, 128, 0),
                ),
            ],
            layouts={
                "horizontal": LayoutNodeConfig(
                    direction="horizontal",
                    children=[
                        LayoutNodeConfig(camera="ee_cam"),
                        LayoutNodeConfig(camera="ifm_camera1"),
                        LayoutNodeConfig(
                            direction="horizontal",
                            children=[
                                LayoutNodeConfig(camera="front_monitor_cam"),
                                LayoutNodeConfig(camera="back_monitor_cam"),
                            ]
                        ),
                    ]
                ),
            },
            active_layout="horizontal",
        )
        composer = MultiViewComposer(config)

    # Define camera colors for visualization
    camera_colors = {
        "ee_cam": (50, 50, 150),              # Dark red
        "ifm_camera1": (50, 150, 50),         # Dark green
        "ifm_camera2": (150, 50, 50),         # Dark blue
        "front_monitor_cam": (50, 150, 150),  # Dark yellow
        "back_monitor_cam": (150, 50, 150),   # Dark magenta
        "boxwall_monitor_cam": (80, 120, 80), # Olive
    }

    # Create and update synthetic images for each camera
    print(f"\nCameras in use: {composer.get_camera_names()}")

    for cam_name in composer.get_camera_names():
        cam_config = composer.get_camera_config(cam_name)
        if cam_config:
            h, w = cam_config.resolution[:2]
            color = camera_colors.get(cam_name, (100, 100, 100))
            img = create_synthetic_image(h, w, color, cam_name)
            composer.update_camera_image(cam_name, img, active=True)

    # Status options for random selection
    status_options = ["Stopped", "SCANNING", "NAVIGATING", "UNLOADING", "FINISHED"]

    print("\nRunning demo loop with random sensor values...")
    print("Press 'q' or ESC to quit\n")

    frame_count = 0
    start_time = time.time()

    while True:
        # Update dynamic data with random values
        laser_distance = random.uniform(20.0, 50.0)  # mm
        laser_active = random.random() > 0.1  # 90% chance active
        pressure_manifold = random.uniform(0.3, 0.8)  # bar
        pressure_base = random.uniform(0.2, 0.5)  # bar
        robot_status = random.choice(status_options)
        is_manual_review = random.random() > 0.5  # 50% chance

        composer.update_dynamic_data(
            laser_distance=laser_distance,
            laser_active=laser_active,
            pressure_manifold=pressure_manifold,
            pressure_base=pressure_base,
            robot_status=robot_status,
            is_manual_review=is_manual_review,
        )

        # Generate the output frame
        frames = composer.generate_frame()

        # Display result
        cv2.imshow("Multi-View Composer Example", frames[0])

        frame_count += 1

        # Print stats every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frame {frame_count}: FPS={fps:.1f} | "
                  f"laser={laser_distance:.1f}mm | "
                  f"status={robot_status} | "
                  f"manual={is_manual_review}")

        # Check for quit key
        key = cv2.waitKey(33) & 0xFF  # ~30 FPS
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

    # Cleanup
    composer.shutdown()
    print(f"\nDone. Rendered {frame_count} frames.")


if __name__ == "__main__":
    main()
