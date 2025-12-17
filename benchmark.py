#!/usr/bin/env python3
"""Benchmark script for multi_view_composer.

Uses synthetic images in memory - no disk I/O required.
"""

import time
import argparse
import random
import cv2
import numpy as np

from multi_view_composer import MultiViewComposer, load_config


def create_synthetic_image(height: int, width: int, color: tuple, label: str) -> np.ndarray:
    """Create a synthetic colored image with a label."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 1.0, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, label, (text_x, text_y), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def run_benchmark(num_frames: int = 50, config_path: str = "config.yaml"):
    print("=" * 70)
    print("MULTI-VIEW COMPOSER BENCHMARK")
    print("=" * 70)

    # Load config and create composer
    config = load_config(config_path)
    composer = MultiViewComposer(config)

    # Camera colors for visualization
    camera_colors = {
        "ee_cam": (50, 50, 150),
        "ifm_camera1": (50, 150, 50),
        "ifm_camera2": (150, 50, 50),
        "front_monitor_cam": (50, 150, 150),
        "back_monitor_cam": (150, 50, 150),
        "boxwall_monitor_cam": (80, 120, 80),
    }

    # Create synthetic images for each camera
    print(f"\nCameras: {composer.get_camera_names()}")
    for cam_name in composer.get_camera_names():
        cam_config = composer.get_camera_config(cam_name)
        if cam_config:
            h, w = cam_config.resolution[:2]
            color = camera_colors.get(cam_name, (100, 100, 100))
            img = create_synthetic_image(h, w, color, cam_name)
            composer.update_camera_image(cam_name, img, active=True)

    # Status options for random selection
    status_options = ["Stopped", "SCANNING", "NAVIGATING", "UNLOADING", "FINISHED"]

    # Warm up
    composer.update_dynamic_data(
        laser_distance=35.0,
        laser_active=True,
        pressure_manifold=0.5,
        pressure_base=0.3,
        robot_status="Stopped",
        is_manual_review=False,
    )
    _ = composer.generate_frame()

    # Benchmark
    print(f"\nRunning {num_frames} frames...")
    print("-" * 40)

    start = time.perf_counter()
    for i in range(num_frames):
        # Update dynamic data with random values
        composer.update_dynamic_data(
            laser_distance=random.uniform(20.0, 50.0),
            laser_active=random.random() > 0.1,
            pressure_manifold=random.uniform(0.3, 0.8),
            pressure_base=random.uniform(0.2, 0.5),
            robot_status=random.choice(status_options),
            is_manual_review=random.random() > 0.5,
        )
        output = composer.generate_frame()
    elapsed = time.perf_counter() - start

    fps = num_frames / elapsed
    ms_per_frame = (elapsed / num_frames) * 1000

    print(f"  Frames: {num_frames}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  ms/frame: {ms_per_frame:.2f}")
    print(f"  Text overlays: {len(config.text_overlays)}")
    print(f"  Layouts: {len(config.layouts)}")

    composer.shutdown()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nFPS: {fps:.1f}")
    print(f"ms/frame: {ms_per_frame:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multi_view_composer")
    parser.add_argument("-n", "--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    run_benchmark(args.frames, args.config)
