#!/usr/bin/env python3
"""Benchmark script for multi_view_composer.

Generates temporary sample images for testing and cleans them up afterward.
"""

import os
import time
import cv2
import argparse

from multi_view_composer import generate_sample_images, cleanup_sample_images


def run_benchmark(num_frames=50, show_display=False):
    print("=" * 70)
    print("MULTI-VIEW COMPOSER BENCHMARK")
    print("=" * 70)

    # Get script directory for generating sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate sample images
    print("\nGenerating sample images...")
    sample_dir = generate_sample_images(script_dir, num_frames=2)
    print(f"  Created: {sample_dir}")

    try:
        # Mock display if not showing
        if not show_display:
            cv2.imshow = lambda *args: None
            cv2.waitKey = lambda *args: 0
            cv2.destroyAllWindows = lambda: None

        # ============ Benchmark ============
        print("\n[1] MultiViewComposer")
        print("-" * 40)

        from viewer import Viewer

        viewer = Viewer(config_path="config.yaml")

        # Warm up
        viewer._load_all_camera_images()
        _ = viewer.composer.generate_frame()

        # Benchmark
        start = time.perf_counter()
        for i in range(num_frames):
            viewer._load_all_camera_images()
            output = viewer.composer.generate_frame()
        elapsed = time.perf_counter() - start

        fps = num_frames / elapsed
        print(f"  Frames: {num_frames}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  ms/frame: {(elapsed/num_frames)*1000:.2f}")
        print(f"  Text overlays: {len(viewer.viewer_config.text_overlays)}")
        print(f"  Layouts: {len(viewer.viewer_config.layouts)}")

        viewer.composer.shutdown()

        # ============ Summary ============
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nFPS: {fps:.1f}")
        print(f"ms/frame: {(elapsed/num_frames)*1000:.2f}")

    finally:
        # Clean up sample images
        print("\nCleaning up sample images...")
        cleanup_sample_images(script_dir)
        print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multi_view_composer")
    parser.add_argument("-n", "--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--display", action="store_true", help="Show display (requires X)")
    args = parser.parse_args()

    run_benchmark(args.frames, args.display)
