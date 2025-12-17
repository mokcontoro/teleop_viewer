#!/usr/bin/env python3
"""Benchmark script for teleop viewer versions.

Generates temporary sample images for testing and cleans them up afterward.
"""

import os
import time
import cv2
import argparse

from teleop_view_image_generator import generate_sample_images, cleanup_sample_images


def run_benchmark(num_frames=50, show_display=False):
    print("=" * 70)
    print("TELEOP VIEWER BENCHMARK")
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

        # ============ Benchmark Original ============
        print("\n[1] Original teleop_viewer.py")
        print("-" * 40)

        try:
            from teleop_viewer import TeleopViewerDnD, load_config

            config = load_config()
            viewer1 = TeleopViewerDnD(config)

            # Warm up
            viewer1.load_images_from_files()
            for item in viewer1.compressed_image_items.values():
                viewer1.convert_msg_to_cv2(item)
                viewer1.put_text_on_image(item)
            _ = viewer1.image_concat_trees[0].get_tree_image(viewer1)

            # Benchmark
            start = time.perf_counter()
            for i in range(num_frames):
                viewer1.load_images_from_files()
                for item in viewer1.compressed_image_items.values():
                    viewer1.convert_msg_to_cv2(item)
                    viewer1.put_text_on_image(item)
                output1 = viewer1.image_concat_trees[0].get_tree_image(viewer1)
            elapsed1 = time.perf_counter() - start

            fps1 = num_frames / elapsed1
            print(f"  Frames: {num_frames}")
            print(f"  Time: {elapsed1:.3f}s")
            print(f"  FPS: {fps1:.1f}")
            print(f"  ms/frame: {(elapsed1/num_frames)*1000:.2f}")
            has_original = True
        except ImportError:
            print("  [SKIPPED] teleop_viewer.py not found")
            fps1 = None
            elapsed1 = None
            has_original = False

        # ============ Benchmark Improved ============
        print("\n[2] Improved (ViewerConfig - flexible overlays)")
        print("-" * 40)

        from teleop_viewer_improved import TeleopViewer

        viewer2 = TeleopViewer(config_path="config.yaml")

        # Warm up
        viewer2._load_all_camera_images()
        _ = viewer2.generator.generate_frame()

        # Benchmark
        start = time.perf_counter()
        for i in range(num_frames):
            viewer2._load_all_camera_images()
            output2 = viewer2.generator.generate_frame()
        elapsed2 = time.perf_counter() - start

        fps2 = num_frames / elapsed2
        print(f"  Frames: {num_frames}")
        print(f"  Time: {elapsed2:.3f}s")
        print(f"  FPS: {fps2:.1f}")
        print(f"  ms/frame: {(elapsed2/num_frames)*1000:.2f}")
        print(f"  Text overlays: {len(viewer2.viewer_config.text_overlays)}")
        print(f"  Layouts: {len(viewer2.viewer_config.layouts)}")

        viewer2.generator.shutdown()

        # ============ Comparison ============
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print(f"\n{'Version':<35} {'FPS':>8} {'ms/frame':>10} {'Speedup':>10}")
        print("-" * 65)

        if has_original:
            print(f"{'Original teleop_viewer.py':<35} {fps1:>8.1f} {(elapsed1/num_frames)*1000:>10.2f} {'1.00x':>10}")
            baseline = fps1
            print(f"{'Improved (ViewerConfig)':<35} {fps2:>8.1f} {(elapsed2/num_frames)*1000:>10.2f} {fps2/baseline:>9.2f}x")
        else:
            print(f"{'Improved (ViewerConfig)':<35} {fps2:>8.1f} {(elapsed2/num_frames)*1000:>10.2f} {'N/A':>10}")

    finally:
        # Clean up sample images
        print("\nCleaning up sample images...")
        cleanup_sample_images(script_dir)
        print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark teleop viewers")
    parser.add_argument("-n", "--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--display", action="store_true", help="Show display (requires X)")
    args = parser.parse_args()

    run_benchmark(args.frames, args.display)
