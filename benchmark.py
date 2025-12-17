#!/usr/bin/env python3
"""Benchmark script for teleop viewer versions."""

import time
import cv2
import argparse

def run_benchmark(num_frames=50, show_display=False):
    print("=" * 70)
    print("TELEOP VIEWER BENCHMARK")
    print("=" * 70)
    
    # Mock display if not showing
    if not show_display:
        cv2.imshow = lambda *args: None
        cv2.waitKey = lambda *args: 0
        cv2.destroyAllWindows = lambda: None

    # ============ Benchmark Original ============
    print("\n[1] Original teleop_viewer.py")
    print("-" * 40)
    
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

    # ============ Benchmark Improved ============
    print("\n[2] Improved teleop_viewer_improved.py")
    print("-" * 40)
    
    from teleop_viewer_improved import OptimizedTeleopViewer, load_config as load_config2
    
    config2 = load_config2()
    viewer2 = OptimizedTeleopViewer(config2)
    
    # Warm up
    for name in viewer2.cameras:
        viewer2._process_camera(name)
    _ = viewer2._concatenate_layout(0)
    
    # Benchmark
    start = time.perf_counter()
    for i in range(num_frames):
        for name in viewer2.cameras:
            viewer2._process_camera(name)
        output2 = viewer2._concatenate_layout(0)
    elapsed2 = time.perf_counter() - start
    
    fps2 = num_frames / elapsed2
    print(f"  Frames: {num_frames}")
    print(f"  Time: {elapsed2:.3f}s")
    print(f"  FPS: {fps2:.1f}")
    print(f"  ms/frame: {(elapsed2/num_frames)*1000:.2f}")

    # ============ Comparison ============
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    speedup = fps2 / fps1
    print(f"  Original:  {fps1:.1f} FPS ({(elapsed1/num_frames)*1000:.2f} ms/frame)")
    print(f"  Improved:  {fps2:.1f} FPS ({(elapsed2/num_frames)*1000:.2f} ms/frame)")
    print(f"  Speedup:   {speedup:.2f}x faster")
    
    viewer2.executor.shutdown(wait=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark teleop viewers")
    parser.add_argument("-n", "--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--display", action="store_true", help="Show display (requires X)")
    args = parser.parse_args()
    
    run_benchmark(args.frames, args.display)
