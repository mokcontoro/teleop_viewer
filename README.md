# Multi-View Composer

A high-performance multi-view image composer with configurable text overlays and flexible YAML-based layouts.

## Features

- Multi-camera image processing and display
- **Fully configurable text overlays** via YAML (templates, colors, conditions)
- **Flexible layout system** - define camera arrangements in config
- Automatic camera filtering - only processes cameras used in layouts
- Image caching and parallel processing for high performance
- **On-demand sample image generation** - no static test files needed

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run the viewer (generates sample images automatically if needed)
python viewer.py

# With custom config
python viewer.py -c config.yaml

# Run example with random dynamic values
python example.py
```

### Controls
- `q` or `ESC`: Quit
- `n`: Next frame

## Benchmark

Run the benchmark (automatically generates and cleans up test images):
```bash
python benchmark.py -n 50
```

## Configuration

All settings are configured via `config.yaml`:

### Text Overlays

Define unlimited text overlays with templates, variables, and conditional colors:

```yaml
text_overlays:
  - id: status
    template: "Status: {robot_status} ({mode})"
    cameras: [back_monitor_cam, front_monitor_cam]
    variables:
      mode:
        type: conditional
        conditions:
          - when: "{is_manual_review} == true"
            value: "Manual"
          - else: "Auto"
    color_rules:
      - when: "{is_manual_review} == true"
        color: [51, 153, 255]  # BGR orange
      - else: [255, 128, 0]

  - id: laser
    template: "Dist: {distance_display}"
    cameras: [back_monitor_cam, ifm_camera2]
    variables:
      distance_cm:
        type: formula
        expr: "{laser_distance} * 0.1"
      distance_display:
        type: conditional
        conditions:
          - when: "{laser_active} == false"
            value: "N/A"
          - else:
            format: "{distance_cm:.2f}cm"
```

### Layouts

Define camera arrangements as trees:

```yaml
layouts:
  horizontal:
    direction: horizontal
    children:
      - direction: vertical
        children:
          - camera: ee_cam
          - camera: ifm_camera1
      - camera: ifm_camera2
      - direction: horizontal
        children:
          - camera: front_monitor_cam
          - camera: back_monitor_cam

active_layout: "horizontal"
```

Only cameras defined in layouts are processed (unused cameras are skipped for performance).

### Available Template Variables

From dynamic data:
- `{laser_distance}` - float (mm)
- `{laser_active}` - bool
- `{pressure_manifold}` - float (bar)
- `{pressure_base}` - float (bar)
- `{robot_status}` - string
- `{is_manual_review}` - bool

## Programmatic Usage

```python
from multi_view_composer import MultiViewComposer

# Create composer from config file
composer = MultiViewComposer("config.yaml")

# Feed camera images
composer.update_camera_image("ee_cam", cv2_image, active=True)

# Update dynamic data for overlays
composer.update_dynamic_data(
    laser_distance=25.0,
    robot_status="SCANNING",
)

# Generate output frame
frames = composer.generate_frame()
```

### Generating Test Images Programmatically

```python
from multi_view_composer import generate_sample_images, cleanup_sample_images

# Generate synthetic test images
sample_dir = generate_sample_images("./", num_frames=2)

# ... run your tests ...

# Clean up when done
cleanup_sample_images("./")
```

Or use the context manager:

```python
from multi_view_composer import SampleImageContext

with SampleImageContext() as sample_dir:
    # Images are available in sample_dir
    # ... run tests ...
# Images are automatically cleaned up
```

## Project Structure

```
multi_view_composer/
├── viewer.py                 # Main viewer application
├── example.py                # Example with random dynamic values
├── benchmark.py              # Performance benchmark
├── config.yaml               # Configuration file
└── multi_view_composer/      # Core package
    ├── __init__.py
    ├── generator.py          # MultiViewComposer class
    ├── camera.py             # Camera configurations
    ├── layout.py             # Layout management
    ├── overlays.py           # Overlay rendering
    ├── config.py             # Configuration dataclasses
    ├── template_engine.py    # Template rendering engine
    └── sample_images.py      # Synthetic image generation
```
