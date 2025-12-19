# Multi-View Composer

A high-performance multi-view image composer with configurable text overlays and flexible YAML-based layouts.

## Features

- **Custom camera definitions** - define any cameras with resolution and rotation
- **Fully configurable text overlays** - templates, computed variables, conditional colors
- **Flexible layout system** - tree-based horizontal/vertical arrangements
- **Any dynamic data** - pass any variables via `update_dynamic_data()`
- **YAML validation** - helpful error messages for configuration issues
- **Logging support** - configurable logging for debugging
- High performance (~600+ FPS) with parallel processing

## Installation

```bash
git clone https://github.com/mokcontoro/multi_view_composer.git
cd multi_view_composer/
pip install ./
```

## Quick Start

```bash
# Run the example
python examples/example.py

# Run with debug logging
python examples/example.py --debug

# Run benchmark
python examples/benchmark.py -n 100
```
## Uninstallation

```bash
pip uninstall multi_view_composer
```


## Basic Usage

```python
from multi_view_composer import MultiViewComposer

# Load config
composer = MultiViewComposer("config.yaml")

# Feed camera images
composer.update_camera_image("cam_left", cv2_image, active=True)

# Update dynamic data (any keyword arguments)
composer.update_dynamic_data(
    temperature=25.5,
    speed_ms=10.0,
    mode="auto",
    level=75,
)

# Generate output
frames = composer.generate_frame()
cv2.imshow("Output", frames[0])

# Cleanup
composer.shutdown()
```

## Configuration

### Cameras

Define cameras with their resolution and optional rotation:

```yaml
cameras:
  cam_left:
    resolution: [480, 640]  # height, width
  cam_right:
    resolution: [480, 640]
    rotate: 90              # Optional: 90, 180, 270, -90
    centermark: true        # Optional: draw crosshair
```

### Layouts

Arrange cameras using horizontal/vertical tree structures:

```yaml
layouts:
  main:
    direction: horizontal
    children:
      - direction: vertical
        children:
          - camera: cam_top
          - camera: cam_bottom
      - camera: cam_right

active_layout: main
```

Visual result:
```
+----------+----------+
| cam_top  |          |
+----------+ cam_right|
|cam_bottom|          |
+----------+----------+
```

### Text Overlays

#### Basic Structure

```yaml
text_overlays:
  - id: my_overlay
    template: "Value: {my_var:.1f}"
    cameras: [cam_left, cam_right]
    color: [255, 200, 100]  # BGR
```

#### Formula Variables

Compute values from dynamic data:

```yaml
variables:
  speed_kmh:
    type: formula
    expr: "{speed_ms} * 3.6"  # Supports +, -, *, /
```

#### Conditional Variables

Return different values based on conditions:

```yaml
variables:
  status_text:
    type: conditional
    conditions:
      - when: "{mode} == 'auto'"
        value: "AUTO"
      - when: "{mode} == 'manual'"
        value: "MANUAL"
      - else: "STANDBY"
```

Use `format` for formatted output:

```yaml
conditions:
  - when: "{active} == false"
    value: "N/A"
  - else:
    format: "{value:.2f}cm"
```

#### Conditional Colors

Change color based on thresholds:

```yaml
color_rules:
  - when: "{level} >= 70"
    color: [0, 255, 0]    # Green
  - when: "{level} >= 30"
    color: [0, 200, 255]  # Yellow
  - else: [0, 0, 255]     # Red
```

#### Visibility Conditions

Show overlay only when condition is true:

```yaml
- id: warning
  template: "WARNING: {message}"
  visible_when: "{show_warning} == true"
```

#### Per-Overlay Styling

```yaml
style:
  font_scale: 0.7
  thickness: 2
  box_height: 30
  background_color: [0, 0, 60]
```

#### Default Style

```yaml
default_overlay_style:
  font: "HERSHEY_SIMPLEX"
  font_scale: 0.8
  thickness: 2
  box_height: 40
  padding_left: 5
  padding_top: 30
  background_color: [0, 0, 0]
```

### Condition Operators

- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean: `true`, `false`
- Strings: `{var} == 'value'` (single quotes)

## Logging

```python
from multi_view_composer import MultiViewComposer, setup_logging
import logging

# Basic setup
setup_logging(level=logging.INFO)

# Debug mode
setup_logging(level=logging.DEBUG)

# File logging
file_handler = logging.FileHandler("composer.log")
setup_logging(level=logging.DEBUG, handler=file_handler)
```

## Error Handling

```python
from multi_view_composer import MultiViewComposer, ConfigError

try:
    composer = MultiViewComposer("config.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
```

Common errors:
- `ConfigError: Configuration file not found: config.yaml`
- `ConfigError: Configuration must define at least one layout in 'layouts'`
- `ConfigError: layouts.main.children: must have at least 2 children`

## Project Structure

```
multi_view_composer/
├── pyproject.toml          # Package configuration
├── README.md
├── examples/
│   ├── example.py          # Example demonstrating all features
│   ├── example_config.yaml # Example configuration
│   └── benchmark.py        # Performance benchmark
└── multi_view_composer/    # Core package
    ├── __init__.py         # Public API exports
    ├── generator.py        # MultiViewComposer class
    ├── camera.py           # Camera handling
    ├── layout.py           # Layout management
    ├── overlays.py         # Overlay rendering
    ├── config.py           # Configuration & YAML loading
    ├── template_engine.py  # Template rendering
    ├── logging_config.py   # Logging utilities
    └── tests/              # Unit tests
```

## API Reference

### MultiViewComposer

```python
composer = MultiViewComposer(config)  # config: str path or ViewerConfig object
composer.update_camera_image(name, image, active=True)  # Update camera image
composer.update_dynamic_data(**kwargs)  # Update overlay variables
frames = composer.generate_frame()  # Generate output frames
composer.get_camera_names()  # List of camera names
composer.get_camera_config(name)  # Get CameraConfig for a camera
composer.shutdown()  # Clean up resources
```

### Exports

```python
from multi_view_composer import (
    # Main class
    MultiViewComposer,

    # Configuration
    ViewerConfig, CameraDefinition, OverlayStyle,
    TextOverlayConfig, ColorRule, VariableConfig,
    CentermarkConfig, BorderConfig, LayoutNodeConfig,
    ConfigError, load_config,

    # Logging
    setup_logging, get_logger,
)
```
