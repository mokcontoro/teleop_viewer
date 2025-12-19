"""Tests for generator module."""

import pytest
import numpy as np

from multi_view_composer import (
    MultiViewComposer,
    CameraConfig,
    CameraDefinition,
    ViewerConfig,
    LayoutNodeConfig,
)


def create_test_config(num_layouts: int = 1) -> ViewerConfig:
    """Create a test configuration with layouts."""
    # Define cameras
    cameras = {
        "cam1": CameraDefinition(name="cam1", resolution=(480, 640)),
        "cam2": CameraDefinition(name="cam2", resolution=(480, 640)),
        "cam3": CameraDefinition(name="cam3", resolution=(480, 640)),
    }

    # Create a simple horizontal layout
    horizontal_layout = LayoutNodeConfig(
        direction="horizontal",
        children=[
            LayoutNodeConfig(camera="cam1"),
            LayoutNodeConfig(camera="cam2"),
        ]
    )

    layouts = {"main": horizontal_layout}

    # Add vertical layout if requested
    if num_layouts > 1:
        vertical_layout = LayoutNodeConfig(
            direction="vertical",
            children=[
                LayoutNodeConfig(camera="cam1"),
                LayoutNodeConfig(camera="cam3"),
            ]
        )
        layouts["vertical"] = vertical_layout

    return ViewerConfig(
        cameras=cameras,
        layouts=layouts,
        active_layout="main",
    )


@pytest.fixture
def default_config():
    """Create default configuration."""
    return create_test_config(num_layouts=1)


@pytest.fixture
def composer(default_config):
    """Create a composer instance."""
    comp = MultiViewComposer(default_config)
    yield comp
    comp.shutdown()


@pytest.fixture
def sample_image():
    """Create a sample camera image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestMultiViewComposerInit:
    def test_initializes_with_config(self):
        config = create_test_config(num_layouts=1)
        comp = MultiViewComposer(config)

        assert comp is not None
        assert comp.num_layouts == 1
        comp.shutdown()

    def test_multi_layout_mode(self):
        config = create_test_config(num_layouts=2)
        comp = MultiViewComposer(config)

        assert comp.num_layouts == 2
        comp.shutdown()

    def test_initializes_only_used_cameras(self, composer):
        """Only cameras referenced in layout should be initialized."""
        cameras = composer.get_camera_names()

        # These are in the layout
        assert "cam1" in cameras
        assert "cam2" in cameras

        # cam3 is NOT in the single-layout config
        assert "cam3" not in cameras

    def test_raises_error_without_layouts(self):
        """Should raise error if no layouts defined."""
        config = ViewerConfig(
            cameras={},
            layouts={},
        )
        with pytest.raises(ValueError, match="No layouts defined"):
            MultiViewComposer(config)

    def test_auto_creates_cameras_not_in_config(self):
        """Cameras in layout but not in config should be auto-created."""
        layouts = {
            "main": LayoutNodeConfig(
                direction="horizontal",
                children=[
                    LayoutNodeConfig(camera="undefined_cam1"),
                    LayoutNodeConfig(camera="undefined_cam2"),
                ]
            )
        }
        config = ViewerConfig(cameras={}, layouts=layouts, active_layout="main")
        comp = MultiViewComposer(config)

        assert "undefined_cam1" in comp.get_camera_names()
        cam = comp.get_camera_config("undefined_cam1")
        assert cam.resolution == (480, 640, 3)  # Default resolution
        comp.shutdown()


class TestUpdateCameraImage:
    def test_updates_existing_camera(self, composer, sample_image):
        result = composer.update_camera_image("cam1", sample_image, active=True)

        assert result is True
        cam = composer.get_camera_config("cam1")
        assert cam.active is True
        assert cam.raw_image is not None

    def test_rejects_unknown_camera(self, composer, sample_image):
        result = composer.update_camera_image("unknown_camera", sample_image)

        assert result is False

    def test_sets_inactive(self, composer, sample_image):
        composer.update_camera_image("cam1", sample_image, active=False)

        cam = composer.get_camera_config("cam1")
        assert cam.active is False


class TestUpdateDynamicData:
    def test_updates_custom_data(self, composer):
        composer.update_dynamic_data(temperature=42.0, my_value=100)

        assert composer.dynamic_data["temperature"] == 42.0
        assert composer.dynamic_data["my_value"] == 100

    def test_updates_multiple_values(self, composer):
        composer.update_dynamic_data(
            speed=35.0,
            active=False,
            level=80,
            status="running"
        )

        assert composer.dynamic_data["speed"] == 35.0
        assert composer.dynamic_data["active"] is False
        assert composer.dynamic_data["level"] == 80
        assert composer.dynamic_data["status"] == "running"


class TestGenerateFrame:
    def test_generates_single_frame(self, composer, sample_image):
        composer.update_camera_image("cam1", sample_image, active=True)

        frames = composer.generate_frame()

        assert len(frames) == 1
        assert frames[0] is not None
        assert len(frames[0].shape) == 3

    def test_generates_two_frames_multi_layout(self, sample_image):
        config = create_test_config(num_layouts=2)
        comp = MultiViewComposer(config)

        comp.update_camera_image("cam1", sample_image, active=True)
        frames = comp.generate_frame()

        assert len(frames) == 2
        comp.shutdown()

    def test_handles_no_active_cameras(self, composer):
        frames = composer.generate_frame()

        # Should still return frames (with placeholders)
        assert len(frames) == 1
        assert frames[0] is not None


class TestGetCameraConfig:
    def test_returns_config_for_existing_camera(self, composer):
        config = composer.get_camera_config("cam1")

        assert config is not None
        assert isinstance(config, CameraConfig)
        assert config.name == "cam1"

    def test_returns_none_for_unknown_camera(self, composer):
        config = composer.get_camera_config("unknown_camera")

        assert config is None


class TestShutdown:
    def test_shutdown_completes(self):
        config = create_test_config()
        comp = MultiViewComposer(config)
        comp.shutdown()

        assert True
