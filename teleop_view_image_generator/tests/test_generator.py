"""Tests for generator module."""

import pytest
import numpy as np

from teleop_view_image_generator import TeleopImageGenerator, CameraConfig


@pytest.fixture
def default_config():
    """Create default configuration."""
    return {
        "resolutions": {
            "ee_cam": (480, 848, 3),
            "ifm": (800, 1280, 3),
            "monitor_cam": (720, 1280, 3),
            "recovery_cam": (530, 848, 3),
        },
        "hardware": {
            "old_elbow_cam": True,
            "camera_mount": "D",
        },
        "use_vertical": False,
    }


@pytest.fixture
def generator(default_config):
    """Create a generator instance."""
    gen = TeleopImageGenerator(default_config)
    yield gen
    gen.shutdown()


@pytest.fixture
def sample_image():
    """Create a sample camera image."""
    return np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)


class TestTeleopImageGeneratorInit:
    def test_initializes_with_config(self, default_config):
        gen = TeleopImageGenerator(default_config)

        assert gen is not None
        assert gen.num_layouts == 1  # use_vertical=False
        gen.shutdown()

    def test_vertical_mode(self, default_config):
        default_config["use_vertical"] = True
        gen = TeleopImageGenerator(default_config)

        assert gen.num_layouts == 2
        gen.shutdown()

    def test_initializes_cameras(self, generator):
        cameras = generator.get_camera_names()

        expected_cameras = [
            "ee_cam", "ifm_camera1", "ifm_camera2",
            "front_monitor_cam", "back_monitor_cam",
            "A1_cam1", "A1_cam2", "boxwall_monitor_cam"
        ]
        for cam in expected_cameras:
            assert cam in cameras


class TestUpdateCameraImage:
    def test_updates_existing_camera(self, generator, sample_image):
        result = generator.update_camera_image("ee_cam", sample_image, active=True)

        assert result is True
        cam = generator.get_camera_config("ee_cam")
        assert cam.active is True
        assert cam.raw_image is not None

    def test_rejects_unknown_camera(self, generator, sample_image):
        result = generator.update_camera_image("unknown_camera", sample_image)

        assert result is False

    def test_sets_inactive(self, generator, sample_image):
        generator.update_camera_image("ee_cam", sample_image, active=False)

        cam = generator.get_camera_config("ee_cam")
        assert cam.active is False


class TestUpdateSensorData:
    def test_updates_laser_distance(self, generator):
        generator.update_sensor_data(laser_distance=42.0)

        assert generator.sensor_data.laser_distance == 42.0

    def test_updates_robot_status(self, generator):
        generator.update_sensor_data(robot_status="NAVIGATING")

        assert generator.sensor_data.robot_status == "NAVIGATING"

    def test_updates_multiple_values(self, generator):
        generator.update_sensor_data(
            laser_distance=35.0,
            laser_active=False,
            pressure_manifold=0.8,
            robot_status="SCANNING"
        )

        assert generator.sensor_data.laser_distance == 35.0
        assert generator.sensor_data.laser_active is False
        assert generator.sensor_data.pressure_manifold == 0.8
        assert generator.sensor_data.robot_status == "SCANNING"


class TestGenerateFrame:
    def test_generates_single_frame(self, generator, sample_image):
        # Set up some camera images
        generator.update_camera_image("ee_cam", sample_image, active=True)

        frames = generator.generate_frame()

        assert len(frames) == 1
        assert frames[0] is not None
        assert len(frames[0].shape) == 3

    def test_generates_two_frames_vertical_mode(self, default_config, sample_image):
        default_config["use_vertical"] = True
        gen = TeleopImageGenerator(default_config)

        gen.update_camera_image("ee_cam", sample_image, active=True)
        frames = gen.generate_frame()

        assert len(frames) == 2
        gen.shutdown()

    def test_handles_no_active_cameras(self, generator):
        # Don't set any camera images
        frames = generator.generate_frame()

        # Should still return frames (with placeholders)
        assert len(frames) == 1
        assert frames[0] is not None


class TestGetCameraConfig:
    def test_returns_config_for_existing_camera(self, generator):
        config = generator.get_camera_config("ee_cam")

        assert config is not None
        assert isinstance(config, CameraConfig)
        assert config.name == "ee_cam"

    def test_returns_none_for_unknown_camera(self, generator):
        config = generator.get_camera_config("unknown_camera")

        assert config is None


class TestShutdown:
    def test_shutdown_completes(self, default_config):
        gen = TeleopImageGenerator(default_config)
        gen.shutdown()

        # Should not raise any exceptions
        assert True
