"""Tests for overlays module."""

import pytest
import numpy as np
import cv2

from teleop_view_image_generator.overlays import (
    SensorData,
    draw_centermark,
    draw_status_overlay,
    draw_laser_overlay,
    draw_pressure_overlay,
    draw_border,
    draw_camera_overlays
)


@pytest.fixture
def sample_image():
    """Create a sample 480x640 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sensor_data():
    """Create sample sensor data."""
    return SensorData(
        laser_distance=35.0,
        laser_active=True,
        pressure_manifold=0.5,
        pressure_base=0.3,
        robot_status="SCANNING",
        is_manual_review=True
    )


class TestDrawCentermark:
    def test_draws_crosshair(self, sample_image):
        img = sample_image.copy()
        draw_centermark(img)

        # Check that some pixels are non-zero (crosshair was drawn)
        assert np.any(img > 0)

    def test_center_position(self, sample_image):
        img = sample_image.copy()
        draw_centermark(img, color=(255, 0, 255))

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # Check center pixel has the crosshair color
        assert img[cy, cx, 0] == 255  # Blue channel
        assert img[cy, cx, 2] == 255  # Red channel


class TestDrawStatusOverlay:
    def test_draws_status_text(self, sample_image):
        img = sample_image.copy()
        next_y = draw_status_overlay(img, "SCANNING", True, 0)

        assert next_y == 40
        # Check that something was drawn (black background replaced)
        assert np.any(img[:40, :] > 0)

    def test_different_statuses(self, sample_image):
        for status in ["SCANNING", "NAVIGATING", "UNLOADING", "FINISHED", "ERROR"]:
            img = sample_image.copy()
            draw_status_overlay(img, status, False, 0)
            assert np.any(img > 0)


class TestDrawLaserOverlay:
    def test_active_laser_close(self, sample_image):
        img = sample_image.copy()
        next_y = draw_laser_overlay(img, 250.0, True, 0)  # 25cm

        assert next_y == 40
        assert np.any(img[:40, :] > 0)

    def test_inactive_laser(self, sample_image):
        img = sample_image.copy()
        draw_laser_overlay(img, 0.0, False, 0)
        assert np.any(img[:40, :] > 0)

    def test_laser_out_of_range(self, sample_image):
        img = sample_image.copy()
        draw_laser_overlay(img, 500.0, True, 0)  # 50cm, out of range
        assert np.any(img[:40, :] > 0)


class TestDrawPressureOverlay:
    def test_draws_pressure(self, sample_image):
        img = sample_image.copy()
        next_y = draw_pressure_overlay(img, 0.5, 0.3, 0)

        assert next_y == 40
        assert np.any(img[:40, :] > 0)


class TestDrawBorder:
    def test_draws_border(self, sample_image):
        img = sample_image.copy()
        draw_border(img, color=(255, 255, 255))

        h, w = img.shape[:2]
        # Check corners have border color
        assert np.all(img[0, 0] == 255)
        assert np.all(img[0, w-1] == 255)
        assert np.all(img[h-1, 0] == 255)


class TestDrawCameraOverlays:
    def test_back_monitor_cam_overlays(self, sample_image, sensor_data):
        img = sample_image.copy()
        overlay_types = ["status", "laser", "pressure"]

        draw_camera_overlays(
            img, "back_monitor_cam", overlay_types, sensor_data,
            tree_index=0, draw_centermark_flag=False
        )

        # Check overlays were drawn (top 120 pixels should have content)
        assert np.any(img[:120, :] > 0)

    def test_no_overlays_on_tree_index_1(self, sample_image, sensor_data):
        img = sample_image.copy()
        overlay_types = ["status", "laser", "pressure"]

        draw_camera_overlays(
            img, "back_monitor_cam", overlay_types, sensor_data,
            tree_index=1, draw_centermark_flag=False
        )

        # Should be unchanged (black)
        assert not np.any(img > 0)

    def test_centermark_on_ee_cam(self, sample_image, sensor_data):
        img = sample_image.copy()

        draw_camera_overlays(
            img, "ee_cam", [], sensor_data,
            tree_index=0, draw_centermark_flag=True
        )

        # Centermark should be drawn
        assert np.any(img > 0)
