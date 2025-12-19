"""Tests for overlays module."""

import pytest
import numpy as np

from multi_view_composer import draw_centermark, draw_border
from multi_view_composer.config import CentermarkConfig, BorderConfig


@pytest.fixture
def sample_image():
    """Create a sample 480x640 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestDrawCentermark:
    def test_draws_crosshair(self, sample_image):
        config = CentermarkConfig(enabled=True)
        img = sample_image.copy()
        draw_centermark(img, config)

        # Check that some pixels are non-zero (crosshair was drawn)
        assert np.any(img > 0)

    def test_disabled_does_nothing(self, sample_image):
        config = CentermarkConfig(enabled=False)
        img = sample_image.copy()
        draw_centermark(img, config)

        assert not np.any(img > 0)

    def test_center_position(self, sample_image):
        config = CentermarkConfig(enabled=True, color=(255, 0, 255))
        img = sample_image.copy()
        draw_centermark(img, config)

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # Check center pixel has the crosshair color
        assert img[cy, cx, 0] == 255  # Blue channel
        assert img[cy, cx, 2] == 255  # Red channel


class TestDrawBorder:
    def test_draws_border(self, sample_image):
        config = BorderConfig(enabled=True, color=(255, 255, 255))
        img = sample_image.copy()
        draw_border(img, config)

        h, w = img.shape[:2]
        # Check corners have border color
        assert np.all(img[0, 0] == 255)
        assert np.all(img[0, w - 1] == 255)
        assert np.all(img[h - 1, 0] == 255)

    def test_disabled_does_nothing(self, sample_image):
        config = BorderConfig(enabled=False)
        img = sample_image.copy()
        draw_border(img, config)

        assert not np.any(img > 0)
