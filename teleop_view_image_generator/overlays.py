"""Overlay rendering for teleop viewer images."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorData:
    """Sensor data for overlays."""
    laser_distance: float = 35.0  # mm
    laser_active: bool = True
    pressure_manifold: float = 0.5  # bar
    pressure_base: float = 0.3  # bar
    robot_status: str = "Stopped"
    is_manual_review: bool = False


def draw_centermark(img: np.ndarray, color: tuple = (255, 0, 255), thickness: int = 4) -> None:
    """
    Draw crosshair centermark on image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        color: BGR color tuple
        thickness: Line thickness
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(w * 0.025)
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)


def draw_status_overlay(
    img: np.ndarray,
    robot_status: str,
    is_manual_review: bool,
    y_offset: int = 0
) -> int:
    """
    Draw robot status overlay on image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        robot_status: Current robot status string
        is_manual_review: Whether in manual review mode
        y_offset: Y position offset

    Returns:
        Next y_offset after this overlay
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    text = f"Status: {robot_status}"
    if robot_status == "SCANNING":
        text += " (Manual)" if is_manual_review else " (Auto)"
        color = (51, 153, 255) if is_manual_review else (255, 128, 0)
    elif robot_status in ("NAVIGATING", "UNLOADING", "FINISHED"):
        color = (255, 128, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(img, (0, y_offset), (w, y_offset + 40), (0, 0, 0), -1)
    cv2.putText(img, text, (5, y_offset + 30), font, 0.8, color, 2, cv2.LINE_AA)

    return y_offset + 40


def draw_laser_overlay(
    img: np.ndarray,
    laser_distance: float,
    laser_active: bool,
    y_offset: int = 40
) -> int:
    """
    Draw laser distance overlay on image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        laser_distance: Laser distance in mm
        laser_active: Whether laser sensor is active
        y_offset: Y position offset

    Returns:
        Next y_offset after this overlay
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    dist_cm = laser_distance * 0.1
    if not laser_active:
        text = "Laser: N/A"
        color = (0, 0, 255)
    elif dist_cm > 44:
        text = "Dist: N/A"
        color = (0, 0, 255)
    elif dist_cm > 31:
        text = f"Dist: {dist_cm:.2f}cm"
        color = (255, 0, 0)
    else:
        text = f"Dist: {dist_cm:.2f}cm"
        color = (0, 255, 0)

    cv2.rectangle(img, (0, y_offset), (w, y_offset + 40), (0, 0, 0), -1)
    cv2.putText(img, text, (5, y_offset + 30), font, 0.8, color, 2, cv2.LINE_AA)

    return y_offset + 40


def draw_pressure_overlay(
    img: np.ndarray,
    pressure_manifold: float,
    pressure_base: float,
    y_offset: int = 80
) -> int:
    """
    Draw vacuum pressure overlay on image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        pressure_manifold: Manifold pressure in bar
        pressure_base: Base pressure in bar
        y_offset: Y position offset

    Returns:
        Next y_offset after this overlay
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    text = f"Z1: {pressure_manifold:.4f} bar | Z2: {pressure_base:.4f} bar"
    cv2.rectangle(img, (0, y_offset), (w, y_offset + 40), (0, 0, 0), -1)
    cv2.putText(img, text, (5, y_offset + 30), font, 0.7, (255, 128, 0), 2, cv2.LINE_AA)

    return y_offset + 40


def draw_border(img: np.ndarray, color: tuple = (255, 255, 255), thickness: int = 1) -> None:
    """
    Draw border around image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        color: BGR color tuple
        thickness: Border thickness
    """
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)


def draw_camera_overlays(
    img: np.ndarray,
    camera_name: str,
    overlay_types: list,
    sensor_data: SensorData,
    tree_index: int = 0,
    draw_centermark_flag: bool = False
) -> None:
    """
    Draw all configured overlays on a camera image (in-place).

    Args:
        img: BGR image to draw on (modified in-place)
        camera_name: Name of the camera
        overlay_types: List of overlay types to draw ["status", "laser", "pressure"]
        sensor_data: SensorData object with current values
        tree_index: Layout index (overlays only on index 0)
        draw_centermark_flag: Whether to draw centermark
    """
    # Centermark
    if draw_centermark_flag:
        draw_centermark(img)

    # Only draw text overlays on tree_index 0 and back_monitor_cam
    if tree_index != 0 or camera_name != "back_monitor_cam":
        return

    y_offset = 0

    if "status" in overlay_types:
        y_offset = draw_status_overlay(
            img,
            sensor_data.robot_status,
            sensor_data.is_manual_review,
            y_offset
        )

    if "laser" in overlay_types:
        y_offset = draw_laser_overlay(
            img,
            sensor_data.laser_distance,
            sensor_data.laser_active,
            y_offset
        )

    if "pressure" in overlay_types:
        y_offset = draw_pressure_overlay(
            img,
            sensor_data.pressure_manifold,
            sensor_data.pressure_base,
            y_offset
        )
