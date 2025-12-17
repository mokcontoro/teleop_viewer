"""Layout and image concatenation logic with tree-based size computation."""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, Optional, Tuple, List


class Direction(Enum):
    """Concatenation direction."""
    VERTICAL = 0
    HORIZONTAL = 1


@dataclass
class LayoutNode:
    """Node in the layout tree."""
    height: int
    width: int
    camera: Optional[str] = None  # Camera name for leaf nodes
    direction: Optional[Direction] = None  # Direction for junction nodes
    left: Optional[LayoutNode] = None  # First child (top or left)
    right: Optional[LayoutNode] = None  # Second child (bottom or right)

    def resize(self, new_height: int, new_width: int, target_sizes: Dict[str, Tuple[int, int]]):
        """Recursively resize this node and all children."""
        if self.camera is not None:
            # Leaf node - update target size
            self.height = new_height
            self.width = new_width
            target_sizes[self.camera] = (new_height, new_width)
        else:
            # Junction node - distribute resize to children
            if self.direction == Direction.VERTICAL:
                # Vertical: distribute height proportionally
                ratio = new_height / self.height
                left_height = round(self.left.height * ratio)
                right_height = new_height - left_height
                self.height = new_height
                self.width = new_width
                self.left.resize(left_height, new_width, target_sizes)
                self.right.resize(right_height, new_width, target_sizes)
            else:
                # Horizontal: distribute width proportionally
                ratio = new_width / self.width
                left_width = round(self.left.width * ratio)
                right_width = new_width - left_width
                self.height = new_height
                self.width = new_width
                self.left.resize(new_height, left_width, target_sizes)
                self.right.resize(new_height, right_width, target_sizes)

    def get_image(self, get_camera_image: Callable[[str], np.ndarray]) -> np.ndarray:
        """Recursively get concatenated image from this node."""
        if self.camera is not None:
            return get_camera_image(self.camera)
        else:
            left_img = self.left.get_image(get_camera_image)
            right_img = self.right.get_image(get_camera_image)
            if self.direction == Direction.VERTICAL:
                return cv2.vconcat([left_img, right_img])
            else:
                return cv2.hconcat([left_img, right_img])


def make_camera_node(camera: str, height: int, width: int) -> LayoutNode:
    """Create a leaf node for a camera."""
    return LayoutNode(height=height, width=width, camera=camera)


def make_junction_node(
    node1: LayoutNode,
    node2: LayoutNode,
    direction: Direction,
    target_sizes: Dict[str, Tuple[int, int]]
) -> LayoutNode:
    """
    Create a junction node connecting two nodes.

    This implements the same resizing logic as the original ImageConcatTree.
    The smaller node is kept as-is, and the larger one is scaled down to match.
    """
    if direction == Direction.VERTICAL:
        # For vertical: match widths
        if node1.width > node2.width:
            # Scale down node1 to match node2's width
            scale = node2.width / node1.width
            new_height = round(node1.height * scale)
            node1.resize(new_height, node2.width, target_sizes)
        elif node1.width < node2.width:
            # Scale down node2 to match node1's width
            scale = node1.width / node2.width
            new_height = round(node2.height * scale)
            node2.resize(new_height, node1.width, target_sizes)

        total_height = node1.height + node2.height
        total_width = node1.width  # They now match

    else:  # HORIZONTAL
        # For horizontal: match heights
        if node1.height > node2.height:
            # Scale down node1 to match node2's height
            scale = node2.height / node1.height
            new_width = round(node1.width * scale)
            node1.resize(node2.height, new_width, target_sizes)
        elif node1.height < node2.height:
            # Scale down node2 to match node1's height
            scale = node1.height / node2.height
            new_width = round(node2.width * scale)
            node2.resize(node1.height, new_width, target_sizes)

        total_height = node1.height  # They now match
        total_width = node1.width + node2.width

    return LayoutNode(
        height=total_height,
        width=total_width,
        direction=direction,
        left=node1,
        right=node2
    )


def compute_horizontal_layout_sizes(
    camera_sizes: Dict[str, Tuple[int, int]]
) -> Tuple[LayoutNode, Dict[str, Tuple[int, int]]]:
    """
    Compute target sizes for horizontal layout using tree-based algorithm.

    Args:
        camera_sizes: Dict mapping camera name to (height, width) after rotation

    Returns:
        Tuple of (root LayoutNode, dict of camera -> target size)
    """
    target_sizes: Dict[str, Tuple[int, int]] = {}

    # Create camera nodes with initial sizes
    ee_node = make_camera_node("ee_cam", *camera_sizes["ee_cam"])
    target_sizes["ee_cam"] = camera_sizes["ee_cam"]

    boxwall_node = make_camera_node("boxwall_monitor_cam", *camera_sizes["boxwall_monitor_cam"])
    target_sizes["boxwall_monitor_cam"] = camera_sizes["boxwall_monitor_cam"]

    ifm1_node = make_camera_node("ifm_camera1", *camera_sizes["ifm_camera1"])
    target_sizes["ifm_camera1"] = camera_sizes["ifm_camera1"]

    ifm2_node = make_camera_node("ifm_camera2", *camera_sizes["ifm_camera2"])
    target_sizes["ifm_camera2"] = camera_sizes["ifm_camera2"]

    front_node = make_camera_node("front_monitor_cam", *camera_sizes["front_monitor_cam"])
    target_sizes["front_monitor_cam"] = camera_sizes["front_monitor_cam"]

    back_node = make_camera_node("back_monitor_cam", *camera_sizes["back_monitor_cam"])
    target_sizes["back_monitor_cam"] = camera_sizes["back_monitor_cam"]

    # Build tree (same structure as original):
    # 1. boxwall on top of ifm1 (vertical)
    boxwall_ifm1 = make_junction_node(boxwall_node, ifm1_node, Direction.VERTICAL, target_sizes)

    # 2. ee on top of (boxwall + ifm1) (vertical)
    ee_boxwall_ifm1 = make_junction_node(ee_node, boxwall_ifm1, Direction.VERTICAL, target_sizes)

    # 3. (ee + boxwall + ifm1) beside ifm2 (horizontal)
    left_section = make_junction_node(ee_boxwall_ifm1, ifm2_node, Direction.HORIZONTAL, target_sizes)

    # 4. front beside back (horizontal)
    monitors = make_junction_node(front_node, back_node, Direction.HORIZONTAL, target_sizes)

    # 5. left_section beside monitors (horizontal)
    root = make_junction_node(left_section, monitors, Direction.HORIZONTAL, target_sizes)

    return root, target_sizes


def compute_vertical_layout_sizes(
    camera_sizes: Dict[str, Tuple[int, int]]
) -> Tuple[LayoutNode, Dict[str, Tuple[int, int]]]:
    """
    Compute target sizes for vertical layout using tree-based algorithm.

    Args:
        camera_sizes: Dict mapping camera name to (height, width) after rotation

    Returns:
        Tuple of (root LayoutNode, dict of camera -> target size)
    """
    target_sizes: Dict[str, Tuple[int, int]] = {}

    # Create camera nodes
    ee_node = make_camera_node("ee_cam", *camera_sizes["ee_cam"])
    target_sizes["ee_cam"] = camera_sizes["ee_cam"]

    ifm1_node = make_camera_node("ifm_camera1", *camera_sizes["ifm_camera1"])
    target_sizes["ifm_camera1"] = camera_sizes["ifm_camera1"]

    ifm2_node = make_camera_node("ifm_camera2", *camera_sizes["ifm_camera2"])
    target_sizes["ifm_camera2"] = camera_sizes["ifm_camera2"]

    front_node = make_camera_node("front_monitor_cam", *camera_sizes["front_monitor_cam"])
    target_sizes["front_monitor_cam"] = camera_sizes["front_monitor_cam"]

    # Build vertical tree:
    # 1. ifm2 beside front_monitor (horizontal)
    top = make_junction_node(ifm2_node, front_node, Direction.HORIZONTAL, target_sizes)

    # 2. ifm1 beside ee (horizontal)
    bottom = make_junction_node(ifm1_node, ee_node, Direction.HORIZONTAL, target_sizes)

    # 3. top above bottom (vertical)
    root = make_junction_node(top, bottom, Direction.VERTICAL, target_sizes)

    return root, target_sizes


class LayoutManager:
    """Manages layout computation and image concatenation."""

    def __init__(self, camera_sizes: Dict[str, Tuple[int, int]], use_vertical: bool = False):
        """
        Initialize layout manager.

        Args:
            camera_sizes: Dict mapping camera name to (height, width) after rotation
            use_vertical: Whether to also compute vertical layout
        """
        self.num_layouts = 2 if use_vertical else 1
        self.roots: List[Optional[LayoutNode]] = [None] * self.num_layouts
        self.target_sizes: List[Dict[str, Tuple[int, int]]] = [{} for _ in range(self.num_layouts)]

        # Compute horizontal layout (tree_index=0)
        self.roots[0], self.target_sizes[0] = compute_horizontal_layout_sizes(camera_sizes)

        # Compute vertical layout if needed (tree_index=1)
        if use_vertical:
            self.roots[1], self.target_sizes[1] = compute_vertical_layout_sizes(camera_sizes)

    def get_target_size(self, camera_name: str, tree_index: int = 0) -> Tuple[int, int]:
        """Get the computed target size for a camera."""
        return self.target_sizes[tree_index].get(camera_name, (480, 640))

    def concatenate(
        self,
        get_image: Callable[[str], np.ndarray],
        tree_index: int = 0
    ) -> np.ndarray:
        """Concatenate images using the pre-computed layout tree."""
        if self.roots[tree_index] is not None:
            return self.roots[tree_index].get_image(get_image)
        return np.zeros((480, 640, 3), dtype=np.uint8)


# Keep simple functions for backwards compatibility
def vconcat_resize(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Vertically concatenate images, resizing to match widths."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if w1 != w2:
        if w1 > w2:
            scale = w2 / w1
            img1 = cv2.resize(img1, (w2, int(h1 * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            scale = w1 / w2
            img2 = cv2.resize(img2, (w1, int(h2 * scale)), interpolation=cv2.INTER_LINEAR)

    return cv2.vconcat([img1, img2])


def hconcat_resize(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Horizontally concatenate images, resizing to match heights."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2:
        if h1 > h2:
            scale = h2 / h1
            img1 = cv2.resize(img1, (int(w1 * scale), h2), interpolation=cv2.INTER_LINEAR)
        else:
            scale = h1 / h2
            img2 = cv2.resize(img2, (int(w2 * scale), h1), interpolation=cv2.INTER_LINEAR)

    return cv2.hconcat([img1, img2])


def create_placeholder(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Create a black placeholder image."""
    return np.zeros((height, width, channels), dtype=np.uint8)


# Deprecated - kept for backwards compatibility
def concatenate_horizontal_layout(
    get_image: Callable[[str], np.ndarray],
    tree_index: int = 0
) -> np.ndarray:
    """Concatenate using simple resize approach (deprecated)."""
    ee = get_image("ee_cam")
    boxwall = get_image("boxwall_monitor_cam")
    ifm1 = get_image("ifm_camera1")
    ifm2 = get_image("ifm_camera2")
    front = get_image("front_monitor_cam")
    back = get_image("back_monitor_cam")

    boxwall_ifm1 = vconcat_resize(boxwall, ifm1)
    left_col = vconcat_resize(ee, boxwall_ifm1)
    left_section = hconcat_resize(left_col, ifm2)
    monitors = hconcat_resize(front, back)
    return hconcat_resize(left_section, monitors)


def concatenate_vertical_layout(
    get_image: Callable[[str], np.ndarray],
    tree_index: int = 1
) -> np.ndarray:
    """Concatenate using simple resize approach (deprecated)."""
    ee = get_image("ee_cam")
    ifm1 = get_image("ifm_camera1")
    ifm2 = get_image("ifm_camera2")
    front = get_image("front_monitor_cam")

    top_row = hconcat_resize(ifm2, front)
    bottom_row = hconcat_resize(ifm1, ee)
    return vconcat_resize(top_row, bottom_row)
