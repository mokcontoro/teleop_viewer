"""Layout and image concatenation logic with tree-based size computation."""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LayoutNodeConfig
else:
    # Runtime import to avoid circular imports
    LayoutNodeConfig = None


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

    def resize(
        self, new_height: int, new_width: int, target_sizes: Dict[str, Tuple[int, int]]
    ):
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
    target_sizes: Dict[str, Tuple[int, int]],
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
        right=node2,
    )


def build_layout_from_config(
    layout_config: LayoutNodeConfig,
    camera_sizes: Dict[str, Tuple[int, int]],
    target_sizes: Dict[str, Tuple[int, int]],
) -> LayoutNode:
    """
    Build a layout tree from a LayoutNodeConfig.

    Args:
        layout_config: Configuration for the layout tree
        camera_sizes: Dict mapping camera name to (height, width)
        target_sizes: Dict to populate with computed target sizes

    Returns:
        Root LayoutNode of the built tree
    """
    if layout_config.camera is not None:
        # Leaf node - camera
        camera = layout_config.camera
        if camera in camera_sizes:
            height, width = camera_sizes[camera]
        else:
            # Default size for unknown cameras
            height, width = 480, 640
        target_sizes[camera] = (height, width)
        return make_camera_node(camera, height, width)

    # Junction node - has children
    if len(layout_config.children) < 2:
        raise ValueError("Junction node must have at least 2 children")

    direction = (
        Direction.HORIZONTAL
        if layout_config.direction == "horizontal"
        else Direction.VERTICAL
    )

    # Build children recursively
    child_nodes = []
    child_weights = []
    for child_config in layout_config.children:
        child_node = build_layout_from_config(child_config, camera_sizes, target_sizes)
        child_nodes.append(child_node)
        child_weights.append(child_config.weight)

    # Check if any child has weights specified
    has_weights = any(w is not None for w in child_weights)

    if has_weights:
        # Use weighted distribution
        result = build_weighted_junction(
            child_nodes, child_weights, direction, target_sizes
        )
    else:
        # Use original pairwise combination
        result = child_nodes[0]
        for i in range(1, len(child_nodes)):
            result = make_junction_node(result, child_nodes[i], direction, target_sizes)

    return result


def build_weighted_junction(
    child_nodes: List[LayoutNode],
    weights: List[Optional[float]],
    direction: Direction,
    target_sizes: Dict[str, Tuple[int, int]],
) -> LayoutNode:
    """
    Build a junction node with weighted size distribution.

    Args:
        child_nodes: List of child LayoutNodes
        weights: List of weights (None means auto-calculate based on size)
        direction: Direction of the junction
        target_sizes: Dict to update with computed target sizes

    Returns:
        Root LayoutNode combining all children with weighted sizes
    """
    n = len(child_nodes)

    # Calculate natural sizes for children without explicit weights
    if direction == Direction.HORIZONTAL:
        # For horizontal: distribute width, match heights
        ref_height = min(node.height for node in child_nodes)
        natural_widths = []
        for node in child_nodes:
            # Scale width to match reference height
            scale = ref_height / node.height
            natural_widths.append(round(node.width * scale))
        total_natural = sum(natural_widths)

        # Convert weights: None -> proportional to natural size
        resolved_weights = []
        for i, w in enumerate(weights):
            if w is not None:
                resolved_weights.append(w)
            else:
                resolved_weights.append(natural_widths[i] / total_natural)

        # Normalize weights to sum to 1.0
        weight_sum = sum(resolved_weights)
        normalized_weights = [w / weight_sum for w in resolved_weights]

        # Calculate total width based on natural sizes and weights
        # Find the child that constrains the total size the most
        total_width = 0
        for i, node in enumerate(child_nodes):
            scale = ref_height / node.height
            scaled_width = round(node.width * scale)
            implied_total = scaled_width / normalized_weights[i]
            if total_width == 0 or implied_total < total_width:
                total_width = int(implied_total)

        # Distribute width according to weights
        allocated_widths = []
        remaining = total_width
        for i in range(n - 1):
            w = round(total_width * normalized_weights[i])
            allocated_widths.append(w)
            remaining -= w
        allocated_widths.append(remaining)  # Last child gets remainder

        # Resize all children to their allocated sizes
        for i, node in enumerate(child_nodes):
            node.resize(ref_height, allocated_widths[i], target_sizes)

        # Build the result by combining nodes pairwise
        result = child_nodes[0]
        for i in range(1, n):
            result = LayoutNode(
                height=ref_height,
                width=result.width + child_nodes[i].width,
                direction=Direction.HORIZONTAL,
                left=result,
                right=child_nodes[i],
            )

    else:  # VERTICAL
        # For vertical: distribute height, match widths
        ref_width = min(node.width for node in child_nodes)
        natural_heights = []
        for node in child_nodes:
            # Scale height to match reference width
            scale = ref_width / node.width
            natural_heights.append(round(node.height * scale))
        total_natural = sum(natural_heights)

        # Convert weights: None -> proportional to natural size
        resolved_weights = []
        for i, w in enumerate(weights):
            if w is not None:
                resolved_weights.append(w)
            else:
                resolved_weights.append(natural_heights[i] / total_natural)

        # Normalize weights to sum to 1.0
        weight_sum = sum(resolved_weights)
        normalized_weights = [w / weight_sum for w in resolved_weights]

        # Calculate total height based on natural sizes and weights
        total_height = 0
        for i, node in enumerate(child_nodes):
            scale = ref_width / node.width
            scaled_height = round(node.height * scale)
            implied_total = scaled_height / normalized_weights[i]
            if total_height == 0 or implied_total < total_height:
                total_height = int(implied_total)

        # Distribute height according to weights
        allocated_heights = []
        remaining = total_height
        for i in range(n - 1):
            h = round(total_height * normalized_weights[i])
            allocated_heights.append(h)
            remaining -= h
        allocated_heights.append(remaining)  # Last child gets remainder

        # Resize all children to their allocated sizes
        for i, node in enumerate(child_nodes):
            node.resize(allocated_heights[i], ref_width, target_sizes)

        # Build the result by combining nodes pairwise
        result = child_nodes[0]
        for i in range(1, n):
            result = LayoutNode(
                height=result.height + child_nodes[i].height,
                width=ref_width,
                direction=Direction.VERTICAL,
                left=result,
                right=child_nodes[i],
            )

    return result


def compute_layout_from_config(
    layout_config: LayoutNodeConfig, camera_sizes: Dict[str, Tuple[int, int]]
) -> Tuple[LayoutNode, Dict[str, Tuple[int, int]]]:
    """
    Compute layout sizes from a config.

    Args:
        layout_config: Configuration for the layout tree
        camera_sizes: Dict mapping camera name to (height, width)

    Returns:
        Tuple of (root LayoutNode, dict of camera -> target size)
    """
    target_sizes: Dict[str, Tuple[int, int]] = {}
    root = build_layout_from_config(layout_config, camera_sizes, target_sizes)
    return root, target_sizes


class LayoutManager:
    """Manages layout computation and image concatenation."""

    def __init__(
        self,
        camera_sizes: Dict[str, Tuple[int, int]],
        layout_configs: Dict[str, LayoutNodeConfig],
        active_layout: str = "horizontal",
        output_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize layout manager.

        Args:
            camera_sizes: Dict mapping camera name to (height, width) after rotation
            layout_configs: Dict of layout configs from YAML
            active_layout: Name of the active layout
            output_size: Optional (height, width) constraint for the final mosaic.
                When provided, the natural layout is uniformly rescaled so the
                concatenated output matches this size. All per-camera target
                sizes are propagated through the layout tree. This avoids
                composing a larger mosaic and shrinking it after the fact.
        """
        self.layout_names = list(layout_configs.keys())
        self.num_layouts = len(self.layout_names)
        self.roots: List[Optional[LayoutNode]] = [None] * self.num_layouts
        self.target_sizes: List[Dict[str, Tuple[int, int]]] = [
            {} for _ in range(self.num_layouts)
        ]

        for i, name in enumerate(self.layout_names):
            self.roots[i], self.target_sizes[i] = compute_layout_from_config(
                layout_configs[name], camera_sizes
            )

        # Constrain final mosaic size if requested.
        if output_size is not None:
            target_h, target_w = output_size
            for i in range(self.num_layouts):
                root = self.roots[i]
                if root is not None:
                    root.resize(target_h, target_w, self.target_sizes[i])

        # Set active layout index
        self.active_layout_index = 0
        if active_layout in self.layout_names:
            self.active_layout_index = self.layout_names.index(active_layout)

    def get_target_size(self, camera_name: str, tree_index: int = 0) -> Tuple[int, int]:
        """Get the computed target size for a camera."""
        return self.target_sizes[tree_index].get(camera_name, (480, 640))

    def concatenate(
        self, get_image: Callable[[str], np.ndarray], tree_index: int = 0
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
            img1 = cv2.resize(
                img1, (w2, int(h1 * scale)), interpolation=cv2.INTER_LINEAR
            )
        else:
            scale = w1 / w2
            img2 = cv2.resize(
                img2, (w1, int(h2 * scale)), interpolation=cv2.INTER_LINEAR
            )

    return cv2.vconcat([img1, img2])


def hconcat_resize(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Horizontally concatenate images, resizing to match heights."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2:
        if h1 > h2:
            scale = h2 / h1
            img1 = cv2.resize(
                img1, (int(w1 * scale), h2), interpolation=cv2.INTER_LINEAR
            )
        else:
            scale = h1 / h2
            img2 = cv2.resize(
                img2, (int(w2 * scale), h1), interpolation=cv2.INTER_LINEAR
            )

    return cv2.hconcat([img1, img2])


def create_placeholder(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Create a black placeholder image."""
    return np.zeros((height, width, channels), dtype=np.uint8)
