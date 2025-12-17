#!/usr/bin/python3

from __future__ import annotations
import cv2
import numpy as np
import yaml
import logging
import time
import os
import glob

from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("Teleop Viewer")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_path)

    if not os.path.exists(config_file):
        logger.warning(f"Config file not found at {config_file}, using defaults")
        return get_default_config()

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_default_config() -> dict:
    """Return default configuration values."""
    return {
        "resolutions": {
            "ee_cam": [480, 848, 3],
            "ifm": [800, 1280, 3],
            "monitor_cam": [720, 1280, 3],
            "recovery_cam": [530, 848, 3],
        },
        "hardware": {
            "old_elbow_cam": True,
            "camera_mount": "D",
        },
        "use_vertical": False,
        "input_directory": "./sample_images",
        "fps": 10,
        "window_name": "Teleop Viewer",
        "sensors": {
            "laser_distance": 35.0,
            "laser_active": True,
            "pressure_manifold": 0.5,
            "pressure_base": 0.3,
            "robot_status": "SCANNING",
            "is_manual_review": True,
        },
    }


# Load configuration
CONFIG = load_config()

# Extract resolution tuples from config
ee_cam_resolution = tuple(CONFIG["resolutions"]["ee_cam"])
ifm_resolution = tuple(CONFIG["resolutions"]["ifm"])
monitor_cam_resolution = tuple(CONFIG["resolutions"]["monitor_cam"])
recovery_cam_resolution = tuple(CONFIG["resolutions"]["recovery_cam"])

# Hardware configuration
old_elbow_cam = CONFIG["hardware"]["old_elbow_cam"]
camera_mount = CONFIG["hardware"]["camera_mount"]

# GPU acceleration (hardcoded to false)
USE_GPU = False


# Custom data class to replace ROS CompressedImage
@dataclass
class ImageData:
    """Container for compressed image data (replaces ROS CompressedImage)."""
    data: bytes  # Raw image bytes (JPEG/PNG)
    format: str  # Image format ("jpeg", "png", etc.)

    def __init__(self, data: bytes = b"", format: str = "jpeg"):
        self.data = data
        self.format = format


# Stores information about the images and how they will be manipulated
@dataclass
class CompressedImageItem:
    camera_name: str  # Camera identifier (e.g., "ee_cam", "ifm_camera1")
    image_path: str  # Path to image file or directory
    active: bool  # Is the image available/loaded
    data: ImageData  # Image data container
    resolution: Tuple[int, int, int]  # Resolution of original image
    rotate: Optional[
        int
    ]  # How the image should be rotated (e.g. cv2.ROTATE_90_COUNTERCLOCKWISE)
    centermark: bool  # If the image should have a centermark added
    desired_resolutions: List[
        Tuple[int, int, int]
    ]  # List of the desired resolutions after image concatenation, one per tree
    cv2_images: List[
        np.ndarray
    ]  # The images after resizing and adding text/marks, one per tree
    feedback_text_functions: Optional[
        List[Callable[[int], FeedbackTextData]]
    ]  # The functions that should be called to define what text data should be added to the image


# The
@dataclass
class FeedbackTextData:
    display_text: str
    position: Tuple[int, int]
    font: int
    font_scale: float
    font_color: Tuple[int, int, int]
    thickness: int
    line_type: int
    desired_width: int
    desired_height: int


# Indicates which image should be resized during concatenation, first, second, or neither
class ImageChanged(Enum):
    NEITHER = 0
    ONE = 1
    TWO = 2


class ImageLoader:
    """Loads images from file system for each camera."""

    def __init__(self, input_directory: str):
        self.input_directory = input_directory
        self.camera_images: dict[str, List[str]] = {}  # camera_name -> list of image paths
        self.camera_indices: dict[str, int] = {}  # camera_name -> current index
        self._scan_directories()

    def _scan_directories(self):
        """Scan input directory for camera subdirectories and their images."""
        if not os.path.exists(self.input_directory):
            logger.warning(f"Input directory does not exist: {self.input_directory}")
            return

        # Look for camera subdirectories
        camera_names = [
            "ee_cam", "ifm_camera1", "ifm_camera2",
            "front_monitor_cam", "back_monitor_cam",
            "A1_cam1", "A1_cam2", "boxwall_monitor_cam", "recovery_cam"
        ]

        for camera_name in camera_names:
            camera_dir = os.path.join(self.input_directory, camera_name)
            if os.path.isdir(camera_dir):
                # Find all image files in the directory
                image_files = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    image_files.extend(glob.glob(os.path.join(camera_dir, ext)))
                    image_files.extend(glob.glob(os.path.join(camera_dir, ext.upper())))

                if image_files:
                    self.camera_images[camera_name] = sorted(image_files)
                    self.camera_indices[camera_name] = 0
                    logger.info(f"Found {len(image_files)} images for {camera_name}")

    def load_image(self, camera_name: str) -> Optional[ImageData]:
        """Load the current image for a camera."""
        if camera_name not in self.camera_images:
            return None

        images = self.camera_images[camera_name]
        if not images:
            return None

        idx = self.camera_indices[camera_name]
        image_path = images[idx]

        try:
            with open(image_path, "rb") as f:
                data = f.read()

            # Determine format from extension
            ext = os.path.splitext(image_path)[1].lower()
            format_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".bmp": "bmp"}
            img_format = format_map.get(ext, "jpeg")

            return ImageData(data=data, format=img_format)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def advance_frame(self, camera_name: str) -> bool:
        """Advance to the next frame for a camera. Returns True if wrapped around."""
        if camera_name not in self.camera_images:
            return False

        images = self.camera_images[camera_name]
        if not images:
            return False

        self.camera_indices[camera_name] = (self.camera_indices[camera_name] + 1) % len(images)
        return self.camera_indices[camera_name] == 0

    def has_images(self, camera_name: str) -> bool:
        """Check if a camera has any images available."""
        return camera_name in self.camera_images and len(self.camera_images[camera_name]) > 0

    def get_image_path(self, camera_name: str) -> str:
        """Get the directory path for a camera."""
        return os.path.join(self.input_directory, camera_name)


"""
Binary tree structure. Stores Concat nodes which are one of two types, camera nodes (contain an image), or junction nodes, which contain how
two images are concatenated (vertically or horizontally). All nodes contain image of the overall width and height in pixels of their child nodes, which is the
sum of all the concatenated images. The camera nodes have no children, and their parents must be junction nodes, the junction nodes have two children,
and those children can either be another junction node (set of concatenated images), or camera nodes

Example Set of concatenated images:
-----------------------------------------------------
|                   |           |                   |
|                   |           |         E         |
|       A           |           |                   |
|                   |           |--------------------
|-------------------|    D      |                   |
|         |         |           |         F         |
|         |         |           |                   |
|    B    |    C    |           |--------------------
|         |         |           |         G         |
|         |         |           |                   |
-----------------------------------------------------

Tree: (0 is a vertical junction node, 1 is a Horizontal)
(Left node is left in a horizontal relationship, left node is top for vertical)

It is easist to start bottom up, left to right:
1. B is horizontally connected to C
2. A is vertically connected to (B,C)
3. (A,B,C) is horizontally connected to D
4. F is vertically connected to G
5. E is vertiically connected to (F,G)
6 (A,B,C,D) is horizontally connected to (E,F,G)


                    1
                  /   \ 
                 1     0
               /  \   / \ 
              0    D  E   0
            /  \         / \ 
           A    1       F   G
               / \ 
              B   C

The tree must be made by first making camera nodes, and then by making the junction nodes that connect them. When you make a junction node, the desired
resolution of the connected nodes and all children are recursively resized to fit together. When you go to get the image from the tree, the images from
the tree nodes are recursively concatenated to make one big image.
"""


class ImageConcatTree:
    root: Optional[ConcatNode]
    tree_index: int

    def __init__(self):
        self.root = None
        self.tree_index = 0

    @staticmethod
    def make_camera_node(
        teleop_obj: TeleopViewerDnD, camera_name: str, tree_index: int
    ) -> ConcatNode:
        """Makes a node that holds one camera's image and size"""
        return ImageConcatTree.ConcatNode(
            total_height=teleop_obj.compressed_image_items[
                camera_name
            ].desired_resolutions[tree_index][0],
            total_width=teleop_obj.compressed_image_items[
                camera_name
            ].desired_resolutions[tree_index][1],
            junction=None,
            camera=camera_name,
        )

    @staticmethod
    def make_juction_node(
        node1: ConcatNode,
        node2: ConcatNode,
        direction: Direction,
        teleop_obj: TeleopViewerDnD,
        tree_index: int,
    ) -> ImageConcatTree.ConcatNode:
        """Makes a node that holds the connection between two nodes, which can either be a camera node or a junction node.
        Recursively resizes all nodes connected to the two nodes being joined"""
        changed = ImageChanged.NEITHER
        if direction == ImageConcatTree.Direction.VERTICAL:
            if node1.total_width > node2.total_width:
                changed = ImageChanged.ONE
                scale_ratio = float(node2.total_width) / float(node1.total_width)
                des_width = node2.total_width
                des_height = round(node1.total_height * scale_ratio)

                total_height = des_height + node2.total_height
                total_width = des_width

            elif node1.total_width < node2.total_width:
                changed = ImageChanged.TWO
                scale_ratio = float(node1.total_width) / float(node2.total_width)
                des_width = node1.total_width
                des_height = round(node2.total_height * scale_ratio)
                total_height = des_height + node1.total_height
                total_width = des_width

            else:
                changed = ImageChanged.NEITHER
                total_height = node1.total_height + node2.total_height
                total_width = node1.total_width
        else:  # HORIZONTAL
            if node1.total_height > node2.total_height:
                changed = ImageChanged.ONE
                scale_ratio = float(node2.total_height) / float(node1.total_height)
                des_height = node2.total_height
                des_width = round(node1.total_width * scale_ratio)

                total_height = des_height
                total_width = des_width + node2.total_width
            elif node1.total_height < node2.total_height:
                changed = ImageChanged.TWO
                scale_ratio = float(node1.total_height) / float(node2.total_height)
                des_height = node1.total_height
                des_width = round(node2.total_width * scale_ratio)
                total_height = des_height
                total_width = des_width + node1.total_width

            else:
                changed = ImageChanged.NEITHER
                total_height = node1.total_height
                total_width = node1.total_width + node2.total_width

        if changed == ImageChanged.ONE:
            node1.resize_node(des_height, des_width, teleop_obj, tree_index)
        elif changed == ImageChanged.TWO:
            node2.resize_node(des_height, des_width, teleop_obj, tree_index)

        return ImageConcatTree.ConcatNode(
            total_height=total_height,
            total_width=total_width,
            junction=ImageConcatTree.Junction(
                direction=direction, node1=node1, node2=node2
            ),
            camera=None,
        )

    def print_tree_sizes(self):
        """For debug only, print the sizes of all the nodes in the tree"""
        if self.root is not None:
            self.root.print_sizes()

    def get_tree_image(self, teleop_obj: TeleopViewerDnD) -> np.ndarray:
        """Recursively concatenates images in the tree to get the final image"""
        if self.root is not None:
            return self.root.get_node_image(teleop_obj, self.tree_index)
        else:
            return np.ndarray((480, 480, 3))

    class ConcatNode:
        """Node of the tree"""

        total_height: int
        total_width: int
        junction: Optional[ImageConcatTree.Junction]
        camera: Optional[str]

        def __init__(
            self,
            total_height=0,
            total_width=0,
            junction=None,
            camera=None,
        ):
            self.total_height = total_height
            self.total_width = total_width
            self.junction = junction
            self.camera = camera

        def resize_node(
            self,
            des_height: int,
            des_width: int,
            teleop_obj: TeleopViewerDnD,
            tree_index: int,
        ):
            """Resizes the node to the desired height and width.
            If the node is a camera node, it just sets the size.
            If the node is a junction node, it calculates what the deisred sizes should be for each child and recursively resizes those nodes.
            """
            if self.junction is None:
                assert self.camera is not None
                self.total_height = des_height
                self.total_width = des_width
                teleop_obj.compressed_image_items[self.camera].desired_resolutions[
                    tree_index
                ] = (des_height, des_width, 3)

            else:
                if self.junction.direction == ImageConcatTree.Direction.VERTICAL:
                    resize_ratio = float(des_height) / float(self.total_height)
                    node1_des_height = round(
                        self.junction.node1.total_height * resize_ratio
                    )
                    node2_des_height = des_height - node1_des_height
                    self.total_height = des_height
                    self.total_width = des_width
                    self.junction.node1.resize_node(
                        node1_des_height, des_width, teleop_obj, tree_index
                    )
                    self.junction.node2.resize_node(
                        node2_des_height, des_width, teleop_obj, tree_index
                    )
                else:  # HORIZONTAL
                    resize_ratio = float(des_width) / float(self.total_width)
                    node1_des_width = round(
                        self.junction.node1.total_width * resize_ratio
                    )
                    node2_des_width = des_width - node1_des_width
                    self.total_height = des_height
                    self.total_width = des_width
                    self.junction.node1.resize_node(
                        des_height, node1_des_width, teleop_obj, tree_index
                    )
                    self.junction.node2.resize_node(
                        des_height, node2_des_width, teleop_obj, tree_index
                    )

        def resize_node_with_padding(
            self,
            des_height: int,
            des_width: int,
            teleop_obj: TeleopViewerDnD,
            pad_side: ImageConcatTree.PadSide,
            tree_index: int,
        ):
            """Not used yet, can be used to add some padding on the sides or top or bottom.
            For use if you want a black space with text or to resize to fit a specified resolution without changing the aspect ratio
            """
            if pad_side == ImageConcatTree.PadSide.NONE:
                self.resize_node(des_height, des_width, teleop_obj, tree_index)
            else:  # pad_side != ImageConcatTree.PadSide.NONE
                current_aspect_ratio = float(self.total_width) / float(
                    self.total_height
                )
                des_aspect_ratio = float(des_width) / float(des_height)

                if des_aspect_ratio == current_aspect_ratio:
                    self.resize_node(des_height, des_width, teleop_obj, tree_index)
                elif des_aspect_ratio > current_aspect_ratio:
                    # desired is wider, pad the sides; make new height desired height
                    self.resize_node(
                        des_height,
                        round(des_height * current_aspect_ratio),
                        teleop_obj,
                        tree_index,
                    )
                else:
                    # desired is taller, pad the top/bottom
                    self.resize_node(
                        round(des_width / current_aspect_ratio),
                        des_width,
                        teleop_obj,
                        tree_index,
                    )

        def print_sizes(self):
            """For debug only, recursively print sizes of node and all children nodes"""
            print(
                f"Junction dir: {self.junction.direction if self.junction is not None else 'None'}"
            )
            print(f"Name: {self.camera}")
            print(f"Size: ({self.total_height}, {self.total_width})\n")

            if self.junction is not None:
                self.junction.node1.print_sizes()
                self.junction.node2.print_sizes()

        def get_node_image(
            self, teleop_obj: TeleopViewerDnD, tree_index: int
        ) -> np.ndarray:
            """Get the image of the node. The tree index indicates which tree it is from, because the actual images are not stored in the tree"""
            if self.camera is not None:
                return teleop_obj.compressed_image_items[self.camera].cv2_images[
                    tree_index
                ]
            else:
                assert self.junction is not None
                # junction
                return TeleopViewerDnD.concat_image_no_resize(
                    self.junction.node1.get_node_image(teleop_obj, tree_index),
                    self.junction.node2.get_node_image(teleop_obj, tree_index),
                    self.junction.direction,
                )

    @dataclass
    class Junction:
        direction: ImageConcatTree.Direction
        node1: ImageConcatTree.ConcatNode
        node2: ImageConcatTree.ConcatNode

    class Direction(Enum):
        VERTICAL = 0
        HORIZONTAL = 1

    class PadSide(Enum):
        NONE = 0
        TOP = 1
        BOTTOM = 2
        BOTH = 3


class TeleopViewerDnD:
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("TeleopViewerDnD")
        self.config = config or CONFIG

        # Get settings from config
        self.use_vertical = self.config.get("use_vertical", False)
        self.fps = self.config.get("fps", 10)
        self.window_name = self.config.get("window_name", "Teleop Viewer")

        # Initialize image loader
        input_dir = self.config.get("input_directory", "./sample_images")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(script_dir, input_dir)
        self.image_loader = ImageLoader(input_dir)

        # If the vertical teleop is toggled, we need to make two trees, otherwise one
        self.num_layouts = 2 if self.use_vertical else 1

        # Each tree defines a set of concatenated images to publish
        self.image_concat_trees = [ImageConcatTree() for _ in range(self.num_layouts)]

        self.define_compressed_image_items()
        self.make_horizontal_image_tree()
        if self.use_vertical:
            self.logger.debug("Vertical teleop is toggled")
            self.make_vertical_image_tree()

        # Initialize sensor values from config (simulated values)
        sensors = self.config.get("sensors", {})
        self.laser_distance = sensors.get("laser_distance", 35.0)
        self.pressure_manifold = sensors.get("pressure_manifold", 0.5)
        self.pressure_base = sensors.get("pressure_base", 0.3)
        self.laser_sensor_active = sensors.get("laser_active", True)
        self.robot_status = sensors.get("robot_status", "Stopped")
        self.is_manual_review_mode = sensors.get("is_manual_review", True)

        self.frame_counter = 1
        self.running = True

    def define_compressed_image_items(self):
        """Define attributes for each image and how they will be manipulated"""

        def rotate_tuple(x: Tuple[int, int, int]) -> Tuple[int, int, int]:
            return x[1], x[0], x[2]

        self.compressed_image_items = {
            "ee_cam": CompressedImageItem(
                camera_name="ee_cam",
                image_path=self.image_loader.get_image_path("ee_cam"),
                active=self.image_loader.has_images("ee_cam"),
                data=ImageData(),
                resolution=ee_cam_resolution,
                rotate=None,
                centermark=True,
                desired_resolutions=[ee_cam_resolution] * self.num_layouts,
                cv2_images=[np.ndarray(ee_cam_resolution)] * self.num_layouts,
                feedback_text_functions=None,
            ),
            "ifm_camera1": CompressedImageItem(
                camera_name="ifm_camera1",
                image_path=self.image_loader.get_image_path("ifm_camera1"),
                active=self.image_loader.has_images("ifm_camera1"),
                data=ImageData(),
                resolution=ifm_resolution
                if camera_mount == "D"
                else recovery_cam_resolution,
                rotate=None,
                centermark=False,
                desired_resolutions=[
                    ifm_resolution if camera_mount == "D" else recovery_cam_resolution
                ]
                * self.num_layouts,
                cv2_images=[
                    np.ndarray(
                        ifm_resolution
                        if camera_mount == "D"
                        else recovery_cam_resolution
                    )
                ]
                * self.num_layouts,
                feedback_text_functions=None,
            ),
            "ifm_camera2": CompressedImageItem(
                camera_name="ifm_camera2",
                image_path=self.image_loader.get_image_path("ifm_camera2"),
                active=self.image_loader.has_images("ifm_camera2"),
                data=ImageData(),
                resolution=ifm_resolution,
                rotate=cv2.ROTATE_90_COUNTERCLOCKWISE if old_elbow_cam else None,
                centermark=False,
                desired_resolutions=(
                    [rotate_tuple(ifm_resolution)] * self.num_layouts
                    if old_elbow_cam
                    else [ifm_resolution] * self.num_layouts
                ),
                cv2_images=(
                    [np.ndarray(rotate_tuple(ifm_resolution))] * self.num_layouts
                    if old_elbow_cam
                    else [np.ndarray(ifm_resolution)] * self.num_layouts
                ),
                feedback_text_functions=[
                    self.make_laser_distance_text,
                ],
            ),
            "front_monitor_cam": CompressedImageItem(
                camera_name="front_monitor_cam",
                image_path=self.image_loader.get_image_path("front_monitor_cam"),
                active=self.image_loader.has_images("front_monitor_cam"),
                data=ImageData(),
                resolution=monitor_cam_resolution,
                rotate=cv2.ROTATE_90_COUNTERCLOCKWISE,
                centermark=False,
                desired_resolutions=(
                    [rotate_tuple(monitor_cam_resolution)] * self.num_layouts
                ),
                cv2_images=[np.ndarray(monitor_cam_resolution)] * self.num_layouts,
                feedback_text_functions=[
                    self.make_robot_status_text,
                    self.make_vacuum_pressure_text,
                ],
            ),
            "back_monitor_cam": CompressedImageItem(
                camera_name="back_monitor_cam",
                image_path=self.image_loader.get_image_path("back_monitor_cam"),
                active=self.image_loader.has_images("back_monitor_cam"),
                data=ImageData(),
                resolution=monitor_cam_resolution,
                rotate=cv2.ROTATE_90_CLOCKWISE if old_elbow_cam else None,
                centermark=False,
                desired_resolutions=(
                    [rotate_tuple(monitor_cam_resolution)] * self.num_layouts
                ),
                cv2_images=[np.ndarray(monitor_cam_resolution)] * self.num_layouts,
                feedback_text_functions=[
                    self.make_robot_status_text,
                    self.make_vacuum_pressure_text,
                    self.make_laser_distance_text,
                ],
            ),
            "A1_cam1": CompressedImageItem(
                camera_name="A1_cam1",
                image_path=self.image_loader.get_image_path("A1_cam1"),
                active=self.image_loader.has_images("A1_cam1"),
                data=ImageData(),
                resolution=ee_cam_resolution,
                rotate=cv2.ROTATE_90_CLOCKWISE,
                centermark=False,
                desired_resolutions=[rotate_tuple(ee_cam_resolution)]
                * self.num_layouts,
                cv2_images=[np.ndarray(rotate_tuple(ee_cam_resolution))]
                * self.num_layouts,
                feedback_text_functions=None,
            ),
            "A1_cam2": CompressedImageItem(
                camera_name="A1_cam2",
                image_path=self.image_loader.get_image_path("A1_cam2"),
                active=self.image_loader.has_images("A1_cam2"),
                data=ImageData(),
                resolution=ee_cam_resolution,
                rotate=cv2.ROTATE_90_COUNTERCLOCKWISE,
                centermark=False,
                desired_resolutions=[rotate_tuple(ee_cam_resolution)]
                * self.num_layouts,
                cv2_images=[np.ndarray(rotate_tuple(ee_cam_resolution))]
                * self.num_layouts,
                feedback_text_functions=None,
            ),
            "boxwall_monitor_cam": CompressedImageItem(
                camera_name="boxwall_monitor_cam",
                image_path=self.image_loader.get_image_path("boxwall_monitor_cam"),
                active=self.image_loader.has_images("boxwall_monitor_cam"),
                data=ImageData(),
                resolution=monitor_cam_resolution,
                rotate=None,
                centermark=False,
                desired_resolutions=[monitor_cam_resolution] * self.num_layouts,
                cv2_images=[np.ndarray(monitor_cam_resolution)] * self.num_layouts,
                feedback_text_functions=None,
            ),
        }

    def make_horizontal_image_tree(self):
        """Makes the landscape version of the image concat tree"""

        # Define each tree by makeing nodes for each sub image and the concatenation relationships between them
        tree_index = 0
        front_monitor_cam_node = ImageConcatTree.make_camera_node(
            self, "front_monitor_cam", tree_index
        )
        back_monitor_cam_node = ImageConcatTree.make_camera_node(
            self, "back_monitor_cam", tree_index
        )
        recovery_cam_node = ImageConcatTree.make_camera_node(
            self, "ifm_camera1", tree_index
        )
        elbow_cam_node = ImageConcatTree.make_camera_node(
            self, "ifm_camera2", tree_index
        )
        ee_cam_node = ImageConcatTree.make_camera_node(self, "ee_cam", tree_index)

        boxwall_cam_node = ImageConcatTree.make_camera_node(
            self, "boxwall_monitor_cam", tree_index
        )

        # Stack the boxwall cam on top of the recovery cam
        # This makes up the right junction
        boxwall_recovery_vertical_junction_node = ImageConcatTree.make_juction_node(
            boxwall_cam_node,
            recovery_cam_node,
            ImageConcatTree.Direction.VERTICAL,
            self,
            tree_index,
        )

        # Stack the EE cam on top of the monitor cams
        ee_monitor_vertical_junction_node = ImageConcatTree.make_juction_node(
            ee_cam_node,
            boxwall_recovery_vertical_junction_node,
            ImageConcatTree.Direction.VERTICAL,
            self,
            tree_index,
        )

        # Horizontally concatenate the EE cam/monitor cam junction with the elbow cam
        # This makes up the left junction
        left_junction_node = ImageConcatTree.make_juction_node(
            ee_monitor_vertical_junction_node,
            elbow_cam_node,
            ImageConcatTree.Direction.HORIZONTAL,
            self,
            tree_index,
        )

        # Horizontally concatenate front and back monitor cams
        monitor_cams_horizontal_junction_node = ImageConcatTree.make_juction_node(
            front_monitor_cam_node,
            back_monitor_cam_node,
            ImageConcatTree.Direction.HORIZONTAL,
            self,
            tree_index,
        )

        # Horizontally concatenate the left and right junctions
        all_junction_node = ImageConcatTree.make_juction_node(
            left_junction_node,
            monitor_cams_horizontal_junction_node,
            ImageConcatTree.Direction.HORIZONTAL,
            self,
            tree_index,
        )
        # TODO resize the whole node
        # all_junction_node.resize_node(480, 1080, self, tree_index)
        # all_junction_node.resize_node_with_padding(480, 1080, self, ImageConcatTree.PadSide.BOTH, tree_index)

        self.image_concat_trees[tree_index].root = all_junction_node
        self.image_concat_trees[tree_index].tree_index = tree_index

    def make_vertical_image_tree(self):
        """Makes the landscape version of the image concat tree"""

        tree_index = 1
        front_monitor_cam_node = ImageConcatTree.make_camera_node(
            self, "front_monitor_cam", tree_index
        )
        recovery_cam_node = ImageConcatTree.make_camera_node(
            self, "ifm_camera1", tree_index
        )
        elbow_cam_node = ImageConcatTree.make_camera_node(
            self, "ifm_camera2", tree_index
        )
        ee_cam_node = ImageConcatTree.make_camera_node(self, "ee_cam", tree_index)

        # Horizontally concatenate elbow cam and front monitor cam
        top_horizontal_junction_node = ImageConcatTree.make_juction_node(
            elbow_cam_node,
            front_monitor_cam_node,
            ImageConcatTree.Direction.HORIZONTAL,
            self,
            tree_index,
        )

        # Horizontally concatenate recovery cam and EE cam
        bottom_horizontal_junction_node = ImageConcatTree.make_juction_node(
            recovery_cam_node,
            ee_cam_node,
            ImageConcatTree.Direction.HORIZONTAL,
            self,
            tree_index,
        )

        # Horizontally concatenate the elbow cam and the right junction
        all_junction_node = ImageConcatTree.make_juction_node(
            top_horizontal_junction_node,
            bottom_horizontal_junction_node,
            ImageConcatTree.Direction.VERTICAL,
            self,
            tree_index,
        )

        self.image_concat_trees[tree_index].root = all_junction_node
        self.image_concat_trees[tree_index].tree_index = tree_index

    @staticmethod
    def draw_contours(img: np.ndarray):
        # Draw the boundary contour on the image
        height, width, _ = img.shape
        boundary_contour = [
            np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        ]
        cv2.drawContours(
            img, boundary_contour, -1, (255, 255, 255), 1
        )  # White color, thickness=1

    def draw_overlays(
        self,
        cv2_img: np.ndarray,
        compressed_image_item: CompressedImageItem,
        tree_index: int,
    ):
        """Draws overlays on top of the image"""
        if compressed_image_item.centermark:
            # Draw an aim mark at the center of the image
            h, w, _ = cv2_img.shape
            center = (w // 2, h // 2)
            # Set size to 2.5% of the image width
            size = int(w * 0.025)
            cv2_img = self.draw_aim_mark(
                cv2_img,
                center,  # need to adjust position
                size=size,
                color=(255, 0, 255),
                thickness=4,
            )

    def convert_msg_to_cv2(self, compressed_image_item: CompressedImageItem):
        if (
            compressed_image_item.active
            and "jpeg" in compressed_image_item.data.format.lower()
        ):
            """Converts the compressed image message to cv2 array. Resizes the image per the desired size for each tree defined.
            Uses gpu accelaration if available"""
            np_arr = np.frombuffer(compressed_image_item.data.data, np.uint8)
            cv2_img_orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if not USE_GPU:
                cpu_cv2_img = cv2.cvtColor(cv2_img_orig, cv2.COLOR_BGR2RGB)
            else:
                # Upload into a GPU mat
                gpu_np_arr = cv2.cuda_GpuMat()
                gpu_np_arr.upload(cv2_img_orig)
                # Convert color in GPU
                gpu_cv2_img = cv2.cuda.cvtColor(gpu_np_arr, cv2.COLOR_BGR2RGB)

            for tree_index in range(self.num_layouts):
                if compressed_image_item.rotate is not None:
                    width, height = (
                        compressed_image_item.desired_resolutions[tree_index][0],
                        compressed_image_item.desired_resolutions[tree_index][1],
                    )
                else:
                    width, height = (
                        compressed_image_item.desired_resolutions[tree_index][1],
                        compressed_image_item.desired_resolutions[tree_index][0],
                    )

                if not USE_GPU:
                    cv2_img = cv2.resize(cpu_cv2_img, (width, height))

                else:
                    gpu_cv2_img = cv2.cuda.cvtColor(gpu_np_arr, cv2.COLOR_BGR2RGB)

                    gpu_resized_img = cv2.cuda.resize(gpu_cv2_img, (width, height))
                    # Download back to CPU memory
                    cv2_img = gpu_resized_img.download()

                if compressed_image_item.rotate is not None:
                    cv2_img = cv2.rotate(cv2_img, compressed_image_item.rotate)

                self.draw_overlays(cv2_img, compressed_image_item, tree_index)
                self.draw_contours(cv2_img)
                compressed_image_item.cv2_images[tree_index] = cv2_img

        else:
            for tree_index in range(self.num_layouts):
                cv2_img = np.zeros(
                    compressed_image_item.desired_resolutions[tree_index],
                    dtype=np.uint8,
                )
                self.draw_contours(cv2_img)

                compressed_image_item.cv2_images[tree_index] = cv2_img

    def make_laser_distance_text(
        self, tree_index: int, compressed_image_item: CompressedImageItem
    ) -> FeedbackTextData:
        """Define text data parameters (for use with cv2.putText) for the laser distance feedback"""
        UPPER_LIMIT_CM = 44
        LOWER_LIMIT_CM = 31
        if self.laser_sensor_active:
            if self.laser_distance * 0.1 > UPPER_LIMIT_CM:
                display_text = "Dist: " + "N/A"
                font_color = (255, 0, 0)
            elif (
                self.laser_distance * 0.1 > LOWER_LIMIT_CM
                and self.laser_distance * 0.1 < UPPER_LIMIT_CM
            ):
                display_text = "Dist: " + f"{self.laser_distance * 0.1:.2f}" + "cm"
                font_color = (0, 0, 255)
            else:
                display_text = "Dist: " + f"{self.laser_distance * 0.1:.2f}" + "cm"
                font_color = (0, 255, 0)
        else:
            display_text = "Laser data is missing"
            font_color = (255, 0, 0)

        font_scale = 1
        thickness = 2

        # Determine position based on tree index and camera
        if (
            tree_index == 0
            and compressed_image_item == self.compressed_image_items["back_monitor_cam"]
        ):
            x, y = 0, 80
        elif (
            tree_index == 1
            and compressed_image_item == self.compressed_image_items["ifm_camera2"]
        ):
            x, y = 0, compressed_image_item.desired_resolutions[tree_index][0]
        else:
            return None

        return FeedbackTextData(
            display_text=display_text,
            position=(x, y),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=font_scale,
            font_color=font_color,
            thickness=thickness,
            line_type=cv2.LINE_AA,
            desired_width=compressed_image_item.desired_resolutions[tree_index][1],
            desired_height=40,
        )

    def make_robot_status_text(
        self, tree_index: int, compressed_image_item: CompressedImageItem
    ) -> FeedbackTextData:
        """Define text data parameters (for use with cv2.putText) for the robot status"""
        display_text = "Status: " + self.robot_status
        font_color = (255, 0, 0)
        if self.robot_status == "SCANNING":
            display_text += " "
            if self.is_manual_review_mode:
                display_text += "(Manual)"
                font_color = (255, 153, 51)
            else:
                display_text += "(Auto)"
                font_color = (0, 128, 255)
        elif (
            self.robot_status == "NAVIGATING"
            or self.robot_status == "UNLOADING"
            or self.robot_status == "FINISHED"
        ):
            font_color = (0, 128, 255)

        if (
            tree_index == 0
            and compressed_image_item == self.compressed_image_items["back_monitor_cam"]
        ):
            x, y = 0, 40
        elif (
            tree_index == 1
            and compressed_image_item
            == self.compressed_image_items["front_monitor_cam"]
        ):
            x, y = 0, compressed_image_item.desired_resolutions[tree_index][0] - 40
        else:
            return None

        return FeedbackTextData(
            display_text=display_text,
            position=(x, y),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=2,
            font_color=font_color,
            thickness=2,
            line_type=cv2.LINE_AA,
            desired_width=compressed_image_item.desired_resolutions[tree_index][1],
            desired_height=40,
        )

    def make_vacuum_pressure_text(
        self, tree_index: int, compressed_image_item: CompressedImageItem
    ) -> FeedbackTextData:
        """Define text data parameters (for use with cv2.putText) for the vacuum pressure"""
        font_color = (0, 128, 255)
        display_text = f"Z1: {self.pressure_manifold:.4f} bar | Z2: {self.pressure_base:.4f} bar"

        if (
            tree_index == 0
            and compressed_image_item == self.compressed_image_items["back_monitor_cam"]
        ):
            x, y = 0, 120
        elif (
            tree_index == 1
            and compressed_image_item
            == self.compressed_image_items["front_monitor_cam"]
        ):
            x, y = 0, compressed_image_item.desired_resolutions[tree_index][0]
        else:
            return None

        return FeedbackTextData(
            display_text=display_text,
            position=(x, y),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1.0,
            font_color=font_color,
            thickness=2,
            line_type=cv2.LINE_AA,
            desired_width=compressed_image_item.desired_resolutions[tree_index][1],
            desired_height=40,
        )

    def draw_text_with_background(
        self,
        tree_index: int,
        compressed_image_item: CompressedImageItem,
        text_data: FeedbackTextData,
    ) -> FeedbackTextData:
        """Draws text on top of a black background rectangle and automatically adjusts the font size to fit within the desired dimensions"""

        # Origin (bottom left) of the text
        text_org = (text_data.position[0], text_data.position[1])

        # Define rectangle coordinates for the background (exact desired dimensions)
        padding = 5  # Internal padding to prevent text from touching edges
        rect_bottom_left = (text_org[0], text_org[1])
        rect_top_right = (
            text_org[0] + text_data.desired_width,
            text_org[1] - text_data.desired_height,
        )

        # Draw black filled rectangle as background
        cv2.rectangle(
            compressed_image_item.cv2_images[tree_index],
            rect_bottom_left,
            rect_top_right,
            (0, 0, 0),
            -1,
        )

        # Adjust font scale to maximize text size within available space (desired dimensions minus internal padding)
        font_scale = text_data.font_scale
        min_font_scale = 0.1  # Prevent text from becoming too small
        target_width = (
            text_data.desired_width - 2 * padding
        )  # Account for internal padding
        target_height = (
            text_data.desired_height - 2 * padding
        )  # Account for internal padding

        # Iteratively reduce font scale to fit within desired width and height
        while font_scale >= min_font_scale:
            (text_width, text_height), baseline = cv2.getTextSize(
                text_data.display_text, text_data.font, font_scale, text_data.thickness
            )
            # Check if text fits within desired width and height (accounting for baseline)
            if text_width <= target_width and (text_height + baseline) <= target_height:
                break
            font_scale -= 0.05  # Reduce font scale gradually for precision

        # If no suitable scale is found, use minimum scale
        font_scale = max(min_font_scale, font_scale)

        # Update text_data with the adjusted font scale
        text_data.font_scale = font_scale

        # Adjust thickness based on font scale for readability
        text_data.thickness = max(2, min(4, int(font_scale * text_data.thickness)))

        # Center text vertically within the bounding box (accounting for internal padding)
        (text_width, text_height), baseline = cv2.getTextSize(
            text_data.display_text,
            text_data.font,
            text_data.font_scale,
            text_data.thickness,
        )

        # Calculate the available text area within the rectangle (excluding internal padding)
        text_area_top_y = text_org[1] - text_data.desired_height + padding
        text_area_bottom_y = text_org[1] - padding
        text_area_center_y = (text_area_top_y + text_area_bottom_y) / 2

        # Position baseline so that text is visually centered in the available text area
        # Text visual center is at baseline_y - (text_height - baseline)/2
        text_y = int(text_area_center_y + (text_height - baseline) / 2)
        text_x = text_org[0] + padding  # Apply horizontal padding
        text_data.position = (text_x, text_y)

        # Draw the text on top of the background
        cv2.putText(
            compressed_image_item.cv2_images[tree_index],
            text_data.display_text,
            text_data.position,
            text_data.font,
            text_data.font_scale,
            text_data.font_color,
            text_data.thickness,
            text_data.line_type,
        )

        return text_data

    def put_text_on_image(self, compressed_image_item: CompressedImageItem):
        """Adds text to the images, once for each tree (output image)"""
        if compressed_image_item.feedback_text_functions is not None:
            for feedback_text_function in compressed_image_item.feedback_text_functions:
                for tree_index in range(self.num_layouts):
                    feedback_text_data = feedback_text_function(
                        tree_index, compressed_image_item
                    )
                    if feedback_text_data is not None:
                        self.draw_text_with_background(
                            tree_index, compressed_image_item, feedback_text_data
                        )

    def load_images_from_files(self):
        """Load images from files for all cameras."""
        for name, compressed_image_item in self.compressed_image_items.items():
            image_data = self.image_loader.load_image(name)
            if image_data is not None:
                compressed_image_item.data = image_data
                compressed_image_item.active = True
            else:
                compressed_image_item.active = False

    def advance_all_frames(self):
        """Advance to the next frame for all cameras (for video-like playback)."""
        for name in self.compressed_image_items.keys():
            self.image_loader.advance_frame(name)

    def run(self):
        """Main loop with OpenCV display."""
        frame_time = 1.0 / self.fps
        self.logger.info(f"Starting viewer at {self.fps} FPS. Press 'q' or ESC to quit.")

        while self.running:
            start_time = time.time()

            try:
                # Load images from files
                self.load_images_from_files()

                # Process each image
                for compressed_image_item in self.compressed_image_items.values():
                    self.convert_msg_to_cv2(compressed_image_item)
                    self.put_text_on_image(compressed_image_item)

                self.frame_counter += 1

                # Display concatenated images
                for tree_index in range(self.num_layouts):
                    concatenated_tree_image = self.image_concat_trees[
                        tree_index
                    ].get_tree_image(self)

                    # Convert RGB to BGR for OpenCV display
                    display_image = cv2.cvtColor(concatenated_tree_image, cv2.COLOR_RGB2BGR)

                    # Determine window name
                    if tree_index == 0:
                        window_name = self.window_name
                    else:
                        window_name = f"{self.window_name} (Vertical)"

                    # Display the image
                    cv2.imshow(window_name, display_image)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.running = False
                elif key == ord('n'):  # 'n' for next frame (manual advance)
                    self.advance_all_frames()

                # Frame rate control
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Advance frames for video-like playback (optional, comment out for single image)
                # self.advance_all_frames()

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.exception(f"Exception in main loop: {e}")
                break

        cv2.destroyAllWindows()
        self.logger.info("Viewer stopped")

    @staticmethod
    def concat_image_no_resize(
        cv_img1: np.ndarray, cv_img2: np.ndarray, axis: ImageConcatTree.Direction
    ) -> np.ndarray:
        if axis == ImageConcatTree.Direction.VERTICAL:
            concat_img = cv2.vconcat([cv_img1, cv_img2])
        elif axis == ImageConcatTree.Direction.HORIZONTAL:
            concat_img = cv2.hconcat([cv_img1, cv_img2])
        return concat_img

    def resize_and_pad(
        self, img: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """
        Resize an image to fit within specific dimensions while maintaining aspect ratio,
        and add black margins to fill the remaining space.

        :param img: Input image
        :param target_width: Desired width of the output image
        :param target_height: Desired height of the output image
        """
        # Get original dimensions
        original_height, original_width = img.shape[:2]

        # Calculate the scaling factor to fit the image
        scale_width = target_w / original_width
        scale_height = target_h / original_height
        scale = min(scale_width, scale_height)

        # Compute new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image while maintaining aspect ratio
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Create a new black canvas with target dimensions
        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate the top-left corner for centering the image
        x_offset = (target_w - new_width) // 2
        y_offset = (target_h - new_height) // 2

        # Place the resized image onto the canvas
        padded_img[
            y_offset : y_offset + new_height, x_offset : x_offset + new_width
        ] = resized_img

        return padded_img

    @staticmethod
    def draw_aim_mark(
        img: np.ndarray,
        center: Tuple[int, int],
        size: int = 10,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Draw an aim mark (crosshair) on the image.

        Parameters:
        - image: Input image to draw on.
        - center: Center coordinates of the aim mark.
        - size: Size of the aim mark (half length of each line).
        - color: Color of the aim mark (BGR format).
        - thickness: Thickness of the lines.
        """
        x, y = center
        cv2.line(img, (x - size, y), (x + size, y), color, thickness)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness)

        return img

def main():
    """Main entry point for the teleop viewer."""
    import argparse

    parser = argparse.ArgumentParser(description="Teleop Viewer - ROS-independent version")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Input directory containing camera images (overrides config)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate in Hz (overrides config)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.input:
        config["input_directory"] = args.input
    if args.fps:
        config["fps"] = args.fps

    # Create and run viewer
    viewer = TeleopViewerDnD(config)
    viewer.run()


if __name__ == "__main__":
    main()
