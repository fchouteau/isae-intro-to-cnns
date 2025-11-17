"""
Drawing bounding boxes on images helpers
"""
import cv2
import matplotlib.colors
import numpy as np

from khumeia.data.item import SatelliteImage
from khumeia.roi.tile import BoundingBox, LabelledTile, PredictionTile, Tile
from khumeia.utils import roi_list_utils

__all__ = ["draw_bbox_on_image", "draw_bboxes_on_image", "draw_item", "draw_item_with_tiles", "draw_item_with_results"]


def _convert_color(color_str: str):
    """
    https://matplotlib.org/examples/color/named_colors.html
    Args:
        color_str:

    Returns:

    """
    if isinstance(color_str, str):
        rgb_color = matplotlib.colors.to_rgb(color_str)
        rgb_color = tuple(map(lambda x: int(255 * x), rgb_color))
        return rgb_color
    elif isinstance(color_str, tuple) and isinstance(color_str[0], float):
        rgb_color = tuple(map(lambda x: int(255 * x), color_str))
        return rgb_color
    else:
        return color_str


def draw_bbox_on_image(image: np.ndarray, bbox: BoundingBox, color: str = "lime", thickness: int = 2) -> np.ndarray:
    """
    Draw one BoundingBox to an image using cv2.rectangle
    Args:
        image: A (h,w,3) 8-bit array representing the image
        bbox:
        color: A matplotlib color compatible color
        thickness: A thickness value

    Returns:
        The same `image` but with the bounding box drawn on it

    """
    color = _convert_color(color)
    cv2.rectangle(image, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color=color, thickness=thickness)
    return image


def draw_bboxes_on_image(image: np.ndarray, bboxes: [BoundingBox], color="lime", thickness=2) -> np.ndarray:
    """
    Draw BoundingBoxes to an image using cv2.rectangle
    """
    color = _convert_color(color)
    for bbox in bboxes:
        image = draw_bbox_on_image(image, bbox, color=color, thickness=thickness)
    return image


def draw_item(item: SatelliteImage) -> np.ndarray:
    """
    Draw an item labels on its image
    """
    image = item.image
    labels = item.labels
    image = draw_bboxes_on_image(image, labels, color="lime")
    return image


def draw_tile(item: SatelliteImage, tile: Tile):
    image = tile.get_data(item.image)
    labels = tile.get_labels(item.labels)
    image = draw_bboxes_on_image(image, labels, color="lime")

    return image


def draw_item_with_tiles(item: SatelliteImage, tiles: [LabelledTile] = None) -> np.ndarray:
    """
    Draw an item labels on its images as well as the tiles in tiles
    """
    image = draw_item(item)
    if tiles is not None:
        tiles = roi_list_utils.filter_tiles_by_item(tiles, item.key)
        tiles_bg = roi_list_utils.filter_tiles_by_label(tiles, "background")
        image = draw_bboxes_on_image(image, tiles_bg, color="red")
        tiles_ac = roi_list_utils.filter_tiles_by_label(tiles, "aircraft")
        image = draw_bboxes_on_image(image, tiles_ac, color="blue")

    return image


def draw_item_with_results(item: SatelliteImage, results: [PredictionTile] = None) -> np.ndarray:
    """
    Draw an item labels on its images as well as the PredictionTiles in Tiles
    """
    image = draw_item(item)
    if results is not None:
        tiles = list(filter(lambda tile: tile.item_id == item.key, results))
        true_positives = list(filter(lambda tile: tile.is_true_positive, tiles))
        false_positives = list(filter(lambda tile: tile.is_false_positive, tiles))
        false_negatives = list(filter(lambda tile: tile.is_false_negative, tiles))
        image = draw_bboxes_on_image(image, true_positives, color="lime")
        image = draw_bboxes_on_image(image, false_positives, color="blue")
        image = draw_bboxes_on_image(image, false_negatives, color="red")

    return image
