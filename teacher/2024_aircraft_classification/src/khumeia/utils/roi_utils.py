import rtree.index

from khumeia.roi.bounding_box import BoundingBox
from khumeia.roi.groundtruth import Groundtruth


def make_index(bboxes: [BoundingBox]) -> rtree.index.Index:
    """

    Returns:
        object:
    """
    indx = rtree.index.Index()
    for i, bbox in enumerate(bboxes):
        indx.insert(i, bbox.bounds)

    return indx


def get_label_from_bboxes_center(tile, bboxes, strict=True, margin_from_bounds=0):
    """
    Center of target inside bbox mode
    Args:
        tile (BoundingBox):
        bboxes (list[Groundtruth]):
        strict (bool):
        margin_from_bounds(float):

    Returns:

    """
    for bbox in bboxes:
        # TODO: Assignment is trivial in single-class problem but does not work in multi-class
        if tile.contains_point(bbox.center, strict=strict, margin=margin_from_bounds):
            return bbox.label
    return "background"


def get_label_from_bboxes_ioa(tile, bboxes, ioa_threshold=0.0):
    """
    Intersection over area mode
    Args:
        tile (BoundingBox):
        bboxes (list[Groundtruth]):
        ioa_threshold:

    Returns:

    """
    area = tile.area
    for bbox in bboxes:
        # TODO: Assignment is trivial in single-class problem but does not work in multi-class
        area_ = tile.intersection(bbox).area
        if area_ / min(area, bbox.area) > ioa_threshold:
            return bbox.label
    return "background"
