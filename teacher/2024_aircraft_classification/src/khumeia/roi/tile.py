import itertools
from typing import Callable, Optional

import numpy as np

from khumeia.roi.bounding_box import BoundingBox
from khumeia.roi.groundtruth import Groundtruth
from khumeia.utils import roi_utils

__all__ = ["Tile", "LabelledTile", "PredictionTile"]


class Tile(BoundingBox):
    """
    Tiles are bbox that represent a location on a large image.
    Functionally different than groundtruth
    """

    def __init__(
        self,
        item_id: str,
        x_min: int,
        y_min: int,
        width: int,
        height: int,
        padding: int = 0,
        data_transform_fn: Optional[Callable] = None,
    ):
        """"""
        super(Tile, self).__init__(x_min=x_min, y_min=y_min, width=width, height=height)
        self.item_id = item_id
        self.padding = int(padding)
        self.data_transform_fn = data_transform_fn

    @classmethod
    def get_tiles_for_item(
        cls,
        item_id: str,
        im_shape: [int],
        tile_shape: [int],
        padding=0,
        stride=1.0,
        offset=(0, 0),
        data_transform_fn=None,
    ) -> ["Tile"]:
        """

        Args:
            item_id:
            im_shape:
            tile_shape:
            padding:
            stride:
            offset:
            data_transform_fn: If you want to data augment. is applied like this: data = data_transform_fn(data)
                It can be an imgaug sequence for example...
                It could be used to resample all windows to the same dimension !

        Returns:

        """
        im_h, im_w = im_shape[:2]

        off_h, off_w = offset[:2]

        tile_h, tile_w = tile_shape[:2]
        stride_h, stride_w = int(tile_h * stride), int(tile_w * stride)
        max_i = int(np.floor(float(im_h - tile_h) / stride_h)) + 1
        max_j = int(np.floor(float(im_w - tile_w) / stride_w)) + 1
        tiles = []

        for i, j in itertools.product(range(max_i), range(max_j)):
            tile = cls(
                item_id=item_id,
                x_min=off_w + j * stride_w,
                y_min=off_h + i * stride_h,
                width=tile_w,
                height=tile_h,
                padding=padding,
                data_transform_fn=data_transform_fn,
            )
            tiles.append(tile)
        return tiles

    @property
    def padded_bounds(self) -> [int]:
        """

        Returns:

        """
        return (
            self.x_min - self.padding,
            self.y_min - self.padding,
            self.x_min + self.height + self.padding,
            self.y_min + self.width + self.padding,
        )

    def get_item(self, dataset):
        for item in dataset.items:
            if self.item_id == item.image_id:
                return item

        return None

    def get_data(self, image: np.ndarray) -> np.ndarray:
        """
        Get the data with padding management from a dataset
        Note: since padded bounds can go to outside the image height, width, there are calculations
        to correctly pad the array with zeros

        Args:
            image:

        Returns:
            np.ndarray RGB 8 bits from Dataset

        """
        if image is None:
            return None

        h, w, d = image.shape[:3]
        x_tl, y_tl, x_br, y_br = self.padded_bounds

        data = np.zeros((self.height + 2 * self.padding, self.width + 2 * self.padding, d), dtype=image.dtype)

        # Compute what we want to extract from the dataset (padding can be negative)
        x_tl_w, y_tl_w, x_br_w, y_br_w = max(0, x_tl), max(0, y_tl), min(w, x_br), min(h, y_br)

        # Compute offset on numpy array
        x_tl_o, y_tl_o, x_br_o, y_br_o = x_tl_w - x_tl, y_tl_w - y_tl, x_br_w - x_br, y_br_w - y_br

        data[y_tl_o : (y_br_o or None), x_tl_o : (x_br_o or None), :] = image[y_tl_w:y_br_w, x_tl_w:x_br_w, :]

        if self.data_transform_fn is None:
            return data
        else:
            return self.data_transform_fn(data)

    def get_labels(self, labels: [Groundtruth]) -> [Groundtruth]:
        labels_index = roi_utils.make_index(labels)
        intersecting_labels = list(labels_index.intersection(self.bounds))
        intersecting_labels = [labels[i] for i in intersecting_labels]
        intersecting_labels = [gt.intersection(self) for gt in intersecting_labels]
        intersecting_labels = self.bboxes_to_relative_coords(intersecting_labels)

        return intersecting_labels

    def bboxes_to_absolute_coords(self, bboxes: [BoundingBox]) -> [BoundingBox]:
        """

        Args:
            bboxes:

        Returns:

        """
        return [bbox.translate(self.padded_bounds[0], self.padded_bounds[1]) for bbox in bboxes]

    def bboxes_to_relative_coords(self, bboxes: [BoundingBox]) -> [BoundingBox]:
        """

        Args:
            bboxes:

        Returns:

        """
        return [bbox.translate(-self.padded_bounds[0], -self.padded_bounds[1]) for bbox in bboxes]

    def filter_inside(self, bboxes: [BoundingBox], to_relative_coordinates=False) -> [BoundingBox]:
        """

        Args:
            bboxes:
            to_relative_coordinates:

        Returns:

        """
        bboxes_inside = [bbox for bbox in bboxes if self.contains(bbox)]
        if to_relative_coordinates:
            bboxes_inside = self.bboxes_to_relative_coords(bboxes_inside)

        return bboxes_inside


class LabelledTile(Tile):
    """
    A tile with label assigned
    """

    def __init__(self, item_id, x_min, y_min, width, height, padding=0, label=None, data_transform_fn=None):
        """

        Args:
            item_id:
            x_min:
            y_min:
            width:
            height:
            padding:
            label:
            data_transform_fn:
        """
        super(LabelledTile, self).__init__(
            item_id, x_min, y_min, width, height, padding=padding, data_transform_fn=data_transform_fn
        )
        self.label = label

    @classmethod
    def from_tile_and_label(cls, tile, label):
        return cls(
            item_id=tile.item_id,
            x_min=tile.x_min,
            y_min=tile.y_min,
            width=tile.width,
            height=tile.height,
            padding=tile.padding,
            label=label,
            data_transform_fn=tile.data_transform_fn,
        )

    @classmethod
    def from_tile_and_groundtruths(
        cls,
        tile: Tile,
        groundtruths: [Groundtruth],
        label_assignment_mode="center",
        ioa_threshold=0.5,
        margin_from_bounds=0,
    ):
        if label_assignment_mode == "center":
            label = roi_utils.get_label_from_bboxes_center(
                tile, groundtruths, strict=True, margin_from_bounds=margin_from_bounds
            )
        else:
            label = roi_utils.get_label_from_bboxes_ioa(tile, groundtruths, ioa_threshold=ioa_threshold)

        return cls.from_tile_and_label(tile, label)


class PredictionTile(LabelledTile):
    """
    A labelled tile that contains a prediction...
    """

    def __init__(
        self, item_id, x_min, y_min, width, height, padding=0, predicted_label=None, label=None, data_transform_fn=None
    ):
        """

        Args:
            item_id:
            x_min:
            y_min:
            width:
            height:
            padding:
            label:
            data_transform_fn:
        """
        super(PredictionTile, self).__init__(
            item_id, x_min, y_min, width, height, padding=padding, label=label, data_transform_fn=data_transform_fn
        )
        self.predicted_label = predicted_label

    @classmethod
    def from_labelled_tile_and_prediction(cls, labelled_tile, prediction):
        """

        Args:
            labelled_tile (LabelledTile):
            prediction (str):

        Returns:

        """
        return cls(
            labelled_tile.item_id,
            labelled_tile.x_min,
            labelled_tile.y_min,
            labelled_tile.width,
            labelled_tile.height,
            padding=labelled_tile.padding,
            label=labelled_tile.label,
            predicted_label=prediction,
            data_transform_fn=labelled_tile.data_transform_fn,
        )

    @property
    def is_correct(self):
        return self.label == self.predicted_label

    @property
    def is_background(self):
        return self.label == "background"

    @property
    def is_true_positive(self):
        """
        TP = correct prediction and gt label not background
        Returns:

        """
        return self.is_correct and not self.is_background

    @property
    def is_false_positive(self):
        """
        FP: incorrect prediction and gt label is background
        Returns:

        """
        return not self.is_correct and self.is_background

    @property
    def is_false_negative(self):
        """
        FN: incorrect prediction and gt label is not background
        Returns:

        """
        return not self.is_correct and not self.is_background
