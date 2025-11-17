import json
from typing import Callable, Optional

import numpy as np

from khumeia.data.item import SatelliteImage
from khumeia.roi.groundtruth import Groundtruth
from khumeia.roi.tile import LabelledTile, Tile
from khumeia.utils.roi_utils import make_index

__all__ = ["TilesGenerator", "RandomTiles", "CenteredTiles", "SlidingWindow"]


class TilesGenerator:
    """
    Generate a list of labelled patches from a satellite image
    """

    def __init__(
        self,
        tile_size: int = 64,
        padding: int = 0,
        data_transform_fn: Optional[Callable] = None,
    ):
        self.tile_size = tile_size
        self.padding = padding
        self.data_transform_fn = data_transform_fn

    def get_tiles_for_item(self, item: SatelliteImage) -> [LabelledTile]:
        raise NotImplementedError

    def __call__(self, item: SatelliteImage) -> [LabelledTile]:
        return self.get_tiles_for_item(item)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

    def __str__(self):
        d = dict()
        d["class"] = self.__class__.__name__
        d.update(self.__dict__)
        return json.dumps(d, indent=4)


class RandomTiles(TilesGenerator):
    def __init__(
        self,
        num_tiles=0,
        tile_size: int = 64,
        padding: int = 0,
        label_assignment_mode: str = "center",
        intersection_over_area_threshold: float = 0.5,
        margin_from_bounds: int = 0,
        discard_background: bool = False,
        data_transform_fn: Optional[Callable] = None,
    ):
        super(RandomTiles, self).__init__(tile_size=tile_size, padding=padding, data_transform_fn=data_transform_fn)
        self.num_tiles = num_tiles
        self.label_assignment_mode = label_assignment_mode
        self.ioa_threshold = intersection_over_area_threshold
        self.margin_from_bounds = margin_from_bounds
        self.discard_background = discard_background

    def get_tiles_for_item(self, item: SatelliteImage) -> [LabelledTile]:
        """
            Apply the sliding window on a full satellite images to generate a list of tiles
            Compares the tiles to the item's groundtruths
            Tiles that matches the declared conditions are assigned with the groundtruth's label

        Args:
            item: input item

        Returns:
            A list of tile with their labels

        """
        h, w = item.shape[:2]

        labels = item.labels
        labels_index = make_index(labels)

        def _get_labelled_tile(t: Tile):
            intersecting_labels = list(labels_index.intersection(t.bounds))
            intersecting_labels = [labels[i] for i in intersecting_labels]
            return LabelledTile.from_tile_and_groundtruths(
                t,
                intersecting_labels,
                label_assignment_mode=self.label_assignment_mode,
                ioa_threshold=self.ioa_threshold,
                margin_from_bounds=self.margin_from_bounds,
            )

        x0s = np.random.randint(0, w - self.tile_size, self.num_tiles)
        y0s = np.random.randint(0, h - self.tile_size, self.num_tiles)

        tiles = []

        for x0, y0 in zip(x0s, y0s):
            tile = Tile(
                item.key,
                x_min=x0,
                y_min=y0,
                width=self.tile_size,
                height=self.tile_size,
                padding=self.padding,
                data_transform_fn=self.data_transform_fn,
            )

            tile = _get_labelled_tile(tile)

            tiles.append(tile)

        if self.discard_background:
            tiles = filter(lambda tile: tile.label != "background", tiles)

        return list(tiles)


class SlidingWindow(TilesGenerator):
    """

    Sliding windows play an integral role in object classification, as they allow us to localize exactly “where” in an
    image an object resides.

    Sliding window approaches are simple in concept, a bounding box of the desired size(s) slides across the test image
    and at each location applies an image classifier to the current window

    ![example1](https://cdn-images-1.medium.com/max/800/1*FHEOyHm1BTWyygQcgvNSXQ.png)

    Sample cutouts of a sliding window iterating from top to bottom (Imagery Courtesy of DigitalGlobe)

    ![example2](https://cdn-images-1.medium.com/max/800/1*BkQLxT_FVz6XqHul5qezEw.gif)

    Sliding window shown iterating across an image (left).
    An image classifier is applied to these cutouts and anything resembling a boat is saved as a positive (right)
    (Imagery Courtesy of DigitalGlobe)

    """

    def __init__(
        self,
        tile_size: int = 64,
        padding: int = 0,
        stride: int = 64,
        label_assignment_mode: str = "center",
        intersection_over_area_threshold: float = 0.5,
        margin_from_bounds: int = 0,
        discard_background: bool = False,
        data_transform_fn: Optional[Callable] = None,
    ):
        """

        Args:
            tile_size(int): tile size (h,w) in pixels
            padding(int):  padding in pixels. best keep it to 0
            stride(int): Stride ("pas") in pixels
            label_assignment_mode: "center" or "ioa",
                if center: If a tile contains a groundtruth's center it gets its label
                if ioa: Calculates the intersection over min(area_tile,area_groundtruth), if the ioa > threshold, then
                assigns
            margin_from_bounds: internal margin to use if "center" is selected
            intersection_over_area_threshold(float): threshold
            data_transform_fn: Useful to generate augmented samples or to apply a specific preprocessing
        """
        super(SlidingWindow, self).__init__(tile_size=tile_size, padding=padding, data_transform_fn=data_transform_fn)
        self.stride = stride
        self.label_assignment_mode = label_assignment_mode
        self.ioa_threshold = intersection_over_area_threshold
        self.margin_from_bounds = margin_from_bounds
        self.discard_background = discard_background

    def get_tiles_for_item(self, item: SatelliteImage) -> [LabelledTile]:
        """
            Apply the sliding window on a full satellite images to generate a list of tiles
            Compares the tiles to the item's groundtruths
            Tiles that matches the declared conditions are assigned with the groundtruth's label

        Args:
            item: input item

        Returns:
            A list of tile with their labels

        """

        labels = item.labels

        tiles = Tile.get_tiles_for_item(
            item.key,
            item.shape,
            tile_shape=(self.tile_size, self.tile_size),
            padding=self.padding,
            stride=float(self.stride) / self.tile_size,
            data_transform_fn=self.data_transform_fn,
        )

        labels_index = make_index(labels)

        def _get_labelled_tile(t: Tile):
            intersecting_labels = list(labels_index.intersection(t.bounds))
            intersecting_labels = [labels[i] for i in intersecting_labels]
            return LabelledTile.from_tile_and_groundtruths(
                t,
                intersecting_labels,
                label_assignment_mode=self.label_assignment_mode,
                ioa_threshold=self.ioa_threshold,
                margin_from_bounds=self.margin_from_bounds,
            )

        tiles_with_labels = map(_get_labelled_tile, tiles)

        if self.discard_background:
            tiles_with_labels = filter(lambda tile: tile.label != "background", tiles_with_labels)

        return list(tiles_with_labels)


class CenteredTiles(TilesGenerator):
    """
    Generate tiles centered on groundtruth objects
    """

    def __init__(
        self,
        tile_size: int = 64,
        padding: int = 0,
        data_transform_fn: Optional[Callable] = None,
    ):
        super(CenteredTiles, self).__init__(
            tile_size=tile_size,
            padding=padding,
            data_transform_fn=data_transform_fn,
        )

    def get_tiles_for_item(self, item: SatelliteImage) -> [LabelledTile]:
        """
            Apply the sliding window on a full satellite images to generate a list of tiles
            Compares the tiles to the item's groundtruths
            Tiles that matches the declared conditions are assigned with the groundtruth's label

        Args:
            item: input item

        Returns:
            A list of tile with their labels

        """

        labels = item.labels  # type:[Groundtruth]

        tiles_with_labels = []

        for label in labels:
            x, y = label.center
            x, y = int(x), int(y)
            x, y = max(0, x - self.tile_size // 2), max(0, y - self.tile_size // 2)

            tile = LabelledTile(
                item_id=item.key,
                x_min=x,
                y_min=y,
                height=self.tile_size,
                width=self.tile_size,
                padding=self.padding,
                data_transform_fn=self.data_transform_fn,
                label=label.label,
            )

            tiles_with_labels.append(tile)

        return tiles_with_labels
