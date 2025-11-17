from pathlib import Path

import numpy as np

from khumeia.data.item import SatelliteImage
from khumeia.helpers.visualisation import draw_tile
from khumeia.roi import Groundtruth
from khumeia.roi.tile import LabelledTile, Tile
from khumeia.utils import io_utils

__all__ = ["ItemTileDumper", "ImageItemTileDumper", "NpArrayTileDumper"]


class ItemTileDumper:
    def __init__(self, item: SatelliteImage, dump_objects: bool = False):
        self.item = item
        self.dump_objects = dump_objects

    @property
    def image(self) -> np.ndarray:
        return self.item.image

    @property
    def labels(self) -> [Groundtruth]:
        return self.item.labels

    def __call__(self, tile: Tile):
        return self.dump_tile(tile)

    def dump_tile(self, tile: Tile):
        raise NotImplementedError


class ImageItemTileDumper(ItemTileDumper):
    def __init__(self, item: SatelliteImage, output_dir: Path, save_format="jpg", dump_objects: bool = False):
        super().__init__(item=item, dump_objects=dump_objects)
        self.output_dir = Path(output_dir)
        self.save_format = save_format

    def dump_tile(self, tile: LabelledTile):
        if tile.item_id == self.item.key:
            (self.output_dir / tile.label).mkdir(exist_ok=True, parents=True)

            tile_data = tile.get_data(self.image)

            tile_basename = "{}_{}.{}".format(self.item.key, tile.key, self.save_format)

            io_utils.imsave(self.output_dir / tile.label / tile_basename, tile_data)

            if self.dump_objects:
                tile_labels = tile.get_labels(self.labels)

                lbls_basename = "{}_{}.json".format(self.item.key, tile.key)

                io_utils.lbsave(self.output_dir / tile.label / lbls_basename, tile_labels)

                preview = draw_tile(self.item, tile)

                io_utils.imsave(self.output_dir / tile.label / f"{self.item.key}_{tile.key}_preview.jpg", preview)

                return self.output_dir / tile.label / tile_basename, self.output_dir / tile.label / lbls_basename

            else:
                return self.output_dir / tile.label / tile_basename, tile.label

        else:
            return None


class NpArrayTileDumper(ItemTileDumper):
    def dump_tile(self, tile: LabelledTile):
        if tile.item_id == self.item.key:
            tile_data = tile.get_data(self.image)

            if self.dump_objects:
                tile_labels = tile.get_labels(self.labels)

                tile_labels = np.asarray(
                    [[label.x_min, label.y_min, label.width, label.height] for label in tile_labels]
                )

                return tile_data, tile_labels
            else:
                tile_label = 0 if tile.label == "background" else 1
                return tile_data, tile_label
        else:
            return None
