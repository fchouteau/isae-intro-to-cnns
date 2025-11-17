import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from khumeia.roi.groundtruth import Groundtruth
from khumeia.utils import io_utils


@dataclass
class Item:
    """
    An item is a container for an image and its labels
    """

    @property
    def key(self):
        raise NotImplementedError

    @property
    def image(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError


@dataclass
class SatelliteImage(Item):
    """
    Contains the necessary information to define a satellite images
    Contains image id, image file and label file
    Contains image and labels as properties
    (cached via joblib to avoid loading the same image n times and to avoid ram overflow)
    The labels are automatically parsed as BoundingBoxes

    image_id: the image identifier (generally the filename...)
    image_file: path to the .jpg image file
    label_file: path to the .json label file
    """

    image_id: str
    image_file: Path
    label_file: Path

    @classmethod
    def from_image_id_and_path(cls, image_id: str, path: Path) -> "SatelliteImage":
        """

        Args:
            image_id: the filename
            path: the root directory to parse when looking for .jpg and .json files

        Returns:

        """
        image_file = path / "{}.jpg".format(image_id)
        label_file = path / "{}.json".format(image_id)
        return cls(image_id=image_id, image_file=image_file, label_file=label_file)

    @property
    def key(self) -> str:
        """
        An unique identifier of the Item class used to for matching

        Returns:
            str: the image_id

        """
        return self.image_id

    @property
    def image(self) -> np.ndarray:
        """
        Read image data (wrapper around skimage.imread)
        Returns:
            np.ndarray: the image data as a int8 (h,w,3) np.ndarray
        """
        image = io_utils.imread(self.image_file)
        return image

    @property
    def labels(self) -> [Groundtruth]:
        """
        Get the labels of a satellite image (load json and decode labels)

        Returns:
            list(Groundtruth): A list of bounding boxes corresponding to the labels

        """
        return io_utils.read_labels(self.label_file)

    @property
    def shape(self) -> (int, int, int):
        """
        Get the shape of the array (wrapper around self.image.shape)

        Returns:
            tuple: h,w,c

        """
        return self.image.shape

    def __str__(self):
        d = dict()
        d["class"] = self.__class__.__name__
        d["image_id"] = self.image_id
        d["image_shape"] = self.shape
        d["nb_labels"] = len(self.labels)
        d["image_file"] = str(self.image_file)
        d["label_file"] = str(self.label_file)
        return json.dumps(d, indent=4)

    def __repr__(self):
        return self.__str__()
