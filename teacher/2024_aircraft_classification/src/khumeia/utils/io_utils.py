import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import skimage.io

from khumeia.roi.groundtruth import Groundtruth
from khumeia.utils import memory


def _read_json(json_file: Path) -> Dict:
    with open(json_file, "r") as f:
        return json.load(f)


if memory is not None:
    imread = memory.cache(skimage.io.imread)
    read_json = memory.cache(_read_json)
else:
    imread = skimage.io.imread
    read_json = _read_json


def read_labels(labels_file: Path, default_label="aircraft") -> [Groundtruth]:
    labels_data = read_json(labels_file)
    labels = []
    for label in labels_data["markers"]:
        x, y, w = label["x"], label["y"], label["w"]
        labels.append(Groundtruth(x_min=x, y_min=y, width=w, height=w, label=default_label))

    return labels


def imsave(fname: Path, arr: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(fname, arr)


def lbsave(fname: Path, labels: [Groundtruth]):

    data = []
    for label in labels:
        data.append(dict(x=label.x_min, y=label.y_min, w=label.width))

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
