import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from khumeia import LOGGER

__all__ = ["download_data", "make_image_ids", "make_labels"]


def download_data(raw_data_dir: Path):
    LOGGER.info("Downloading data to {}".format(str(raw_data_dir)))
    cmds = [
        "gsutil -m rsync -r gs://planespotting-data-public/USGS_public_domain_photos/ {}".format(
            str(raw_data_dir / "trainval")
        ),
        "gsutil -m rsync -r gs://planespotting-data-public/USGS_public_domain_photos_eval/ {}".format(
            str(raw_data_dir / "eval")
        ),
    ]

    cmds = [shlex.split(cmd) for cmd in cmds]

    outliers = [
        "USGS_TUC1l.jpg",
        "USGS_TUC1l.json",
        "USGS_TUC1s.jpg",
        "USGS_TUC1s.json",
        "USGS_TUC2s.jpg",
        "USGS_TUC2s.json",
        "USGS_TUC3s.jpg",
        "USGS_TUC3s.json",
        "USGS_TUC4s.jpg",
        "USGS_TUC4s.json",
        "USGS_DMA.jpg",
        "USGS_DMA2.jpg",
    ]
    raw_data_dir.mkdir(exist_ok=True)
    (raw_data_dir / "trainval").mkdir(exist_ok=True)
    (raw_data_dir / "eval").mkdir(exist_ok=True)

    for cmd in cmds:
        LOGGER.info(" ".join(cmd))
        subprocess.check_call(cmd, shell=False)
    for outlier in outliers:
        try:
            (raw_data_dir / "trainval" / outlier).unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        try:
            (raw_data_dir / "eval" / outlier).unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def make_labels(raw_data_dir: Path, fold="trainval"):
    df = pd.read_csv(raw_data_dir / "{}_ids.csv".format(fold))
    trainval_labels = []
    eval_labels = []
    image_ids = list(df["image_id"].unique())

    for image_id in image_ids:
        fold = list(df[df["image_id"] == image_id]["fold"])[0]
        image_path = raw_data_dir / fold / f"{image_id}.jpg"
        label_path = image_path.with_suffix(".json")

        with open(label_path, "r") as f:
            labels = json.load(f)

        for label in labels["markers"]:
            x, y, w = label["x"], label["y"], label["w"]
            if fold == "trainval":
                trainval_labels.append({"image_id": image_id, "x": x, "y": y, "size": w})
            else:
                eval_labels.append({"image_id": image_id, "x": x, "y": y, "size": w})

    pd.DataFrame(trainval_labels).to_csv(
        raw_data_dir / f"{fold}_labels.csv",
        index=False,
        index_label=None,
    )


def make_image_ids(raw_data_dir: Path):
    train_images = (raw_data_dir / "trainval").glob("*.jpg")
    dataset = []
    for image_file in train_images:
        image_id = image_file.stem
        dataset.append({"image_id": image_id, "fold": "trainval"})
    pd.DataFrame(dataset).to_csv(raw_data_dir / "trainval_ids.csv", index_label=None, index=False)

    eval_images = (raw_data_dir / "eval").glob("*.jpg")
    dataset = []
    for image_file in eval_images:
        image_id = image_file.stem
        dataset.append({"image_id": image_id, "fold": "eval"})
    pd.DataFrame(dataset).to_csv(raw_data_dir / "eval_ids.csv", index_label=None, index=False)
