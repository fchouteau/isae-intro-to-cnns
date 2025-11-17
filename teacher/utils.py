from pathlib import Path
from typing import Callable, List, Tuple
import itertools
import random
import shlex
import subprocess

import cv2
import joblib
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
import os
import rasterio
import kornia.geometry
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torchgeo.datasets import CloudCoverDetection
from torchvision.ops.misc import interpolate

# Configuration constants
MAX_R = 3.0  # max reflectance
MID_R = 0.13
SAT = 1.2
GAMMA = 1.8

# Gamma adjustment constants
G_OFF = 0.01
G_OFF_POW = G_OFF**GAMMA
G_OFF_RANGE = (1 + G_OFF) ** GAMMA - G_OFF_POW


def s_adj(a):
    return adj_gamma(adj(a, MID_R, 1, MAX_R))


def adj_gamma(b):
    return ((b + G_OFF) ** GAMMA - G_OFF_POW) / G_OFF_RANGE


def sat_enh(r, g, b):
    avg_s = (r + g + b) / 3.0 * (1 - SAT)
    return [
        np.clip(avg_s + r * SAT,0,1),
        np.clip(avg_s + g * SAT,0,1),
        np.clip(avg_s + b * SAT,0,1),
    ]


def adj(a, tx, ty, max_c):
    ar = np.clip(a / max_c, 0, 1)
    return (
        ar
        * (ar * (tx / max_c + ty - 1) - ty)
        / (ar * (2 * tx / max_c - 1) - tx / max_c)
    )


def sRGB(c):
    """Vectorized sRGB conversion using numpy broadcasting."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c**0.41666666666) - 0.055)


def image_to_enchanced_tci(image):
    """
    Process an entire image using pure numpy broadcasting.

    Args:
        image_data: Dictionary with numpy arrays for each band

    Returns:
        Processed RGBA image as numpy array
    """
    # reflectance floating
    r, g, b = image[:3, :, :].astype(np.float32) / 10000.0

    # Apply adjustments to each band (s_adj works with broadcasting)
    r = s_adj(r)
    g = s_adj(g)
    b = s_adj(b)

    # Saturation enhancement
    avg_s = (r + g + b) / 3.0 * (1 - SAT)

    r_enh = np.clip(avg_s + r * SAT, 0.0, 1.0)
    g_enh = np.clip(avg_s + g * SAT, 0.0, 1.0)
    b_enh = np.clip(avg_s + b * SAT, 0.0, 1.0)

    # Apply sRGB conversion (now uses broadcasting via np.where)
    r_srgb = sRGB(r_enh)
    g_srgb = sRGB(g_enh)
    b_srgb = sRGB(b_enh)

    # Stack the results
    img = np.stack([r_srgb, g_srgb, b_srgb], axis=-1)

    return (img * 255).astype(np.uint8)


def overlay_img_msk(img, msk, coefficients=[0.5, 0.5]):
    """
    Overlay a raw IMG and its MSK (0.5 * IMG_In_Visible + 0.5 * Msk)
    Args:
        img(np.ndarray): Img array (from imread)
        msk(np.ndarray): BGR mask

    Returns:
        np.ndarray containing the overlay (it's a copy)
    """
    tmp = np.copy(img).astype(np.float32)
    tmp2 = np.copy(msk).astype(np.float32)
    if len(tmp2.shape) == 2:
        idxs = tmp2 != 0.0
    else:
        c = tmp2.shape[2]
        idxs = ~np.all(tmp2 == [0.0 for _ in range(c)], axis=-1)

    tmp[idxs] = coefficients[0] * tmp[idxs] + coefficients[1] * tmp2[idxs]
    tmp = np.clip(tmp, 0.0, 255.0)
    tmp = tmp.astype(np.uint8)
    del tmp2

    return tmp


def color_to_rgb(color_str: str):
    if isinstance(color_str, str):
        # https://matplotlib.org/3.1.0/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
        color = np.asarray(matplotlib.colors.to_rgb(color_str))
        # color = np.asarray((color[2], color[1], color[0]))
        color = (255.0 * color).astype(np.uint8)
        color = tuple([int(c) for c in color])
        return color
    else:
        return color_str


def class_mask_to_color_mask(msk: np.ndarray, class_colors: List):
    if len(msk.shape) == 3:
        msk = msk[:, :, 0]
    color_mask = np.asarray(class_colors)[msk]

    return color_mask


def color_image_to_class_mask(msk: np.ndarray, class_colors: List):
    h, w = msk.shape[:2]

    class_mask = np.zeros((h, w)).astype(np.uint8)

    for cls_index, cls_color in enumerate(class_colors):
        cls_occ = np.all(msk == cls_color, axis=-1)
        class_mask[cls_occ] = cls_index + 1

    return class_mask

class CloudCoverDatasetCustom(CloudCoverDetection):
    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            bands=CloudCoverDetection.rgb_bands,
            transforms=transforms,
            download=download,
        )

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        sample = super().__getitem__(item)

        img, msk = sample["image"], sample["mask"]

        img = kornia.geometry.rescale(
            img.float(), factor=0.5, interpolation="bilinear", antialias=True
        )
        msk = kornia.geometry.rescale(msk.float(), factor=0.5, interpolation="nearest")

        img = image_to_enchanced_tci(img.numpy())

        msk = msk.numpy().astype(np.uint8)

        return img, msk

    @staticmethod
    def plot(
        sample: Tuple[Tensor, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ):

        img, mask = sample

        msk = class_mask_to_color_mask(mask, class_colors=[[0, 0, 0], [255, 0, 0]])

        prv = overlay_img_msk(img, msk)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

        axs[0].imshow(img)
        axs[0].axis("off")
        axs[1].imshow(mask, cmap="viridis")
        axs[1].axis("off")
        axs[2].imshow(prv)
        axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")
            axs[2].set_title("Preview")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

def compute_metadata(dataset, item, tile_size, only_central):
    metas = []

    try:
        img, msk = dataset[item]
        h, w = img.shape[0], img.shape[1]
    except:
        return metas

    if only_central:
        _, msk_roi = (
            img[
                h // 2 - tile_size // 2 : h // 2 + tile_size // 2,
                w // 2 - tile_size // 2 : w // 2 + tile_size // 2,
                :,
            ],
            msk[
                h // 2 - tile_size // 2 : h // 2 + tile_size // 2,
                w // 2 - tile_size // 2 : w // 2 + tile_size // 2,
            ],
        )

        ncn_roi = msk_roi.sum().astype(np.float32) / (msk_roi.shape[0] * msk_roi.shape[1])

        meta = dict(item=item, kh=-1, kw=-1, ncn=round(float(ncn_roi), 2), tile_size=tile_size)

        metas.append(meta)

    else:

        for kh, kw in itertools.product(range(h // tile_size), range(w // tile_size)):
            _, msk_roi = (
                img[kh * tile_size : (kh + 1) * tile_size, kw * tile_size: (kw + 1) * tile_size, :],
                msk[kh * tile_size : (kh + 1) * tile_size, kw * tile_size: (kw + 1) * tile_size],
            )

            ncn_roi = msk_roi.sum().astype(np.float32) / (msk_roi.shape[0] * msk_roi.shape[1])

            meta = dict(item=item, kh=kh, kw=kw, ncn=round(float(ncn_roi), 2), tile_size=tile_size)

            metas.append(meta)

    return metas


parallel_metadata = joblib.delayed(compute_metadata)

def get_classification_sample(dataset, roi_meta):
    img, msk = dataset[roi_meta.item]
    h, w = img.shape[0], img.shape[1]

    ts = roi_meta.tile_size

    kh, kw = roi_meta.kh, roi_meta.kw

    lbl_roi = 0 if roi_meta.clear == True else 1

    if kh == -1 or kw == -1 :
        img_roi = img[
            h // 2 - ts // 2 : h // 2 + ts // 2,
            w // 2 - ts // 2 : w // 2 + ts // 2,
            :,
        ]
        _ = msk[
            h // 2 - ts // 2 : h // 2 + ts // 2,
            w // 2 - ts // 2 : w // 2 + ts // 2,
        ]

    else:
        img_roi = img[kh * ts : (kh + 1) * ts, kw * ts: (kw + 1) * ts, :]
        _ = msk[kh * ts : (kh + 1) * ts, kw * ts: (kw + 1) * ts]

    return img_roi, lbl_roi


parallel_classification_sample = joblib.delayed(get_classification_sample)

def get_segmentation_sample(dataset, roi_meta):
    img, msk = dataset[roi_meta.item]
    h, w = img.shape[0], img.shape[1]

    ts = roi_meta.tile_size

    kh, kw = roi_meta.kh, roi_meta.kw

    if kh == -1 or kw == -1 :
        img_roi = img[
            h // 2 - ts // 2 : h // 2 + ts // 2,
            w // 2 - ts // 2 : w // 2 + ts // 2,
            :,
        ]
        msk_roi = msk[
            h // 2 - ts // 2 : h // 2 + ts // 2,
            w // 2 - ts // 2 : w // 2 + ts // 2,
        ]

    else:
        img_roi = img[kh * ts : (kh + 1) * ts, kw * ts: (kw + 1) * ts, :]
        msk_roi = msk[kh * ts : (kh + 1) * ts, kw * ts: (kw + 1) * ts]

    return img_roi, msk_roi


parallel_segmentation_sample = joblib.delayed(get_segmentation_sample)
