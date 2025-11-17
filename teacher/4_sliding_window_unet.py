# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: py312-isae
#     language: python
#     name: py312-isae
# ---

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# # Session 2 : Using the sliding window technique to predict on larger images
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" align="left" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>&nbsp;| Florient Chouteau | <a href="https://supaerodatascience.github.io/deep-learning/">https://supaerodatascience.github.io/deep-learning/</a>
#
# In this session we will load our previously detected model and apply it on large images using the sliding window technique.
#
# The sliding window technique is a method to convert a classifier into detector. It can be illustrated by a single gif:
#
# ![sw](https://storage.googleapis.com/fchouteau-isae-deep-learning/static/sliding_window.gif)
#
# For more information about the sliding window technique refer to this excellent article:
#
# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
#
# *Note:* We are training our model to recognize images at a single scale. Satellite imagery more or less prevents the foreground/background effect that sometimes require large changes in scale between training and testing for "normal" photography. So you can ignore the bits about the image pyramid on this issue (it is very good for general culture though, and can be applied in other use cases, or if we used multiscale training to "zoom" small aircrafts for example)
# %%
# Put your imports here
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

# %%
# Global variables
tiles_dataset_url = "https://storage.googleapis.com/fchouteau-isae-deep-learning/tiles_cloud_segmentation_dataset_2025.npz"

# %% [markdown]
# ## Download the dataset

# %%
# Download data
ds = np.lib.npyio.DataSource("/tmp/")
f = ds.open(tiles_dataset_url, "rb")
eval_tiles = np.load(f)
eval_tiles = eval_tiles["eval_tiles"]

# %% [markdown]
# ## Data Exploration
#
# - Plot some of the images
# - The images are not labelled to prevent any "competition", the objective is just to apply it.

# %%
eval_tiles.shape

# %%
grid_size = 4
grid = np.zeros((grid_size * 256, grid_size * 256, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = np.copy(eval_tiles[np.random.randint(0, eval_tiles.shape[0])])
        grid[i * 256 : (i + 1) * 256, j * 256 : (j + 1) * 256, :] = tile

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %% [markdown]
# ## Dense classification with sliding window

# %% [markdown]
# ### Reload your model
#
# - Using the routines detailed in the previous notebook, upload the scripted model corresponding to the best training (don't forget to save it on the other notebooks) then reload the model
#
# - Find the mean / std of the dataset you trained with to normalize the images !

# %%
# from google.colab import files

# uploaded = files.upload()

# for fn in uploaded.keys():
#     print(
#         'User uploaded file "{name}" with length {length} bytes'.format(
#             name=fn, length=len(uploaded[fn])
#         )
#     )

# %% {"editable": true, "slideshow": {"slide_type": ""}}
import torch.jit

MODEL = torch.jit.load("scripted_model.pt", map_location="cpu")
MODEL = MODEL.cpu().eval()

# %%
# Put your training mean and std here !

MEAN = ...

STD = ...

image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

# %% {"tags": ["solution"]}
MEAN = [0.54284222, 0.56890255, 0.51767049]

STD = [0.23642953, 0.20100785, 0.22810791]

image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


# %% [markdown]
# ### Implement the sliding window
#
# Intuitively, it's about applying an aircraft classifier trained on 64x64 pictures of aircraft or "anything else" as a detector.
#
# Our network structure more or less prevents applying it to the full 512x512 images, and even if it could (you may be able to do it with global pooling layers...) this would not bring much information ("there is at least one aircraft in this region" sometimes is not sufficient).
#
# So the idea is to "slide" our 64x64 classifier on the image and collect the coordinates where "aircraft" is returned. Those should be the locations of our aircrafts;
#
# You could view your model as a big convolution returning "aircraft / not aircraft". Its kernel size is 64x64, there are one or two filters depending on if you coded with softmax or crossentropy. You then just have to decide on the stride of this convolution... And to keep in mind how to go back to coordinates to plot your aircrafts afterwards ;)
#
# There are a lot of degrees of freedom when developping sliding windows. A sliding window with a too small "step" will only provide noisy overlapping detections. A step too large will make you miss some objects.
#
# It's up to you to find acceptable parameters.
#
# *Note*: The dataset labels were generated so that an image is considered an aircraft **if and only if the center of an aircraft lies in the center 32x32** of the 64x64 image

# %%
def apply_model_on_large_image(
    img: np.ndarray, model: nn.Module, patch_size=64, patch_stride=32
):
    h, w, c = img.shape
    coords = []

    for i0 in range(0, h - patch_size + 1, patch_stride):
        for j0 in range(0, w - patch_size + 1, patch_stride):
            patch = img[i0 : i0 + patch_size, j0 : j0 + patch_size]
            patch = image_transforms(patch).unsqueeze(0)

            with torch.no_grad():
                y_pred = model(patch)
                y_pred = y_pred[0, 0].cpu().numpy()
                if y_pred > 0.5:
                    coords.append((i0 + 32, j0 + 32))
    return coords


# %% [markdown]
# ### Apply the sliding window on the dataset and visualize results

# %%
k = np.random.randint(eval_tiles.shape[0])
image = np.copy(eval_tiles[k])

# %%
results = apply_model_on_large_image(image, MODEL)


# %%
def plot_results_on_image(image: np.ndarray, results: results):
    color = (0, 255, 0)

    image0 = np.copy(image)

    for ic, jc in results:
        image = cv2.rectangle(
            image, (jc - 32, ic - 32), (jc + 32, ic + 32), color, thickness=2
        )
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    ax[0].imshow(image0)
    ax[1].imshow(image)
    plt.show()


# %%
plot_results_on_image(image, results)

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# ## Image segmentation
#
# Now we will do the same with our U-Net.
#
# Our U-Net is fully convolutional, there are no mathematical constraints on the image size, only memory constraints. But one 256x256 image should fit in RAM :)
#
# We can say that the sliding window is "built-in".
#
# So instead of doing the sliding window manually, we will just pass the entire image and get the masks

# %%
from typing import List

import matplotlib


def overlay_img_msk(img, msk, color="red"):
    """Overlay an image with a colored mask"""
    c = color_to_rgb(color)
    msk = class_mask_to_color_mask(msk, class_colors=[[0, 0, 0], c])
    return overlay_img_rgb_msk(img, msk)


def overlay_img_prd(img, prd, cmap="viridis"):
    # Create overlays for ground truth and prediction without threshold
    cm = plt.get_cmap(cmap)
    pred_mask = (cm(prd)[:, :, :3] * 255.0).astype(np.uint8)

    # Overlay predicted mask on image
    pred_overlay = overlay_img_rgb_msk(img, pred_mask)

    return pred_overlay


def overlay_img_rgb_msk(img, msk, coefficients=[0.5, 0.5]):
    """
    Overlay a raw IMG and its MSK (0.5 * IMG_In_Visible + 0.5 * Msk)
    Args:
        img(np.ndarray): Img array (from imread)
        msk(np.ndarray): RGB mask

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
    """Convert color name to RGB tuple"""
    if isinstance(color_str, str):
        # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        color = np.asarray(matplotlib.colors.to_rgb(color_str))
        color = (255.0 * color).astype(np.uint8)
        color = tuple([int(c) for c in color])
        return color
    else:
        return color_str


def class_mask_to_color_mask(msk: np.ndarray, class_colors: List):
    """Convert class indices to colored mask"""
    if len(msk.shape) == 3:
        msk = msk[:, :, 0]
    color_mask = np.asarray(class_colors)[msk]
    return color_mask


def color_image_to_class_mask(msk: np.ndarray, class_colors: List):
    """Convert colored mask to class indices"""
    h, w = msk.shape[:2]
    class_mask = np.zeros((h, w)).astype(np.uint8)
    for cls_index, cls_color in enumerate(class_colors):
        cls_occ = np.all(msk == cls_color, axis=-1)
        class_mask[cls_occ] = cls_index + 1
    return class_mask


# %% [markdown]
# ### Reload your model
#
# - Using the routines detailed in the previous notebook, upload the scripted model corresponding to the best training (don't forget to save it on the other notebooks) then reload the model
#
# - Find the mean / std of the dataset you trained with to normalize the images !

# %%
# from google.colab import files

# uploaded = files.upload()

# for fn in uploaded.keys():
#     print(
#         'User uploaded file "{name}" with length {length} bytes'.format(
#             name=fn, length=len(uploaded[fn])
#         )
#     )

# %%
import torch.jit

UNET_MODEL = torch.jit.load("unet_scripted_model.pt", map_location="cpu")
UNET_MODEL = UNET_MODEL.cpu().eval()


# %%
def apply_unet_model_on_large_image(img: np.ndarray, model: nn.Module):
    h, w, c = img.shape

    img = image_transforms(img).unsqueeze(0)

    with torch.no_grad():
        y_pred = model(img)
        print(y_pred)
        mask = (y_pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    return mask


# %% {"editable": true, "slideshow": {"slide_type": ""}}
def plot_mask_on_image(image: np.ndarray, mask: np.ndarray):

    color = (0, 255, 0)

    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    ax[0].imshow(image)
    ax[1].imshow(overlay_img_msk(image, mask, color="red"))

    plt.show()


# %% [markdown]
# ### Apply segmentation model

# %%
image = np.copy(eval_tiles[k])

# %%
mask = apply_unet_model_on_large_image(image, UNET_MODEL)

# %%
plot_mask_on_image(image, mask)

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# # Et voilà !
