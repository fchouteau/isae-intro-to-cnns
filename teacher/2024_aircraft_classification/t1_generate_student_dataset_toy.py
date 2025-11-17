# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib widget

# %% [markdown]
# # Génération du jeu de données "toy"
#
# Jeu de données équilibré et images mélangées

# %%
import sys
from pathlib import Path

# %%
# add khumeia
sys.path.append("./src/")
sys.path = list(set(sys.path))

# %%
# setup env variable
raw_data_dir = Path("./data/usgs/").resolve()
TRAINVAL_DATA_DIR = raw_data_dir / "trainval"
EVAL_DATA_DIR = raw_data_dir / "eval"
TRAINVAL_DATA_DIR

# %% [markdown]
# ## Using Khumeia

# %%
from khumeia import helpers

# %%
trainval_dataset = helpers.dataset_generation.items_dataset_from_path(TRAINVAL_DATA_DIR)

# %% [markdown]
# ## Dataset parsing using khumeia

# %%
import random

from matplotlib import pyplot as plt

from khumeia.helpers.visualisation import draw_item_with_tiles
from khumeia.roi.tiles_generator import CenteredTiles, RandomTiles
from khumeia.utils.roi_list_utils import filter_tiles_by_item

# %%
MAX_ITEMS = None
TILE_SIZE = 64
SPARSE_TILE_STRIDE = 64
DENSE_TILE_STRIDE = 32
MARGIN = 16

# %%
trainval_dataset.items = trainval_dataset.items[
    : min(len(trainval_dataset), MAX_ITEMS or len(trainval_dataset))
]

# %%
random_tiles = RandomTiles(
    tile_size=TILE_SIZE,
    num_tiles=1024,
    margin_from_bounds=MARGIN,
)

centered_tiles = CenteredTiles(
    tile_size=TILE_SIZE,
)

# %%
trainval_tiles = helpers.dataset_generation.generate_candidate_tiles_from_items(
    trainval_dataset,
    sliding_windows=[centered_tiles, random_tiles],
    n_jobs=4,
)
# %% [markdown]
# Display one image

# %%
_item_ids = list(set(item.item_id for item in trainval_tiles))
print(_item_ids)

# %%
_item_id = random.randint(0, len(_item_ids) - 1)
# _item_id = "USGS_CVG"

_tiles = filter_tiles_by_item(trainval_tiles, "USGS_CVG")
_item = trainval_dataset.filter(lambda item: item.image_id == "USGS_CVG")[0]

_img = draw_item_with_tiles(_item, _tiles)

plt.imshow(_img)
plt.show()

# %% [markdown]
# ## Toy Dataset Generation
# Let's generate our first dataset

# %%
NB_FG_TRAINVAL_TILES = 1792
TRAIN_TEST_SPLIT = 0.75

# %%
import random

import numpy as np

from khumeia.roi.tiles_sampler import *

# %%
train_stratified_sampler = BackgroundToPositiveRatioSampler(
    nb_positive_tiles_max=NB_FG_TRAINVAL_TILES,
    background_to_positive_ratio=1,
    with_replacement=False,
    shuffle=True,
)

trainval_tiles_sampled = helpers.dataset_generation.sample_tiles_from_candidates(
    trainval_tiles, tiles_samplers=[train_stratified_sampler]
)

# %%
# Split both and shuffle
trainval_fg_tiles = trainval_tiles_sampled.filter(lambda t: t.label != "background")
random.shuffle(trainval_fg_tiles.items)
trainval_bg_tiles = trainval_tiles_sampled.filter(lambda t: t.label == "background")
random.shuffle(trainval_bg_tiles.items)

# %%
trainval_fg_array = helpers.dataset_generation.dump_dataset_tiles(
    tiles_dataset=trainval_fg_tiles, items_dataset=trainval_dataset
)
trainval_bg_array = helpers.dataset_generation.dump_dataset_tiles(
    tiles_dataset=trainval_bg_tiles, items_dataset=trainval_dataset
)

# %%
# Split into train-test while keep class repartition
trainval_fg_images = np.asarray([i[0] for i in trainval_fg_array.items])
trainval_bg_images = np.asarray([i[0] for i in trainval_bg_array.items])

n_fg = trainval_fg_images.shape[0]
n_bg = trainval_bg_images.shape[0]
train_fg_images, test_fg_images = (
    trainval_fg_images[: int(TRAIN_TEST_SPLIT * n_fg)],
    trainval_fg_images[int(TRAIN_TEST_SPLIT * n_fg) :],
)
train_fg_labels, test_fg_labels = [1 for _ in train_fg_images], [
    1 for _ in test_fg_images
]
train_bg_images, test_bg_images = (
    trainval_bg_images[: int(TRAIN_TEST_SPLIT * n_bg)],
    trainval_bg_images[int(TRAIN_TEST_SPLIT * n_bg) :],
)
train_bg_labels, test_bg_labels = [0 for _ in train_bg_images], [
    0 for _ in test_bg_images
]

train_images = np.concatenate([train_fg_images, train_bg_images], axis=0)
train_labels = np.concatenate([train_fg_labels, train_bg_labels], axis=0)
test_images = np.concatenate([test_fg_images, test_bg_images], axis=0)
test_labels = np.concatenate([test_fg_labels, test_bg_labels], axis=0)

train_indexes = np.arange(train_images.shape[0])
np.random.shuffle(train_indexes)
test_indexes = np.arange(test_images.shape[0])
np.random.shuffle(test_indexes)

train_images = train_images[train_indexes]
train_labels = train_labels[train_indexes]
test_images = test_images[test_indexes]
test_labels = test_labels[test_indexes]
# %%
print(train_images.shape)
print(train_labels.shape)
# %%
print(test_images.shape)
print(test_labels.shape)

# %%
# Save as dict of nparrays
dataset_path = Path("./data") / "toy_aircraft_dataset_2023.npz"

with open(dataset_path, "wb") as f:
    np.savez_compressed(
        f,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

# %%
# upload to gcp
import shlex
import subprocess

cmd = "gsutil -m cp -r {} gs://fchouteau-isae-deep-learning/".format(
    str(dataset_path.resolve())
)
print(cmd)
subprocess.check_call(cmd, shell=True)
# %% [markdown]
# ## Reload and check everything is ok

# %%
import numpy as np

# %%
# try to reload using numpy datasource
ds = np.DataSource("/tmp/")
f = ds.open(
    "https://storage.googleapis.com/fchouteau-isae-deep-learning/toy_aircraft_dataset_2023.npz",
    "rb",
)
toy_dataset = np.load(f)
train_images = toy_dataset["train_images"]
train_labels = toy_dataset["train_labels"]
test_images = toy_dataset["test_images"]
test_labels = toy_dataset["test_labels"]
print(train_images.shape)

# %%
np.unique(train_labels, return_counts=True)

# %%
np.unique(test_labels, return_counts=True)

# %%
# plot them
import cv2
from matplotlib import pyplot as plt

# %%
grid_size = 8
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = train_images[i * grid_size + j]
        label = train_labels[i * grid_size + j]
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        tile = cv2.rectangle(tile, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = tile

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %%
grid_size = 8
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
indexes = np.random.choice(len(test_images), size=(grid_size * grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        idx = indexes[i * grid_size + j]
        tile = test_images[idx]
        label = test_labels[idx]
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        tile = cv2.rectangle(tile, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = tile

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()
