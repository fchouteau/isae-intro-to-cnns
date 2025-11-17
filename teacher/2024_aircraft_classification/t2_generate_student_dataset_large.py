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
# # %matplotlib widget

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

# %% [markdown]
# ## Using Khumeia

# %%
from khumeia import helpers

# %%
trainval_dataset = helpers.dataset_generation.items_dataset_from_path(TRAINVAL_DATA_DIR)
eval_dataset = helpers.dataset_generation.items_dataset_from_path(EVAL_DATA_DIR)

# %% [markdown]
# ## Dataset parsing using khumeia

# %%
from khumeia.roi.tiles_generator import CenteredTiles, RandomTiles, SlidingWindow

# %%
MAX_ITEMS = None
TILE_SIZE = 64
SPARSE_TILE_STRIDE = 64
DENSE_TILE_STRIDE = 16
MARGIN = 16

# %%
trainval_dataset.items = trainval_dataset.items[
    : min(len(trainval_dataset), MAX_ITEMS or len(trainval_dataset))
]

train_dataset, test_dataset = helpers.dataset_generation.split_dataset(
    trainval_dataset, proportion=0.8
)
test_dataset = test_dataset.extend(eval_dataset)

# %%
print("TRAIN :{}".format(list(set(item.key for item in train_dataset.items))))
print("VAL: {}".format(list(set(item.key for item in test_dataset.items))))

# %%
sliding_window_dense = SlidingWindow(
    tile_size=TILE_SIZE,
    stride=DENSE_TILE_STRIDE,
    margin_from_bounds=MARGIN,
    discard_background=True,
)

sliding_window_sparse = SlidingWindow(
    tile_size=TILE_SIZE,
    stride=SPARSE_TILE_STRIDE,
    margin_from_bounds=MARGIN,
    discard_background=False,
)

random_tiles = RandomTiles(
    tile_size=TILE_SIZE, num_tiles=1024, margin_from_bounds=MARGIN
)

centered_tiles = CenteredTiles(tile_size=TILE_SIZE)

sliding_windows = [
    sliding_window_dense,
    sliding_window_sparse,
    random_tiles,
    centered_tiles,
]

# %%
train_tiles = helpers.dataset_generation.generate_candidate_tiles_from_items(
    train_dataset, sliding_windows=sliding_windows, n_jobs=4
)
# %%
random_tiles = RandomTiles(
    tile_size=TILE_SIZE, num_tiles=10 * 1024, margin_from_bounds=MARGIN
)

centered_tiles = CenteredTiles(tile_size=TILE_SIZE)

test_sliding_windows = [random_tiles, centered_tiles]

test_tiles = helpers.dataset_generation.generate_candidate_tiles_from_items(
    test_dataset, sliding_windows=test_sliding_windows, n_jobs=4
)

# %% [markdown]
# ## Big Dataset Generation
# Let's generate our first dataset

# %%
import os

import numpy as np

from khumeia.roi.tiles_sampler import *

# %%
SAMPLING_RATIO = 19
NB_POSITIVE_TRAIN_TILES = 4608
NB_POSITIVE_TEST_TILES = 1024

# %%
train_stratified_sampler = BackgroundToPositiveRatioPerItemSampler(
    nb_positive_tiles_max=NB_POSITIVE_TRAIN_TILES,
    background_to_positive_ratio=SAMPLING_RATIO,
    with_replacement=True,
    shuffle=True,
)

train_tiles_sampled = helpers.dataset_generation.sample_tiles_from_candidates(
    train_tiles, tiles_samplers=[train_stratified_sampler]
)

# %%
test_stratified_sampler = BackgroundToPositiveRatioPerItemSampler(
    nb_positive_tiles_max=NB_POSITIVE_TEST_TILES,
    background_to_positive_ratio=SAMPLING_RATIO,
    with_replacement=True,
    shuffle=True,
)

test_tiles_sampled = helpers.dataset_generation.sample_tiles_from_candidates(
    test_tiles, tiles_samplers=[test_stratified_sampler]
)

# %%
train_array = helpers.dataset_generation.dump_dataset_tiles(
    tiles_dataset=train_tiles_sampled, items_dataset=train_dataset
)

# %%
test_array = helpers.dataset_generation.dump_dataset_tiles(
    tiles_dataset=test_tiles_sampled, items_dataset=test_dataset
)

# %%
train_images = np.asarray([i[0] for i in train_array.items])
train_labels = np.asarray([i[1] for i in train_array.items])

# %%
print(train_images.shape)
print(train_labels.shape)

# %%
test_images = np.asarray([i[0] for i in test_array.items])
test_labels = np.asarray([i[1] for i in test_array.items])

# %%
print(test_images.shape)
print(test_labels.shape)

# %%
# Shuffle arrays
train_indexes = np.arange(train_images.shape[0])
np.random.shuffle(train_indexes)
test_indexes = np.arange(test_images.shape[0])
np.random.shuffle(test_indexes)

train_images = train_images[train_indexes]
train_labels = train_labels[train_indexes]
test_images = test_images[test_indexes]
test_labels = test_labels[test_indexes]

# %%
# Save as dict of nparrays
dataset_path = Path("./data") / "large_aircraft_dataset_2023.npz"

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
    os.path.abspath(dataset_path)
)
print(cmd)
subprocess.check_call(cmd, shell=True)
# %% [markdown]
# ## Reload and check everything is ok

# %%
# try to reload using numpy datasource
ds = np.DataSource("/tmp/")
f = ds.open(
    "https://storage.googleapis.com/fchouteau-isae-deep-learning/large_aircraft_dataset_2023.npz",
    "rb",
)
large_dataset = np.load(f)

train_images = large_dataset["train_images"]
train_labels = large_dataset["train_labels"]
test_images = large_dataset["test_images"]
test_labels = large_dataset["test_labels"]
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
indexes = np.random.choice(len(train_images), size=(grid_size * grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        idx = indexes[i * grid_size + j]
        tile = train_images[idx]
        label = train_labels[idx]
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
grid_2 = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
indexes = np.random.choice(len(test_images), size=(grid_size * grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        idx = indexes[i * grid_size + j]
        tile = test_images[idx]
        label = test_labels[idx]
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        tile = cv2.rectangle(tile, (0, 0), (64, 64), color, thickness=2)
        grid_2[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = tile

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid_2)
plt.show()
