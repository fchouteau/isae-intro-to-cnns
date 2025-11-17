# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib notebook

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import tqdm

# %%
# add khumeia
sys.path.append("./src/")
sys.path = list(set(sys.path))

# %%
# setup env variable
raw_data_dir = Path("./data/usgs/").absolute()
TRAINVAL_DATA_DIR = raw_data_dir / "trainval"
EVAL_DATA_DIR = raw_data_dir / "eval"

# %%
# List files
for root, dirs, files in os.walk(os.path.join(os.environ.get("TP_DATA"), "usgs")):
    for file in files:
        print(os.path.join(root, file))
# %%
# Setup variables
image_ids = pd.read_csv(os.path.join(raw_data_dir, "trainval_ids.csv"))
train_labels = pd.read_csv(os.path.join(raw_data_dir, "trainval_labels.csv"))

# %%
print(
    "Number of images in train dataset \n{}".format(
        train_labels["image_id"].value_counts()
    )
)
print(
    "Description of labels \n{}".format(
        scipy.stats.describe(train_labels["image_id"].value_counts())
    )
)
train_labels["size"].describe()

# %% [markdown]
# ## Using Khumeia

# %%
from khumeia import helpers

# %%
trainval_dataset = helpers.dataset_generation.items_dataset_from_path(TRAINVAL_DATA_DIR)

# %%
for satellite_image in trainval_dataset:
    print(satellite_image)

# %% [markdown]
# ## Plotting histograms and descriptions

# %%
# Let's write a histogram function
from matplotlib import pyplot as plt


def plot_histogram(dataset, n_bins=256):
    """
    Plotting histogram over a dataset
    Args:
        dataset(khumeia.data.Dataset): dataset
        n_bins(int): number of bins for histogram

    Returns:
        The histogram
    """
    mean_hist_r = [0 for _ in range(n_bins)]
    mean_hist_g = [0 for _ in range(n_bins)]
    mean_hist_b = [0 for _ in range(n_bins)]

    for image_item in tqdm.tqdm(dataset, desc="computing histograms..."):
        img = image_item.image

        hist_r, _ = np.histogram(img[:, :, 0], bins=n_bins, density=True)
        hist_g, _ = np.histogram(img[:, :, 1], bins=n_bins, density=True)
        hist_b, _ = np.histogram(img[:, :, 2], bins=n_bins, density=True)
        mean_hist_r = np.sum([mean_hist_r, hist_r], axis=0)
        mean_hist_g = np.sum([mean_hist_g, hist_g], axis=0)
        mean_hist_b = np.sum([mean_hist_b, hist_b], axis=0)

    mean_hist_r /= len(image_ids)
    mean_hist_g /= len(image_ids)
    mean_hist_b /= len(image_ids)

    plt.bar(np.arange(len(mean_hist_r)), mean_hist_r, color="red", width=1, alpha=0.5)
    plt.bar(np.arange(len(mean_hist_g)), mean_hist_g, color="green", width=1, alpha=0.5)
    plt.bar(np.arange(len(mean_hist_b)), mean_hist_b, color="blue", width=1, alpha=0.5)
    plt.show()


# %%
# # Plot the histogram for the 10 images
# plot_histogram(trainval_dataset[:10])


# %%
def describe_dataset(dataset):
    """
    Print image id and image shape and nb of labels per item
    Args:
        dataset:

    Returns:

    """
    for image_item in dataset:
        print(
            "item id: {} - shape: {} - nb labels:{}".format(
                image_item.image_id, image_item.shape, len(image_item.labels)
            )
        )


# %%
# Describe the dataset
describe_dataset(trainval_dataset)

# %% [markdown]
# ## Some data visualisation
# Let's plot an image (using khumeia helpers) and its labels

# %%
item = trainval_dataset[2]
print(item)
image = item.image
labels = item.labels

# %%
image = helpers.visualisation.draw_bboxes_on_image(image, labels, color="green")
plt.figure(figsize=(10, 10))
plt.title(item.image_id)
plt.imshow(image)
plt.show()
