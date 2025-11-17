# ---
# jupyter:
#   jupytext:
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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import random
import subprocess
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from utils import (
    CloudCoverDatasetCustom,
    class_mask_to_color_mask,
    overlay_img_msk,
    parallel_classification_sample,
    parallel_metadata,
    parallel_segmentation_sample,
)

# %%
trainval_dataset = CloudCoverDatasetCustom(
    root="data",
    split="train",
    transforms=None,
    download=True,
)

eval_dataset = CloudCoverDatasetCustom(
    root="data",
    split="test",
    transforms=None,
    download=True,
)


# %%
# trainval_dataset._download()

# %%
# eval_dataset._download()

# %%
len(trainval_dataset), len(eval_dataset)

# %%
k = np.random.randint(len(trainval_dataset))
print(k)
img, msk = trainval_dataset[k]

fig = CloudCoverDatasetCustom.plot(trainval_dataset[k])

# %%
ts = 64

# %%
kh, kw = np.random.randint(256 // ts), np.random.randint(256 // ts)
kw, kw

# %%
roi = img[kh * ts : (kh + 1) * ts, (kw) * ts : (kw + 1) * ts, :]

# %%
plt.imshow(roi)
plt.show()

# %% [markdown]
# ## Dataset Indexing

# %%
if not Path("data/train.csv").exists():
    it = tqdm.tqdm(range(len(trainval_dataset)))

    trainval_df = joblib.Parallel(n_jobs=8)(
        parallel_metadata(trainval_dataset, k, ts, False) for k in it
    )

    trainval_df = [i for l in trainval_df for i in l]
    trainval_df = pd.DataFrame(trainval_df)
    trainval_df.to_csv("data/train.csv", index=None)
else:
    trainval_df = pd.read_csv("data/train.csv")

# %%
if not Path("data/eval.csv").exists():
    it = tqdm.tqdm(range(len(eval_dataset)))

    eval_df = joblib.Parallel(n_jobs=8)(
        parallel_metadata(eval_dataset, k, ts, False) for k in it
    )

    eval_df = [i for l in eval_df for i in l]
    eval_df = pd.DataFrame(eval_df)
    eval_df.to_csv("data/eval.csv", index=None)
else:
    eval_df = pd.read_csv("data/eval.csv")

# %% [markdown]
# # Make 64x64 image segmentation dataset

# %%
# Stratification
trainval_df["obscured"] = trainval_df.apply(lambda r: r["ncn"] > 0.75, axis=1)
trainval_df["partial"] = trainval_df.apply(lambda r: 0.75 > r["ncn"] > 0.25, axis=1)
trainval_df["clear"] = trainval_df.apply(lambda r: r["ncn"] <= 0.25, axis=1)

# %%
# Sampling
_dfs = []

stratification = [("clear", 6400), ("partial", 3200), ("obscured", 3200)]

for label, items in stratification:
    _df = trainval_df[trainval_df[label] == True]
    _df = _df.sample(n=items, replace=items >= len(_df))
    _dfs.append(_df)

trainval_df_segmentation = pd.concat(_dfs)

for label, items in stratification:
    print(label, trainval_df_segmentation[label].value_counts())

# %%
# NPZs
it = tqdm.tqdm(
    trainval_df_segmentation.itertuples(), total=len(trainval_df_segmentation)
)

trainval_items = joblib.Parallel(n_jobs=10)(
    parallel_segmentation_sample(trainval_dataset, item) for item in it
)

random.shuffle(trainval_items)

trainval_images, trainval_labels = zip(*trainval_items)

trainval_images = np.stack(trainval_images, axis=0)

trainval_labels = np.stack(trainval_labels, axis=0)

# %%
trainval_images.shape, trainval_labels.shape

# %%
# Stratification
eval_df["obscured"] = eval_df.apply(lambda r: r["ncn"] > 0.75, axis=1)
eval_df["partial"] = eval_df.apply(lambda r: 0.75 > r["ncn"] > 0.25, axis=1)
eval_df["clear"] = eval_df.apply(lambda r: r["ncn"] <= 0.25, axis=1)

# %%
# Sampling
_dfs = []

stratification = [("clear", 640), ("partial", 320), ("obscured", 320)]

for label, items in stratification:
    _df = eval_df[eval_df[label] == True]
    _df = _df.sample(n=items, replace=items >= len(_df))
    _dfs.append(_df)

eval_df_segmentation = pd.concat(_dfs)

for label, items in stratification:
    print(label, eval_df_segmentation[label].value_counts())

# %%
# NPZs
it = tqdm.tqdm(eval_df_segmentation.itertuples(), total=len(eval_df_segmentation))

eval_items = joblib.Parallel(n_jobs=10)(
    parallel_segmentation_sample(eval_dataset, item) for item in it
)

random.shuffle(eval_items)

eval_images, eval_labels = zip(*eval_items)

eval_images = np.stack(eval_images, axis=0)

eval_labels = np.stack(eval_labels, axis=0)

# %%
# Save as dict of nparrays
dataset_path = Path("./data") / "toy_cloud_segmentation_2025.npz"

with open(dataset_path, "wb") as f:
    np.savez_compressed(
        f,
        train_images=trainval_images,
        train_labels=trainval_labels,
        test_images=eval_images,
        test_labels=eval_labels,
    )

# %%
# upload to gcp
cmd = "gcloud storage cp -r {} gs://fchouteau-isae-deep-learning/".format(
    str(dataset_path.resolve())
)
print(cmd)
subprocess.check_call(cmd, shell=True)

# %%
# !rm -rf /tmp/storage.googleapis.com/

# %%
# try to reload using numpy datasource
ds = np.lib.npyio.DataSource("/tmp/")
f = ds.open(
    "https://storage.googleapis.com/fchouteau-isae-deep-learning/toy_cloud_segmentation_2025.npz",
    "rb",
)
toy_dataset = np.load(f)
train_images = toy_dataset["train_images"]
train_labels = toy_dataset["train_labels"]
test_images = toy_dataset["test_images"]
test_labels = toy_dataset["test_labels"]

# %%
print(train_images.shape, test_images.shape)

# %%
print(train_labels.shape, test_labels.shape)

# %%
grid_size = 8
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        img = train_images[i * grid_size + j]
        msk = train_labels[i * grid_size + j]
        msk = class_mask_to_color_mask(msk, class_colors=[[0, 0, 0], [255, 0, 0]])
        tile = overlay_img_msk(img, msk)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = tile

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
ax.axis("off")
plt.savefig("cloud_samples_segmentation.png")
plt.show()

# %%
