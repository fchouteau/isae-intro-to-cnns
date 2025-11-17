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
if not Path("data/eval_full.csv").exists():
    it = tqdm.tqdm(range(len(eval_dataset)))

    eval_df = joblib.Parallel(n_jobs=8)(
        parallel_metadata(eval_dataset, k, 256, True) for k in it
    )

    eval_df = [i for l in eval_df for i in l]
    eval_df = pd.DataFrame(eval_df)
    eval_df.to_csv("data/eval_full.csv", index=None)
else:
    eval_df = pd.read_csv("data/eval_full.csv")

# %%
# Stratification
eval_df["obscured"] = eval_df.apply(lambda r: r["ncn"] > 0.75, axis=1)
eval_df["partial"] = eval_df.apply(lambda r: 0.75 > r["ncn"] > 0.25, axis=1)
eval_df["clear"] = eval_df.apply(lambda r: r["ncn"] <= 0.25, axis=1)

# %%
# Sampling
_dfs = []

stratification = [("clear", 16), ("partial", 8), ("obscured", 8)]

for label, items in stratification:
    _df = eval_df[eval_df[label] == True]
    _df = _df.sample(n=items, replace=items >= len(_df), random_state=2025)
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

eval_tiles = np.stack(eval_images, axis=0)

eval_labels = np.stack(eval_labels, axis=0)

# %%
# Save as dict of nparrays
dataset_path = Path("./data") / "tiles_cloud_segmentation_dataset_2025.npz"

with open(dataset_path, "wb") as f:
    np.savez_compressed(
        f,
        eval_tiles=eval_tiles,
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
    "https://storage.googleapis.com/fchouteau-isae-deep-learning/tiles_cloud_segmentation_dataset_2025.npz",
    "rb",
)
toy_dataset = np.load(f)
eval_tiles = toy_dataset["eval_tiles"]

# %%
eval_tiles.shape

# %%
grid_size = 4
grid = np.zeros((grid_size * 256, grid_size * 256, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = eval_tiles[i * grid_size + j]
        grid[i * 256 : (i + 1) * 256, j * 256 : (j + 1) * 256, :] = tile

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
ax.axis("off")
plt.savefig("cloud_tiles.png")
plt.show()

# %%
