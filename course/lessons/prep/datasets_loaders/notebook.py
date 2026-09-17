# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: "CSCS PyTorch Course \u2014 CPU"
#     language: python
#     name: cscs-pytorch-course-cpu
# ---

# %% [markdown]
# # Datasets and data loaders
#
# A `Dataset` defines one sample. A `DataLoader` defines how samples become an
# epoch of mini-batches.

# %%
import torch
from torch.utils.data import DataLoader

from lessons.prep.datasets_loaders.reference import dataset_loader_report, make_points
from lessons.prep.datasets_loaders.solution import PairDataset

# %%
features, targets = make_points(samples=20)
dataset = PairDataset(features, targets)
print("samples:", len(dataset))
print("one sample:", dataset[3])

# %%
generator = torch.Generator().manual_seed(7)
loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator, num_workers=0)
for batch_id, (batch_features, batch_targets) in enumerate(loader):
    print(batch_id, batch_features.shape, batch_targets.shape)

# %% [markdown]
# Seeding the DataLoader's own generator makes shuffle order reproducible without
# coupling it to model initialization. `num_workers=0` is the portable preparation path.

# %%
report = dataset_loader_report(PairDataset)
print(report)
assert report["status"] == "pass"
