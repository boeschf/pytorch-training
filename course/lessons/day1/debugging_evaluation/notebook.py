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
# # Debugging and evaluation
#
# Inspect invariants before changing the optimizer. Most Day 1 failures are
# visible in shape, dtype, device, model state, or the held-out learning curve.

# %%
import torch

from lessons.day1.cnn.solution import build_cnn
from lessons.day1.debugging_evaluation.reference import (
    classification_issues,
    debugging_report,
    evaluation_accuracy,
)
from lessons.day1.debugging_evaluation.solution import repair_batch

# %%
model = build_cnn()
model.eval()
broken_images = torch.zeros(8, 8, 8, dtype=torch.int32)
broken_targets = torch.arange(8, dtype=torch.float32) % 2
print(*classification_issues(model, broken_images, broken_targets), sep="\n- ")

# %%
images, targets = repair_batch(broken_images, broken_targets)
print("fixed:", images.shape, images.dtype, targets.shape, targets.dtype)
print("issues:", classification_issues(model, images, targets))
print("accuracy is now measurable:", evaluation_accuracy(model, images, targets))

# %% [markdown]
# A valid batch is not evidence of learning. Compare training and held-out loss
# over time; confirm both the pre-training baseline and final result; then inspect
# examples where predictions and targets disagree.

# %%
report = debugging_report(repair_batch)
print(report)
assert report["status"] == "pass"
