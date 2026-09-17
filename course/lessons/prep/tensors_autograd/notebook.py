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
# # Tensors and autograd
#
# This lesson adapts the repository's original PyTorch overview into a bounded,
# executable gradient check.

# %%
import torch

from lessons.prep.tensors_autograd.reference import autograd_report
from lessons.prep.tensors_autograd.solution import squared_error

# %%
values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("shape:", values.shape)
print("dtype:", values.dtype)
print("device:", values.device)
print("second column:", values[:, 1])

# %% [markdown]
# Autograd records operations involving tensors with `requires_grad=True`.
# `backward()` starts at a scalar and accumulates derivatives in leaf tensors.

# %%
inputs = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
targets = torch.tensor([0.5, -1.0, 2.0])
loss = squared_error(inputs, targets)
loss.backward()
print("loss:", loss.item())
print("gradient:", inputs.grad)
print("expected:", 2 * (inputs.detach() - targets))

# %%
report = autograd_report(squared_error, "cpu")
print(report)
assert report["status"] == "pass"
