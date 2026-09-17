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
# # From tensors to a trained MLP
#
# This notebook is a thin educational view over the tested lesson. The complete
# implementation is beside it in `reference.py`; the notebook inspects inputs and
# results without maintaining another training loop.

# %%
from pprint import pprint

from lessons.day1.mlp.reference import MLP, make_dataset, train_mlp
from lessons.prep.tensor_device.reference import tensor_device_report

# %% [markdown]
# ## Check tensors and the CPU device
#
# This checked-in path is intentionally CPU-only. The same diagnostic can select
# CUDA under the `cuda` or `alps-gh200` profile.

# %%
tensor_report = tensor_device_report("cpu")
pprint(tensor_report)
assert tensor_report["status"] == "pass"

# %% [markdown]
# ## Inspect the deterministic classification problem
#
# Four noisy clusters form an XOR-like problem. A linear classifier cannot
# separate both classes, so the MLP needs its hidden layer and nonlinearity.

# %%
split = make_dataset(samples=512, seed=7)
print("train:", split.train_features.shape, split.train_targets.shape)
print("evaluation:", split.eval_features.shape, split.eval_targets.shape)
print("first features:\n", split.train_features[:4])
print("first targets:", split.train_targets[:4])

# %%
model = MLP()
print(model)

# %% [markdown]
# ## Train and evaluate
#
# `train_mlp` executes the reference loop used by the training CLI, smoke runs,
# and slides. The adjacent exercise and solution focus on one optimization step.

# %%
metrics = train_mlp(requested_device="cpu", seed=7)
pprint(metrics)
assert metrics["status"] == "pass"
assert metrics["final_train_loss"] < metrics["initial_train_loss"]
assert metrics["eval_accuracy"] >= 0.95

# %% [markdown]
# The learning signal is observable: training loss decreases and held-out
# accuracy exceeds the contract. The next exercise asks you to implement the
# five operations in one optimization step yourself.
