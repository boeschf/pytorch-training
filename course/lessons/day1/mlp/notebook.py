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
# implementation is beside it in `training.py`; the notebook inspects inputs,
# learning curves, and the learned classification boundary.

# %%
from pprint import pprint

import matplotlib.pyplot as plt
import torch

from lessons.day1.mlp.training import MLP, make_dataset, run_mlp
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
# `run_mlp` is the single training loop used by the CLI, notebook, smoke runs,
# and exercise. It defaults to the completed step from `solution.py`; the
# exercise injects the participant's implementation and a progress callback.

# %%
run = run_mlp(requested_device="cpu", seed=7)
metrics = run.report
pprint(metrics)
assert metrics["status"] == "pass"
assert metrics["final_train_loss"] < metrics["initial_train_loss"]
assert metrics["eval_accuracy"] >= 0.95

# %% [markdown]
# ## Plot the learning curves
#
# Training and evaluation loss should decrease together. A widening gap would
# indicate that the model is fitting the training samples without generalizing.

# %%
epochs = [snapshot.epoch for snapshot in run.history]
train_losses = [snapshot.train_loss for snapshot in run.history]
eval_losses = [snapshot.eval_loss for snapshot in run.history]

figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(epochs, train_losses, marker="o", label="training")
axis.plot(epochs, eval_losses, marker="o", label="evaluation")
axis.set(xlabel="epoch", ylabel="cross-entropy loss", title="MLP learning curves")
axis.grid(alpha=0.3)
axis.legend()
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Plot the learned classifier
#
# The background shows the model's predicted class across the input plane. The
# points are held-out examples colored by their true class.

# %%
eval_features = run.split.eval_features
eval_targets = run.split.eval_targets
plot_limit = (
    max(
        run.split.train_features.abs().max().item(),
        eval_features.abs().max().item(),
    )
    + 0.2
)
axis_values = torch.linspace(-plot_limit, plot_limit, 200)
grid_x, grid_y = torch.meshgrid(axis_values, axis_values, indexing="xy")
grid = torch.column_stack((grid_x.ravel(), grid_y.ravel())).to(run.device)

with torch.no_grad():
    grid_predictions = run.model(grid).argmax(dim=1).cpu().reshape(grid_x.shape)
    eval_predictions = run.model(eval_features.to(run.device)).argmax(dim=1).cpu()

accuracy = (eval_predictions == eval_targets).float().mean().item()
figure, axis = plt.subplots(figsize=(6, 5))
axis.contourf(
    grid_x.numpy(),
    grid_y.numpy(),
    grid_predictions.numpy(),
    levels=(-0.5, 0.5, 1.5),
    cmap="coolwarm",
    alpha=0.25,
)
colors = plt.get_cmap("coolwarm")
for class_id in (0, 1):
    class_points = eval_features[eval_targets == class_id]
    axis.scatter(
        class_points[:, 0].numpy(),
        class_points[:, 1].numpy(),
        color=colors(float(class_id)),
        edgecolors="black",
        label=f"true class {class_id}",
    )
axis.set(
    xlabel="feature 1",
    ylabel="feature 2",
    title=f"Held-out classification — accuracy {accuracy:.1%}",
)
axis.set_aspect("equal")
axis.legend(title="held-out labels")
figure.tight_layout()
plt.show()

# %% [markdown]
# The loss curves show optimization over time; the decision regions show what
# the final model learned. The next exercise asks you to implement the five
# operations in one optimization step yourself.
