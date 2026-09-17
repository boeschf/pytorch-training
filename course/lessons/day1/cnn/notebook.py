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
# # Compact convolutional classifier
#
# The legacy CIFAR-10 notebook required a download and a long training run. This
# adaptation keeps its image → convolution → pooling → logits learning path, but
# uses deterministic 8×8 stripe images so every participant can run it offline.

# %%
from pprint import pprint

import matplotlib.pyplot as plt

from lessons.day1.cnn.solution import build_cnn
from lessons.day1.cnn.training import make_image_dataset, run_cnn

# %%
split = make_image_dataset(samples=256, seed=11)
print("training images:", split.train_images.shape)
print("training targets:", split.train_targets.shape)
print(build_cnn())

# %%
figure, axes = plt.subplots(2, 4, figsize=(8, 4))
for axis, image, target in zip(
    axes.flat,
    split.train_images[:8],
    split.train_targets[:8],
    strict=True,
):
    axis.imshow(image.squeeze(0), cmap="gray", vmin=-0.4, vmax=1.4)
    axis.set_title(f"class {target.item()}")
    axis.axis("off")
figure.suptitle("Offline stripe-image samples")
figure.tight_layout()
plt.show()

# %% [markdown]
# Shape flow: `[batch, 1, 8, 8] → Conv2d → [batch, 4, 8, 8] → Pool →
# [batch, 4, 4, 4] → Flatten → [batch, 64] → Linear → [batch, 2]`.

# %%
run = run_cnn(build_model=build_cnn, requested_device="cpu")
pprint(run.report)
assert run.report["status"] == "pass"
assert run.report["eval_accuracy"] >= 0.95

# %%
epochs = [snapshot.epoch for snapshot in run.history]
eval_losses = [snapshot.eval_loss for snapshot in run.history]
accuracies = [snapshot.eval_accuracy for snapshot in run.history]
figure, loss_axis = plt.subplots(figsize=(7, 4))
accuracy_axis = loss_axis.twinx()
loss_axis.plot(epochs, eval_losses, marker="o", color="tab:blue", label="evaluation loss")
accuracy_axis.plot(epochs, accuracies, marker="s", color="tab:orange", label="accuracy")
loss_axis.set(xlabel="epoch", ylabel="cross-entropy loss")
accuracy_axis.set(ylabel="held-out accuracy", ylim=(0, 1.05))
figure.suptitle("CNN evaluation signal")
figure.tight_layout()
plt.show()
