"""Construct a compact convolutional image classifier.

Outcome: connect convolution, activation, pooling, flattening, and classification.
Prerequisite: the MLP lesson plus Dataset and DataLoader preparation.
Expected runtime: 25 minutes; CPU self-check completes in under one minute.
Resources: CPU or one GPU; synthetic offline images; no dataset download.
"""

from __future__ import annotations

from torch import nn

from lessons.day1.cnn.training import run_cnn
from lessons.prep.tensor_device.reference import DeviceRequest


def build_cnn() -> nn.Module:
    """Return a model mapping `[batch, 1, 8, 8]` images to two logits."""
    raise NotImplementedError(
        "compose Conv2d, ReLU, MaxPool2d, Flatten, and Linear for two classes"
    )


def main(requested_device: DeviceRequest = "auto") -> int:
    """Train the participant model and check held-out classification."""
    try:
        report = run_cnn(build_model=build_cnn, requested_device=requested_device).report
    except NotImplementedError as error:
        print(f"Status: INCOMPLETE — {error}")
        return 2
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
