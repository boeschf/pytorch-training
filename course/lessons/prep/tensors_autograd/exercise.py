"""Build a scalar squared-error objective for autograd.

Outcome: create a computation graph, reduce a tensor to a scalar, and obtain gradients.
Prerequisite: tensor shapes, indexing, dtype, and device selection.
Expected runtime: 15 minutes; self-check runs in under 10 seconds.
Resources: CPU; optionally one GPU; no network or downloaded data.
"""

from __future__ import annotations

import torch

from lessons.prep.tensor_device.reference import DeviceRequest
from lessons.prep.tensors_autograd.reference import autograd_report


def squared_error(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the sum of squared element-wise errors."""
    raise NotImplementedError("subtract targets, square each error, and reduce to one scalar")


def main(requested_device: DeviceRequest = "auto") -> int:
    """Run the participant objective through the gradient self-check."""
    try:
        report = autograd_report(squared_error, requested_device)
    except NotImplementedError as error:
        print(f"Status: INCOMPLETE — {error}")
        return 2
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
